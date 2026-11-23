"""Teach/correct scoring: compare a live attempt against a known target
sign's reference, instead of open-set classification.

This is a closed-set problem (the user was shown `target_word` and asked
to sign it), so it should use a more targeted, more robust signal than the
coarse 100-way classifier softmax: cosine similarity between the attempt's
pooled embedding and the target word's reference centroid, computed in
EXACTLY the same space the model was trained in (forward_features, same
preprocessing as scripts/build_sign_references.py).

Preprocessing must match build_sign_references.py / app/backend/main.py's
run_inference exactly:
  1. canonicalize_handedness(raw, target="right")   -- on RAW keypoints
  2. normalize_per_part(canonicalized)
  3. pad/truncate to MAX_SEQ_LEN=150, run through the same checkpoint
Any divergence here would make live-attempt embeddings live in a
different space than the reference centroids, silently invalidating the
cosine similarity.
"""

from typing import Optional

import json
from pathlib import Path

import numpy as np
import torch

from mouna.coach.reliability import check_reliability
from mouna.coach import cp_profile
from mouna.coach.parameters import compare_discrete, extract_dominant_hand_signal, find_divergent_frame
from mouna.data.handedness import canonicalize_handedness, detect_dominant_hand
from mouna.data.part_normalize import normalize_per_part

MAX_SEQ_LEN = 150  # must match build_sign_references.py and main.py's run_inference

# ---------------------------------------------------------------------------
# Thresholds -- chosen empirically, not guessed.
#
# We measured cosine similarity between individual confirmed-ok clip
# embeddings and their OWN word's reference centroid ("within-word") vs.
# against OTHER words' centroids ("across-word"), over all 555 confirmed-ok
# clips across the 20 currently-covered words:
#
#   within-word : mean 0.868, std 0.149, p10 = 0.718
#   across-word : mean 0.027, std 0.151, p90 = 0.212, max 0.698 (one outlier)
#
# The two distributions are well separated with a gap roughly between 0.3
# and 0.6 (across-word p90 ~0.21, within-word p10 ~0.72), so:
#   >= GOOD_SIMILARITY_THRESHOLD        -> looks like a solid attempt
#   >= WRONG_SIGN_SIMILARITY_THRESHOLD  -> recognizable but needs work
#   <  WRONG_SIGN_SIMILARITY_THRESHOLD  -> essentially in "different sign"
#                                          territory (across-word range)
GOOD_SIMILARITY_THRESHOLD = 0.6
WRONG_SIGN_SIMILARITY_THRESHOLD = 0.3

# Coarse classifier confidence gate -- matches CONF_THRESH in app/backend/main.py
# so "coarse_prediction" is reported consistently with the existing /ws/detect
# behavior (below this the coarse head itself doesn't trust its own top-1).
COARSE_CONF_THRESH = 0.25


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def _hand_count(label: str) -> str:
    """Collapse detect_dominant_hand's output to what actually matters for
    a hand-count comparison: 'two' (both hands genuinely active), 'one'
    (left or right only), or 'none'. 'left' vs 'right' is NOT a real
    mismatch -- canonicalize_handedness already normalizes which hand is
    "dominant" to a fixed convention, so comparing raw 'left'/'right'
    labels between an attempt and a reference would flag false positives.
    What's real is whether one of them used both hands and the other
    didn't."""
    if label == "both":
        return "two"
    if label in ("left", "right"):
        return "one"
    return "none"  # 'neither' -- no hand detected


def score_attempt(
    raw_keypoints_225: np.ndarray,
    target_word: str,
    references: dict,
    model,
    idx_to_label: Optional[dict] = None,
    reference_bucket_index: Optional[dict] = None,
) -> dict:
    """Score a live attempt's RAW (pre-canonicalization) keypoints against
    `target_word`'s reference.

    Args:
        raw_keypoints_225: (T, 225) raw MediaPipe keypoints, NOT yet
            canonicalized or normalized -- same convention as the
            `keypoints` payload already accepted by /ws/detect.
        target_word: the word the user was shown and asked to sign.
        references: the dict loaded from data/references/sign_references.pkl
            (word -> {"embedding_centroid", "representative_sequence", "n_references"}).
        model: loaded PoseTransformer (eval mode), same checkpoint as
            build_sign_references.py / main.py (data/best_model_v4canon.pth).
        idx_to_label: index->label map from the checkpoint
            (ckpt["idx_to_label"]). If omitted, falls back to
            `model.idx_to_label` if the caller stashed it there.

    Returns:
        {
          "target": str,
          "coarse_prediction": str,
          "coarse_confidence": float,
          "embedding_similarity": float,
          "hand_count_match": bool,
          "parameters": {
              "handshape":   {"target": str, "yours": str, "match": bool},
              "location":    {"target": str, "yours": str, "match": bool},
              "orientation": {"target": str, "yours": str, "match": bool},
          } | None,  # None if either side's hand couldn't be read for geometry
          "overall_verdict": "good" | "needs_work" | "wrong_sign",
          "notes": [str, ...],
        }

    `parameters` is diagnostic/explanatory ONLY, same stance as the coarse
    classifier below -- it does not affect `overall_verdict`. Only 3
    parameters, not the full 5: movement and facial expression failed/haven't
    passed the same within-word-vs-across-word validation the other 3 did
    (see mouna.coach.parameters's module docstring) and are deliberately
    left out rather than shown unvalidated.
    """
    if idx_to_label is None:
        idx_to_label = getattr(model, "idx_to_label", None)
    if idx_to_label is None:
        raise ValueError(
            "idx_to_label not provided and not found on model -- pass it explicitly "
            "or set model.idx_to_label after loading the checkpoint."
        )

    if target_word not in references:
        # Nothing to compare against -- this word hasn't been through QA yet
        # (build_sign_references.py only covers words with confirmed-ok clips).
        return {
            "target": target_word,
            "coarse_prediction": None,
            "coarse_confidence": None,
            "embedding_similarity": None,
            "hand_count_match": None,
            "parameters": None,
            "divergent_frames": {},
            "overall_verdict": "needs_work",
            "notes": [f"No reference data yet for '{target_word}' -- can't score this sign."],
        }

    raw = np.asarray(raw_keypoints_225, dtype=np.float32)

    # --- same preprocessing pipeline as training / build_sign_references.py ---
    canon = canonicalize_handedness(raw, target="right")
    normed = normalize_per_part(canon)

    T = min(len(normed), MAX_SEQ_LEN)
    padded = normed[:T]
    if T < MAX_SEQ_LEN:
        padded = np.vstack(
            [padded, np.zeros((MAX_SEQ_LEN - T, padded.shape[1]), dtype=np.float32)]
        )
    device = next(model.parameters()).device
    x = torch.from_numpy(padded).unsqueeze(0).float().to(device)
    lengths = torch.tensor([T], device=device)

    with torch.no_grad():
        pooled = model.forward_features(x, lengths)  # (1, d_model) -- same call model.forward() makes internally
        embedding = pooled[0].cpu().numpy()
        logits = model.classifier(pooled)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top1_idx = int(np.argmax(probs))
    coarse_prediction = idx_to_label[top1_idx]
    coarse_confidence = float(probs[top1_idx])

    # --- embedding similarity vs. the target's reference centroid ---
    centroid = references[target_word]["embedding_centroid"]
    embedding_similarity = _cosine_similarity(embedding, centroid)

    # --- hand-count match ---
    # Both sides must be read from RAW (pre-normalize_per_part) keypoints --
    # detect_dominant_hand's own docstring requires this (it checks for exact
    # zero to mean "undetected," which normalize_per_part's interpolation
    # would otherwise mask). representative_sequence_raw_canon (added by
    # build_sign_references_v2.py) is the reference-side raw form; the old
    # representative_sequence is already part-normalized and was being fed
    # into this call incorrectly before that field existed.
    reference_raw_canon = references[target_word].get("representative_sequence_raw_canon")
    attempt_hand_raw = detect_dominant_hand(raw)
    reference_hand_raw = detect_dominant_hand(
        reference_raw_canon if reference_raw_canon is not None else references[target_word]["representative_sequence"]
    )
    attempt_count = _hand_count(attempt_hand_raw)
    reference_count = _hand_count(reference_hand_raw)
    hand_count_match = attempt_count == reference_count

    # --- decomposed parameter breakdown (handshape/location/orientation) ---
    # Diagnostic only -- see this function's docstring. None if either side's
    # hand geometry can't be read (e.g. attempt never showed a hand).
    parameters = None
    divergent_frames: dict = {}
    looks_like: Optional[str] = None
    if reference_raw_canon is not None:
        attempt_signal = extract_dominant_hand_signal(canon)
        reference_signal = extract_dominant_hand_signal(reference_raw_canon)
        if attempt_signal is not None and reference_signal is not None:
            parameters = compare_discrete(attempt_signal, reference_signal, references[target_word].get("target_buckets"))
            # Only for params that actually mismatched -- a "where it went
            # wrong" snapshot for something that matched isn't useful.
            for param, result in parameters.items():
                if result["match"] is False:
                    frame_idx = find_divergent_frame(attempt_signal, reference_signal, param)
                    if frame_idx is not None:
                        divergent_frames[param] = frame_idx

            # "This looks more like X" -- real, not hypothetical: a 5-way
            # bucket collision (basketball/bed/decide/hot/need) was found
            # and confirmed against actual reference data in the
            # 2026-11-01 research pass. If the attempt's own 3-parameter
            # bucket triple matches a DIFFERENT known word, surface it so
            # the learner can see exactly which other sign they drifted
            # toward -- not just that this one didn't match.
            # Only when the attempt actually MISSED the target on at least
            # one parameter: if it matched all three, every other word in
            # that bucket ties with the target (e.g. who/apple/orange/how
            # share one triple), so naming one of them as "closer" is false.
            if reference_bucket_index and any(r["match"] is False for r in parameters.values()):
                attempt_triple = (
                    parameters["handshape"]["yours"],
                    parameters["location"]["yours"],
                    parameters["orientation"]["yours"],
                )
                candidates = [w for w in reference_bucket_index.get(attempt_triple, []) if w != target_word]
                if candidates:
                    looks_like = sorted(candidates)[0]

    # --- verdict ---
    # Design decision: the coarse classifier (open-set, 100-way softmax)
    # does NOT get to override the verdict at all -- it is reported and
    # surfaced as a note for diagnostic context only. Reasoning: the coarse
    # head is a much blunter instrument here -- it's making a 1-of-100
    # decision, while embedding_similarity is a targeted 1-of-1 comparison
    # against the specific sign the user was actually asked to perform. A
    # correct-but-atypical attempt can easily lose a 100-way argmax to a
    # visually-similar neighbor sign while still sitting well inside that
    # target's embedding neighborhood (see the within-word std of 0.149
    # above -- there's real spread even among confirmed-correct clips; our
    # own self-test hit exactly this case -- see compare.py's module
    # self-test notes). So embedding_similarity against the KNOWN target,
    # together with hand_count_match, is the whole verdict signal.
    notes: list[str] = []
    reliability = check_reliability()
    if parameters:
        for name, result in parameters.items():
            result["reliable"] = reliability.get(name, {}).get("reliable", True)

    if embedding_similarity >= GOOD_SIMILARITY_THRESHOLD:
        verdict = "good"
    elif embedding_similarity >= WRONG_SIGN_SIMILARITY_THRESHOLD:
        verdict = "needs_work"
    else:
        verdict = "wrong_sign"

    # Only worth surfacing "this looks like X" when the attempt ISN'T
    # already judged good -- a correct attempt landing in the same coarse
    # bucket as another word is diagnostic noise, not something to flag.
    if verdict == "good":
        looks_like = None

    variation_hands = references[target_word].get("variation_hands") or []
    attempt_n = {"one": 1, "two": 2}.get(attempt_count)
    if not hand_count_match and attempt_n in variation_hands:
        # The learner signed the sign's accepted other form (e.g. one-handed
        # SAME). Say so, rather than calling it a wrong hand count.
        notes.append(
            f"used the {'one' if attempt_n == 1 else 'two'}-handed version of this sign -- that's correct ASL too, "
            f"but practice here uses the {'two' if attempt_n == 1 else 'one'}-handed version shown in the example."
        )
    elif not hand_count_match:
        notes.append(
            f"used {'two hands' if attempt_count == 'two' else ('one hand' if attempt_count == 'one' else 'no detected hand')}, "
            f"this sign typically uses {'two hands' if reference_count == 'two' else ('one hand' if reference_count == 'one' else 'a hand')}."
        )
        # A hand-count mismatch blocks a "good" verdict only when the check is
        # measured as reliable (mouna.coach.reliability). Benchmark: it wrongly
        # flagged 22% of correct signing, which alone graded 1 in 5 correct
        # attempts "needs work" -- so until it improves it's only a hint.
        if verdict == "good" and reliability["hand_count"]["reliable"]:
            verdict = "needs_work"

    if coarse_prediction != target_word:
        notes.append(
            f"coarse classifier's top guess was '{coarse_prediction}' "
            f"({coarse_confidence:.2f} confidence), not '{target_word}' -- "
            "informational only, embedding similarity drives the verdict."
        )

    if parameters:
        for name, result in parameters.items():
            if result["match"] is False and result.get("why") and result.get("reliable", True):
                notes.append(result["why"])

    if looks_like:
        notes.append(f"Your attempt's handshape/location/orientation actually matched '{looks_like}' more closely than '{target_word}'.")

    if not notes:
        notes.append("Looks good -- matches the target sign's reference closely.")

    return {
        "target": target_word,
        "coarse_prediction": coarse_prediction,
        "coarse_confidence": coarse_confidence,
        "embedding_similarity": embedding_similarity,
        "check_reliability": reliability,
        "cp_findings": cp_findings(canon, target_word, reliability),
        "hand_count_match": hand_count_match,
        "parameters": parameters,
        "divergent_frames": divergent_frames,
        "looks_like": looks_like,
        "overall_verdict": verdict,
        "notes": notes,
    }


_PROFILES: dict | None = None
_FAMILY = {"handshape": "fingers", "location": "location_profile", "movement": "movement_profile"}


def cp_findings(canon: np.ndarray, target_word: str, reliability: dict) -> list[dict]:
    """The learner's own CPs vs the range fluent signers use for this sign
    (mouna.coach.cp_profile): what specifically differs, each marked reliable
    or not by the judgement benchmark."""
    global _PROFILES
    if _PROFILES is None:
        try:
            _PROFILES = json.loads((Path(__file__).resolve().parents[3] / "data/references/cp_profiles.json").read_text())
        except (OSError, ValueError):
            _PROFILES = {}
    prof = _PROFILES.get(target_word)
    feats = cp_profile.features(canon) if prof else None
    if not feats:
        return []
    out = []
    for f in cp_profile.compare(feats, prof, target_word):
        fam = _FAMILY.get(f["part"], "movement_profile")
        out.append({**f, "reliable": bool(reliability.get(fam, {}).get("reliable", False))})
    return out
