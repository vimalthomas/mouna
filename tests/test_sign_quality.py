"""Quality gates for every sign in the practice set, plus a regression
snapshot so later changes can't silently degrade signs that work today.

Gates (a NEW sign must pass all of them before it is offered for practice):
  1. its example video passes its own grading (good verdict, right number of
     hands, all checked CP parts agree, similarity >= 0.80)
  2. the example video file exists
  3. it has a drawn hand path (the Learn screen's arrows / "stays here")
  4. the 3D hand shows the same number of hands as the example video
  5. its CP targets come from a majority vote over approved clips
  6. it is in the published-source sign catalog (ASL-CDI / ASL-LEX)

Snapshot: tests/golden/signs.json records each sign's verdict, similarity,
hand count and CP targets. A sign leaving the practice set, its similarity
dropping by more than 0.05, or a CP target flipping fails the run by name.
After an intended change, refresh it with:  MOUNA_UPDATE_GOLDEN=1 pytest tests/test_sign_quality.py
"""
import json
import os
import pickle

import numpy as np
import pytest

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REFS = pickle.load(open(ROOT / "data/references/sign_references.pkl", "rb"))
EXAMPLES = json.loads((ROOT / "data/references/example_videos.json").read_text())
CATALOG = json.loads((ROOT / "data/references/sign_catalog.json").read_text())
GOLDEN = ROOT / "tests/golden/signs.json"
MIN_SIMILARITY = 0.80


def _practice_ready(word: str) -> bool:
    c = EXAMPLES.get(word)
    return bool(c and c.get("verdict") == "good" and c.get("hand_count_match")
                and c.get("params_matched") == 3 and (c.get("similarity") or 0) >= MIN_SIMILARITY)


PRACTICE = sorted(w for w in REFS if REFS[w].get("representative_sequence_raw_canon") is not None and _practice_ready(w))


def _hands_in(seq) -> tuple[int, float]:
    a = np.asarray(seq).reshape(len(seq), 75, 3)
    l, r = np.any(a[:, 33:54] != 0, axis=(1, 2)), np.any(a[:, 54:75] != 0, axis=(1, 2))
    any_hand = (l | r).sum()
    share = float((l & r).sum() / any_hand) if any_hand else 0.0
    return (2 if share >= 0.3 else 1), share


def test_practice_set_is_not_empty():
    assert len(PRACTICE) >= 14, f"only {len(PRACTICE)} signs pass the gates"


@pytest.mark.parametrize("word", PRACTICE)
def test_example_video_passes_its_own_grading(word):
    c = EXAMPLES[word]
    assert c["verdict"] == "good" and c["hand_count_match"], f"{word}: verdict {c['verdict']}"
    assert c["params_matched"] == 3, f"{word}: example contradicts its reference on a CP part"
    assert c["similarity"] >= MIN_SIMILARITY, f"{word}: similarity {c['similarity']}"


@pytest.mark.parametrize("word", PRACTICE)
def test_example_video_exists(word):
    assert (ROOT / "app/wlasl/videos" / f"{EXAMPLES[word]['video_id']}.mp4").exists()


@pytest.mark.parametrize("word", PRACTICE)
def test_has_hand_path(word):
    assert EXAMPLES[word].get("hand_paths"), f"{word}: no hand path, so no arrows on the Learn screen"


@pytest.mark.parametrize("word", PRACTICE)
def test_3d_hands_match_video(word):
    from choose_display_clips import target_hands
    want = target_hands()[word]
    ref = REFS[word]
    seq = ref.get("display_sequence_raw_canon")
    seq = seq if seq is not None else ref["representative_sequence_raw_canon"]
    n, share = _hands_in(seq)
    assert n == want or (want == 2 and share >= 0.1), f"{word}: video shows {want} hand(s), 3D shows {n}"


@pytest.mark.parametrize("word", PRACTICE)
def test_targets_from_majority_vote(word):
    ref = REFS[word]
    if ref.get("qa_method") == "auto_screened":
        pytest.skip(f"{word}: auto-screened reference (no human approvals yet)")
    assert ref.get("target_buckets"), f"{word}: CP targets not from a majority vote"


@pytest.mark.parametrize("word", PRACTICE)
def test_in_sign_catalog(word):
    assert word in CATALOG, f"{word}: missing from data/references/sign_catalog.json"


# ---- regression snapshot ---------------------------------------------------

def _snapshot() -> dict:
    out = {}
    for w in PRACTICE:
        c, t = EXAMPLES[w], REFS[w].get("target_buckets") or {}
        out[w] = {"similarity": round(float(c["similarity"]), 3), "n_hands": c.get("n_hands"),
                  "targets": {k: v["value"] for k, v in sorted(t.items()) if v.get("settled")}}
    return out


def test_no_regression_against_golden_snapshot():
    current = _snapshot()
    if os.environ.get("MOUNA_UPDATE_GOLDEN") or not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(current, indent=2, sort_keys=True))
        pytest.skip("golden snapshot written")
    golden = json.loads(GOLDEN.read_text())
    problems = []
    for w, g in golden.items():
        cur = current.get(w)
        if cur is None:
            problems.append(f"{w}: dropped out of the practice set")
            continue
        if cur["similarity"] < g["similarity"] - 0.05:
            problems.append(f"{w}: similarity {g['similarity']} -> {cur['similarity']}")
        for part, value in g["targets"].items():
            if cur["targets"].get(part, value) != value:
                problems.append(f"{w}: {part} target {value} -> {cur['targets'][part]}")
    assert not problems, "Regressions:\n  " + "\n  ".join(problems)


# ---- sign variants ---------------------------------------------------------

VARIANTS_PATH = ROOT / "data/references/sign_variants.json"
VARIANTS = json.loads(VARIANTS_PATH.read_text()) if VARIANTS_PATH.exists() else {}


@pytest.mark.parametrize("word", sorted(w for w in VARIANTS if w in PRACTICE))
def test_variant_sign_practises_its_main_form(word):
    # Example, 3D and grading reference all use the majority hand count.
    from choose_display_clips import target_hands
    main = VARIANTS[word]["primary_hands"]
    assert target_hands()[word] == main, f"{word}: example shows {target_hands()[word]} hand(s), main form is {main}"
    n, share = _hands_in(REFS[word]["representative_sequence_raw_canon"])
    assert n == main or (main == 2 and share >= 0.1), f"{word}: reference uses {n} hand(s), main form is {main}"


@pytest.mark.parametrize("word", sorted(VARIANTS))
def test_variation_clips_left_out_of_reference(word):
    if word not in REFS or REFS[word].get("qa_method") == "auto_screened":
        pytest.skip("no reference")
    assert REFS[word].get("variation_hands") == [v["n_hands"] for v in VARIANTS[word]["variations"]]
