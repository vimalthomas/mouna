"""Decomposed, target-relative parameter scoring: handshape, location,
orientation -- computed directly from keypoint geometry, no retraining
required. Supplements (does not replace) compare.py's embedding-similarity
verdict, which stays the single source of truth for pass/fail -- these are
diagnostic/explanatory, same design stance already taken for the coarse
classifier.

These are DISCRETE bucket classifiers, not continuous distance scores. An
earlier continuous version (mean distance, DTW, etc. -- see
scripts/calibrate_parameter_thresholds.py) was tried first and showed NO
clean within/across-sign separation for any of 4 metrics. Discretizing the
same underlying geometry into coarse buckets recovered real signal for 3 of
4 (validated via bucket-agreement rate: do within-word clip pairs land in
the same bucket more often than across-word pairs? -- see
scripts/calibrate_discrete_buckets.py):
  handshape   (open/half/closed)              +23.6pp gap -- real
  orientation (dominant palm-normal axis)      +25.6pp gap -- real
  location    (face/chest/neutral region)      +11.9pp gap -- real
  movement    (speed tertile)                  +4.7pp gap  -- NOT real, excluded

Movement is intentionally absent from this module. Speed-only bucketing
didn't separate signs (+4.7pp). A direction-based version (net
displacement's dominant axis+sign, same style as orientation, see
scripts/calibrate_movement_direction.py) was also tried and also failed
(+4.4pp, within=0.286 vs across=0.242) -- flattening a clip to a single
start->end vector loses too much of the real path shape (curves, repeated
taps) to separate signs. Don't add a movement score here without the
same validation the other three went through; if retried, a real path-
shape signal (e.g. DTW-aligned trajectory comparison, not a single
summary vector) is the more promising untried direction.

Must operate on keypoints AFTER canonicalize_handedness but BEFORE
normalize_per_part -- see part_normalize.py's docstring for why body/hand
normalization is kept separate; location needs hand-vs-body relative
position, which normalize_per_part's hand-centering throws away.

Reference side: scripts/build_sign_references_v2.py stores this exact
form as references[word]["representative_sequence_raw_canon"].
"""

from typing import Optional

import numpy as np

from mouna.data.part_normalize import _interpolate_missing  # reuse validated gap-filling, not reimplement it

WRIST = 0
MIDDLE_MCP = 9
INDEX_MCP = 5
PINKY_MCP = 17
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
NOSE = 0

FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}

# --- empirically validated bucket boundaries (scripts/calibrate_discrete_buckets.py) ---
HANDSHAPE_LOW_CUT = 1.345   # openness tertile cutoffs, from the 554-clip calibration run
HANDSHAPE_HIGH_CUT = 1.692
LOCATION_NEUTRAL_MIN_DIST = 0.9  # beyond this from BOTH face and chest anchors -> neutral_space


def _body_frame(pose_raw: np.ndarray):
    """pose_raw: (T, 33, 3) canonicalized raw pose. Returns mid_shoulder (T,3)
    and a single clip-level body_scale, same formula as part_normalize.py."""
    pose = _interpolate_missing(pose_raw)
    mid_shoulder = (pose[:, LEFT_SHOULDER] + pose[:, RIGHT_SHOULDER]) / 2.0
    shoulder_dist = np.linalg.norm(pose[:, LEFT_SHOULDER] - pose[:, RIGHT_SHOULDER], axis=-1)
    valid = shoulder_dist[shoulder_dist > 1e-6]
    scale = float(np.mean(valid)) if len(valid) else 1.0
    return pose, mid_shoulder, (scale if scale > 1e-6 else 1.0)


def _normalize_hand(hand_raw: np.ndarray) -> np.ndarray:
    """Wrist-centered, scaled by wrist-to-middle-MCP distance -- identical
    formula to part_normalize.py's nested normalize_hand(). Duplicated
    (not imported) because that helper is nested inside normalize_per_part
    and not exposed at module level."""
    hand = _interpolate_missing(hand_raw)
    wrist = hand[:, WRIST]
    mcp = hand[:, MIDDLE_MCP]
    dist = np.linalg.norm(mcp - wrist, axis=-1)
    valid = dist[dist > 1e-6]
    scale = float(np.mean(valid)) if len(valid) else 1.0
    if scale < 1e-6:
        scale = 1.0
    return (hand - wrist[:, None, :]) / scale


def extract_dominant_hand_signal(flat_225_canon: np.ndarray) -> Optional[dict]:
    """flat_225_canon: (T, 225) keypoints AFTER canonicalize_handedness
    (target='right'), BEFORE normalize_per_part. Returns per-frame geometric
    trajectories for the dominant (right-slot) hand plus the nose's position
    in the same body-relative frame (needed for the location bucket's
    face-region anchor), or None if the hand was never detected."""
    T = flat_225_canon.shape[0]
    arr = flat_225_canon.reshape(T, 75, 3).astype(np.float32)
    pose_raw, rh_raw = arr[:, :33], arr[:, 54:75]

    present = np.any(rh_raw != 0, axis=(1, 2))
    if present.sum() < 2:
        return None

    pose, mid_shoulder, body_scale = _body_frame(pose_raw)
    rh_norm = _normalize_hand(rh_raw)
    wrist_raw = _interpolate_missing(rh_raw)[:, WRIST]

    # x,y only -- the wrist's z comes from the HandLandmarker model, mid_shoulder's
    # z comes from the separate PoseLandmarker model; the two depth estimates are
    # not on a comparable scale (confirmed empirically: nose z ~-2 vs hand z ~+1
    # for the same clips, an anatomically implausible gap), so mixing them into
    # one 3D distance corrupts it with cross-model noise. Image-plane position
    # alone is still a real, comparable signal.
    location_traj = ((wrist_raw - mid_shoulder) / body_scale)[:, :2]  # (T, 2), body-relative
    nose_rel = ((pose[:, NOSE] - mid_shoulder) / body_scale)[:, :2]   # (T, 2), same frame

    curl = np.stack(
        [np.linalg.norm(rh_norm[:, tip], axis=-1) for tip in FINGER_TIPS.values()],
        axis=1,
    )  # (T, 5) -- bigger = more extended, smaller = more curled

    v1, v2 = rh_norm[:, INDEX_MCP], rh_norm[:, PINKY_MCP]
    normal = np.cross(v1, v2)
    norm_len = np.linalg.norm(normal, axis=-1, keepdims=True)
    norm_len = np.where(norm_len < 1e-6, 1.0, norm_len)
    orientation_traj = normal / norm_len  # (T, 3), unit vectors

    return {
        "location_traj": location_traj[present],
        "handshape_curl": curl[present],
        "orientation_traj": orientation_traj[present],
        "nose_rel_mean": nose_rel.mean(axis=0),
        # original-array frame indices the above rows came from -- needed to map
        # "which valid-frame index had the worst deviation" back to an actual
        # frame for the skeleton snapshot feature (see find_divergent_frame).
        "valid_frame_indices": np.where(present)[0],
    }


def find_divergent_frame(attempt_signal: dict, reference_signal: dict, param: str) -> Optional[int]:
    """Which of the ATTEMPT's original frames deviated most from the
    reference's typical value for `param`? Returns an index into the
    attempt's original (pre-filtering) frame array, or None if `param`
    isn't one of the three with a per-frame trajectory to compare. Used to
    pick a real, specific "here's where it went wrong" snapshot, not a
    generic evenly-sampled frame."""
    if param == "handshape":
        ref_mean = reference_signal["handshape_curl"].mean(axis=0)
        per_frame_dist = np.linalg.norm(attempt_signal["handshape_curl"] - ref_mean, axis=-1)
    elif param == "location":
        ref_mean = reference_signal["location_traj"].mean(axis=0)
        per_frame_dist = np.linalg.norm(attempt_signal["location_traj"] - ref_mean, axis=-1)
    elif param == "orientation":
        ref_mean = reference_signal["orientation_traj"].mean(axis=0)
        ref_mean = ref_mean / max(np.linalg.norm(ref_mean), 1e-6)
        att = attempt_signal["orientation_traj"]
        att_norm = att / np.maximum(np.linalg.norm(att, axis=-1, keepdims=True), 1e-6)
        per_frame_dist = -np.einsum("ij,j->i", att_norm, ref_mean)  # more negative cosine = further off
    else:
        return None
    worst = int(np.argmax(per_frame_dist))
    return int(attempt_signal["valid_frame_indices"][worst])


def classify_handshape_bucket(signal: dict) -> str:
    openness = float(np.mean(signal["handshape_curl"].mean(axis=0)))
    if openness < HANDSHAPE_LOW_CUT:
        return "closed"
    if openness < HANDSHAPE_HIGH_CUT:
        return "half"
    return "open"


def classify_location_bucket(signal: dict) -> str:
    location_mean = signal["location_traj"].mean(axis=0)
    nose_rel = signal["nose_rel_mean"]
    chest_anchor = np.zeros(2)
    d_face = float(np.linalg.norm(location_mean - nose_rel))
    d_chest = float(np.linalg.norm(location_mean - chest_anchor))
    if d_face > LOCATION_NEUTRAL_MIN_DIST and d_chest > LOCATION_NEUTRAL_MIN_DIST:
        return "neutral_space"
    return "face_region" if d_face < d_chest else "chest_region"


def classify_orientation_bucket(signal: dict) -> str:
    direction = signal["orientation_traj"].mean(axis=0)
    axis = int(np.argmax(np.abs(direction)))
    sign = "+" if direction[axis] >= 0 else "-"
    return f"{sign}{['x', 'y', 'z'][axis]}"


HANDSHAPE_PLAIN = {
    # The handshape bucket only measures overall openness, so "closed"
    # covers fists AND pointing shapes (index out, rest curled) -- phrase it
    # as "mostly curled in", never "fist".
    "open": "your fingers extended open",
    "half": "your fingers half-bent",
    "closed": "most of your fingers curled in",
}
LOCATION_PLAIN = {
    "face_region": "near your face",
    "chest_region": "near your chest",
    "neutral_space": "out in neutral space in front of you",
}
# Orientation buckets are the dominant axis of the mean palm normal in the
# CANONICAL frame (dominant hand = the "right_hand" array, MediaPipe image
# axes: +y down, +z away from the camera). Two facts make these phrases
# safe to state, both measured on the 23 reference signs, not assumed:
#   - +cross(index-wrist, pinky-wrist) points out of the PALM for this
#     array: fingertips curl to that side in 96% of signs;
#   - the dominant hand sits on the -x side of mid-shoulder in 23/23
#     signs, so +x points from that hand toward the body's midline.
# Phrased relative to the body (midline / you / out), not left/right, so
# it stays true for left-handed signers, whose attempts are mirrored into
# the same canonical frame before bucketing. Caveat: this is the palm's
# MAIN direction averaged over the whole sign; signs whose palm rotates
# partway through are summarized by their dominant direction.
ORIENTATION_PLAIN = {
    "+x": "your palm facing sideways, toward the middle of your body",
    "-x": "your palm facing sideways, away from your body",
    "+y": "your palm facing down",
    "-y": "your palm facing up",
    "+z": "your palm facing you",
    "-z": "your palm facing out, toward the person you're signing to",
}
PLAIN = {"handshape": HANDSHAPE_PLAIN, "location": LOCATION_PLAIN, "orientation": ORIENTATION_PLAIN}

# Imperative "how to do it" phrasing for the prep checklist.
HOWTO = {
    "handshape": {
        "open": "Open your hand with your fingers extended.",
        "half": "Bend your fingers about halfway, as in the picture.",
        "closed": "Curl most of your fingers in. Check the picture for any finger that stays out.",
    },
    # No "location" entry on purpose: the location buckets are consistent
    # (same classifier on attempt and reference, so match/mismatch is a fair
    # comparison) but their REGION NAMES are not trustworthy yet -- 0 of the
    # 9 face/head reference signs (apple, who, orange, ...) classify as
    # face_region, because the bucket averages the wrist over the whole
    # clip against a "chest" anchor that is really the shoulder line.
    # Until location is recalibrated with labels, the UI uses the sign's
    # own description for where to sign, and mismatches point at the ghost.
    "orientation": {
        "+x": "Turn your palm sideways, facing the middle of your body.",
        "-x": "Turn your palm sideways, facing away from your body.",
        "+y": "Turn your palm to face down.",
        "-y": "Turn your palm to face up.",
        "+z": "Turn your palm to face you.",
        "-z": "Turn your palm to face out, toward the person you're signing to.",
    },
}


def plain_bucket(param: str, bucket: str) -> str | None:
    """Plain description of a bucket value ("your palm facing up"); None for
    location, whose region names aren't trustworthy yet (see HOWTO)."""
    if param == "location":
        return None
    return PLAIN.get(param, {}).get(bucket, bucket)


def howto_bucket(param: str, bucket: str) -> str | None:
    """Imperative instruction for a target bucket ("Turn your palm to face up.")."""
    return HOWTO.get(param, {}).get(bucket)


def _describe_mismatch(param: str, target: str, yours: str) -> str:
    """Plain-language reason a mismatched parameter is wrong, so the UI never
    has to show raw bucket codes like "+x" (see ORIENTATION_PLAIN for why
    the orientation directions can now be stated)."""
    if param == "handshape":
        return f"This sign needs {HANDSHAPE_PLAIN[target]}, but you had {HANDSHAPE_PLAIN[yours]}."
    if param == "location":
        # Region names withheld -- see the HOWTO comment above.
        return "Your hand was in a different place than in the target sign. Move it to where the white ghost hand is on your video."
    if param == "orientation":
        return f"This sign needs {ORIENTATION_PLAIN[target]}, but you had {ORIENTATION_PLAIN[yours]}."
    return ""


CLASSIFIERS = {
    "handshape": lambda s: classify_handshape_bucket(s),
    "location": lambda s: classify_location_bucket(s),
    "orientation": lambda s: classify_orientation_bucket(s),
}
# A part counts as settled for a sign when at least this share of its
# approved clips agree; below it, signers genuinely differ (or the sign sits
# on a bucket border), so the part isn't graded for that sign.
MIN_AGREEMENT = 0.6


def majority_targets(signals: list[dict]) -> dict:
    """Target bucket per parameter by majority vote over ALL approved clips
    of a sign, instead of whatever a single representative clip happens to
    show (a single clip near a bucket border flipped e.g. who's handshape
    between closed and half, contradicting its own example video).
      {param: {"value": bucket, "agreement": share, "n": clips,
               "votes": {bucket: count}, "settled": agreement >= MIN_AGREEMENT}}"""
    from collections import Counter
    out = {}
    for name, classify in CLASSIFIERS.items():
        votes = Counter(classify(s) for s in signals if s is not None)
        n = sum(votes.values())
        if not n:
            continue
        value, top = votes.most_common(1)[0]
        out[name] = {"value": value, "agreement": round(top / n, 3), "n": n,
                     "votes": dict(votes), "settled": top / n >= MIN_AGREEMENT}
    return out


def compare_discrete(attempt_signal: dict, reference_signal: dict, targets: dict | None = None) -> dict:
    """Bucket both sides and report target vs. yours + match, for each of
    the 3 validated parameters, plus a plain-language "why" when they
    mismatch. No score number, no verdict -- this is explanatory detail
    alongside compare.py's embedding-driven verdict, not a replacement for
    it."""
    out = {}
    for name, classify in (
        ("handshape", classify_handshape_bucket),
        ("location", classify_location_bucket),
        ("orientation", classify_orientation_bucket),
    ):
        t = (targets or {}).get(name)
        target_bucket = t["value"] if t else classify(reference_signal)
        yours_bucket = classify(attempt_signal)
        if t and not t.get("settled", True):
            # Approved signers split on this part: don't grade it.
            out[name] = {
                "target": target_bucket, "yours": yours_bucket, "match": None, "uncertain": True,
                "why": "Signers do this part in different ways, so it isn't graded for this sign.",
                "target_plain": None, "yours_plain": plain_bucket(name, yours_bucket),
            }
            continue
        match = target_bucket == yours_bucket
        out[name] = {
            "target": target_bucket,
            "yours": yours_bucket,
            "match": match,
            "why": None if match else _describe_mismatch(name, target_bucket, yours_bucket),
            "target_plain": plain_bucket(name, target_bucket),
            "yours_plain": plain_bucket(name, yours_bucket),
        }
    return out


def chart_series(signal: dict, n_points: int = 40) -> dict:
    """Per-attempt chart data (same signals the CP buckets use):
      finger_openness: mean fingertip-to-wrist distance per finger, in hand
        lengths (bigger = more extended) -- the per-finger view of CP1;
      travel: distance of the dominant wrist from where it started, in
        shoulder widths, resampled to n_points over the sign's own 0..1
        time -- its peaks are the big movements (CP2/CP3)."""
    curl = signal["handshape_curl"]
    loc = signal["location_traj"]
    travel = np.linalg.norm(loc - loc[0], axis=-1)
    x = np.linspace(0, 1, len(travel)) if len(travel) > 1 else np.array([0.0])
    grid = np.linspace(0, 1, n_points)
    resampled = np.interp(grid, x, travel) if len(travel) > 1 else np.zeros(n_points)
    return {
        "finger_openness": [round(float(v), 3) for v in curl.mean(axis=0)],
        "travel": [round(float(v), 3) for v in resampled],
    }
