"""Unit tests for coaching logic fixed during development. Fast, no video or
model needed. Each test names the real bug it guards against."""
import asyncio

import numpy as np

from mouna.coach.parameters import (
    HOWTO, ORIENTATION_PLAIN, PLAIN, _describe_mismatch, compare_discrete, majority_targets,
)


# ---- plain language: learners never see raw bucket codes -----------------

def test_no_raw_codes_in_learner_text():
    texts = [v for d in PLAIN.values() for v in d.values()] + [v for d in HOWTO.values() for v in d.values()]
    for o in ORIENTATION_PLAIN:
        texts.append(_describe_mismatch("orientation", o, "+z" if o != "+z" else "-z"))
    for t in texts:
        for code in ("+x", "-x", "+y", "-y", "+z", "-z", "_region", "neutral_space"):
            assert code not in t, f"raw code {code!r} leaked into: {t}"


def test_location_names_are_withheld():
    # Location buckets put every face sign at "chest" (0/9 face signs), so
    # region names must not reach learners until location is recalibrated.
    assert "location" not in HOWTO
    why = _describe_mismatch("location", "face_region", "chest_region")
    assert "chest" not in why and "face" not in why


def test_handshape_never_says_fist_for_closed():
    # "closed" covers pointing shapes (GO's index finger); "fist" was wrong.
    assert "fist" not in HOWTO["handshape"]["closed"].lower()


# ---- majority-vote targets ------------------------------------------------

def _signal(curl, loc, normal):
    T = 6
    return {
        "handshape_curl": np.full((T, 5), curl, dtype=np.float32),
        "location_traj": np.tile(np.array(loc, dtype=np.float32), (T, 1)),
        "orientation_traj": np.tile(np.array(normal, dtype=np.float32), (T, 1)),
        "nose_rel_mean": np.array([0.0, -0.8], dtype=np.float32),
        "valid_frame_indices": np.arange(T),
    }


def test_majority_vote_settled_and_split():
    sigs = [_signal(2.0, [0.1, 0.1], [0, 0, 1])] * 8 + [_signal(0.5, [0.1, 0.1], [1, 0, 0])] * 3
    t = majority_targets(sigs)
    assert t["handshape"]["value"] == "open" and t["handshape"]["settled"]
    assert t["handshape"]["n"] == 11 and abs(t["handshape"]["agreement"] - 8 / 11) < 1e-3
    split = majority_targets([_signal(2.0, [0, 0], [0, 0, 1])] * 5 + [_signal(0.5, [0, 0], [0, 0, 1])] * 5)
    assert not split["handshape"]["settled"]


def test_split_part_is_not_graded():
    attempt = _signal(0.5, [0.1, 0.1], [0, 0, 1])
    ref = _signal(2.0, [0.1, 0.1], [0, 0, 1])
    targets = {"handshape": {"value": "open", "agreement": 0.5, "n": 10, "settled": False, "votes": {}}}
    out = compare_discrete(attempt, ref, targets)
    assert out["handshape"]["match"] is None and out["handshape"]["uncertain"]
    assert out["orientation"]["match"] is True


# ---- coaching messages ----------------------------------------------------

def test_ungraded_sign_does_not_claim_a_match():
    # Bug: a sign with no reference got "all parts matched".
    from services import coach_writer
    msg, _ = asyncio.run(coach_writer.write_feedback("thursday", "", {"embedding_similarity": None, "parameters": None}))
    assert "can't grade" in msg and "matched" not in msg


def test_uncertain_part_is_neither_right_nor_a_fix():
    from services import coach_writer
    score = {"overall_verdict": "needs_work", "embedding_similarity": 0.6, "hand_count_match": True, "parameters": {
        "handshape": {"match": None, "uncertain": True, "why": "Signers do this part in different ways"},
        "orientation": {"match": False, "why": "Turn your palm."},
        "location": {"match": True, "why": None},
    }}
    msg, _ = asyncio.run(coach_writer.write_feedback("who", "L-hand circles at lips", score))
    assert "Most important: Turn your palm." in msg
    assert "Handshape: signers vary" in msg


# ---- hands ----------------------------------------------------------------

def test_duplicate_hand_is_detected_but_pressed_hands_are_not():
    from services.keypoint_service import is_duplicate_hand
    hand = np.random.default_rng(0).random((21, 2)) * 0.1 + 0.4
    assert is_duplicate_hand(hand, hand + 0.003)          # same hand twice
    mirrored = hand.copy()
    mirrored[:, 0] = 2 * hand[:, 0].mean() - hand[:, 0]   # palms pressed together
    assert not is_duplicate_hand(hand, mirrored)


def test_hand_paths_follow_position_not_label():
    # Bug: on "accident" MediaPipe swapped left/right labels mid-sign, so both
    # drawn paths ran the same way. Tracks must follow each physical hand.
    from pick_example_videos import hand_paths
    frames, times = [], []
    for i in range(20):
        t = i / 19
        a = np.zeros((75, 3), dtype=np.float32)
        left_x, right_x = 0.2 + 0.25 * t, 0.8 - 0.25 * t     # hands converge
        pts_l = np.tile([left_x, 0.5, 0.0], (21, 1))
        pts_r = np.tile([right_x, 0.5, 0.0], (21, 1))
        if i >= 10:                                          # labels swap halfway
            pts_l, pts_r = pts_r, pts_l
        a[33:54], a[54:75] = pts_l, pts_r
        frames.append(a.reshape(-1).tolist())
        times.append(i / 10)
    paths = hand_paths(frames, times)
    assert len(paths) == 2
    left, right = paths
    assert left["x"][-1] > left["x"][0] and right["x"][-1] < right["x"][0]
