#!/usr/bin/env python3
"""LLM judge over structured evidence, vs the vision-only LLM judge and the
Score tab's own flags -- on the same 280 attempts as llm_vision_v2.

For each attempt the model gets the measured evidence for the target sign
(mouna.coach.param_scores: level and score per parameter, the joint angles /
positions furthest outside the fluent range with that range, the worst CP,
how reliable each score is on held-out signs, the whole-sign similarity),
and answers same / different / unsure per part like the vision judge.

  arm "text":   evidence only
  arm "vision": evidence + the same 3-row key-frame image as llm_vision_judge

Scored with llm_vision_judge.score() yardsticks; "ours" = the Score tab
(needs_work = different; hands = the hand-count check).
Writes llm_evidence_<arm>.json and llm_evidence_rows.csv.
Run from repo root: ./venv/bin/python scripts/llm_evidence_judge.py [arms]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
ARMS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["text", "vision"]
_argv, sys.argv = sys.argv, sys.argv[:1]          # llm_vision_judge reads argv at import
import llm_vision_judge as V  # noqa: E402  (loads settings, chdirs to ROOT)
sys.argv = _argv
OUT = V.OUT
PARTS = V.PARTS
CP_WORDS = {"palm_x": "palm facing sideways", "palm_y": "palm facing up/down", "palm_z": "palm facing toward/away",
            "finger_thumb": "thumb bend", "finger_index": "index finger bend", "finger_middle": "middle finger bend",
            "finger_ring": "ring finger bend", "finger_little": "little finger bend", "thumb_out": "thumb position",
            "loc_x": "hand side-to-side position", "loc_y": "hand height", "dir_x": "side-to-side motion",
            "dir_y": "up-down motion", "strokes": "number of strokes", "travel": "size of movement", "gap": "hands touching"}

PROMPT = """You judge an attempt at the American Sign Language sign "{sign}".
Measurements compare the attempt with fluent signers of "{sign}" (not with one video).
Each parameter has a level (fluent / close / needs_work), a 0-100 score, the
measurements furthest outside the fluent range, and how trustworthy that score
is on held-out data. Scores marked "estimate" are hints and can be wrong;
tracking noise also produces small deviations in correct signing.

{evidence}
{image_note}
Decide, per part, whether the attempt's part is DIFFERENT from how fluent signers
do "{sign}" in a way that changes the sign:
- handshape: fingers extended/bent, thumb position
- orientation: which way the palm faces
- location: where the hand is relative to face and body
- hands: one or two hands used
Say "different" only with clear support; "same" otherwise; "unsure" if the evidence is missing.
Reply with JSON only:
{{"handshape": "...", "orientation": "...", "location": "...", "hands": "...",
  "how": "<for any 'different': what the attempt does instead, in a few words; else empty>"}}"""

IMAGE_NOTE = """An image is attached. ROW 1 and ROW 2: two fluent signers doing "{sign}"; their
differences are normal variation. ROW 3: the attempt. Use it together with the measurements."""


def evidence_text(ps: dict, sim: float | None, hands_match, rel: dict) -> str:
    lines = [f"Whole-sign similarity to fluent signers: {sim:.2f} (good >= 0.60)" if sim == sim and sim is not None else ""]
    for p in ("handshape", "location", "orientation", "movement"):
        e = ps.get(p) or {}
        if not e.get("level"):
            lines.append(f"- {e.get('label', p)}: not measured")
            continue
        trust = "trusted" if e.get("reliable") else "estimate"
        r = rel.get(p)
        if r:
            trust += f"; on held-out signs it flags {r['test_false_alarm']:.0%} of correct signing and catches {r['test_catches']:.0%} of real differences"
        s = f"- {e['label']}: {e['level']} (score {e['score']}; {trust})"
        ev = e.get("evidence") or []
        if ev:
            s += "; furthest outside: " + "; ".join(
                f"{x['dim']} {x['yours']}{'°' if x['unit'] == '°' else ''} vs fluent {x['fluent'][0]}-{x['fluent'][1]}{'°' if x['unit'] == '°' else ' shoulder widths'}"
                for x in ev[:3])
        elif e.get("cp_worst") and e["level"] != "fluent":
            s += f"; worst measurement: {CP_WORDS.get(e['cp_worst'], e['cp_worst'])}"
        lines.append(s)
    if hands_match is not None:
        lines.append(f"- Hand count: {'matches' if hands_match else 'differs from'} what fluent signers use (an estimate; flags about 22% of correct signing)")
    lines.append("- Facial expression: not measured")
    return "\n".join(x for x in lines if x)


def ask(client, types, prompt: str, img: bytes | None) -> dict | None:
    contents = ([types.Part.from_bytes(data=img, mime_type="image/jpeg")] if img else []) + [prompt]
    last = ""
    for attempt in range(4):
        for model in V.MODELS:
            try:
                r = client.models.generate_content(model=model, contents=contents,
                                                   config=types.GenerateContentConfig(response_mime_type="application/json", max_output_tokens=1500))
                d = json.loads(r.text)
                if isinstance(d, list):
                    d = next((x for x in d if isinstance(x, dict)), {})
                return {**{k: str(d.get(k, "unsure")).lower() for k in PARTS}, "note": d.get("how", ""), "model": model}
            except Exception as e:  # noqa: BLE001
                last = str(e)[:120]
                print(f"    {model}: {last}", flush=True)
        time.sleep(6 * (attempt + 1))
    print("  gave up:", last, flush=True)
    return None


def main() -> None:
    from google import genai
    from google.genai import types
    from build_sign_references_v2 import load_raw_lookup
    from mouna.coach.param_scores import _refs, parameter_scores
    from mouna.data.handedness import canonicalize_handedness
    client = genai.Client(api_key=V.settings.gemini_api_key,
                          http_options=types.HttpOptions(timeout=30000, retry_options=types.HttpRetryOptions(attempts=1)))
    raw = load_raw_lookup()
    sizes = json.loads((ROOT / "data/references/video_sizes.json").read_text())
    rel = _refs()["calibration"]
    base = pd.read_csv(OUT / "llm_vision_v2_rows.csv")
    bench = pd.read_csv(OUT / "rows.csv.gz")[["target", "video_id", "similarity", "check_hand_count"]]
    base = base.merge(bench.drop(columns="check_hand_count").drop_duplicates(["target", "video_id"]), on=["target", "video_id"], how="left")
    dec = pd.read_csv(ROOT / "data/silver/qa/review_decisions.csv")
    approved = set(dec[dec.decision == "ok"].video_id.astype(str))
    v4 = pd.read_parquet(ROOT / "data/silver/keypoints/keypoints_official_split_v4canon.parquet", columns=["video_id", "gloss"])
    v4["vid"] = v4.video_id.astype(str)
    second = {}
    for g, sub in v4[v4.vid.isin(approved)].groupby("gloss"):
        for v in sub.vid:
            if v != V.EXAMPLES.get(g, {}).get("video_id") and V.video_path(v) and v in raw:
                second[g] = v
                break
    part_file = OUT / "llm_evidence_rows.partial.csv"
    done = pd.read_csv(part_file) if part_file.exists() else pd.DataFrame()
    out = done.to_dict("records")
    seen = {(o["target"], str(o["video_id"]), o["arm"]) for o in out}
    print(f"{len(base)} attempts x {ARMS}; {len(seen)} already done", flush=True)
    for i, r in enumerate(base.itertuples()):
        vid = str(r.video_id)
        canon = canonicalize_handedness(np.asarray(raw[vid], dtype=np.float32), target="right")
        ps = parameter_scores(canon, r.target, *sizes[vid]) or {}
        hm = None if pd.isna(r.check_hand_count) else not (r.check_hand_count is True or str(r.check_hand_count) == "True")
        ev = evidence_text(ps, r.similarity, hm, rel)
        ours = {"ours_handshape": (ps.get("handshape") or {}).get("level") == "needs_work",
                "ours_location": (ps.get("location") or {}).get("level") == "needs_work",
                "ours_orientation": (ps.get("orientation") or {}).get("level") == "needs_work",
                "ours_hands": r.check_hand_count}
        img = None
        for arm in ARMS:
            if (r.target, vid, arm) in seen:
                continue
            if arm == "vision" and img is None:
                tvid = V.EXAMPLES[r.target]["video_id"]
                r2 = second.get(r.target)
                img = V.sheet(ROOT / f"app/wlasl/videos/{tvid}.mp4", V.video_path(vid), V.track(tvid, raw, None), V.track(vid, raw, None),
                              ref2=(V.video_path(r2), V.track(r2, raw, None)) if r2 else None)
                if img is None:
                    continue
            prompt = PROMPT.format(sign=r.target, evidence=ev, image_note=IMAGE_NOTE.format(sign=r.target) if arm == "vision" else "")
            ans = ask(client, types, prompt, img if arm == "vision" else None)
            if ans is None:
                continue
            out.append({"arm": arm, **{c: getattr(r, c) for c in ("target", "attempt_sign", "video_id", "same", "group",
                                                                  "differs_handshape", "differs_location", "differs_hands")},
                        **ours, **{f"llm_{k}": ans[k] for k in PARTS}, "llm_note": ans["note"], "llm_model": ans["model"],
                        "evidence": ev})
            time.sleep(1.0)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(base)}", flush=True)
            pd.DataFrame(out).to_csv(part_file, index=False)
    res = pd.DataFrame(out)
    res.to_csv(OUT / "llm_evidence_rows.csv", index=False)
    for arm in res.arm.unique():
        a = res[res.arm == arm].rename(columns={"ours_handshape": "check_handshape", "ours_location": "check_location",
                                                "ours_hands": "check_hand_count", "ours_orientation": "check_orientation"})
        s = V.score(a)
        (OUT / f"llm_evidence_{arm}.json").write_text(json.dumps(s, indent=2))
        print(arm, json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
