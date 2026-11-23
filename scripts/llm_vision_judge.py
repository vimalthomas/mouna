#!/usr/bin/env python3
"""Can a vision LLM judge sign parts better than our geometric checks?

Takes a sample of the judgement benchmark's attempts (dataset clips only,
never learner video), shows the model one image -- top row: 3 moments of the
target sign's example video; bottom row: 3 moments of the attempt -- and asks,
per part, same / different / unsure. Scored exactly like the checks in
scripts/judgement_benchmark.py (catches when ASL-LEX says the part differs,
false alarms on correct attempts and on other signs sharing the part).

Uses the Gemini key in app/backend/.env (GEMINI_API_KEY).
Writes data/references/judgement_benchmark/llm_vision.json
Run from repo root:  ./venv/bin/python scripts/llm_vision_judge.py [n_per_group]
"""
import io
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "app" / "backend")]
os.chdir(ROOT / "app" / "backend")
from config import settings  # noqa: E402
os.chdir(ROOT)

OUT = ROOT / "data/references/judgement_benchmark"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest"]
PARTS = ("handshape", "orientation", "location", "hands")
EXAMPLES = json.loads((ROOT / "data/references/example_videos.json").read_text())
random.seed(11)

PROMPT = """You compare two people signing in American Sign Language.
TOP row: three moments of a fluent signer doing the sign "{sign}" (the reference).
BOTTOM row: three moments of another person attempting a sign.
Compare ONLY what you can see in the stills, for the signing hand(s):
- handshape: which fingers are extended/bent, thumb position
- orientation: which way the palm faces
- location: where the hand is relative to the face and body
- hands: one hand or two hands used
Ignore clothing, background, camera angle, body size and speed.
For each part answer "same", "different" or "unsure". Answer "different" only
when the difference is clear. Reply with JSON only:
{{"handshape": "...", "orientation": "...", "location": "...", "hands": "...", "note": "<one short sentence>"}}"""


def video_path(vid: str) -> Path | None:
    for p in (ROOT / f"data/raw/asl_citizen/videos/{vid}", ROOT / f"app/wlasl/videos/{vid}.mp4",
              ROOT / f"data/raw/msasl/videos/{vid}.mp4", ROOT / f"data/raw/msasl/videos/{vid}"):
        if p.exists():
            return p
    return None


def frames(path: Path, at=(0.3, 0.5, 0.7), height=220) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    out = []
    for a in at:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * a))
        ok, f = cap.read()
        if not ok:
            continue
        out.append(cv2.resize(f, (int(f.shape[1] * height / f.shape[0]), height)))
    return out


def sheet(target_path: Path, attempt_path: Path) -> bytes | None:
    rows = []
    for p in (target_path, attempt_path):
        fs = frames(p)
        if len(fs) < 2:
            return None
        rows.append(np.hstack(fs))
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    gap = np.full((8, w, 3), 255, np.uint8)
    ok, buf = cv2.imencode(".jpg", np.vstack([rows[0], gap, rows[1]]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


def ask(client, types, img: bytes, sign: str) -> dict | None:
    for attempt in range(6):
        for model in MODELS:
            try:
                r = client.models.generate_content(
                    model=model,
                    contents=[types.Part.from_bytes(data=img, mime_type="image/jpeg"), PROMPT.format(sign=sign)],
                    config=types.GenerateContentConfig(response_mime_type="application/json", max_output_tokens=300),
                )
                d = json.loads(r.text)
                return {**{k: str(d.get(k, "unsure")).lower() for k in PARTS}, "note": d.get("note", ""), "model": model}
            except Exception as e:  # noqa: BLE001 -- busy / quota / bad JSON: try the next model, then wait
                last = str(e)[:120]
        time.sleep(8 * (attempt + 1))
    print("  gave up:", last, flush=True)
    return None


def main() -> None:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=settings.gemini_api_key)
    df = pd.read_csv(OUT / "rows.csv.gz")
    df["path"] = df.video_id.astype(str).map(video_path)
    df = df[df.path.notna() & df.target.map(lambda t: (ROOT / f"app/wlasl/videos/{EXAMPLES[t]['video_id']}.mp4").exists())]
    wrong = df[~df.same]
    groups = {
        "correct": df[df.same],
        "handshape_differs": wrong[wrong.differs_handshape.eq(True)],
        "location_differs": wrong[wrong.differs_location.eq(True)],
        "hands_differ": wrong[wrong.differs_hands.eq(True)],
        "shares_parts": wrong[wrong.differs_handshape.eq(False) | wrong.differs_location.eq(False) | wrong.differs_hands.eq(False)],
    }
    sample = pd.concat([g.sample(min(N, len(g)), random_state=3).assign(group=k) for k, g in groups.items()]).drop_duplicates("video_id")
    print(f"{len(sample)} attempts to judge", flush=True)

    out = []
    for i, r in enumerate(sample.itertuples()):
        img = sheet(ROOT / f"app/wlasl/videos/{EXAMPLES[r.target]['video_id']}.mp4", r.path)
        if img is None:
            continue
        ans = ask(client, types, img, r.target)
        if ans is None:
            continue
        out.append({**{c: getattr(r, c) for c in ("target", "attempt_sign", "video_id", "same", "group",
                                                   "differs_handshape", "differs_location", "differs_hands",
                                                   "check_handshape", "check_location", "check_orientation", "check_hand_count")},
                    **{f"llm_{k}": ans[k] for k in PARTS}, "llm_note": ans["note"], "llm_model": ans["model"]})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(sample)}", flush=True)
        time.sleep(1.5)

    res = pd.DataFrame(out)
    res.to_csv(OUT / "llm_vision_rows.csv", index=False)
    summary = score(res)
    (OUT / "llm_vision.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def score(res: pd.DataFrame) -> dict:
    """Same yardsticks as the geometric checks, on the same attempts."""
    def r(s):
        s = s.dropna().astype(bool)
        return {"rate": round(float(s.mean()), 3) if len(s) else None, "n": int(len(s))}
    flag = lambda col: res[col].map(lambda v: None if v == "unsure" else v == "different")  # noqa: E731
    out = {"judged": int(len(res))}
    pairs = {"handshape": ("handshape", "check_handshape"), "location": ("location", "check_location"),
             "hands": ("hands", "check_hand_count"), "orientation": (None, "check_orientation")}
    same, wrong = res.same.astype(bool), ~res.same.astype(bool)
    for part, (lex, ours) in pairs.items():
        llm_col = f"llm_{part}"
        f_llm = flag(llm_col)
        f_ours = res[ours].map(lambda v: None if pd.isna(v) else bool(v) if not isinstance(v, str) else v == "True")
        row = {"unsure_rate": round(float((res[llm_col] == "unsure").mean()), 3),
               "llm_false_alarm_on_correct": r(f_llm[same]), "ours_false_alarm_on_correct": r(f_ours[same])}
        if lex:
            d = wrong & res[f"differs_{lex}"].eq(True)
            s = wrong & res[f"differs_{lex}"].eq(False)
            row.update({"llm_catches": r(f_llm[d]), "ours_catches": r(f_ours[d]),
                        "llm_false_alarm_other_signs": r(f_llm[s]), "ours_false_alarm_other_signs": r(f_ours[s])})
        out[part] = row
    return out


if __name__ == "__main__":
    main()
