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
MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest"]   # Flash first (paid tier); Lite only if Flash is overloaded
PARTS = ("handshape", "orientation", "location", "hands")
EXAMPLES = json.loads((ROOT / "data/references/example_videos.json").read_text())
random.seed(11)

PROMPT = """You compare people signing in American Sign Language.
ROW 1 and ROW 2: two different fluent signers doing the sign "{sign}" (references).
Differences between rows 1 and 2 are NORMAL variation (style, camera, body, speed).
ROW 3: another person attempting the sign.
For the signing hand(s), compare row 3 with rows 1 and 2:
- handshape: which fingers are extended/bent, thumb position
- orientation: which way the palm faces
- location: where the hand is relative to the face and body
- hands: one hand or two hands used
A part is "different" ONLY if row 3 clearly differs from BOTH references, by more
than the references differ from each other, in a way that would change the sign.
Otherwise "same"; if you can't see it, "unsure".
Reply with JSON only:
{{"handshape": "...", "orientation": "...", "location": "...", "hands": "...",
  "how": "<for any 'different': what row 3 does instead, in a few words; else empty>"}}"""


def video_path(vid: str) -> Path | None:
    for p in (ROOT / f"data/raw/asl_citizen/videos/{vid}", ROOT / f"app/wlasl/videos/{vid}.mp4",
              ROOT / f"data/raw/msasl/videos/{vid}.mp4", ROOT / f"data/raw/msasl/videos/{vid}"):
        if p.exists():
            return p
    return None


def track(vid: str, raw: dict, skin_dir: Path) -> np.ndarray | None:
    """(T, 75, 3) normalized keypoints for a clip: the raw tracking for dataset
    clips, or the example's 3D-skin data (pixels -> normalized) for targets."""
    if vid in raw:
        return np.asarray(raw[vid], dtype=np.float32).reshape(-1, 75, 3)
    return None


def key_frames(path: Path, kp: np.ndarray | None, n_show=3, height=300) -> list[np.ndarray]:
    """Frames while the hands are up, cropped to the face and hands."""
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    W, H = cap.get(3) or 1, cap.get(4) or 1
    idx = [int(n * a) for a in (0.3, 0.5, 0.7)]
    box = None
    if kp is not None and len(kp):
        hands = np.any(kp[:, 33:75] != 0, axis=(1, 2))
        up = np.where(hands)[0]
        if len(up) >= 3:
            lo, hi = up[0], up[-1]
            idx = [int((lo + (hi - lo) * a) * n / len(kp)) for a in (0.25, 0.5, 0.75)]
        pts = kp[:, [0, 2, 5, 7, 8, 11, 12, 15, 16] + list(range(33, 75)), :2].reshape(-1, 2)
        pts = pts[np.all(pts > 0, axis=1) & np.all(pts < 1, axis=1)]
        if len(pts) > 10:
            x0, y0 = np.percentile(pts, 2, axis=0); x1, y1 = np.percentile(pts, 98, axis=0)
            cx, cy, half = (x0 + x1) / 2 * W, (y0 + y1) / 2 * H, max((x1 - x0) * W, (y1 - y0) * H) * 0.62
            box = (int(max(0, cx - half)), int(max(0, cy - half)), int(min(W, cx + half)), int(min(H, cy + half)))
    out = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(n - 1, i)))
        ok, f = cap.read()
        if not ok:
            continue
        if box:
            f = f[box[1]:box[3], box[0]:box[2]]
        if f.size:
            out.append(cv2.resize(f, (int(f.shape[1] * height / max(1, f.shape[0])), height)))
    return out[:n_show]


def sheet(target_path: Path, attempt_path: Path, t_kp=None, a_kp=None, ref2: tuple | None = None) -> bytes | None:
    rows = []
    clips = [(target_path, t_kp)] + ([ref2] if ref2 else []) + [(attempt_path, a_kp)]
    for p, kp in clips:
        fs = key_frames(p, kp)
        if len(fs) < 2:
            return None
        rows.append(np.hstack(fs))
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    gap = np.full((10, w, 3), 255, np.uint8)
    stacked = [x for r in rows for x in (r, gap)][:-1]
    ok, buf = cv2.imencode(".jpg", np.vstack(stacked), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes() if ok else None


def ask(client, types, img: bytes, sign: str) -> dict | None:
    for attempt in range(6):
        for model in MODELS:
            try:
                r = client.models.generate_content(
                    model=model,
                    contents=[types.Part.from_bytes(data=img, mime_type="image/jpeg"), PROMPT.format(sign=sign)],
                    config=types.GenerateContentConfig(response_mime_type="application/json", max_output_tokens=1500),
                )
                d = json.loads(r.text)
                if isinstance(d, list):   # some replies wrap the object in a list
                    d = next((x for x in d if isinstance(x, dict)), {})
                return {**{k: str(d.get(k, "unsure")).lower() for k in PARTS}, "note": d.get("how") or d.get("note", ""), "model": model}
            except Exception as e:  # noqa: BLE001 -- busy / quota / bad JSON: try the next model, then wait
                last = str(e)[:120]
                print(f"    {model}: {last}", flush=True)
        time.sleep(8 * (attempt + 1))
    print("  gave up:", last, flush=True)
    return None


def main() -> None:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=settings.gemini_api_key, http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)))
    sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
    from build_sign_references_v2 import load_raw_lookup
    raw = load_raw_lookup()
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
    dec = pd.read_csv(ROOT / "data/silver/qa/review_decisions.csv")
    approved = set(dec[dec.decision == "ok"].video_id.astype(str))
    v4 = pd.read_parquet(ROOT / "data/silver/keypoints/keypoints_official_split_v4canon.parquet", columns=["video_id", "gloss"])
    v4["vid"] = v4.video_id.astype(str)
    second = {}
    for g, sub in v4[v4.vid.isin(approved)].groupby("gloss"):
        for v in sub.vid:
            if v != EXAMPLES.get(g, {}).get("video_id") and video_path(v) and v in raw:
                second[g] = v
                break
    df = df[df.target.isin(second)]
    sample = pd.concat([g.sample(min(N, len(g)), random_state=3).assign(group=k) for k, g in groups.items()]).drop_duplicates("video_id")
    print(f"{len(sample)} attempts to judge", flush=True)

    out = []
    for i, r in enumerate(sample.itertuples()):
        tvid = EXAMPLES[r.target]["video_id"]
        r2 = second[r.target]
        img = sheet(ROOT / f"app/wlasl/videos/{tvid}.mp4", r.path, track(tvid, raw, None), track(str(r.video_id), raw, None),
                    ref2=(video_path(r2), track(r2, raw, None)))
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
    tag = os.environ.get("JUDGE_TAG", "v2")
    res.to_csv(OUT / f"llm_vision_{tag}_rows.csv", index=False)
    summary = score(res)
    (OUT / f"llm_vision_{tag}.json").write_text(json.dumps(summary, indent=2))
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
