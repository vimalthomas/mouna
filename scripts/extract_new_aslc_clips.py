#!/usr/bin/env python3
"""Extract raw keypoints for the QA-kept ASL Citizen clips of signs the model
wasn't trained on (see qa_new_aslc_clips.py). Same extractor and frame cap as
extract_asl_citizen_raw.py; writes a separate v2 file so the shared training
parquet is untouched:  data/silver/keypoints_v2/keypoints_aslc_new_raw.parquet
"""
import multiprocessing as mp
import pickle
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "data/silver/qa/new_aslc_clips_qa.csv"
VIDEOS = ROOT / "data/raw/asl_citizen/videos"
OUT = ROOT / "data/silver/keypoints_v2/keypoints_aslc_new_raw.parquet"
_ex = None


def _init():
    global _ex
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from mouna.data.preprocessing import KeypointExtractor
    _ex = KeypointExtractor()


def _one(row):
    try:
        flat = _ex.flatten_keypoints(_ex.extract_from_video(str(VIDEOS / row["video_file"]), max_frames=150))
        assert len(flat), "zero frames"
        return {"video_id": row["video_file"], "gloss": row["word"], "keypoints_pickle": pickle.dumps(flat), "num_frames": len(flat), "success": True}
    except Exception as e:  # noqa: BLE001
        return {"video_id": row["video_file"], "gloss": row["word"], "keypoints_pickle": None, "num_frames": 0, "success": False, "error": str(e)}


def main():
    qa = pd.read_csv(QA)
    rows = qa[qa.keep.astype(str) == "True"].to_dict("records")
    print(f"{len(rows)} QA-kept clips")
    with mp.Pool(6, initializer=_init) as pool:
        res = list(pool.imap_unordered(_one, rows, chunksize=4))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(res)
    df.to_parquet(OUT, index=False)
    ok = df[df.success]
    print(f"Extracted {len(ok)}/{len(df)} -> {OUT.relative_to(ROOT)}")
    print(ok.gloss.value_counts().to_string())


if __name__ == "__main__":
    main()
