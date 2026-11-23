#!/usr/bin/env python3
"""Training set for the 108-class model: v4canon (100 signs, unchanged) plus
the 8 new essential signs from QA-kept ASL Citizen clips, processed exactly
like v4canon (canonicalize_handedness -> normalize_per_part). New-sign clips
use ASL Citizen's signer-disjoint split: train+val -> train, test -> test.
Old val/test rows are untouched, so old-sign accuracy stays comparable.
Writes data/silver/keypoints_v2/keypoints_official_split_v5new8.parquet
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mouna.data.handedness import canonicalize_handedness  # noqa: E402
from mouna.data.part_normalize import normalize_per_part  # noqa: E402

V4 = ROOT / "data/silver/keypoints/keypoints_official_split_v4canon.parquet"
NEW = ROOT / "data/silver/keypoints_v2/keypoints_aslc_new_raw.parquet"
QA = ROOT / "data/silver/qa/new_aslc_clips_qa.csv"
OUT = ROOT / "data/silver/keypoints_v2/keypoints_official_split_v5new8.parquet"

v4 = pd.read_parquet(V4)
split = pd.read_csv(QA).set_index("video_file").asl_citizen_split
rows = []
for _, r in pd.read_parquet(NEW).query("success").iterrows():
    x = normalize_per_part(canonicalize_handedness(np.asarray(pickle.loads(r.keypoints_pickle), dtype=np.float32), target="right"))
    rows.append({"video_id": r.video_id, "gloss": r.gloss, "keypoints_pickle": pickle.dumps(x.astype(np.float32)),
                 "num_frames": len(x), "success": True,
                 "official_split": "test" if split[r.video_id] == "test" else "train", "source": "asl_citizen_new8"})
df = pd.concat([v4, pd.DataFrame(rows)], ignore_index=True)
df.to_parquet(OUT, index=False)
print(len(df), df.gloss.nunique(), df.official_split.value_counts().to_dict())
