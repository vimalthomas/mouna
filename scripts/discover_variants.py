#!/usr/bin/env python3
"""Find signs whose approved clips fall into distinct ways of signing
(variants), for a human to confirm before grading uses them.

For each sign: embed every approved clip with the recognizer, try 2 and 3
clusters, and propose a split only when ALL of these hold:
  - the clusters are well separated (silhouette >= MIN_SILHOUETTE)
  - every cluster has >= MIN_CLIPS clips and >= MIN_SHARE of the sign
  - the clusters differ in something a learner would do differently:
    a settled CP target (handshape / orientation) or the number of hands
Anything else is signer noise and stays one target.

Output: data/references/variant_proposals.json (sign -> clusters, each with
its clip ids, size, CP targets, hand count, and the clip nearest its centre
to show the reviewer). Nothing in the app reads this until confirmed.

Run from repo root:  ./venv/bin/python scripts/discover_variants.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

sys.path[:0] = ["src", "scripts"]
from build_sign_references_v2 import (  # noqa: E402
    DECISIONS_PATH, MAX_SEQ_LEN, MODEL_PATH, V4CANON_PATH, load_raw_lookup,
)
from mouna.coach.parameters import extract_dominant_hand_signal, majority_targets  # noqa: E402
from mouna.data.handedness import canonicalize_handedness  # noqa: E402
from mouna.data.part_normalize import normalize_per_part  # noqa: E402
from mouna.models.transformer import PoseTransformer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/references/variant_proposals.json"
MIN_SILHOUETTE = 0.25
MIN_CLIPS = 3
MIN_SHARE = 0.2
GRADED_PARTS = ("handshape", "orientation")  # location isn't trusted yet


def hands_used(canon: np.ndarray) -> int:
    a = canon.reshape(len(canon), 75, 3)
    l, r = np.any(a[:, 33:54] != 0, axis=(1, 2)), np.any(a[:, 54:75] != 0, axis=(1, 2))
    any_hand = (l | r).sum()
    return 2 if any_hand and (l & r).sum() / any_hand >= 0.3 else 1


def main() -> None:
    ckpt = torch.load(ROOT / MODEL_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = PoseTransformer(**{k: v for k, v in cfg.items() if k != "num_classes"}, num_classes=cfg["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dec = pd.read_csv(ROOT / DECISIONS_PATH)
    ok = set(dec[dec.decision == "ok"].video_id.astype(str))
    v4 = pd.read_parquet(ROOT / V4CANON_PATH, columns=["video_id", "gloss", "source"])
    v4["vid"] = v4.video_id.astype(str)
    v4 = v4[v4.vid.isin(ok) & (v4.source != "openpose_recovered")]
    raw = load_raw_lookup()

    proposals, summary = {}, []
    for word, sub in sorted(v4.groupby("gloss")):
        vids, embs, sigs, hands = [], [], [], []
        for vid in sub.vid:
            flat = raw.get(vid)
            if flat is None:
                continue
            canon = canonicalize_handedness(np.asarray(flat, dtype=np.float32), target="right")
            sig = extract_dominant_hand_signal(canon)
            if sig is None:
                continue
            normed = normalize_per_part(canon)
            T = min(len(normed), MAX_SEQ_LEN)
            x = np.zeros((MAX_SEQ_LEN, normed.shape[1]), dtype=np.float32)
            x[:T] = normed[:T]
            with torch.no_grad():
                e = model.forward_features(torch.from_numpy(x)[None], torch.tensor([T]))[0].numpy()
            vids.append(vid); embs.append(e / (np.linalg.norm(e) + 1e-8)); sigs.append(sig); hands.append(hands_used(canon))
        n = len(vids)
        if n < 2 * MIN_CLIPS:
            summary.append((word, n, "too few clips"))
            continue
        X = np.stack(embs)
        best = None
        for k in (2, 3):
            if n < k * MIN_CLIPS:
                continue
            labels = AgglomerativeClustering(n_clusters=k, linkage="average", metric="cosine").fit_predict(X)
            sizes = np.bincount(labels)
            if sizes.min() < MIN_CLIPS or sizes.min() / n < MIN_SHARE:
                continue
            s = silhouette_score(X, labels, metric="cosine")
            if s >= MIN_SILHOUETTE and (best is None or s > best[0]):
                best = (s, labels)
        if best is None:
            summary.append((word, n, "one way"))
            continue
        s, labels = best
        clusters = []
        for c in range(labels.max() + 1):
            idx = np.where(labels == c)[0]
            cen = X[idx].mean(0)
            rep = idx[int(np.argmax(X[idx] @ cen))]
            t = majority_targets([sigs[i] for i in idx])
            clusters.append({
                "size": int(len(idx)),
                "video_ids": [vids[i] for i in idx],
                "representative_video_id": vids[rep],
                "n_hands": int(round(np.mean([hands[i] for i in idx]))),
                "targets": {p: t[p]["value"] for p in GRADED_PARTS if t.get(p, {}).get("settled")},
            })
        differs = []
        if len({c["n_hands"] for c in clusters}) > 1:
            differs.append("hands")
        for p in GRADED_PARTS:
            vals = {c["targets"].get(p) for c in clusters}
            if None not in vals and len(vals) > 1:
                differs.append(p)
        if not differs:
            summary.append((word, n, f"clusters (sil {s:.2f}) but same CP: noise"))
            continue
        proposals[word] = {"silhouette": round(float(s), 3), "differs_in": differs, "clusters": clusters}
        summary.append((word, n, f"VARIANTS x{len(clusters)} (sil {s:.2f}) differ in {', '.join(differs)}"))

    for w, n, msg in summary:
        print(f"  {w:12s} {n:4d} clips  {msg}")
    OUT.write_text(json.dumps(proposals, indent=2))
    print(f"\n{len(proposals)} signs with proposed variants -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
