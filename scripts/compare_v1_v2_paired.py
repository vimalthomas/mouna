#!/usr/bin/env python3
"""Paired comparison of v1 vs v2 checkpoints on the IDENTICAL held-out test
set (confirmed same 273 video_ids), with McNemar's test — a single aggregate
accuracy delta on a 273-example test set doesn't tell you whether a ~1.4pp gap
is real or noise; this does.
"""

import pickle
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from train_transformer import KeypointDataset, MAX_SEQ_LEN  # noqa: E402
from mouna.models.transformer import PoseTransformer  # noqa: E402

SEED = 42


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = PoseTransformer(
        input_dim=cfg["input_dim"], d_model=cfg["d_model"], nhead=cfg["nhead"],
        num_layers=cfg["num_layers"], dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"], num_classes=cfg["num_classes"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt["label_to_idx"]


def predict_all(model, loader, device):
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            kp = batch["keypoints"].to(device)
            lengths = batch["seq_len"].to(device)
            y = batch["label"]
            out = model(kp, lengths)
            preds.extend(out.argmax(dim=1).cpu().numpy().tolist())
            labels.extend(y.numpy().tolist())
    return np.array(preds), np.array(labels)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    df1 = pd.read_parquet("data/silver/keypoints/keypoints.parquet")
    df1 = df1[df1["success"]].reset_index(drop=True)
    _, temp1 = train_test_split(df1, test_size=0.30, stratify=df1["gloss"], random_state=SEED)
    _, test_df = train_test_split(temp1, test_size=0.50, stratify=temp1["gloss"], random_state=SEED)
    test_df = test_df.reset_index(drop=True)

    model_v1, label_to_idx = load_model("data/best_model_transformer_v1.pth", device)
    model_v2, label_to_idx_v2 = load_model("data/best_model_transformer_v2.pth", device)
    assert label_to_idx == label_to_idx_v2, "label mappings differ between checkpoints — not directly comparable"

    ds = KeypointDataset(test_df, label_to_idx, max_length=MAX_SEQ_LEN, augment=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    preds_v1, labels = predict_all(model_v1, loader, device)
    preds_v2, labels2 = predict_all(model_v2, loader, device)
    assert (labels == labels2).all()

    correct_v1 = preds_v1 == labels
    correct_v2 = preds_v2 == labels

    acc_v1 = correct_v1.mean() * 100
    acc_v2 = correct_v2.mean() * 100
    print(f"v1 test acc (recomputed, paired set): {acc_v1:.1f}%  ({correct_v1.sum()}/{len(labels)})")
    print(f"v2 test acc (recomputed, paired set): {acc_v2:.1f}%  ({correct_v2.sum()}/{len(labels)})")

    # McNemar's test: only discordant pairs matter (v1 right+v2 wrong vs v1 wrong+v2 right)
    v1_right_v2_wrong = int(((correct_v1) & (~correct_v2)).sum())
    v1_wrong_v2_right = int(((~correct_v1) & (correct_v2)).sum())
    both_right = int((correct_v1 & correct_v2).sum())
    both_wrong = int((~correct_v1 & ~correct_v2).sum())
    print(f"\nBoth right: {both_right}  Both wrong: {both_wrong}")
    print(f"v1 right -> v2 wrong (v2 lost): {v1_right_v2_wrong}")
    print(f"v1 wrong -> v2 right (v2 gained): {v1_wrong_v2_right}")

    n = v1_right_v2_wrong + v1_wrong_v2_right
    if n == 0:
        print("\nNo discordant pairs — models identical on this test set.")
        return
    # Exact McNemar (binomial test on discordant pairs)
    from scipy.stats import binomtest
    result = binomtest(v1_wrong_v2_right, n, p=0.5, alternative="two-sided")
    print(f"\nMcNemar's exact test (discordant pairs n={n}): p = {result.pvalue:.4f}")
    if result.pvalue < 0.05:
        print("=> statistically significant at p<0.05 — the improvement looks real, not noise.")
    else:
        print("=> NOT statistically significant at p<0.05 — can't distinguish this from noise "
              "given this test set size. Treat the +1.4pp as suggestive, not confirmed.")


if __name__ == "__main__":
    main()
