#!/usr/bin/env python3
"""Statistical verdict on pretrained-init vs. from-scratch, across the 5-seed
sweep run by run_pretrain_finetune_sweep.sh. Same standard this project has
applied to every other comparison today (v1/v2 data, z-score/shoulder-width
normalization): only call it "validated" if a paired test is significant.
"""

import re
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_LOG = Path("/tmp/mouna_pretrain_sweep_results.log")


def main():
    lines = RESULTS_LOG.read_text().strip().splitlines()
    by_seed = {}
    for line in lines:
        m = re.match(r"SEED=(\d+) VARIANT=(\w+) TEST_ACC=([\d.]+)", line)
        if not m:
            continue
        seed, variant, acc = int(m.group(1)), m.group(2), float(m.group(3))
        by_seed.setdefault(seed, {})[variant] = acc

    seeds = sorted(by_seed)
    scratch = np.array([by_seed[s]["scratch"] for s in seeds])
    pretrained = np.array([by_seed[s]["pretrained"] for s in seeds])
    deltas = pretrained - scratch

    print("seed   scratch   pretrained   delta")
    for s, sc, pt, d in zip(seeds, scratch, pretrained, deltas):
        print(f"{s:4d}  {sc:7.1f}  {pt:10.1f}   {d:+.1f}")

    print()
    print(f"scratch     mean={scratch.mean():.2f}  std={scratch.std(ddof=1):.2f}")
    print(f"pretrained  mean={pretrained.mean():.2f}  std={pretrained.std(ddof=1):.2f}")
    print(f"delta (pretrained-scratch) mean={deltas.mean():.2f}  std={deltas.std(ddof=1):.2f}")
    print(f"pretrained wins: {(deltas>0).sum()}/{len(seeds)} seeds")

    t, p_t = stats.ttest_rel(pretrained, scratch)
    print(f"\npaired t-test: t={t:.3f}  p={p_t:.4f}")

    try:
        w, p_w = stats.wilcoxon(deltas)
        print(f"wilcoxon signed-rank: p={p_w:.4f}")
    except ValueError as e:
        p_w = None
        print(f"wilcoxon signed-rank: not computable ({e})")

    sig = (p_t < 0.05) or (p_w is not None and p_w < 0.05)
    direction = "beats" if deltas.mean() > 0 else "LOSES TO"
    print()
    if sig:
        print(f"=> SIGNIFICANT at p<0.05 on at least one test — pretrained-init "
              f"{direction} from-scratch (mean delta {deltas.mean():+.2f}pp). Real effect, not noise.")
    else:
        print("=> NOT significant at p<0.05 on either test — can't distinguish "
              "this from noise given this test set size. Treat as inconclusive, "
              "not confirmed.")


if __name__ == "__main__":
    main()
