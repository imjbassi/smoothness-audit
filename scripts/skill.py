"""Rank operator skill tiers (better / okay / worse) with the same metrics.

Usage:
    python skill.py scores_mh.csv datasets/can/mh/low_dim_v141.hdf5
"""
import sys
import h5py, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

csv_path, hdf5_path = sys.argv[1], sys.argv[2]
df = pd.read_csv(csv_path).set_index("demo")

lab = {}
with h5py.File(hdf5_path, "r") as f:
    for tier, v in [("worse", 0), ("okay", 1), ("better", 2)]:
        for d in f[f"mask/{tier}"][:]:
            lab[d.decode()] = v

df["skill"] = [lab.get(d, np.nan) for d in df.index]
df = df.dropna(subset=["skill"])
print(f"{len(df)} labeled; tier counts {df.skill.value_counts().to_dict()}")
print("(0=worse, 1=okay, 2=better)\n")

metrics = ["ted", "ted_norm", "sal", "sal_norm", "jerk", "path_len", "path_len_norm", "T"]

bw = df[df.skill != 1]
print("better vs worse  (>0.50 = metric ranks skill correctly)")
for m in metrics:
    print(f"  {m:9s} AUROC {roc_auc_score(bw.skill == 2, -bw[m]):.3f}")

print("\nbetter vs rest")
for m in metrics:
    print(f"  {m:9s} AUROC {roc_auc_score(df.skill == 2, -df[m]):.3f}")

print("\nmean score by tier")
print(df.groupby("skill")[metrics].mean().round(3))
