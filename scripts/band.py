"""Duration-matched skill ranking with bootstrap confidence intervals.

Restricts the multi-human demos to a narrow episode-length band so that
episode duration carries little information, then asks how well each
metric separates 'better' from 'worse' operators inside that band.

Always read the T row first: if T is far from 0.50, the band has NOT
controlled for duration and the other rows cannot be interpreted as
duration-free.

Usage:
    python band.py scores_mh.csv datasets/can/mh/low_dim_v141.hdf5 140 175
    python band.py scores_mh.csv datasets/can/mh/low_dim_v141.hdf5 0 99999
"""
import sys
import h5py, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

METRICS = ["ted", "ted_norm", "sal", "sal_norm", "jerk",
           "path_len", "path_len_norm", "T"]
N_BOOT = 2000
SEED = 0


def main(csv_path, hdf5_path, lo, hi):
    df = pd.read_csv(csv_path).set_index("demo")

    lab = {}
    with h5py.File(hdf5_path, "r") as f:
        for tier, v in [("worse", 0), ("okay", 1), ("better", 2)]:
            for d in f[f"mask/{tier}"][:]:
                lab[d.decode()] = v

    df["skill"] = [lab.get(d, np.nan) for d in df.index]
    df = df.dropna(subset=["skill"])

    band = df[df["T"].between(lo, hi)]
    bw = band[band.skill != 1]
    n_pos = int((bw.skill == 2).sum())
    n_neg = int((bw.skill == 0).sum())

    print(f"\nband T in [{lo}, {hi}]")
    print(f"{len(bw)} demos: {n_pos} better / {n_neg} worse")
    print(f"T range in band: {bw['T'].min():.0f} to {bw['T'].max():.0f}")
    if min(n_pos, n_neg) < 15:
        print(f"WARNING: only {min(n_pos, n_neg)} in the smaller class. "
              f"Intervals will be very wide.")
    if len(bw) == 0 or n_pos == 0 or n_neg == 0:
        print("Not enough data in this band.")
        return

    print("\nbetter vs worse (>0.50 = ranks skill correctly), "
          "95% bootstrap CI")
    y = (bw.skill == 2).values
    rng = np.random.default_rng(SEED)

    for m in [m for m in METRICS if m in bw.columns]:
        s = -bw[m].values  # metrics are badness, negate so higher = better
        point = roc_auc_score(y, s)
        boot = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) < 2:
                continue
            boot.append(roc_auc_score(y[i], s[i]))
        if boot:
            lo_ci, hi_ci = np.percentile(boot, [2.5, 97.5])
            flag = ""
            if m == "T":
                flag = "  <- must be near 0.50 for the band to be a control"
            print(f"  {m:14s} {point:.3f}  [{lo_ci:.3f}, {hi_ci:.3f}]{flag}")
        else:
            print(f"  {m:14s} {point:.3f}  [CI failed]")

    print("\nmean score by tier (within band)")
    cols = [m for m in METRICS if m in bw.columns]
    print(band.groupby("skill")[cols].mean().round(4))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))