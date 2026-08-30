"""Score every demo in a robomimic hdf5, report AUROC vs the success label,
and write filter keys for the downstream training runs.

Usage:
    python score.py --hdf5 datasets/can/paired/low_dim_v141.hdf5 --keep 0.5
"""
import argparse, json
import h5py, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from metrics import ALL


def load(f, k):
    o = f[f"data/{k}/obs"]
    return (o["robot0_eef_pos"][:],
            o["robot0_eef_quat"][:],
            o["robot0_gripper_qpos"][:])


def success(f, k):
    return float(f[f"data/{k}/rewards"][-1] > 0)


def main(a):
    rows = []
    with h5py.File(a.hdf5, "r") as f:
        keys = list(f["data"].keys())
        for k in keys:
            p, q, g = load(f, k)
            row = {"demo": k, "T": len(p), "success": success(f, k)}
            row.update({n: fn(p, q, g) for n, fn in ALL.items()})
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("scores.csv", index=False)

    print(f"\n{len(df)} demos, {int(df.success.sum())} successful "
          f"({df.success.mean():.1%})\n")
    print("AUROC (label=success, score negated so higher score = predicted clean)")
    print("0.50 = chance. Below 0.50 = INVERTED.\n")
    for n in ALL:
        auc = roc_auc_score(df.success, -df[n])
        # correlation with episode length: the confound to rule out
        rho = np.corrcoef(df[n], df["T"])[0, 1]
        print(f"  {n:10s} AUROC {auc:.3f}   corr(score, len) {rho:+.2f}")
    print(f"\n  {'length':10s} AUROC "
          f"{roc_auc_score(df.success, -df['T']):.3f}   <- baseline: episode length alone")

    # filter keys
    n_keep = int(a.keep * len(df))
    sel = {n: df.nsmallest(n_keep, n).demo.tolist() for n in ALL}
    sel["oracle"] = df[df.success == 1].demo.tolist()[:n_keep]
    rng = np.random.default_rng(0)
    sel["random"] = list(rng.choice(df.demo, n_keep, replace=False))
    sel["nocuration"] = list(rng.choice(df.demo, n_keep, replace=False))

    with h5py.File(a.hdf5, "a") as f:
        for n, demos in sel.items():
            name = f"{n}_top{int(a.keep*100)}"
            if f"mask/{name}" in f:
                del f[f"mask/{name}"]
            f.create_dataset(f"mask/{name}",
                             data=np.array(demos, dtype="S"))
    print("\nfilter keys written:",
          ", ".join(f"{n}_top{int(a.keep*100)}" for n in sel))
    json.dump({k: len(v) for k, v in sel.items()}, open("filters.json", "w"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", required=True)
    p.add_argument("--keep", type=float, default=0.5)
    main(p.parse_args())
