"""Collect final training results across all curation conditions and seeds.

Reads robomimic's per-run log.txt files, extracts the rollout success rates,
and reports mean / std / bootstrap CI per condition.

Usage:
    python collect.py
    python collect.py --root results_train
"""
import argparse, glob, json, os, re
import numpy as np
import pandas as pd

RUN_RE = re.compile(r"^(?P<cond>[a-z_]+)_s(?P<seed>\d+)$")


def parse_run(run_dir):
    """Return list of (epoch, success_rate) from a run's log."""
    hits = []
    # robomimic names checkpoints model_epoch_{N}_{env}_success_{X}.pth
    for p in glob.glob(os.path.join(run_dir, "*", "models", "*success*.pth")):
        m = re.search(r"model_epoch_(\d+).*?success_([0-9.]+)\.pth$",
                      os.path.basename(p))
        if m:
            hits.append((int(m.group(1)), float(m.group(2).rstrip("."))))
    if hits:
        return hits
    # fall back to scraping the text log
    for p in glob.glob(os.path.join(run_dir, "*", "logs", "log.txt")):
        txt = open(p, errors="ignore").read()
        for m in re.finditer(r'"Success_Rate":\s*([0-9.]+)', txt):
            hits.append((None, float(m.group(1))))
    return hits


def main(a):
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(a.root, "*"))):
        name = os.path.basename(run_dir)
        m = RUN_RE.match(name)
        if not m:
            continue
        hits = parse_run(run_dir)
        if not hits:
            print(f"  (no results yet for {name})")
            continue
        best = max(h[1] for h in hits)
        final = hits[-1][1]
        rows.append({"condition": m.group("cond"), "seed": int(m.group("seed")),
                     "best_success": best, "final_success": final,
                     "n_evals": len(hits)})

    if not rows:
        print("No completed runs found.")
        return

    df = pd.DataFrame(rows)
    df.to_csv("train_results.csv", index=False)
    print(f"\n{len(df)} runs across {df.condition.nunique()} conditions\n")

    rng = np.random.default_rng(0)
    print("best rollout success rate, mean +/- std [95% bootstrap CI]")
    order = ["oracle", "jerk", "ted", "sal", "path_len",
             "random", "nocuration"]
    for cond in [c for c in order if c in set(df.condition)]:
        v = df[df.condition == cond].best_success.values
        boot = [rng.choice(v, len(v), replace=True).mean() for _ in range(5000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {cond:11s} {v.mean():.3f} +/- {v.std(ddof=1):.3f} "
              f"[{lo:.3f}, {hi:.3f}]   n={len(v)}")

    print("\nper-seed detail")
    print(df.pivot_table(index="condition", columns="seed",
                         values="best_success").round(3))
    print("\nwrote train_results.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results_train")
    main(p.parse_args())
