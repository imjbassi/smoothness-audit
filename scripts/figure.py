"""Main figure: detection AUROC under episode-length controls.

Panel A: robomimic can/mh, better-vs-worse operator AUROC across four
duration bands of increasing tightness. Shaded = 95% bootstrap CI.
Panel B: robomimic can/mg (all episodes exactly 150 steps), AUROC
against the task-success label.

All numbers are from the experiments in this project (scores_*.csv,
band.py output). Rerun band.py / score.py to regenerate them.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- Panel A data: can/mh better-vs-worse, four bands ----------------------
bands = ["full\n(n=200)", "120–220\n(n=98)", "130–190\n(n=74)", "140–175\n(n=40)"]
x = np.arange(4)

series = {
    "SAL":  dict(v=[0.952, 0.793, 0.759, 0.628],
                 lo=[0.922, 0.690, 0.628, 0.378],
                 hi=[0.975, 0.881, 0.875, 0.856],
                 color="#d62728", marker="o"),
    "jerk": dict(v=[0.584, 0.862, 0.906, 0.896],
                 lo=[0.505, 0.783, 0.832, 0.783],
                 hi=[0.665, 0.928, 0.965, 0.980],
                 color="#1f77b4", marker="s"),
    "TED":  dict(v=[0.558, 0.653, 0.631, 0.359],
                 lo=[0.480, 0.522, 0.468, 0.147],
                 hi=[0.636, 0.776, 0.779, 0.583],
                 color="#2ca02c", marker="^"),
    "episode length (T)":
            dict(v=[0.957, 0.801, 0.753, 0.485],
                 lo=[0.929, 0.689, 0.594, 0.227],
                 hi=[0.980, 0.901, 0.891, 0.740],
                 color="#7f7f7f", marker="D"),
}

# ---- Panel B data: can/mg, success label, T constant = 150 -----------------
mg_names = ["jerk", "SAL", "path len", "TED", "length"]
mg_vals  = [0.706, 0.454, 0.421, 0.376, 0.500]
mg_cols  = ["#1f77b4", "#d62728", "#9467bd", "#2ca02c", "#7f7f7f"]

# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.35, 1.0]})

for name, s in series.items():
    ls = "--" if "length" in name else "-"
    ax1.plot(x, s["v"], ls, marker=s["marker"], ms=4.5, lw=1.6,
             color=s["color"], label=name, zorder=3)
    ax1.fill_between(x, s["lo"], s["hi"], color=s["color"], alpha=0.13, lw=0)

ax1.axhline(0.5, color="k", lw=0.8, ls=":", zorder=1)
ax1.set_xticks(x)
ax1.set_xticklabels(bands, fontsize=7.5)
ax1.set_ylim(0.05, 1.02)
ax1.set_ylabel("AUROC (better vs. worse operator)", fontsize=8.5)
ax1.set_xlabel("episode-length band (tighter \u2192)", fontsize=8.5)
ax1.set_title("(a) can/mh: real operator labels", fontsize=9)
ax1.legend(fontsize=7, loc="lower left", framealpha=0.9)
ax1.tick_params(labelsize=8)

bars = ax2.bar(np.arange(len(mg_names)), mg_vals, color=mg_cols, width=0.62)
ax2.axhline(0.5, color="k", lw=0.8, ls=":")
for b, v in zip(bars, mg_vals):
    ax2.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
             ha="center", fontsize=7.5)
ax2.set_xticks(np.arange(len(mg_names)))
ax2.set_xticklabels(mg_names, fontsize=8)
ax2.set_ylim(0.0, 0.8)
ax2.set_ylabel("AUROC (task success)", fontsize=8.5)
ax2.set_title("(b) can/mg: T fixed at 150 steps", fontsize=9)
ax2.tick_params(labelsize=8)

fig.tight_layout()
fig.savefig("fig_main.pdf", bbox_inches="tight")
fig.savefig("fig_main.png", dpi=220, bbox_inches="tight")
print("wrote fig_main.pdf / fig_main.png")
