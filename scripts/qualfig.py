"""Qualitative panel: does TED's contact partition absorb the failure?

Picks a failed can/mg demonstration, plots the end-effector trajectory with
TED's gripper-contact segment boundaries and the per-segment Bezier envelope
overlaid, and reports the per-segment envelope residual.

The Discussion claims that a failure event lands ON a segment boundary and
is therefore well-approximated on both sides. This script tests that claim
visually. IF THE PLOT DOES NOT SHOW THAT, THE DISCUSSION PARAGRAPH IS WRONG
and must be rewritten -- do not force the caption to match the story.

Usage:
    python qualfig.py --hdf5 datasets/can/mg/low_dim_sparse_v141.hdf5
    python qualfig.py --hdf5 ... --demo demo_17     # pick a specific one
"""
import argparse
import h5py, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metrics import contact_segments, bezier_envelope, ted, DT


def load(f, k):
    o = f[f"data/{k}/obs"]
    return (o["robot0_eef_pos"][:], o["robot0_eef_quat"][:],
            o["robot0_gripper_qpos"][:], f[f"data/{k}/rewards"][:])


def pick_demo(f, want_fail=True):
    """Median-TED failed demo: representative, not cherry-picked."""
    cands = []
    for k in f["data"]:
        r = f[f"data/{k}/rewards"][:]
        failed = not (r[-1] > 0)
        if failed == want_fail:
            p, q, g, _ = load(f, k)
            cands.append((ted(p, q, g), k))
    cands.sort()
    return cands[len(cands) // 2][1]


def main(a):
    with h5py.File(a.hdf5, "r") as f:
        key = a.demo or pick_demo(f)
        pos, quat, grip, rew = load(f, key)
        success = rew[-1] > 0

    T = len(pos)
    t = np.arange(T)
    width = np.abs(grip[:, 0] - grip[:, 1])
    segs = contact_segments(grip)

    rot = __import__("scipy.spatial.transform", fromlist=["Rotation"]) \
        .Rotation.from_quat(quat).as_rotvec()
    z = np.hstack([pos, 0.25 * rot])

    env = np.zeros_like(z)
    resid = []
    for s, e in segs:
        seg = z[s:e]
        if len(seg) < 4:
            env[s:e] = seg
            resid.append((s, e, 0.0))
            continue
        m = max(2, int(0.2 * len(seg)))
        parts = [bezier_envelope(seg[:m])]
        if len(seg) > 2 * m:
            parts.append(bezier_envelope(seg[m:-m]))
        parts.append(bezier_envelope(seg[-m:]))
        stacked = np.vstack(parts)[: len(seg)]
        env[s:e] = stacked
        resid.append((s, e, float(np.linalg.norm(seg - stacked, axis=1).mean())))

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(6.5, 5.2), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0, 1.2]})

    labels = ["x", "y", "z"]
    for i in range(3):
        ax1.plot(t, pos[:, i], lw=1.3, label=f"EE {labels[i]}", zorder=3)
        ax1.plot(t, env[:, i], lw=1.0, ls="--", color="k", alpha=0.55,
                 zorder=4, label="Bezier envelope" if i == 0 else None)
    for s, e in segs[1:]:
        ax1.axvline(s, color="crimson", lw=1.0, alpha=0.75, zorder=2)
    ax1.set_ylabel("EE position (m)", fontsize=8.5)
    ax1.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.9)
    ax1.set_title(
        f"{key}  ({'success' if success else 'FAILURE'});  "
        f"red lines = TED contact-segment boundaries", fontsize=9)
    ax1.tick_params(labelsize=8)

    ax2.plot(t, width, lw=1.3, color="#444444")
    for s, e in segs[1:]:
        ax2.axvline(s, color="crimson", lw=1.0, alpha=0.75)
    ax2.set_ylabel("gripper\nwidth", fontsize=8.5)
    ax2.tick_params(labelsize=8)

    per_step = np.linalg.norm(z - env, axis=1)
    ax3.plot(t, per_step, lw=1.2, color="#1f77b4")
    for s, e in segs[1:]:
        ax3.axvline(s, color="crimson", lw=1.0, alpha=0.75)
    for s, e, r in resid:
        ax3.hlines(r, s, e, color="darkorange", lw=1.8, zorder=5)
    ax3.set_ylabel("envelope\nresidual", fontsize=8.5)
    ax3.set_xlabel("timestep", fontsize=8.5)
    ax3.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig("fig_qual.pdf", bbox_inches="tight")
    fig.savefig("fig_qual.png", dpi=200, bbox_inches="tight")

    print(f"\ndemo {key}  ({'success' if success else 'FAILURE'}), T={T}")
    print(f"TED score: {ted(pos, quat, grip):.4f}")
    print(f"{len(segs)} contact segments, boundaries at "
          f"{[s for s, _ in segs[1:]]}")
    print("\nper-segment mean envelope residual "
          "(low = segment well-approximated, i.e. 'looks smooth'):")
    for s, e, r in resid:
        print(f"  [{s:4d},{e:4d})  len {e-s:4d}   residual {r:.4f}")
    print("\nCHECK: is the failure event at or near a boundary, with LOW "
          "residual on both sides?\nIf not, the Discussion mechanism claim "
          "is wrong and needs rewriting.")
    print("\nwrote fig_qual.pdf / fig_qual.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", required=True)
    p.add_argument("--demo", default=None)
    main(p.parse_args())
