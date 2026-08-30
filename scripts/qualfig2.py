import argparse
import h5py, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from metrics import contact_segments, bezier_envelope, ted

def load(f, k):
    o = f[f"data/{k}/obs"]
    return (o["robot0_eef_pos"][:], o["robot0_eef_quat"][:],
            o["robot0_gripper_qpos"][:], f[f"data/{k}/rewards"][:])

def envelope_of(pos, quat, grip):
    rot = R.from_quat(quat).as_rotvec()
    z = np.hstack([pos, 0.25 * rot])
    env = np.zeros_like(z)
    resid = []
    for s, e in contact_segments(grip):
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
        st = np.vstack(parts)[: len(seg)]
        env[s:e] = st
        resid.append((s, e, float(np.linalg.norm(seg - st, axis=1).mean())))
    return z, env, resid, contact_segments(grip)

def column(axtop, axbot, pos, quat, grip, title, score, color):
    z, env, resid, segs = envelope_of(pos, quat, grip)
    t = np.arange(len(pos))
    first = title.startswith("Success")
    for i, lab in enumerate(["x", "y", "z"]):
        axtop.plot(t, pos[:, i], lw=1.2, zorder=3,
                   label=f"EE {lab}" if first else None)
        axtop.plot(t, env[:, i], lw=0.9, ls="--", color="k", alpha=0.5,
                   zorder=4, label="envelope" if (i == 0 and first) else None)
    for s, _ in segs[1:]:
        axtop.axvline(s, color="crimson", lw=0.8, alpha=0.6, zorder=2)
    axtop.set_title(f"{title}   TED = {score:.4f}", fontsize=9, color=color)
    axtop.tick_params(labelsize=7.5)
    per = np.linalg.norm(z - env, axis=1)
    axbot.plot(t, per, lw=1.0, color="#1f77b4")
    for s, e, r in resid:
        axbot.hlines(r, s, e, color="darkorange", lw=2.0, zorder=5)
    for s, _ in segs[1:]:
        axbot.axvline(s, color="crimson", lw=0.8, alpha=0.6)
    axbot.set_ylim(-0.02, 0.85)
    axbot.set_xlabel("timestep", fontsize=8.5)
    axbot.tick_params(labelsize=7.5)
    return max(r for _, _, r in resid)

def main(a):
    with h5py.File(a.hdf5, "r") as f:
        sp, sq, sg, _ = load(f, a.success)
        fp, fq, fg, _ = load(f, a.failure)
        s_score = ted(sp, sq, sg); f_score = ted(fp, fq, fg)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.0), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 1.2]})
    smax = column(axes[0][0], axes[1][0], sp, sq, sg,
                  f"Success ({a.success})", s_score, "#2b7a2b")
    fmax = column(axes[0][1], axes[1][1], fp, fq, fg,
                  f"Failure ({a.failure})", f_score, "#a52020")
    axes[0][0].set_ylabel("EE position (m)", fontsize=8.5)
    axes[1][0].set_ylabel("envelope\nresidual", fontsize=8.5)
    axes[0][0].legend(fontsize=6.5, ncol=2, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig("fig_qual_pair.pdf", bbox_inches="tight")
    fig.savefig("fig_qual_pair.png", dpi=220, bbox_inches="tight")
    print(f"success {a.success}: TED {s_score:.4f}, max seg residual {smax:.4f}")
    print(f"failure {a.failure}: TED {f_score:.4f}, max seg residual {fmax:.4f}")
    print("wrote fig_qual_pair.pdf / fig_qual_pair.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", required=True)
    p.add_argument("--success", default="demo_1832")
    p.add_argument("--failure", default="demo_753")
    main(p.parse_args())
