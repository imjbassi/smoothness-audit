"""Demonstration-quality metrics for the curation audit.

Convention: every function returns a BADNESS score (higher = worse).
SAL is negated internally so all metrics point the same direction.

v2: adds length-normalized variants (sal_norm, ted_norm, path_len_norm)
to test how much of each metric's apparent signal is episode duration.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from fastdtw import fastdtw

DT = 0.05  # robomimic control rate is 20 Hz


# ---------------------------------------------------------------- cheap metrics

def path_length(pos):
    return float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())


def path_length_norm(pos):
    """Mean per-step displacement. Removes the duration effect."""
    return path_length(pos) / max(len(pos) - 1, 1)


def mean_jerk(pos, dt=DT):
    if len(pos) < 4:
        return 0.0
    d3 = pos[3:] - 3 * pos[2:-1] + 3 * pos[1:-2] - pos[:-3]
    return float(np.linalg.norm(d3 / dt**3, axis=1).mean())


def sal(pos, dt=DT, eps=1e-5):
    """Spectral arc length of the speed profile. Returns BADNESS (= -SAL)."""
    v = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt
    if len(v) < 4:
        return 0.0
    T = len(v)
    V = np.abs(np.fft.rfft(v))
    f = np.fft.rfftfreq(T, d=dt)
    L = np.log(V + eps)
    arc = np.sqrt(np.diff(f) ** 2 + np.diff(L) ** 2).sum()
    return float(arc)  # SAL = -arc, so badness = arc


def sal_norm(pos, dt=DT):
    """SAL per timestep. The arc length sums over T/2 spectral bins, so the
    raw score grows with episode duration regardless of execution quality."""
    return sal(pos, dt) / max(len(pos) - 1, 1)


# ---------------------------------------------------------------------- TED

def contact_segments(grip, thresh=None):
    """Partition timesteps by gripper open/closed state.

    grip: (T,2) gripper joint positions. Width = |q0 - q1|.
    Returns list of (start, end) index pairs.
    """
    w = np.abs(grip[:, 0] - grip[:, 1])
    if thresh is None:
        thresh = 0.5 * (w.max() + w.min())
    closed = (w < thresh).astype(int)
    bounds = [0] + list(np.where(np.diff(closed) != 0)[0] + 1) + [len(closed)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _bezier(ctrl, n):
    from math import comb
    K = len(ctrl) - 1
    t = np.linspace(0, 1, n)[:, None]
    return sum(comb(K, i) * (1 - t) ** (K - i) * t**i * ctrl[i]
               for i in range(K + 1))


def bezier_envelope(x, K0=10, M_ref=20):
    """Greedy refine-and-keep-best Bezier fit (rinse Alg. 1, simplified)."""
    n = len(x)
    if n < 4:
        return x.copy()
    idx = sorted(set(np.linspace(0, n - 1, min(K0, n)).astype(int)))
    best, best_score = None, np.inf
    for _ in range(M_ref):
        curve = _bezier(x[idx], n)
        resid = np.linalg.norm(x - curve, axis=1)
        score = resid.mean() + 0.1 * len(idx) / n
        if score < best_score:
            best, best_score = curve, score
        j = int(np.argmax(resid))
        if j in idx or len(idx) >= n:
            break
        idx = sorted(idx + [j])
    return best


def ted(pos, quat, grip, w_ori=0.25, r=0.2):
    """Trajectory-Envelope Distance. Higher = less smooth = worse.

    Already length-normalized by |P| (the DTW path length), but |P| grows
    with T, so ted_norm below is a stricter control.
    """
    rot = R.from_quat(quat).as_rotvec()
    z = np.hstack([pos, w_ori * rot])
    env = np.zeros_like(z)
    for s, e in contact_segments(grip):
        seg = z[s:e]
        if len(seg) < 4:
            env[s:e] = seg
            continue
        m = max(2, int(r * len(seg)))
        parts = [bezier_envelope(seg[:m])]
        if len(seg) > 2 * m:
            parts.append(bezier_envelope(seg[m:-m]))
        parts.append(bezier_envelope(seg[-m:]))
        stacked = np.vstack(parts)
        env[s:e] = stacked[: len(seg)]
    dist, path = fastdtw(z, env, dist=lambda a, b: np.linalg.norm(a - b))
    return float(dist / len(path))


def ted_norm(pos, quat, grip):
    """TED per timestep."""
    return ted(pos, quat, grip) / max(len(pos) - 1, 1)


ALL = {
    "ted": lambda p, q, g: ted(p, q, g),
    "ted_norm": lambda p, q, g: ted_norm(p, q, g),
    "sal": lambda p, q, g: sal(p),
    "sal_norm": lambda p, q, g: sal_norm(p),
    "jerk": lambda p, q, g: mean_jerk(p),
    "path_len": lambda p, q, g: path_length(p),
    "path_len_norm": lambda p, q, g: path_length_norm(p),
}
