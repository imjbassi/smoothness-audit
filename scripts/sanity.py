import h5py, numpy as np, sys
f = h5py.File(sys.argv[1])
ks = list(f["data"])
print(len(ks), "demos")
print("success rate", np.mean([f[f"data/{k}/rewards"][-1] > 0 for k in ks]))
L = [f[f"data/{k}/actions"].shape[0] for k in ks]
print("len range", min(L), max(L))
print("obs keys", list(f[f"data/{ks[0]}/obs"].keys()))