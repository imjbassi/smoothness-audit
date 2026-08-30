"""Write one robomimic config per (curation condition, seed).

Usage:
    python gen_configs.py --hdf5 datasets/can/mg/low_dim_sparse_v141.hdf5
    bash run_all.sh
"""
import argparse, copy, json, os

CONDITIONS = ["ted", "sal", "jerk", "path_len", "oracle", "random", "nocuration"]
SEEDS = [0, 1, 2, 3, 4]
KEEP = 50  # matches the *_top50 filter keys written by score.py


def base(hdf5, out_dir):
    return {
        "algo_name": "bc",
        "experiment": {
            "name": "PLACEHOLDER",
            "validate": False,
            "logging": {"terminal_output_to_txt": True, "log_tb": True,
                        "log_wandb": False},
            "save": {"enabled": True, "every_n_epochs": 100,
                     "on_best_rollout_success_rate": True},
            "epoch_every_n_steps": 500,
            "rollout": {"enabled": True, "n": 50, "horizon": 400,
                        "rate": 150, "warmstart": 0,
                        "terminate_on_success": True},
            "render_video": False,
        },
        "train": {
            "data": [{"path": hdf5}],
            "output_dir": os.path.abspath(out_dir),
            "num_data_workers": 0,
            "hdf5_cache_mode": "all",
            "hdf5_use_swmr": True,
            "hdf5_filter_key": "PLACEHOLDER",
            "seq_length": 1,
            "dataset_keys": ["actions", "rewards", "dones"],
            "cuda": True,
            "batch_size": 100,
            "num_epochs": 600,
            "seed": 0,
        },
        "algo": {
            "optim_params": {"policy": {
                "optimizer_type": "adam",
                "learning_rate": {"initial": 1e-4, "decay_factor": 0.1,
                                  "epoch_schedule": [],
                                  "scheduler_type": "multistep"},
                "regularization": {"L2": 0.0}}},
            "loss": {"l2_weight": 1.0, "l1_weight": 0.0, "cos_weight": 0.0},
            "actor_layer_dims": [1024, 1024],
            "gmm": {"enabled": True, "num_modes": 5, "min_std": 0.0001,
                    "std_activation": "softplus", "low_noise_eval": True},
        },
        "observation": {"modalities": {"obs": {"low_dim": [
            "robot0_eef_pos", "robot0_eef_quat",
            "robot0_gripper_qpos", "object"]}}},
    }


def main(a):
    os.makedirs("configs", exist_ok=True)
    lines = ["#!/bin/bash", "set -e", ""]
    for cond in CONDITIONS:
        for seed in SEEDS:
            cfg = copy.deepcopy(base(a.hdf5, a.out))
            name = f"{cond}_s{seed}"
            cfg["experiment"]["name"] = name
            cfg["train"]["hdf5_filter_key"] = f"{cond}_top{KEEP}"
            cfg["train"]["seed"] = seed
            path = f"configs/{name}.json"
            with open(path, "w") as f:
                json.dump(cfg, f, indent=4)
            lines.append(f"echo '=== {name} ==='")
            lines.append(f"python robomimic/scripts/train.py --config {path}")
    with open("run_all.sh", "w") as f:
        f.write("\n".join(lines) + "\n")
    n = len(CONDITIONS) * len(SEEDS)
    print(f"wrote {n} configs to configs/ and run_all.sh")
    print("conditions:", ", ".join(CONDITIONS))
    print("seeds:", SEEDS)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", required=True)
    p.add_argument("--out", default="results_train")
    main(p.parse_args())
