#!/usr/bin/env python3
"""
Generate a fully autoregressive MagLIF animation locally. Uses the largest training
step size (max_time_diff=20). Plots the whole trajectory (no frame limit). Memory-safe:
checks available RAM and exits cleanly rather than crashing the machine.

Usage:
  # No-rollout checkpoint (default)
  python scripts/generate_maglif_animation_local.py \\
    --ckpt .ckpt/examples/time_dep/maglif_long_rollout_no_rollouts.best.pt \\
    --data_dir /path/to/parent_of_new_processed_maglif \\
    --output .results/examples/time_dep/maglif_no_rollout_ar20.gif

  # Long-rollout checkpoint (same script, different config + ckpt)
  python scripts/generate_maglif_animation_local.py \\
    --config config/examples/time_dep/maglif_long_rollout_eval.json \\
    --ckpt .ckpt/examples/time_dep/maglif_long_rollout.best.pt \\
    --data_dir /path/to/data \\
    --output .results/examples/time_dep/maglif_long_rollout_ar20.gif

  If --ckpt is omitted, the script tries to download from GCS (gsutil) using the
  path implied by the config (no-rollout or long-rollout).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch


def get_available_memory_mb():
    """Return available memory in MB (conservative: use free + buffers/cache on Linux)."""
    try:
        with open("/proc/meminfo") as f:
            data = f.read()
        total = free = avail = 0
        for line in data.splitlines():
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) // 1024
            elif line.startswith("MemFree:"):
                free = int(line.split()[1]) // 1024
            elif line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) // 1024
        return avail if avail > 0 else free
    except Exception:
        return 0


def estimate_animation_memory_mb(n_frames: int, n_points: int, n_channels: int) -> float:
    """Rough MB for gt_sequence + pred_sequence + matplotlib buffers (float32)."""
    bytes_per_frame = n_points * n_channels * 4  # float32
    seq_mb = (n_frames * bytes_per_frame * 2) / (1024 * 1024)  # gt + pred
    mpl_overhead = 100.0  # matplotlib figures/anim
    return seq_mb * 2 + mpl_overhead  # 2x for safety

# Project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Generate MagLIF AR animation locally (step=20, memory efficient)")
    parser.add_argument("--config", type=str, default="config/examples/time_dep/maglif_long_rollout_no_rollouts_eval.json",
                        help="Path to eval JSON config")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to best.pt. If unset, tries to download from GCS.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override dataset base_path (e.g. local path or /path/to/mount)")
    parser.add_argument("--output", type=str, default=".results/examples/time_dep/maglif_no_rollout_ar20.gif",
                        help="Output GIF path")
    parser.add_argument("--step", type=int, default=20,
                        help="Autoregressive step size (biggest training lag = 20)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Optional cap on AR timesteps (default: none; plot whole trajectory)")
    parser.add_argument("--min_available_mb", type=int, default=1024,
                        help="Abort if available memory drops below this (MB)")
    args = parser.parse_args()

    # Load config
    from omegaconf import OmegaConf
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(REPO_ROOT, config_path)
    with open(config_path) as f:
        config = OmegaConf.create(json.load(f))
    if args.data_dir:
        config.dataset.base_path = args.data_dir.rstrip("/")

    # Resolve checkpoint (infer GCS path from config if needed)
    ckpt_path = args.ckpt
    if not ckpt_path:
        default_ckpt = config.path.ckpt_path
        if not os.path.isabs(default_ckpt):
            default_ckpt = os.path.join(REPO_ROOT, default_ckpt)
        # GCS path: .ckpt/.../name.best.pt -> gs://zen_ml_checkpoints/gaot_<name_no_rollouts|long_rollout>/<name>.best.pt
        ckpt_basename = os.path.basename(default_ckpt)
        if "no_rollouts" in ckpt_basename:
            gcs_prefix = "gaot_maglif_long_rollout_no_rollouts"
        else:
            gcs_prefix = "gaot_maglif_long_rollout"
        gcs_uri = f"gs://zen_ml_checkpoints/{gcs_prefix}/{ckpt_basename}"
        if os.path.isfile(default_ckpt):
            ckpt_path = default_ckpt
        else:
            os.makedirs(os.path.dirname(default_ckpt), exist_ok=True)
            print(f"Checkpoint not found. Trying to download from GCS...")
            ret = os.system(f"gsutil -m cp {gcs_uri} {default_ckpt}")
            if ret != 0:
                print(f"Download failed. Provide --ckpt /path/to/{ckpt_basename}")
                sys.exit(1)
            ckpt_path = default_ckpt
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, ckpt_path)
    if not os.path.isfile(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    config.path.ckpt_path = ckpt_path

    # Ensure path config has absolute paths
    for key in ["ckpt_path", "loss_path", "result_path", "database_path"]:
        p = config.path[key]
        if not os.path.isabs(p):
            config.path[key] = os.path.join(REPO_ROOT, p)
        if key != "ckpt_path":
            os.makedirs(os.path.dirname(config.path[key]), exist_ok=True)

    # Build trainer (same as main.py)
    from main import prepare_arg
    from argparse import Namespace
    arg = Namespace(**OmegaConf.to_container(config, resolve=True))
    arg = prepare_arg(arg)

    from src.trainer.sequential_trainer import SequentialTrainer
    t = SequentialTrainer(arg)
    t.load_ckpt()
    t.model.to(t.device)
    t.model.eval()

    # Find first test file
    base = t.dataset_config.base_path
    name = t.dataset_config.name
    test_dir = os.path.join(base, name, "data", "test")
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.hdf5")))
    if not test_files:
        print(f"No test files in {test_dir}")
        sys.exit(1)
    test_file = test_files[0]
    print(f"Using test file: {test_file}")

    # Load full trajectory (memory: one file only)
    avail_mb = get_available_memory_mb()
    if avail_mb > 0 and avail_mb < args.min_available_mb:
        print(f"[MEMORY] Aborting: available memory {avail_mb} MB < {args.min_available_mb} MB")
        sys.exit(1)
    try:
        u_full, c_full, t_vals = t._load_full_trajectory_from_h5(test_file)
    except MemoryError:
        print("[MEMORY] Out of memory loading trajectory. Exiting safely.")
        sys.exit(1)
    T, N, Cu = u_full.shape
    t_vals = np.asarray(t_vals, dtype=np.float64)
    if T < 2:
        print("Trajectory too short")
        sys.exit(1)

    # Time indices: step=20 (biggest training lag). No frame limit by default.
    step = max(1, args.step)
    max_steps = T - 1
    if args.max_frames is not None and args.max_frames >= 1:
        max_steps = min(max_steps, args.max_frames)
    animation_time_indices = np.arange(0, max_steps + 1, step, dtype=int)
    if animation_time_indices[-1] > max_steps:
        animation_time_indices = animation_time_indices[animation_time_indices <= max_steps]
    if len(animation_time_indices) < 2:
        animation_time_indices = np.array([0, min(step, T - 1)], dtype=int)
    n_frames = len(animation_time_indices)
    est_mb = estimate_animation_memory_mb(n_frames, N, Cu)
    if avail_mb > 0 and est_mb > 0.5 * avail_mb:
        print(f"[MEMORY] Aborting: estimated animation memory {est_mb:.0f} MB > 50% of available ({avail_mb} MB)")
        sys.exit(1)
    print(f"AR step={step}, indices 0..{animation_time_indices[-1]} ({n_frames} frames), est. memory ~{est_mb:.0f} MB")

    # Temporarily set t_values for this file so AR uses correct times
    t_values_save = t.t_values
    t.t_values = t_vals

    try:
        # --- Fully autoregressive: initial state at t=0 only; every later frame is model prediction from previous prediction (no GT feeding). ---
        initial_u_denorm = u_full[0:1].to(t.device)  # [1, N, Cu]
        initial_u_norm = t._normalize_u(initial_u_denorm)

        # Time features for first step (0 -> animation_time_indices[1])
        st_mu = float(t.stats["start_time"]["mean"])
        st_sd = float(t.stats["start_time"]["std"])
        dt_mu = float(t.stats["time_diffs"]["mean"])
        dt_sd = float(t.stats["time_diffs"]["std"])
        i0 = int(animation_time_indices[0])
        i1 = int(animation_time_indices[1])
        t0_val = float(t_vals[i0])
        dt_val = float(t_vals[i1] - t_vals[i0])
        start_norm = (t0_val - st_mu) / (st_sd if st_sd > 0 else 1.0)
        diff_norm = (dt_val - dt_mu) / (dt_sd if dt_sd > 0 else 1.0)
        st_feat = torch.full((1, N, 1), start_norm, dtype=torch.float32, device=t.device)
        dt_feat = torch.full((1, N, 1), diff_norm, dtype=torch.float32, device=t.device)
        x0_autoreg = torch.cat([initial_u_norm, st_feat, dt_feat], dim=-1)  # [1, N, Cu+2]

        # Single rollout from x0; each step feeds previous prediction (fully autoregressive)
        with torch.no_grad():
            pred_autoreg = t._autoregressive_predict_trainer_side(x0_autoreg, animation_time_indices)
        # pred_autoreg: [1, K, N, Cu] with K = len(animation_time_indices) - 1 (no GT in between)
        initial_np = initial_u_denorm[0].cpu().numpy()  # [N, Cu]
        pred_np = pred_autoreg[0].cpu().numpy()         # [K, N, Cu]
        pred_sequence_ar = np.concatenate([initial_np[np.newaxis, :, :], pred_np], axis=0)  # [K+1, N, Cu]
        assert pred_sequence_ar.shape[0] == n_frames, "Fully AR sequence length must match time indices"

        # Ground truth at same indices (for comparison in animation only)
        gt_sequence = u_full[animation_time_indices].cpu().numpy()  # [K+1, N, Cu]

        # Coords (physical)
        if t.coord_mode == "fx":
            coords = t.data_processor.coord_scaler.inverse_transform(t.coord.cpu()).numpy()
        else:
            coords = None

        time_values = [float(t_vals[i]) for i in animation_time_indices]
        u_mean = t.stats["u"]["mean"].cpu().numpy().reshape(1, -1)
        u_std = t.stats["u"]["std"].cpu().numpy().reshape(1, -1)
        names = t.metadata.names.get("u") or [f"Var{i}" for i in range(Cu)]
        domain = getattr(t.metadata, "domain_x", None)

        out_path = args.output
        if not os.path.isabs(out_path):
            out_path = os.path.join(REPO_ROOT, out_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        from src.utils.plotting import create_sequential_animation_1d
        try:
            create_sequential_animation_1d(
                gt_sequence=gt_sequence,
                pred_sequence=pred_sequence_ar,
                coords=coords,
                save_path=out_path,
                input_data=initial_np,
                time_values=time_values,
                interval=80,
                names=names,
                domain=domain,
                show_error=True,
                u_mean=u_mean,
                u_std=u_std,
                max_frames=n_frames,  # use all frames (no downsampling)
                metadata=t.metadata,
            )
        except MemoryError:
            print("[MEMORY] Out of memory during animation creation. Exiting safely.")
            sys.exit(1)
        print(f"Animation saved: {out_path}")
    finally:
        t.t_values = t_values_save


if __name__ == "__main__":
    main()
