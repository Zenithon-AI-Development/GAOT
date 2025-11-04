#!/usr/bin/env python
"""
Generate perturbed initial condition trajectories for uncertainty quantification.
Creates multiple full trajectories from small perturbations of the initial snapshot.
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path


def load_config_and_model(config_path, seed=42):
    """Load config and initialize trainer with model."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from main import FileParser, prepare_arg
    from src.trainer.sequential_trainer import SequentialTrainer
    
    # Parse config
    parser = FileParser(config_path)
    arg = parser.parse_args()
    
    # Prepare arg (sets up paths, etc.)
    arg = prepare_arg(arg)
    
    # Override seed
    arg.setup.seed = seed
    
    # Disable wandb
    if hasattr(arg, 'wandb'):
        arg.wandb['enabled'] = False
    
    trainer = SequentialTrainer(arg)
    
    if arg.setup.ckpt:
        trainer.load_ckpt()
    
    trainer.model.eval()
    trainer.model.to(trainer.device)
    
    return trainer


def load_initial_snapshot(trainer, h5_path):
    """Load initial snapshot and conditioning from HDF5 file."""
    u_full, c_full, t_vals = trainer._load_full_trajectory_from_h5(h5_path)
    
    # Get initial snapshot
    u_0 = u_full[0]  # [N, Cu]
    c_0 = c_full[0]  # [N, Cc]
    
    coords = trainer.coord.cpu().numpy()
    
    return u_0, c_0, t_vals, coords, u_full.shape[0]


def create_perturbed_initial_conditions(u_0, num_perturbations=10, perturbation_scale=0.01):
    """
    Create perturbed versions of initial condition.
    
    Args:
        u_0: [N, Cu] initial snapshot (torch tensor)
        num_perturbations: Number of perturbed versions to create
        perturbation_scale: Scale of perturbation relative to std of each channel
    
    Returns:
        List of [N, Cu] perturbed initial conditions (torch tensors)
    """
    N, Cu = u_0.shape
    perturbed_ics = []
    
    # Compute per-channel statistics for scaling perturbations
    u_std_per_channel = u_0.std(dim=0, keepdim=True)  # [1, Cu]
    
    print(f"\nCreating {num_perturbations} perturbed initial conditions:")
    print(f"  Perturbation scale: {perturbation_scale * 100:.1f}% of channel std")
    print(f"  Channel stds: {u_std_per_channel[0].cpu().numpy()}")
    
    for i in range(num_perturbations):
        # Create random perturbation scaled by channel std
        perturbation = torch.randn_like(u_0) * u_std_per_channel * perturbation_scale
        u_perturbed = u_0 + perturbation
        
        perturbed_ics.append(u_perturbed)
        
        # Print statistics
        rel_change = (perturbation.abs().mean() / u_0.abs().mean()).item()
        print(f"  Perturbation {i+1}: relative change = {rel_change*100:.3f}%")
    
    return perturbed_ics


def predict_full_trajectory_from_ic(trainer, u_0, c_full, t_vals, use_stride=1):
    """
    Predict full trajectory from given initial condition.
    
    Args:
        trainer: Initialized trainer with loaded model
        u_0: [N, Cu] initial condition (torch tensor)
        c_full: [T, N, Cc] full conditioning trajectory (torch tensor)
        t_vals: [T] time values
        use_stride: Prediction stride
    
    Returns:
        predictions: [T_pred, N, Cu] trajectory (numpy array)
        time_indices: Indices of predicted timesteps
    """
    T_full = len(t_vals)
    N, Cu = u_0.shape
    device = trainer.device
    
    # Build time indices with stride
    time_indices = np.arange(0, T_full, use_stride, dtype=int)
    
    # Prepare initial state
    u_0_batch = u_0.unsqueeze(0).to(device)  # [1, N, Cu]
    c_0 = c_full[time_indices[0]].unsqueeze(0).to(device)  # [1, N, Cc]
    
    # Get stats for normalization
    u_mean = trainer.stats["u"]["mean"].to(device)
    u_std = trainer.stats["u"]["std"].to(device)
    c_mean = trainer.stats["c"]["mean"].to(device)
    c_std = trainer.stats["c"]["std"].to(device)
    
    # Normalize initial state
    u_0_norm = (u_0_batch - u_mean) / u_std
    c_0_norm = (c_0 - c_mean) / c_std
    
    # Build x_curr with dummy time features
    dummy_time = torch.zeros(1, N, 2, dtype=torch.float32, device=device)
    x_curr = torch.cat([u_0_norm, c_0_norm, dummy_time], dim=-1)  # [1, N, Cu+Cc+2]
    
    # Call AR prediction
    with torch.no_grad():
        preds_denorm = trainer._autoregressive_predict_trainer_side(x_curr, time_indices)
    
    # preds_denorm: [1, K, N, Cu] where K = len(time_indices) - 1
    pred_seq = preds_denorm[0].cpu().numpy()  # [K, N, Cu]
    
    # Build complete trajectory: initial snapshot + predictions
    full_trajectory = np.zeros((len(time_indices), N, Cu), dtype=np.float32)
    full_trajectory[0] = u_0_batch[0].cpu().numpy()
    full_trajectory[1:] = pred_seq
    
    return full_trajectory, time_indices


def main():
    parser = argparse.ArgumentParser(
        description='Generate perturbed initial condition trajectories for UQ')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--h5file', type=str, required=True,
                       help='Path to HDF5 file (uses only initial snapshot)')
    parser.add_argument('--output_dir', type=str, default='perturbed_trajectories',
                       help='Output directory')
    parser.add_argument('--num_perturbations', type=int, default=10,
                       help='Number of perturbed trajectories to generate')
    parser.add_argument('--perturbation_scale', type=float, default=0.01,
                       help='Perturbation scale (fraction of channel std)')
    parser.add_argument('--stride', type=int, default=1,
                       help='Prediction stride')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for perturbations')
    
    args = parser.parse_args()
    
    print("="*60)
    print("PERTURBED TRAJECTORY GENERATION FOR UQ")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Input: {args.h5file}")
    print(f"Output dir: {args.output_dir}")
    print(f"Number of perturbations: {args.num_perturbations}")
    print(f"Perturbation scale: {args.perturbation_scale}")
    print(f"Stride: {args.stride}")
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("\nLoading model...")
    trainer = load_config_and_model(args.config, seed=args.seed)
    
    # Load initial snapshot
    print(f"\nLoading initial snapshot from: {args.h5file}")
    u_0, c_0, t_vals, coords, T_full = load_initial_snapshot(trainer, args.h5file)
    print(f"  Initial snapshot shape: {u_0.shape}")
    print(f"  Total timesteps in file: {T_full}")
    print(f"  Coordinates shape: {coords.shape}")
    
    # Load full conditioning trajectory (needed for time-dependent conditioning)
    u_full, c_full, _ = trainer._load_full_trajectory_from_h5(args.h5file)
    
    # Create perturbed initial conditions
    perturbed_ics = create_perturbed_initial_conditions(
        u_0, num_perturbations=args.num_perturbations,
        perturbation_scale=args.perturbation_scale)
    
    # Generate trajectories for each perturbation
    all_trajectories = []
    
    for i, u_0_perturbed in enumerate(perturbed_ics):
        print(f"\n{'='*60}")
        print(f"GENERATING TRAJECTORY {i+1}/{args.num_perturbations}")
        print(f"{'='*60}")
        
        trajectory, time_indices = predict_full_trajectory_from_ic(
            trainer, u_0_perturbed, c_full, t_vals, use_stride=args.stride)
        
        all_trajectories.append(trajectory)
        
        print(f"✓ Trajectory {i+1} complete: shape {trajectory.shape}")
    
    # Stack all trajectories
    all_trajectories = np.stack(all_trajectories, axis=0)  # [N_pert, T, N, Cu]
    
    # Compute ensemble statistics
    print(f"\n{'='*60}")
    print("ENSEMBLE STATISTICS")
    print(f"{'='*60}")
    
    ensemble_mean = all_trajectories.mean(axis=0)  # [T, N, Cu]
    ensemble_std = all_trajectories.std(axis=0)    # [T, N, Cu]
    
    print(f"Ensemble shape: {all_trajectories.shape}")
    print(f"Mean shape: {ensemble_mean.shape}")
    print(f"Std shape: {ensemble_std.shape}")
    
    # Print temporal evolution of uncertainty
    print(f"\nTemporal evolution of ensemble std (averaged over space & channels):")
    for t_idx in range(0, len(time_indices), max(1, len(time_indices)//10)):
        mean_std = ensemble_std[t_idx].mean()
        print(f"  t={time_indices[t_idx]:3d}: std = {mean_std:.6e}")
    
    # Save ensemble (NPZ format for easy analysis)
    ensemble_file = os.path.join(args.output_dir, 'perturbed_trajectories_ensemble.npz')
    
    np.savez_compressed(
        ensemble_file,
        trajectories=all_trajectories,        # [N_pert, T, N, Cu]
        ensemble_mean=ensemble_mean,          # [T, N, Cu]
        ensemble_std=ensemble_std,            # [T, N, Cu]
        coordinates=coords,                   # [N, 2]
        time_values=t_vals[time_indices],     # [T]
        time_indices=time_indices,            # [T]
        perturbation_scale=args.perturbation_scale,
        num_perturbations=args.num_perturbations,
        stride=args.stride,
        seed=args.seed
    )
    
    print(f"\n✓ Saved ensemble to: {ensemble_file}")
    
    # Save individual trajectories in HDF5 format (matching ground truth structure)
    print(f"\nSaving individual trajectories in HDF5 format...")
    
    # Import HDF5 save function from AR script
    from generate_autoregressive_predictions import save_predictions_hdf5
    
    # Determine grid shape from coordinates
    N_points = coords.shape[0]
    H = W = int(np.sqrt(N_points))
    grid_shape = (H, W)
    
    for i in range(args.num_perturbations):
        indiv_file = os.path.join(args.output_dir, f'trajectory_pert{i:03d}.hdf5')
        save_predictions_hdf5(
            all_trajectories[i],  # [T, N, Cu]
            coords,
            t_vals[time_indices],
            time_indices,
            indiv_file,
            field_names=['density', 'pressure', 'b_field', 'velocity'],
            grid_shape=grid_shape
        )
    
    print(f"✓ Saved {args.num_perturbations} individual trajectories in HDF5 format")
    
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {args.output_dir}")
    print(f"Ensemble file: perturbed_trajectories_ensemble.npz")
    print(f"Individual files: trajectory_pert000.npz ... trajectory_pert{args.num_perturbations-1:03d}.npz")


if __name__ == '__main__':
    main()

