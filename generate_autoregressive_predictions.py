#!/usr/bin/env python
"""
Generate fully autoregressive predictions with configurable stride.
Ensures proper denormalization before comparing to ground truth.
"""

import os
import sys
import argparse
import numpy as np
import torch
import h5py
from pathlib import Path


def load_config_and_model(config_path, seed=42):
    """Load config and initialize trainer with model."""
    # Import here to avoid issues with module paths
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from main import FileParser, prepare_arg
    from src.trainer.sequential_trainer import SequentialTrainer
    
    # Parse config
    parser = FileParser(config_path)
    arg = parser.parse_args()  # Fixed: use parse_args() not parse()
    
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


def load_full_trajectory(trainer, h5_path):
    """Load full trajectory from HDF5 file."""
    u_full, c_full, t_vals = trainer._load_full_trajectory_from_h5(h5_path)
    coords = trainer.coord.cpu().numpy()
    return u_full, c_full, t_vals, coords


def predict_fully_autoregressive(trainer, u_full, c_full, t_vals, use_stride=1):
    """
    Predict completely autoregressively using trainer's method.
    
    Args:
        trainer: Initialized trainer with loaded model
        u_full: [T, N, Cu] ground truth (denormalized)
        c_full: [T, N, Cc] conditioning (denormalized)
        t_vals: [T] time values
        use_stride: Prediction stride (1=every step, 4=every 4th step)
    
    Returns:
        predictions: [len(time_indices), N, Cu] denormalized predictions
        time_indices: Indices of predicted timesteps
    """
    T, N, Cu = u_full.shape
    device = trainer.device
    
    # Build time indices with stride
    time_indices = np.arange(0, T, use_stride, dtype=int)
    
    print(f"\nPredicting autoregressively with stride={use_stride}:")
    print(f"  Time indices: {time_indices[:5]}...{time_indices[-5:] if len(time_indices) > 10 else time_indices[-3:]}")
    print(f"  Total predictions: {len(time_indices)-1}")
    
    # Prepare initial state (matches TestDataset format)
    u_0 = u_full[time_indices[0]].unsqueeze(0).to(device)  # [1, N, Cu]
    c_0 = c_full[time_indices[0]].unsqueeze(0).to(device)  # [1, N, Cc]
    
    # Get stats for normalization
    u_mean = trainer.stats["u"]["mean"].to(device)
    u_std = trainer.stats["u"]["std"].to(device)
    c_mean = trainer.stats["c"]["mean"].to(device)
    c_std = trainer.stats["c"]["std"].to(device)
    
    # Normalize initial state
    u_0_norm = (u_0 - u_mean) / u_std
    c_0_norm = (c_0 - c_mean) / c_std
    
    # Build x_curr with dummy time features (will be replaced in loop)
    dummy_time = torch.zeros(1, N, 2, dtype=torch.float32, device=device)
    x_curr = torch.cat([u_0_norm, c_0_norm, dummy_time], dim=-1)  # [1, N, Cu+Cc+2]
    
    # Call the trainer's AR prediction method - this handles denormalization internally
    with torch.no_grad():
        preds_denorm = trainer._autoregressive_predict_trainer_side(x_curr, time_indices)
    
    # preds_denorm: [1, K, N, Cu] where K = len(time_indices) - 1
    pred_seq = preds_denorm[0].cpu().numpy()  # [K, N, Cu]
    
    # Build complete trajectory: GT at t=0, then predictions
    full_trajectory = np.zeros((len(time_indices), N, Cu), dtype=np.float32)
    full_trajectory[0] = u_0[0].cpu().numpy()  # GT at first index
    full_trajectory[1:] = pred_seq  # Predictions (already denormalized)
    
    print(f"✓ Autoregressive prediction complete")
    return full_trajectory, time_indices


def compute_errors(predictions, ground_truth, u_mean, u_std):
    """
    Compute error metrics matching compute_batch_errors from sequential_trainer.
    
    Args:
        predictions: [T, N, C] denormalized predictions
        ground_truth: [T, N, C] denormalized ground truth
        u_mean: [C] mean for normalization
        u_std: [C] std for normalization
    """
    errors = {
        'l1': [],
        'l2': [],
        'rel_l1': [],
        'rel_l2': []
    }
    
    print(f"\n[ERROR DEBUG] Checking normalization state:")
    print(f"  GT[0] range: {ground_truth[0].min():.3e} to {ground_truth[0].max():.3e}")
    print(f"  Pred[1] range: {predictions[1].min():.3e} to {predictions[1].max():.3e}")
    print(f"  u_mean: {u_mean}")
    print(f"  u_std: {u_std}")
    
    for t in range(1, len(predictions)):  # Skip t=0 (GT)
        pred_t = predictions[t]  # [N, C] denormalized
        gt_t = ground_truth[t]   # [N, C] denormalized
        
        # Normalize (like compute_batch_errors does)
        gt_norm = (gt_t - u_mean) / u_std
        pred_norm = (pred_t - u_mean) / u_std
        
        # Absolute errors in normalized space
        abs_error_norm = np.abs(pred_norm - gt_norm)
        l1 = abs_error_norm.mean()
        l2 = np.sqrt((abs_error_norm**2).mean())
        
        # Relative errors (error sum / gt sum, like compute_batch_errors)
        error_sum = abs_error_norm.sum()
        gt_sum = np.abs(gt_norm).sum()
        rel_l1 = error_sum / (gt_sum + 1e-10)
        
        # L2 relative
        error_l2_norm = np.sqrt((abs_error_norm**2).sum())
        gt_l2_norm = np.sqrt((gt_norm**2).sum())
        rel_l2 = error_l2_norm / (gt_l2_norm + 1e-10)
        
        errors['l1'].append(l1)
        errors['l2'].append(l2)
        errors['rel_l1'].append(rel_l1)
        errors['rel_l2'].append(rel_l2)
    
    return errors


def save_predictions_hdf5(predictions, coords, t_vals, time_indices, output_path, 
                          field_names=None, grid_shape=(64, 64)):
    """
    Save predictions in HDF5 format matching ground truth structure.
    
    Args:
        predictions: [T, N, C] predictions
        coords: [N, 2] coordinates
        t_vals: [T] time values
        time_indices: [T] time indices
        output_path: Output HDF5 file path
        field_names: List of field names (e.g., ['density', 'pressure', 'b_field', 'velocity'])
        grid_shape: Tuple (H, W) for reshaping data
    """
    T, N, C = predictions.shape
    H, W = grid_shape
    
    if field_names is None:
        field_names = ['density', 'pressure', 'b_field', 'velocity']
    
    # Ensure we have right number of field names
    if len(field_names) < C:
        field_names = field_names + [f'field_{i}' for i in range(len(field_names), C)]
    
    with h5py.File(output_path, 'w') as h5:
        # Create groups
        t0_group = h5.create_group('t0_fields')
        
        # Reshape predictions from [T, N, C] to [T, H, W, C_per_field]
        predictions_hwc = predictions.reshape(T, H, W, C)
        
        # Save each field (assuming 1 channel per field for now)
        for i, fname in enumerate(field_names[:C]):
            # Save as [T, H, W, 1]
            t0_group.create_dataset(fname, data=predictions_hwc[..., i:i+1], 
                                   compression='gzip', compression_opts=4)
        
        # Create dimensions group
        dim_group = h5.create_group('dimensions')
        
        # Time dimension
        time_data = np.zeros((T, 2), dtype=np.float32)
        time_data[:, 0] = t_vals  # Simulation time
        time_data[:, 1] = time_indices  # Frame index
        dim_group.create_dataset('time', data=time_data)
        
        # Spatial dimensions (x and y)
        coords_hw = coords.reshape(H, W, 2)
        x_grid = coords_hw[:, :, 0]
        y_grid = coords_hw[:, :, 1]
        
        # X dimension: [H, W, 1] with x-coordinates
        dim_group.create_dataset('x', data=x_grid[:, :, np.newaxis])
        # Y dimension: [H, W, 1] with y-coordinates  
        dim_group.create_dataset('y', data=y_grid[:, :, np.newaxis])
    
    print(f"✓ Saved to HDF5: {output_path}")
    print(f"  Format: t0_fields/{field_names[:C]}")
    print(f"  Shape per field: ({T}, {H}, {W}, 1)")


def save_predictions_npz(predictions, gt, coords, t_vals, time_indices, output_path, seed=None, stride=1):
    """Save predictions to NPZ file (legacy format)."""
    np.savez_compressed(
        output_path,
        predictions=predictions,  # [T, N, C]
        ground_truth=gt,  # [T, N, C]
        coordinates=coords,  # [N, 2]
        time_values=t_vals,  # [T]
        time_indices=time_indices,  # [T]
        seed=seed,
        stride=stride
    )
    print(f"✓ Saved to NPZ: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate autoregressive predictions')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--h5file', type=str, required=True,
                       help='Path to HDF5 file for prediction')
    parser.add_argument('--output_dir', type=str, default='ar_predictions',
                       help='Output directory for predictions')
    parser.add_argument('--num_seeds', type=int, default=1,
                       help='Number of seeds to run (note: seeds do not affect loaded models)')
    parser.add_argument('--base_seed', type=int, default=42,
                       help='Base seed value')
    parser.add_argument('--stride', type=int, default=1,
                       help='Prediction stride (1=every step, 4=every 4th step)')
    parser.add_argument('--save_format', type=str, default='hdf5', choices=['hdf5', 'npz'],
                       help='Output format: hdf5 (matches ground truth) or npz (legacy)')
    parser.add_argument('--grid_shape', type=int, nargs=2, default=[64, 64],
                       help='Grid shape (H, W) for HDF5 output')
    
    args = parser.parse_args()
    
    print("="*60)
    print("GENERATING AUTOREGRESSIVE PREDICTIONS")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Input: {args.h5file}")
    print(f"Output dir: {args.output_dir}")
    print(f"Number of seeds: {args.num_seeds}")
    print(f"Stride: {args.stride}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each seed
    for seed_idx in range(args.num_seeds):
        seed = args.base_seed + seed_idx
        
        print(f"\n{'='*60}")
        print(f"SEED {seed_idx+1}/{args.num_seeds}: {seed}")
        print(f"{'='*60}")
        
        # Load model
        trainer = load_config_and_model(args.config, seed=seed)
        
        # Load trajectory
        print(f"\nLoading trajectory from: {args.h5file}")
        u_full, c_full, t_vals, coords = load_full_trajectory(trainer, args.h5file)
        print(f"  Shape: {u_full.shape}")
        print(f"  Time range: {t_vals[0]:.6e} to {t_vals[-1]:.6e} s")
        
        # Generate predictions
        predictions, time_indices_used = predict_fully_autoregressive(
            trainer, u_full, c_full, t_vals, use_stride=args.stride)
        
        # Get ground truth at the same time indices
        gt_at_indices = u_full[time_indices_used].cpu().numpy()
        
        # Get normalization stats for error computation
        u_mean_np = trainer.stats["u"]["mean"].cpu().numpy().flatten()
        u_std_np = trainer.stats["u"]["std"].cpu().numpy().flatten()
        
        # Compute errors (using proper normalization like compute_batch_errors)
        print("\nComputing errors...")
        errors = compute_errors(predictions, gt_at_indices, u_mean_np, u_std_np)
        
        # Print overall errors
        print(f"\n  Overall errors:")
        print(f"    Mean Rel-L1: {np.mean(errors['rel_l1']):.6f}")
        print(f"    Mean Rel-L2: {np.mean(errors['rel_l2']):.6f}")
        print(f"    Final Abs-L1: {errors['l1'][-1]:.6e}")
        print(f"    Final Abs-L2: {errors['l2'][-1]:.6e}")
        
        # Print first 10 prediction errors
        T_pred = len(predictions) - 1  # Exclude initial GT
        print(f"\n  Relative errors for first {min(10, T_pred)} autoregressive predictions:")
        for j in range(1, min(11, len(predictions))):
            rel_l1 = errors['rel_l1'][j-1]
            rel_l2 = errors['rel_l2'][j-1]
            time_idx = time_indices_used[j]
            print(f"    t={time_idx:3d} (pred #{j}): Rel-L1={rel_l1:.6f}, Rel-L2={rel_l2:.6f}")
        
        # Save predictions
        suffix = f'_stride{args.stride}' if args.stride > 1 else ''
        
        if args.save_format == 'hdf5':
            output_file = os.path.join(args.output_dir, f'ar_prediction_seed{seed:04d}{suffix}.hdf5')
            save_predictions_hdf5(
                predictions,
                coords,
                t_vals[time_indices_used],
                time_indices_used,
                output_file,
                field_names=['density', 'pressure', 'b_field', 'velocity'],
                grid_shape=tuple(args.grid_shape)
            )
        else:
            output_file = os.path.join(args.output_dir, f'ar_prediction_seed{seed:04d}{suffix}.npz')
            save_predictions_npz(
                predictions, 
                gt_at_indices,
                coords,
                t_vals[time_indices_used],
                time_indices_used,
                output_file,
                seed=seed,
                stride=args.stride
            )
    
    print(f"\n{'='*60}")
    print("ALL PREDICTIONS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

