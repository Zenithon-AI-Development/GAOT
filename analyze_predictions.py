#!/usr/bin/env python3
"""
Analyze saved predictions from GAOT to diagnose animation issues.

Usage:
    python analyze_predictions.py --npz path/to/full_trajectory_predictions.npz
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def analyze_predictions(npz_file):
    """Analyze saved predictions."""
    print("="*60)
    print("ANALYZING PREDICTIONS")
    print("="*60)
    
    data = np.load(npz_file)
    
    input_data = data['input']
    gt_seq = data['gt_sequence']
    pred_seq = data['pred_sequence']
    coords = data['coords']
    time_indices = data['time_indices']
    t_values = data['t_values']
    
    K, N, C = gt_seq.shape
    
    print(f"\nData shapes:")
    print(f"  GT sequence: {gt_seq.shape}")
    print(f"  Pred sequence: {pred_seq.shape}")
    print(f"  Timesteps: {K}")
    print(f"  Points: {N}")
    print(f"  Channels: {C}")
    
    # Check variation
    print(f"\nTemporal variation:")
    for c in range(C):
        gt_means = gt_seq[:, :, c].mean(axis=1)  # Mean over space for each time
        pred_means = pred_seq[:, :, c].mean(axis=1)
        
        gt_temp_std = gt_means.std()
        pred_temp_std = pred_means.std()
        
        print(f"  Channel {c}:")
        print(f"    GT temporal std: {gt_temp_std:.6e}")
        print(f"    Pred temporal std: {pred_temp_std:.6e}")
        print(f"    Ratio: {pred_temp_std/gt_temp_std*100:.2f}%")
        
        # Check if predictions are stuck
        if pred_temp_std < gt_temp_std * 0.01:
            print(f"    ⚠ WARNING: Predictions show little temporal variation!")
    
    # Check errors
    errors = np.abs(gt_seq - pred_seq)
    mean_error_over_time = errors.mean(axis=(1, 2))  # [K] - mean error per timestep
    
    print(f"\nPrediction errors:")
    print(f"  Mean error (first 10 steps): {mean_error_over_time[:10]}")
    print(f"  Mean error (last 10 steps): {mean_error_over_time[-10:]}")
    print(f"  Overall mean: {mean_error_over_time.mean():.6e}")
    print(f"  Overall std: {mean_error_over_time.std():.6e}")
    
    # Plot error evolution
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Error over time
    axes[0].plot(time_indices[1:], mean_error_over_time)
    axes[0].set_xlabel('Timestep Index')
    axes[0].set_ylabel('Mean Absolute Error')
    axes[0].set_title('Prediction Error Evolution')
    axes[0].grid(True)
    
    # GT vs Pred temporal evolution for each channel
    axes[1].set_xlabel('Timestep Index')
    axes[1].set_ylabel('Spatial Mean Value')
    axes[1].set_title('Temporal Evolution (Spatial Means)')
    
    for c in range(C):
        gt_means = gt_seq[:, :, c].mean(axis=1)
        pred_means = pred_seq[:, :, c].mean(axis=1)
        axes[1].plot(time_indices[1:], gt_means, label=f'GT ch{c}', linestyle='-')
        axes[1].plot(time_indices[1:], pred_means, label=f'Pred ch{c}', linestyle='--')
    
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('prediction_analysis.png', dpi=150)
    print(f"\nSaved analysis plot: prediction_analysis.png")
    
    # Frame-by-frame comparison
    print(f"\nFrame-by-frame GT variation:")
    for i in range(0, K, K//10):  # Check 10 frames
        if i > 0:
            diff = np.abs(gt_seq[i] - gt_seq[i-K//10]).mean()
            rel = diff / (np.abs(gt_seq[i]).mean() + 1e-10)
            print(f"  Frame {i-K//10} → {i}: abs={diff:.4e}, rel={rel*100:.2f}%")


def create_simple_animation(npz_file, output='simple_animation.gif', stride=4):
    """Create a simple animation with better contrast."""
    data = np.load(npz_file)
    
    gt_seq = data['gt_sequence'][::stride]  # Subsample
    pred_seq = data['pred_sequence'][::stride]
    coords = data['coords']
    time_indices = data['time_indices'][::stride]
    
    K, N, C = gt_seq.shape
    
    print(f"\nCreating simple animation (stride={stride}):")
    print(f"  Frames: {K}")
    
    # Just show first channel (density) for simplicity
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Use per-channel min/max for better contrast
    c = 0  # First channel
    gt_c = gt_seq[:, :, c]
    pred_c = pred_seq[:, :, c]
    vmin = min(gt_c.min(), pred_c.min())
    vmax = max(gt_c.max(), pred_c.max())
    
    ax1.set_title('Ground Truth')
    ax1.set_aspect('equal')
    s1 = ax1.scatter(coords[:, 0], coords[:, 1], c=gt_c[0], vmin=vmin, vmax=vmax, s=10, cmap='viridis')
    plt.colorbar(s1, ax=ax1)
    
    ax2.set_title('Prediction')
    ax2.set_aspect('equal')
    s2 = ax2.scatter(coords[:, 0], coords[:, 1], c=pred_c[0], vmin=vmin, vmax=vmax, s=10, cmap='viridis')
    plt.colorbar(s2, ax=ax2)
    
    ax3.set_title('Absolute Error')
    ax3.set_aspect('equal')
    error = np.abs(gt_c[0] - pred_c[0])
    s3 = ax3.scatter(coords[:, 0], coords[:, 1], c=error, s=10, cmap='hot')
    plt.colorbar(s3, ax=ax3)
    
    title = fig.suptitle(f'Frame 0/{K}')
    
    def animate(frame):
        s1.set_array(gt_c[frame])
        s2.set_array(pred_c[frame])
        error = np.abs(gt_c[frame] - pred_c[frame])
        s3.set_array(error)
        s3.set_clim(vmin=0, vmax=error.max())  # Dynamic error scale
        title.set_text(f'Frame {frame}/{K}')
        return [s1, s2, s3, title]
    
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, animate, frames=K, interval=200, blit=False)
    anim.save(output, writer='pillow', fps=5, dpi=100)
    plt.close()
    
    print(f"  Saved: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', type=str, required=True,
                       help='Path to full_trajectory_predictions.npz')
    parser.add_argument('--create-animation', action='store_true',
                       help='Also create a simple animation')
    parser.add_argument('--stride', type=int, default=4,
                       help='Stride for animation (4=every 4th frame)')
    
    args = parser.parse_args()
    
    # Analyze
    analyze_predictions(args.npz)
    
    # Create animation if requested
    if args.create_animation:
        create_simple_animation(args.npz, stride=args.stride)


if __name__ == '__main__':
    main()

