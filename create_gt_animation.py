#!/usr/bin/env python3
"""
Create animation from ground truth trajectory (no model predictions).
This demonstrates the expected data format and animation creation process.

Usage:
    python create_gt_animation.py --h5file path/to/file.hdf5 --output animation.gif
"""

import argparse
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def load_full_trajectory(h5file):
    """Load complete trajectory from HDF5 file."""
    print(f"Loading trajectory from: {h5file}")
    
    with h5py.File(h5file, 'r') as f:
        # Load time values
        t_vals = np.asarray(f['dimensions/time'][...])
        T = len(t_vals)
        
        # Load coordinates
        r_coords = np.asarray(f['dimensions/r_coords'][...])
        z_coords = np.asarray(f['dimensions/z_coords'][...])
        Z, R = np.meshgrid(z_coords, r_coords, indexing='ij')
        coords = np.stack([R, Z], axis=-1).reshape(-1, 2)  # [N, 2]
        N = coords.shape[0]
        
        # Load fields
        def load_field(group, name):
            data = np.asarray(f[f'{group}/{name}'][...])
            if data.ndim == 3:  # (T, H, W)
                return data.reshape(T, N)
            elif data.ndim == 4:  # (T, H, W, C)
                return data.reshape(T, N, data.shape[-1])
            else:
                raise ValueError(f"Unexpected shape for {group}/{name}: {data.shape}")
        
        # Load all fields
        fields = {}
        for group in ['t0_fields', 't1_fields']:
            if group in f:
                for field_name in f[group].keys():
                    data = load_field(group, field_name)
                    if data.ndim == 2:
                        data = data[..., None]  # Add channel dimension
                    fields[field_name] = data
        
        # Stack all fields
        field_names = sorted(fields.keys())
        u_full = np.concatenate([fields[name] for name in field_names], axis=-1)  # [T, N, C]
        
    print(f"  Timesteps: {T}")
    print(f"  Spatial points: {N}")
    print(f"  Channels: {len(field_names)} {field_names}")
    print(f"  Shape: {u_full.shape}")
    
    return u_full, coords, t_vals, field_names

from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm, LogNorm, Normalize
from matplotlib.colors import hsv_to_rgb

def create_animation_from_gt(u_sequence, coords, t_values, field_names, save_path, 
                              stride=1, interval=20, colormode='divergent', cmap_name=None, center_zero=True):
    """Create animation from ground truth with enhanced contrast."""
    # Subsample if needed
    if stride > 1:
        u_sequence = u_sequence[::stride]
        t_values = t_values[::stride]
    
    T, N, C = u_sequence.shape
    print(f"\nCreating animation:")
    print(f"  Frames: {T}")
    print(f"  Stride: {stride}")
    print(f"  Interval: {interval}ms")
    
    # choose default colormap if not provided
    if cmap_name is None:
        cmap_name = 'viridis' if colormode == 'single' else 'coolwarm' if colormode == 'divergent' else 'turbo'
    
    # Setup figure
    fig_width = 3 * C
    fig, axes = plt.subplots(1, C, figsize=(fig_width, 3))
    if C == 1:
        axes = [axes]

    # Compute global value ranges for consistent colormaps
    vmin = u_sequence.min(axis=(0, 1))  # [C]
    vmax = u_sequence.max(axis=(0, 1))  # [C]

    # Helper: create norm for channel
    def make_norm(i):
        if colormode == 'divergent':
            # center at 0 if data crosses zero
            center = 0.0 if center_zero else 0.5*(vmin[i] + vmax[i])
            return TwoSlopeNorm(vmin=vmin[i], vcenter=center, vmax=vmax[i])
        elif colormode == 'log':
            return LogNorm(vmin=max(vmin[i], 1e-12), vmax=max(vmax[i], 1e-12))
        else:
            return Normalize(vmin=vmin[i], vmax=vmax[i])

    # Initialize scatter plots (supporting the HSV multi-channel mode)
    scatters = []
    colorbars = []
    # If using HSV mode we need at least 2 channels; we'll pack first two channels
    use_hsv = (colormode == 'hsv')
    if use_hsv and C < 2:
        raise ValueError("HSV mode requires at least 2 channels (uses channels 0 and 1).")

    for i, (ax, name) in enumerate(zip(axes, field_names)):
        ax.set_aspect('equal')
        ax.set_xlabel('r')
        ax.set_ylabel('z')
        ax.set_title(name)

        if use_hsv:
            # initial rgb colors from channels 0 (x) and 1 (y)
            if i == 0:
                # compute hue from angle of (ch0, ch1), value from magnitude
                ch0 = u_sequence[0, :, 0]
                ch1 = u_sequence[0, :, 1]
                angles = np.arctan2(ch1, ch0)  # [-pi, pi]
                norm_angles = (angles + np.pi) / (2 * np.pi)  # [0,1] -> hue
                mags = np.sqrt(ch0**2 + ch1**2)
                # normalize magnitude to [0,1]
                mmin, mmax = mags.min(), mags.max()
                mags_n = (mags - mmin) / (mmax - mmin + 1e-12)
                hsv = np.stack([norm_angles, np.clip(mags_n,0,1), np.clip(mags_n,0,1)], axis=1)
                colors = hsv_to_rgb(hsv)
                scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=5)
                # no scalar colorbar for combined-hsv; optionally show legend or separate colorbars for ch0/ch1
                scatters.append(scatter)
                colorbars.append(None)
            else:
                # additional channels still as single-channel maps
                cmap = cm.get_cmap(cmap_name)
                norm = make_norm(i)
                scatter = ax.scatter(coords[:, 0], coords[:, 1],
                                     c=u_sequence[0, :, i], vmin=vmin[i], vmax=vmax[i],
                                     s=5, cmap=cmap, norm=norm)
                cb = plt.colorbar(scatter, ax=ax)
                scatters.append(scatter)
                colorbars.append(cb)
        else:
            cmap = cm.get_cmap(cmap_name)
            norm = make_norm(i)
            scatter = ax.scatter(coords[:, 0], coords[:, 1],
                                 c=u_sequence[0, :, i],
                                 s=5, cmap=cmap, norm=norm)
            cb = plt.colorbar(scatter, ax=ax)
            scatters.append(scatter)
            colorbars.append(cb)

    # Animation title
    time_text = fig.suptitle(f'Time: {t_values[0]:.6e}', fontsize=12)

    # Update function
    def animate(frame):
        for i, scatter in enumerate(scatters):
            if use_hsv:
                # first axis: combined hue/magnitude
                if i == 0:
                    ch0 = u_sequence[frame, :, 0]
                    ch1 = u_sequence[frame, :, 1]
                    angles = np.arctan2(ch1, ch0)
                    norm_angles = (angles + np.pi) / (2 * np.pi)
                    mags = np.sqrt(ch0**2 + ch1**2)
                    mmin, mmax = mags.min(), mags.max()
                    mags_n = (mags - mmin) / (mmax - mmin + 1e-12)
                    hsv = np.stack([norm_angles, np.clip(mags_n,0,1), np.clip(mags_n,0,1)], axis=1)
                    colors = hsv_to_rgb(hsv)
                    scatter.set_facecolor(colors)
                else:
                    scatter.set_array(u_sequence[frame, :, i])
                    # If using a lognorm or TwoSlopeNorm, you can update clim if you want dynamic scaling:
                    # scatter.set_clim(vmin=..., vmax=...)
                    if colorbars[i] is not None:
                        colorbars[i].update_normal(scatter)
            else:
                scatter.set_array(u_sequence[frame, :, i])
                if colorbars[i] is not None:
                    colorbars[i].update_normal(scatter)

        time_text.set_text(f'Time: {t_values[frame]:.6e} (frame {frame+1}/{T})')
        return scatters + [time_text]
    
    # Create animation
    print(f"  Generating frames...")
    anim = FuncAnimation(fig, animate, frames=T, interval=interval, blit=False)
    
    # Save
    print(f"  Saving to: {save_path}")
    if save_path.endswith('.gif'):
        anim.save(save_path, writer='pillow', fps=1000//interval, dpi=100)
    elif save_path.endswith('.mp4'):
        anim.save(save_path, writer='ffmpeg', fps=1000//interval, dpi=100)
    else:
        save_path_gif = save_path + '.gif'
        anim.save(save_path_gif, writer='pillow', fps=1000//interval, dpi=100)
        print(f"  (Saved as {save_path_gif})")
    
    plt.close(fig)
    print(f"✓ Animation created successfully!")


def main():
    parser = argparse.ArgumentParser(description='Create animation from ground truth HDF5 trajectory')
    parser.add_argument('--h5file', type=str, required=True, help='Path to HDF5 file')
    parser.add_argument('--output', type=str, default='gt_animation.gif', help='Output file path')
    parser.add_argument('--stride', type=int, default=1, help='Temporal stride (1=all frames)')
    parser.add_argument('--interval', type=int, default=20, help='Milliseconds per frame')
    
    args = parser.parse_args()
    
    # Load data
    u_sequence, coords, t_values, field_names = load_full_trajectory(args.h5file)
    
    # Create animation
    create_animation_from_gt(u_sequence, coords, t_values, field_names, 
                            args.output, stride=args.stride, interval=args.interval)


if __name__ == '__main__':
    main()

