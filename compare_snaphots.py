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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
import math
import sys

# Pick a set of expressive colormaps (one per channel). Will cycle if more channels exist.
# These are perceptually-meaningful and visually distinct for multiple-panel displays.
DEFAULT_COLORMAPS = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "coolwarm",
    "twilight",
    "turbo",
]


def load_full_trajectory(h5file):
    """Load complete trajectory from HDF5 file.

    Returns:
        u_full: np.ndarray, shape [T, N, C]
        coords: np.ndarray, shape [N, 2]  (r, z)
        t_vals: np.ndarray, shape [T]
        channel_labels: list of str, length C
    """
    print(f"Loading trajectory from: {h5file}")

    with h5py.File(h5file, "r") as f:
        # Load time values
        t_vals = np.asarray(f["dimensions/time"][...])
        T = len(t_vals)

        # Load coordinates
        r_coords = np.asarray(f["dimensions/r_coords"][...])
        z_coords = np.asarray(f["dimensions/z_coords"][...])
        Z, R = np.meshgrid(z_coords, r_coords, indexing="ij")
        coords = np.stack([R, Z], axis=-1).reshape(-1, 2)  # [N, 2]
        N = coords.shape[0]

        # Helper to load a field group/name and return shape-normalized array
        def load_field(group, name):
            data = np.asarray(f[f"{group}/{name}"][...])
            # Expect (T, H, W) -> (T, N) or (T, H, W, C) -> (T, N, C)
            if data.ndim == 3:  # (T, H, W)
                return data.reshape(T, N)[..., None]  # [T, N, 1]
            elif data.ndim == 4:  # (T, H, W, C)
                H, W, C = data.shape[1], data.shape[2], data.shape[3]
                return data.reshape(T, N, C)  # [T, N, C]
            else:
                raise ValueError(f"Unexpected shape for {group}/{name}: {data.shape}")

        # Load all fields in sorted order for deterministic ordering
        fields = {}  # name -> ndarray [T, N, c]
        for group in ["t0_fields", "t1_fields"]:
            if group in f:
                for field_name in sorted(f[group].keys()):
                    fields[field_name] = load_field(group, field_name)

        # Build per-channel labels (expand multi-channel fields)
        channel_labels = []
        arrays_to_concat = []
        for fname in sorted(fields.keys()):
            arr = fields[fname]  # [T, N, c]
            cdim = arr.shape[-1]
            # make channel labels like "field", "field_0", "field_1", ...
            if cdim == 1:
                channel_labels.append(fname)
            else:
                for ci in range(cdim):
                    channel_labels.append(f"{fname}_{ci}")
            arrays_to_concat.append(arr)

        if not arrays_to_concat:
            raise RuntimeError("No fields found in HDF5 file under t0_fields/t1_fields.")

        # Concatenate on channel dimension -> [T, N, C]
        u_full = np.concatenate(arrays_to_concat, axis=-1)

    print(f"  Timesteps: {T}")
    print(f"  Spatial points: {N}")
    print(f"  Channels (C): {u_full.shape[-1]}")
    print(f"  Channel labels: {channel_labels}")
    print(f"  Shape: {u_full.shape}")

    return u_full, coords, t_vals, channel_labels


def create_animation_from_gt(
    u_sequence,
    coords,
    t_values,
    channel_labels,
    save_path,
    stride=1,
    interval=140 / 702 * 1e-3,
    point_size=8,
):
    """
    Create animation from ground truth sequence.

    Args:
        u_sequence: [T, N, C] array
        coords: [N, 2] array
        t_values: [T] array
        channel_labels: list length C
        save_path: output file path
        stride: temporal stride (1 = all frames, 2 = every other frame, etc.)
        interval: milliseconds per frame (float)
        point_size: scatter point size
    """
    # Subsample if needed
    if stride > 1:
        u_sequence = u_sequence[::stride]
        t_values = t_values[::stride]

    T, N, C = u_sequence.shape
    if len(channel_labels) != C:
        # Fail-safe: make generic labels if the length mismatches
        channel_labels = [f"chan_{i}" for i in range(C)]

    print(f"\nCreating animation:")
    print(f"  Frames: {T}")
    print(f"  Spatial points: {N}")
    print(f"  Channels: {C}")
    print(f"  Stride: {stride}")
    print(f"  Interval (ms): {interval}")

    # Robust axes creation: arrange in a single row; if too wide for display,
    # matplotlib will handle it, and user can adjust figsize or later modify layout.
    fig_width = max(3 * C, 6)
    fig, axes = plt.subplots(1, C, figsize=(fig_width, 3))
    # Ensure axes is always a 1D array
    if C == 1:
        axes = np.atleast_1d(axes)
    else:
        axes = np.array(axes).reshape(-1)

    # Compute per-channel vmin/vmax (global across time & space) and normalizers
    vmin = u_sequence.min(axis=(0, 1))
    vmax = u_sequence.max(axis=(0, 1))
    normalizers = [Normalize(vmin=vmin[i], vmax=vmax[i]) for i in range(C)]

    # Choose colormaps for each channel
    colormaps = []
    for i in range(C):
        cmap = DEFAULT_COLORMAPS[i % len(DEFAULT_COLORMAPS)]
        colormaps.append(cmap)

    # Initialize scatter plots and colorbars
    scatters = []
    colorbars = []
    for i, ax in enumerate(axes[:C]):
        ax.set_aspect("equal")
        ax.set_xlabel("r")
        ax.set_ylabel("z")
        ax.set_title(channel_labels[i])

        # initial values for colors
        vals0 = u_sequence[0, :, i]
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=vals0,
            s=point_size,
            cmap=colormaps[i],
            norm=normalizers[i],
            edgecolors="none",
        )
        # Add colorbar per axis
        cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
        cbar.ax.tick_params(labelsize=8)
        scatters.append(sc)
        colorbars.append(cbar)

    # Hide any extra axes if C < axes created (shouldn't happen but safe)
    for ax in axes[C:]:
        ax.axis("off")

    # Animation title
    time_text = fig.suptitle(f"Time: {t_values[0]:.6e}", fontsize=12)

    def animate(frame):
        """Update function for animation."""
        for i, scatter in enumerate(scatters):
            # Update array - must be 1D of length N
            scatter.set_array(u_sequence[frame, :, i])
            # If global vmin/vmax changed dynamically (it doesn't here), we'd update norm
            # scatter.set_norm(normalizers[i])  # not necessary here
        time_text.set_text(
            f"Time: {t_values[frame]:.6e} (frame {frame+1}/{T})"
        )
        # Return all artists that have changed
        return scatters + [time_text]

    # Create animation. FuncAnimation interval expects ms.
    print("  Generating frames...")
    anim = FuncAnimation(fig, animate, frames=T, interval=interval, blit=False)

    # Save: compute fps from interval (ms -> fps). Guard against zero/too-large values.
    fps = None
    try:
        fps = int(round(1000.0 / float(interval)))
        if fps <= 0:
            raise ValueError
    except Exception:
        # fallback: small default fps
        fps = 10

    print(f"  Saving to: {save_path} (fps={fps})")
    try:
        if save_path.lower().endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=fps, dpi=150)
        elif save_path.lower().endswith(".mp4"):
            anim.save(save_path, writer="ffmpeg", fps=fps, dpi=150)
        else:
            # If unspecified extension, save a gif by default
            out = save_path + ".gif"
            anim.save(out, writer="pillow", fps=fps, dpi=150)
            print(f"  (Saved as {out})")
    except Exception as e:
        print("Error while saving animation:", e, file=sys.stderr)
        # As a fallback, try saving with pillow and default fps
        try:
            fallback = save_path if save_path.lower().endswith(".gif") else save_path + ".gif"
            print(f"  Attempting fallback save to {fallback} ...")
            anim.save(fallback, writer="pillow", fps=fps, dpi=150)
            print(f"  (Saved fallback as {fallback})")
        except Exception as e2:
            print("Fallback save also failed:", e2, file=sys.stderr)
            raise

    plt.close(fig)
    print("✓ Animation created successfully!")


def main():
    parser = argparse.ArgumentParser(description="Create animation from ground truth HDF5 trajectory")
    parser.add_argument("--h5file", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument("--output", type=str, default="gt_animation.gif", help="Output file path")
    parser.add_argument("--stride", type=int, default=1, help="Temporal stride (1=all frames)")
    parser.add_argument(
        "--interval",
        type=float,
        default=140.0 / 702.0 * 1.0,  # default milliseconds per frame (kept simple)
        help="Milliseconds per frame (float). Example: 2.0 (ms)",
    )

    args = parser.parse_args()

    # Load data
    u_sequence, coords, t_values, channel_labels = load_full_trajectory(args.h5file)

    # Create animation
    create_animation_from_gt(
        u_sequence,
        coords,
        t_values,
        channel_labels,
        args.output,
        stride=args.stride,
        interval=args.interval,
    )


if __name__ == "__main__":
    main()

