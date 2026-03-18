#!/usr/bin/env python3
"""
Analyze autocorrelation in MagLIF dataset to find optimal timestep.

The optimal timestep for training is typically where autocorrelation ≈ 0.5,
as this provides a good balance between predictability and information content.
"""
import os
import sys
import argparse
import numpy as np
import h5py
import glob
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path to import GAOT modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import helper functions directly to avoid dependency issues
def _time_from_dimensions(h5: h5py.File) -> np.ndarray:
    """Extract time values from dimensions group."""
    if "dimensions" not in h5 or "time" not in h5["dimensions"]:
        raise KeyError("dimensions/time not found in HDF5")
    t = np.asarray(h5["dimensions/time"][...], dtype=np.float64)
    if t.ndim == 2:
        t = t.squeeze(0)
    return t

def _as_TN_C(arr: np.ndarray, field: str, T_hint: Optional[int]) -> np.ndarray:
    """Normalize field data to (T, N, C) format."""
    x = np.asarray(arr)
    if x.ndim == 3:
        return x[0, :, :, None]
    elif x.ndim == 4:
        return x[0, :, :, :]
    else:
        raise RuntimeError(f"Unsupported rank {x.ndim} for field '{field}'.")

def _read_field_TNC(h5: h5py.File, group: str, field: str, T_hint: Optional[int]) -> np.ndarray:
    """Read a field from HDF5 and convert to (T, N, C) format."""
    if group not in h5:
        raise KeyError(f"Group '{group}' not found. Have: {list(h5.keys())}")
    if field not in h5[group]:
        have = list(h5[group].keys())
        raise KeyError(f"Field '{field}' not in group '{group}'. Have: {have}")
    x = np.array(h5[f"{group}/{field}"][...], copy=False)
    return _as_TN_C(x, field=field, T_hint=T_hint).astype(np.float32)

def _list_field_datasets(h5: h5py.File, group: str) -> List[str]:
    """List all dataset names in a group."""
    if group not in h5:
        return []
    return [k for k, v in h5[group].items() if isinstance(v, h5py.Dataset)]


def compute_autocorrelation(
    data: np.ndarray, lag: int, axis: int = 0
) -> np.ndarray:
    """
    Compute autocorrelation at given lag.
    
    Args:
        data: Array of shape (T, N, C) or (T, N)
        lag: Time lag for autocorrelation
        axis: Time axis (default: 0)
    
    Returns:
        Autocorrelation values, shape (N, C) or (N,)
    """
    if data.shape[axis] <= lag:
        return np.nan
    
    # Flatten spatial dimensions
    if data.ndim == 3:
        T, N, C = data.shape
        data_flat = data.reshape(T, N * C)
        corr_flat = np.zeros(N * C)
        for i in range(N * C):
            x = data_flat[:, i]
            if np.std(x) < 1e-10:
                corr_flat[i] = np.nan
            else:
                corr = np.corrcoef(x[:-lag], x[lag:])[0, 1]
                corr_flat[i] = corr if not np.isnan(corr) else 0.0
        return corr_flat.reshape(N, C)
    elif data.ndim == 2:
        T, N = data.shape
        corr = np.zeros(N)
        for i in range(N):
            x = data[:, i]
            if np.std(x) < 1e-10:
                corr[i] = np.nan
            else:
                c = np.corrcoef(x[:-lag], x[lag:])[0, 1]
                corr[i] = c if not np.isnan(c) else 0.0
        return corr
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")


def analyze_file_autocorrelation(
    filepath: str,
    field_names: List[str],
    max_timestep: int = 20,
    target_autocorr: float = 0.5,
) -> Dict:
    """
    Analyze autocorrelation for a single HDF5 file.
    
    Returns:
        Dictionary with autocorrelation results per field
    """
    results = {}
    
    with h5py.File(filepath, "r") as h5:
        # Get time values
        t_vals = _time_from_dimensions(h5)
        T = len(t_vals)
        
        # Analyze each field
        for field_name in field_names:
            if field_name not in h5.get("fields", {}):
                continue
            
            # Read field data: (T, N, C)
            field_data = _read_field_TNC(h5, "fields", field_name, T)
            
            # Compute autocorrelation for different lags
            autocorrs = []
            valid_lags = []
            
            for lag in range(1, min(max_timestep + 1, T)):
                try:
                    ac = compute_autocorrelation(field_data, lag, axis=0)
                    # Average over spatial dimensions
                    ac_mean = np.nanmean(ac)
                    if not np.isnan(ac_mean):
                        autocorrs.append(ac_mean)
                        valid_lags.append(lag)
                except Exception as e:
                    print(f"  Warning: Failed to compute autocorr for {field_name} at lag {lag}: {e}")
                    continue
            
            if len(autocorrs) > 0:
                results[field_name] = {
                    "lags": np.array(valid_lags),
                    "autocorrs": np.array(autocorrs),
                }
    
    return results


def find_optimal_timestep(
    lags: np.ndarray,
    autocorrs: np.ndarray,
    target: float = 0.5,
    tolerance: float = 0.1,
) -> Optional[int]:
    """
    Find timestep where autocorrelation is closest to target.
    
    Args:
        lags: Array of lag values
        autocorrs: Array of autocorrelation values
        target: Target autocorrelation (default: 0.5)
        tolerance: Acceptable range around target
    
    Returns:
        Optimal lag (timestep) or None if not found
    """
    if len(autocorrs) == 0:
        return None
    
    # Find closest to target
    distances = np.abs(autocorrs - target)
    min_idx = np.argmin(distances)
    
    if distances[min_idx] <= tolerance:
        return int(lags[min_idx])
    
    # If no value within tolerance, return closest
    return int(lags[min_idx])


def analyze_dataset(
    data_dir: str,
    field_names: List[str],
    max_timestep: int = 20,
    target_autocorr: float = 0.5,
    max_files: Optional[int] = None,
) -> Dict:
    """
    Analyze autocorrelation across entire dataset.
    
    Args:
        data_dir: Directory containing HDF5 files
        field_names: List of field names to analyze
        max_timestep: Maximum timestep to analyze
        target_autocorr: Target autocorrelation value
        max_files: Maximum number of files to process (None = all)
    
    Returns:
        Dictionary with aggregated results
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    if max_files is not None:
        files = files[:max_files]
    
    if len(files) == 0:
        raise FileNotFoundError(f"No HDF5 files found in {data_dir}")
    
    print(f"Analyzing {len(files)} files...")
    
    # Aggregate results across files
    field_results = {name: {"lags": [], "autocorrs": []} for name in field_names}
    
    for i, filepath in enumerate(files):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(files)} files...")
        
        try:
            file_results = analyze_file_autocorrelation(
                filepath, field_names, max_timestep, target_autocorr
            )
            
            for field_name, results in file_results.items():
                if field_name in field_results:
                    field_results[field_name]["lags"].extend(results["lags"].tolist())
                    field_results[field_name]["autocorrs"].extend(
                        results["autocorrs"].tolist()
                    )
        except Exception as e:
            print(f"  Warning: Failed to process {filepath}: {e}")
            continue
    
    # Aggregate and compute statistics
    aggregated = {}
    for field_name in field_names:
        if len(field_results[field_name]["lags"]) == 0:
            continue
        
        lags = np.array(field_results[field_name]["lags"])
        autocorrs = np.array(field_results[field_name]["autocorrs"])
        
        # Group by lag and compute mean/std
        unique_lags = np.unique(lags)
        lag_means = []
        lag_stds = []
        lag_counts = []
        
        for lag in unique_lags:
            mask = lags == lag
            ac_vals = autocorrs[mask]
            lag_means.append(np.nanmean(ac_vals))
            lag_stds.append(np.nanstd(ac_vals))
            lag_counts.append(np.sum(mask))
        
        # Find optimal timestep
        optimal = find_optimal_timestep(
            unique_lags, np.array(lag_means), target_autocorr
        )
        
        aggregated[field_name] = {
            "lags": unique_lags,
            "autocorr_mean": np.array(lag_means),
            "autocorr_std": np.array(lag_stds),
            "counts": np.array(lag_counts),
            "optimal_timestep": optimal,
        }
    
    return aggregated


def plot_autocorrelation_analysis(
    results: Dict,
    output_dir: str,
    target_autocorr: float = 0.5,
):
    """
    Create visualization of autocorrelation analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    n_fields = len(results)
    if n_fields == 0:
        print("No results to plot")
        return
    
    # Create subplots
    fig, axes = plt.subplots(
        (n_fields + 2) // 3, 3, figsize=(15, 5 * ((n_fields + 2) // 3))
    )
    if n_fields == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    optimal_timesteps = {}
    
    for idx, (field_name, data) in enumerate(results.items()):
        ax = axes[idx]
        
        lags = data["lags"]
        means = data["autocorr_mean"]
        stds = data["autocorr_std"]
        optimal = data["optimal_timestep"]
        
        # Plot mean with error bars
        ax.errorbar(
            lags,
            means,
            yerr=stds,
            marker="o",
            capsize=3,
            label=f"{field_name} (optimal={optimal})",
        )
        
        # Add target line
        ax.axhline(
            y=target_autocorr,
            color="r",
            linestyle="--",
            label=f"Target ({target_autocorr})",
        )
        
        # Highlight optimal timestep
        if optimal is not None:
            optimal_idx = np.where(lags == optimal)[0]
            if len(optimal_idx) > 0:
                ax.axvline(
                    x=optimal,
                    color="g",
                    linestyle=":",
                    alpha=0.5,
                    label=f"Optimal timestep={optimal}",
                )
        
        ax.set_xlabel("Timestep (lag)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(f"{field_name}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        optimal_timesteps[field_name] = optimal
    
    # Hide unused subplots
    for idx in range(n_fields, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "autocorrelation_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {plot_path}")
    plt.close()
    
    # Create summary plot (all fields together)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for field_name, data in results.items():
        lags = data["lags"]
        means = data["autocorr_mean"]
        optimal = data["optimal_timestep"]
        
        ax.plot(
            lags,
            means,
            marker="o",
            label=f"{field_name} (opt={optimal})",
            alpha=0.7,
        )
    
    ax.axhline(y=target_autocorr, color="r", linestyle="--", label=f"Target ({target_autocorr})")
    ax.set_xlabel("Timestep (lag)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation Analysis - All Fields")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    
    plt.tight_layout()
    summary_path = os.path.join(output_dir, "autocorrelation_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    print(f"Saved summary plot to {summary_path}")
    plt.close()
    
    return optimal_timesteps


def save_results(
    results: Dict,
    optimal_timesteps: Dict,
    output_dir: str,
    target_autocorr: float,
):
    """
    Save analysis results to text file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "autocorrelation_results.txt")
    
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MagLIF Autocorrelation Analysis Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Target autocorrelation: {target_autocorr}\n\n")
        
        f.write("Optimal Timesteps (autocorrelation ≈ 0.5):\n")
        f.write("-" * 80 + "\n")
        for field_name, optimal in sorted(optimal_timesteps.items()):
            if optimal is not None:
                f.write(f"  {field_name:20s}: {optimal:3d}\n")
            else:
                f.write(f"  {field_name:20s}: Not found\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Detailed Results per Field:\n")
        f.write("=" * 80 + "\n\n")
        
        for field_name, data in results.items():
            f.write(f"\n{field_name}:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Optimal timestep: {data['optimal_timestep']}\n")
            f.write(f"  Timestep | Autocorr (mean ± std) | Count\n")
            
            for lag, mean, std, count in zip(
                data["lags"],
                data["autocorr_mean"],
                data["autocorr_std"],
                data["counts"],
            ):
                f.write(f"  {lag:8d} | {mean:7.4f} ± {std:6.4f} | {count:6d}\n")
    
    print(f"Saved results to {output_path}")
    
    # Also save as JSON for programmatic access
    import json
    
    json_results = {}
    for field_name, data in results.items():
        json_results[field_name] = {
            "optimal_timestep": int(data["optimal_timestep"])
            if data["optimal_timestep"] is not None
            else None,
            "lags": data["lags"].tolist(),
            "autocorr_mean": data["autocorr_mean"].tolist(),
            "autocorr_std": data["autocorr_std"].tolist(),
        }
    
    json_path = os.path.join(output_dir, "autocorrelation_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Saved JSON results to {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze autocorrelation in MagLIF dataset to find optimal timestep"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing HDF5 files (train split)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./autocorr_analysis",
        help="Output directory for results and plots",
    )
    parser.add_argument(
        "--target_autocorr",
        type=float,
        default=0.5,
        help="Target autocorrelation value (default: 0.5)",
    )
    parser.add_argument(
        "--max_timestep",
        type=int,
        default=20,
        help="Maximum timestep to analyze (default: 20)",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process (default: all)",
    )
    parser.add_argument(
        "--field_names",
        type=str,
        nargs="+",
        default=None,
        help="Field names to analyze (default: all MagLIF fields)",
    )
    
    args = parser.parse_args()
    
    # Default MagLIF field names
    if args.field_names is None:
        field_names = [
            "Rho",
            "rho_Be",
            "rho_DT",
            "T_elec",
            "T_ion",
            "Rad_Temp",
            "Vel",
            "P_ion",
            "P_elec",
            "n_elec",
            "bmag",
            "jz",
        ]
    else:
        field_names = args.field_names
    
    print("=" * 80)
    print("MagLIF Autocorrelation Analysis")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target autocorrelation: {args.target_autocorr}")
    print(f"Max timestep: {args.max_timestep}")
    print(f"Fields: {', '.join(field_names)}")
    print("=" * 80)
    
    # Analyze dataset
    results = analyze_dataset(
        args.data_dir,
        field_names,
        args.max_timestep,
        args.target_autocorr,
        args.max_files,
    )
    
    if len(results) == 0:
        print("ERROR: No results obtained. Check data directory and field names.")
        return 1
    
    # Create visualizations
    optimal_timesteps = plot_autocorrelation_analysis(
        results, args.output_dir, args.target_autocorr
    )
    
    # Save results
    save_results(results, optimal_timesteps, args.output_dir, args.target_autocorr)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary - Optimal Timesteps:")
    print("=" * 80)
    for field_name, optimal in sorted(optimal_timesteps.items()):
        if optimal is not None:
            print(f"  {field_name:20s}: {optimal:3d}")
        else:
            print(f"  {field_name:20s}: Not found")
    
    # Compute global recommendation (median of optimal timesteps)
    valid_optimals = [v for v in optimal_timesteps.values() if v is not None]
    if len(valid_optimals) > 0:
        global_optimal = int(np.median(valid_optimals))
        print(f"\n  {'Global recommendation':20s}: {global_optimal:3d} (median)")
        print(
            f"\nRecommendation: Use time_step={global_optimal} in your training config"
        )
    
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
