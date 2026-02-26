# -*- coding: utf-8 -*-
# src/datasets/well_h5_sequential_data_processor_maglif.py
"""
Sequential data processor for 1D MagLIF dataset.
Handles radial-only coordinates and specific MagLIF field structure.
"""
import os, glob, h5py, numpy as np, torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from scipy import interpolate
from scipy.stats import norm
from ..utils.scaling import CoordinateScaler

from .sequential_data_processor import SequentialDataProcessor
from .well_h5_pair_dataset_maglif import (
    WellH5PairIterableMagLIF,
    build_time_pairs,
    _mesh_from_dimensions_1d, 
    _time_from_dimensions, 
    _list_field_datasets,
    _as_TN_C,
    _read_field_TNC,
)

# ---------- normalization helpers for log/asinh normalization ----------
def _log_normalize_channel(x: np.ndarray, field_name: str, channel_idx: int) -> Tuple[np.ndarray, float]:
    """
    Apply log normalization to a channel.
    Returns (normalized_data, offset_used).
    """
    # Fields with zeros but no negatives: use small offset
    fields_with_zeros = ["Rad_Temp", "P_ion", "P_elec"]
    # For positive-only fields, use minimal epsilon to avoid log(0)
    offset = 1e-6 if field_name in fields_with_zeros else 1e-10
    x_positive = x + offset
    x_log = np.log(x_positive)
    return x_log, offset

def _asinh_normalize_channel(x: np.ndarray, field_name: str) -> Tuple[np.ndarray, float]:
    """
    Apply asinh normalization to a channel (for signed values).
    Returns (normalized_data, scale_used).
    Uses scale to normalize input before asinh: asinh(x/scale)
    """
    # Compute robust scale based on percentiles to avoid outliers
    abs_x = np.abs(x[x != 0]) if (x == 0).any() else np.abs(x)
    if len(abs_x) > 0:
        scale = np.percentile(abs_x, 95) if len(abs_x) > 10 else np.max(abs_x)
        scale = max(scale, 1e-10)  # Avoid division by zero
    else:
        scale = 1.0
    x_scaled = x / scale
    x_asinh = np.arcsinh(x_scaled)
    return x_asinh, scale

def _log_denormalize_channel(x_norm: np.ndarray, mean: float, std: float, offset: float) -> np.ndarray:
    """Denormalize log-normalized channel: exp(normalized * std + mean) - offset"""
    return np.exp(x_norm * std + mean) - offset

def _asinh_denormalize_channel(x_norm: np.ndarray, mean: float, std: float, scale: float) -> np.ndarray:
    """Denormalize asinh-normalized channel: scale * sinh(normalized * std + mean)"""
    return scale * np.sinh(x_norm * std + mean)

# ---------- helpers to compute stats across TRAIN ----------
def _stream_true_time_stats(files: List[str], time_step: int, max_time_diff: Optional[int]) -> Tuple[float,float,float,float]:
    """Compute time statistics across all training files."""
    n_st=0; st_sum=0.0; st_sumsq=0.0
    n_dt=0; dt_sum=0.0; dt_sumsq=0.0
    for fp in files:
        with h5py.File(fp, "r") as f:
            t = _time_from_dimensions(f).astype(np.float64)
            T = int(t.shape[0])
        ti, to = build_time_pairs(T, max_time_diff, time_step)
        if ti.size == 0: 
            continue
        starts = t[ti]
        diffs  = t[to] - t[ti]

        n_st += int(starts.size)
        st_sum += starts.sum()
        st_sumsq += (starts**2).sum()

        n_dt += int(diffs.size)
        dt_sum += diffs.sum()
        dt_sumsq += (diffs**2).sum()

    if n_st == 0:
        return 0.0, 1.0, 1.0, 1.0
    st_mean = st_sum / n_st
    st_std  = float(np.sqrt(max(st_sumsq / n_st - st_mean**2, 1e-12)))
    if n_dt == 0:
        return st_mean, st_std, 1.0, 1.0
    dt_mean = dt_sum / n_dt
    dt_std  = float(np.sqrt(max(dt_sumsq / n_dt - dt_mean**2, 1e-12)))
    return float(st_mean), float(st_std), float(dt_mean), float(dt_std)

def _stack_u_for_file(h5: h5py.File, field_names: List[str]) -> np.ndarray:
    """Stack all u fields from a file into (T, N, Cu) format."""
    t_vals = _time_from_dimensions(h5)
    T_hint = int(t_vals.shape[0])
    
    chunks = []
    for fname in field_names:
        field_data = _read_field_TNC(h5, "fields", fname, T_hint)
        chunks.append(field_data)
    
    return np.concatenate(chunks, axis=-1)  # (T, N, Cu)

def _u_mean_std(train_files: List[str], field_names: List[str], normalization_mode: str = "standard") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean and std for u fields across training set.
    If normalization_mode == "log", computes stats on log/asinh transformed data.
    If normalization_mode == "quantile", computes stats on quantile-normalized data.
    Returns (u_mean, u_std, norm_params_1, norm_params_2) where norm_params are quantiles per channel.
    """
    if normalization_mode == "standard":
        u_sum = None
        u_sumsq = None
        u_count = 0

        for fp in train_files:
            with h5py.File(fp, "r") as h5:
                uTNC = _stack_u_for_file(h5, field_names)  # (T, N, Cu)
                Cu = uTNC.shape[-1]
                if u_sum is None:
                    u_sum   = np.zeros((Cu,), dtype=np.float64)
                    u_sumsq = np.zeros((Cu,), dtype=np.float64)
                x = uTNC.reshape(-1, Cu).astype(np.float64)  # (T*N, Cu)
                u_sum   += x.sum(axis=0)
                u_sumsq += (x * x).sum(axis=0)
                u_count += x.shape[0]

        u_mean = u_sum / max(1, u_count)
        u_var  = np.maximum(u_sumsq / max(1, u_count) - u_mean*u_mean, 1e-12)
        u_std  = np.sqrt(u_var)
        # For standard normalization, norm_params are dummy zeros
        norm_params_1 = np.zeros((len(field_names),), dtype=np.float32)
        norm_params_2 = np.zeros((len(field_names),), dtype=np.float32)
        return u_mean.astype(np.float32), u_std.astype(np.float32), norm_params_1, norm_params_2
    
    elif normalization_mode == "log":
        # For log normalization: compute stats on transformed data
        u_sum = None
        u_sumsq = None
        u_count = 0
        norm_params_1 = np.zeros((len(field_names),), dtype=np.float32)  # offsets/scales
        norm_params_2 = np.zeros((len(field_names),), dtype=np.float32)  # unused (for future use)
        
        # First pass: determine normalization parameters per channel
        signed_fields = ["Vel", "jz"]
        fields_with_zeros = ["Rad_Temp", "P_ion", "P_elec"]
        
        # Collect data for each channel to compute normalization params
        channel_data = [[] for _ in field_names]
        for fp in train_files[:min(10, len(train_files))]:  # Sample first 10 files for efficiency
            with h5py.File(fp, "r") as h5:
                uTNC = _stack_u_for_file(h5, field_names)  # (T, N, Cu)
                x = uTNC.reshape(-1, len(field_names)).astype(np.float64)  # (T*N, Cu)
                for ch_idx in range(len(field_names)):
                    channel_data[ch_idx].extend(x[:, ch_idx].tolist())
        
        # Compute normalization parameters per channel
        for ch_idx, field_name in enumerate(field_names):
            if field_name in signed_fields:
                # Use asinh normalization for signed fields
                ch_data = np.array(channel_data[ch_idx], dtype=np.float64)
                _, scale = _asinh_normalize_channel(ch_data, field_name)
                norm_params_1[ch_idx] = float(scale)
            else:
                # Use log normalization for other fields
                ch_data = np.array(channel_data[ch_idx], dtype=np.float64)
                _, offset = _log_normalize_channel(ch_data, field_name, ch_idx)
                norm_params_1[ch_idx] = float(offset)
        
        # Second pass: compute mean/std on transformed data
        for fp in train_files:
            with h5py.File(fp, "r") as h5:
                uTNC = _stack_u_for_file(h5, field_names)  # (T, N, Cu)
                Cu = uTNC.shape[-1]
                if u_sum is None:
                    u_sum   = np.zeros((Cu,), dtype=np.float64)
                    u_sumsq = np.zeros((Cu,), dtype=np.float64)
                
                x = uTNC.reshape(-1, Cu).astype(np.float64)  # (T*N, Cu)
                
                # Transform each channel according to its normalization type
                x_transformed = np.zeros_like(x, dtype=np.float64)
                for ch_idx, field_name in enumerate(field_names):
                    if field_name in signed_fields:
                        # asinh normalization
                        scale = norm_params_1[ch_idx]
                        x_transformed[:, ch_idx] = np.arcsinh(x[:, ch_idx] / scale)
                    else:
                        # log normalization
                        offset = norm_params_1[ch_idx]
                        x_transformed[:, ch_idx] = np.log(x[:, ch_idx] + offset)
                
                u_sum   += x_transformed.sum(axis=0)
                u_sumsq += (x_transformed * x_transformed).sum(axis=0)
                u_count += x_transformed.shape[0]

        u_mean = u_sum / max(1, u_count)
        u_var  = np.maximum(u_sumsq / max(1, u_count) - u_mean*u_mean, 1e-12)
        u_std  = np.sqrt(u_var)
        return u_mean.astype(np.float32), u_std.astype(np.float32), norm_params_1, norm_params_2
    
    elif normalization_mode == "quantile":
        # For quantile normalization: compute quantiles per channel, then compute stats on quantile-normalized data
        num_quantiles = 1000
        quantile_levels = np.linspace(0, 1, num_quantiles + 1)[1:-1]  # Exclude 0 and 1 to avoid inf
        
        # First pass: collect all data per channel to compute quantiles
        channel_data = [[] for _ in field_names]
        for fp in train_files:
            with h5py.File(fp, "r") as h5:
                uTNC = _stack_u_for_file(h5, field_names)  # (T, N, Cu)
                x = uTNC.reshape(-1, len(field_names)).astype(np.float64)  # (T*N, Cu)
                for ch_idx in range(len(field_names)):
                    channel_data[ch_idx].extend(x[:, ch_idx].tolist())
        
        # Compute quantiles per channel
        quantiles_per_channel = []
        for ch_idx in range(len(field_names)):
            ch_data = np.array(channel_data[ch_idx], dtype=np.float64)
            if len(ch_data) == 0:
                quantiles = np.zeros(num_quantiles - 1, dtype=np.float32)
            else:
                quantiles = np.quantile(ch_data, quantile_levels).astype(np.float32)
            quantiles_per_channel.append(quantiles)
        
        # Store quantiles in norm_params_1 (flattened, will need to reshape later)
        # We'll store as a 2D array: [num_channels, num_quantiles]
        quantiles_array = np.array(quantiles_per_channel, dtype=np.float32)  # [Cu, num_quantiles]
        norm_params_1 = quantiles_array.flatten()  # Flatten for storage
        norm_params_2 = np.array([num_quantiles - 1, len(field_names)], dtype=np.float32)  # Store shape info
        
        # Second pass: compute mean/std on quantile-normalized data
        u_sum = None
        u_sumsq = None
        u_count = 0
        
        for fp in train_files:
            with h5py.File(fp, "r") as h5:
                uTNC = _stack_u_for_file(h5, field_names)  # (T, N, Cu)
                Cu = uTNC.shape[-1]
                if u_sum is None:
                    u_sum   = np.zeros((Cu,), dtype=np.float64)
                    u_sumsq = np.zeros((Cu,), dtype=np.float64)
                
                x = uTNC.reshape(-1, Cu).astype(np.float64)  # (T*N, Cu)
                
                # Transform each channel using quantile normalization
                x_transformed = np.zeros_like(x, dtype=np.float64)
                for ch_idx in range(Cu):
                    ch_values = x[:, ch_idx]
                    ch_quantiles = quantiles_per_channel[ch_idx]
                    
                    # Map each value to its quantile rank, then to standard normal distribution
                    # Sort quantiles and create mapping
                    sorted_quantiles = np.sort(ch_quantiles)
                    quantile_ranks = np.linspace(0, 1, len(sorted_quantiles))
                    
                    # Create interpolation function: value -> quantile rank
                    interp_func = interpolate.interp1d(
                        sorted_quantiles, quantile_ranks,
                        kind='linear',
                        bounds_error=False,
                        fill_value=(0.0, 1.0)
                    )
                    
                    # Map to quantile ranks
                    ranks = interp_func(ch_values)
                    ranks = np.clip(ranks, 0.0, 1.0)
                    
                    # Map quantile ranks to standard normal distribution using inverse CDF
                    # Clip to avoid inf values at boundaries
                    ranks_clipped = np.clip(ranks, 0.001, 0.999)
                    x_transformed[:, ch_idx] = norm.ppf(ranks_clipped)
                
                u_sum   += x_transformed.sum(axis=0)
                u_sumsq += (x_transformed * x_transformed).sum(axis=0)
                u_count += x_transformed.shape[0]
        
        u_mean = u_sum / max(1, u_count)
        u_var  = np.maximum(u_sumsq / max(1, u_count) - u_mean*u_mean, 1e-12)
        u_std  = np.sqrt(u_var)
        return u_mean.astype(np.float32), u_std.astype(np.float32), norm_params_1, norm_params_2
    else:
        raise ValueError(f"Unknown normalization_mode: {normalization_mode}. Must be 'standard', 'log', or 'quantile'.")


class WellH5SequentialDataProcessorMagLIF(SequentialDataProcessor):
    """
    Data processor for 1D MagLIF dataset.
    - 1D radial coordinates only
    - 12 fields: Rho, rho_Be, rho_DT, T_elec, T_ion, Rad_Temp, Vel, P_ion, P_elec, n_elec, bmag, jz
    - No conditioning (c) data
    - Streaming-based processing to handle large datasets
    """

    def _load_raw_sequential_data(self) -> Dict:
        base = self.dataset_config.base_path
        name = self.dataset_config.name

        split_dirs = {
            "train": os.path.join(base, name, "data", "train"),
            "val":   os.path.join(base, name, "data", "valid"),
            "test":  os.path.join(base, name, "data", "test"),
        }
        for k, d in split_dirs.items():
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Missing split directory: {d}")

        train_files = sorted(glob.glob(os.path.join(split_dirs["train"], "*.hdf5")))
        val_files   = sorted(glob.glob(os.path.join(split_dirs["val"],   "*.hdf5")))
        test_files  = sorted(glob.glob(os.path.join(split_dirs["test"],  "*.hdf5")))
        max_train = getattr(self.dataset_config, "max_train_files", None)
        max_val   = getattr(self.dataset_config, "max_val_files", None)
        max_test  = getattr(self.dataset_config, "max_test_files", None)
        if max_train is not None:
            train_files = train_files[:max_train]
        if max_val is not None:
            val_files = val_files[:max_val]
        if max_test is not None:
            test_files = test_files[:max_test]
        if not train_files:
            raise FileNotFoundError(f"No .hdf5 files found under {split_dirs['train']}")

        # Probe first file to get structure
        with h5py.File(train_files[0], "r") as h5:
            # Get 1D coordinates [N, 1]
            x_fixed_phys = _mesh_from_dimensions_1d(h5).astype(np.float32)
            t_vals = _time_from_dimensions(h5).astype(np.float32)
            
            # Discover fields
            all_fields = _list_field_datasets(h5, "fields")
            print(f"[MagLIF] Found {len(all_fields)} fields: {all_fields}")
            
            # Define the field order we want (12 channels)
            # Map to actual HDF5 field names (exact match from file)
            field_mapping = [
                "Rho",           # channel 0: total density
                "rho_Be",        # channel 1: beryllium density
                "rho_DT",        # channel 2: DT density
                "T_elec",        # channel 3: electron temperature
                "T_ion",         # channel 4: ion temperature
                "Rad_Temp",      # channel 5: radiation temperature
                "Vel",           # channel 6: velocity
                "P_ion",         # channel 7: ion pressure
                "P_elec",        # channel 8: electron pressure
                "n_elec",        # channel 9: electron number density
                "bmag",          # channel 10: magnetic field magnitude
                "jz",            # channel 11: current density
            ]
            
            # Verify all fields exist
            missing = [f for f in field_mapping if f not in all_fields]
            if missing:
                raise ValueError(f"Missing required fields: {missing}. Available: {all_fields}")
            
            field_names = field_mapping

        # Coordinate scaling to [-1, 1]
        from ..utils.helpers_true_domain_1d import true_domain_from_dir_1d
        train_dir = os.path.join(base, name, "data", "train")
        true_domain = true_domain_from_dir_1d(train_dir)  # ((rmin,), (rmax,))

        self.coord_scaler = CoordinateScaler(
            target_range=(-1, 1),
            mode=getattr(self.dataset_config, "coord_scaling", "domain")
        )
        
        # Fit on true domain corners (1D)
        (r0,), (r1,) = true_domain
        self.coord_scaler.fit(torch.tensor([[r0], [r1]], dtype=self.dtype))

        # Store domain for metadata
        self.metadata.domain_x = ((r0,), (r1,))

        # Return unscaled coords (trainer applies scaling once)
        x_fixed = torch.tensor(x_fixed_phys, dtype=self.dtype)

        # Get normalization mode from config
        normalization_mode = getattr(self.dataset_config, "normalization_mode", "standard")
        
        # Compute mean/std over TRAIN
        u_mean, u_std, norm_params_1, norm_params_2 = _u_mean_std(train_files, field_names, normalization_mode)

        # Time stats
        time_step = int(self.time_step if self.time_step is not None else 1)
        max_diff  = None if (self.max_time_diff is None) else int(self.max_time_diff)
        st_m, st_s, dt_m, dt_s = _stream_true_time_stats(train_files, time_step, max_diff)

        # Build stats dict
        self.stats = {
            "u": {
                "mean": torch.tensor(u_mean, dtype=self.dtype).view(1, -1),
                "std":  torch.tensor(u_std,  dtype=self.dtype).clamp_min(1e-8).view(1, -1)
            },
            "start_time": {
                "mean": torch.tensor(st_m, dtype=self.dtype),
                "std":  torch.tensor(st_s + 1e-8, dtype=self.dtype)
            },
            "time_diffs": {
                "mean": torch.tensor(dt_m, dtype=self.dtype),
                "std":  torch.tensor(dt_s + 1e-8, dtype=self.dtype)
            },
            "normalization_mode": normalization_mode,
            "norm_params_1": torch.tensor(norm_params_1, dtype=self.dtype),  # offsets for log, scales for asinh
            "norm_params_2": torch.tensor(norm_params_2, dtype=self.dtype),  # reserved for future use
            "field_names": field_names,  # Store field names for channel-specific normalization
        }

        t_vals_t = torch.tensor(t_vals, dtype=self.dtype)
        N = x_fixed.shape[0]
        Cu = len(field_names)

        # Dummy tensors to signal channel counts
        dummy_u = torch.zeros((1, 2, N, Cu), dtype=self.dtype)
        dummy_c = None  # No conditioning for MagLIF

        # Store field names for reference
        self.field_names = field_names

        return {
            "train": {"u": dummy_u, "c": dummy_c, "x": x_fixed, "t": t_vals_t},
            "val":   {"u": dummy_u, "c": dummy_c, "x": x_fixed, "t": t_vals_t},
            "test":  {"u": dummy_u, "c": dummy_c, "x": x_fixed, "t": t_vals_t},
            "_stream_meta": {
                "split_dirs": split_dirs,
                "field_names": field_names,
                "time_step": time_step,
                "max_time_diff": max_diff,
                "train_files": train_files,
                "val_files": val_files,
                "test_files": test_files,
            }
        }

    def _split_and_normalize_sequential_data(self, raw: Dict, is_variable_coords: bool) -> Dict:
        """Just set t_values; normalization already handled via streaming."""
        self.t_values = raw["train"]["t"].cpu().numpy()
        return raw

    def create_sequential_data_loaders(self, data_splits: Dict, is_variable_coords: bool, **kwargs):
        """Create streaming data loaders."""
        meta = data_splits["_stream_meta"]

        def mk(split):
            sample_ratio = getattr(self.dataset_config, "sample_ratio", None) if split == "train" else None
            files_list = meta.get(f"{split}_files")
            max_s = getattr(self.dataset_config, f"max_{split}_samples", None)
            return WellH5PairIterableMagLIF(
                split_dir=meta["split_dirs"][split],
                stats=self.stats,
                time_step=meta["time_step"],
                max_time_diff=meta["max_time_diff"],
                field_names=meta["field_names"],
                cache_samples=getattr(self.dataset_config, "stream_cache_samples", 1),
                normalization_mode=self.stats.get("normalization_mode", "standard"),
                sample_ratio=sample_ratio,
                files_list=files_list,
                max_samples=max_s,
            )

        def make_loader(dataset):
            return DataLoader(
                dataset,
                batch_size=self.dataset_config.batch_size,
                shuffle=False,
                num_workers=self.dataset_config.num_workers,
                pin_memory=True,
            )

        loaders = {}
        if getattr(self.dataset_config, "train", True):
            loaders["train"] = make_loader(mk("train"))
            loaders["val"]   = make_loader(mk("val"))
        else:
            loaders["train"] = loaders["val"] = None

        loaders["test"] = make_loader(mk("test"))
        self.t_values = data_splits["train"]["t"].cpu().numpy()
        self.runtime_hints = {"use_trainer_autoreg": True}
        return loaders

