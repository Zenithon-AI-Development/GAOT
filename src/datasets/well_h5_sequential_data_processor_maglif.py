# -*- coding: utf-8 -*-
# src/datasets/well_h5_sequential_data_processor_maglif.py
"""
Sequential data processor for 1D MagLIF dataset.
Handles radial-only coordinates and specific MagLIF field structure.
"""
import os, glob, h5py, numpy as np, torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
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

def _u_mean_std(train_files: List[str], field_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and std for u fields across training set."""
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
    return u_mean.astype(np.float32), u_std.astype(np.float32)


class WellH5SequentialDataProcessorMagLIF(SequentialDataProcessor):
    """
    Data processor for 1D MagLIF dataset.
    - 1D radial coordinates only
    - 9 fields: rho_Be, rho_DT, T_elec, T_ion, Vel, P_ion, P_elec, n_elec, bmag
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
            
            # Define the field order we want (9 channels)
            # Based on metadata: Rho, mat(1)%zeff, T_elec, T_ion, Vel, P_ion, P_elec, n_elec, bmag
            # Map to actual HDF5 field names (exact match from file)
            field_mapping = [
                "Rho",           # channel 0: total density
                "mat(1)%zeff",   # channel 1: effective charge
                "T_elec",        # channel 2: electron temperature
                "T_ion",         # channel 3: ion temperature
                "Vel",           # channel 4: velocity
                "P_ion",         # channel 5: ion pressure
                "P_elec",        # channel 6: electron pressure
                "n_elec",        # channel 7: electron number density
                "bmag",          # channel 8: magnetic field magnitude
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

        # Compute mean/std over TRAIN
        u_mean, u_std = _u_mean_std(train_files, field_names)

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
            return WellH5PairIterableMagLIF(
                split_dir=meta["split_dirs"][split],
                stats=self.stats,
                time_step=meta["time_step"],
                max_time_diff=meta["max_time_diff"],
                field_names=meta["field_names"],
                cache_samples=getattr(self.dataset_config, "stream_cache_samples", 1),
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

