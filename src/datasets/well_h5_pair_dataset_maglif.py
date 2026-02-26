# -*- coding: utf-8 -*-
# src/datasets/well_h5_pair_dataset_maglif.py
"""
1D-specific pair dataset for MagLIF data.
Handles radial-only coordinates (no z dimension).
"""
import os, glob, h5py, numpy as np, torch
from torch.utils.data import IterableDataset
from typing import Dict, List, Tuple, Optional

# ---------------- basic helpers (1D-specific) ----------------
def _time_from_dimensions(h5: h5py.File) -> np.ndarray:
    """Extract time values from dimensions group."""
    if "dimensions" not in h5 or "time" not in h5["dimensions"]:
        raise KeyError("dimensions/time not found in HDF5")
    t = np.asarray(h5["dimensions/time"][...], dtype=np.float64)
    # Handle shape (1, T) or (T,)
    if t.ndim == 2:
        t = t.squeeze(0)
    return t

def _mesh_from_dimensions_1d(h5: h5py.File) -> np.ndarray:
    """
    Extract 1D radial coordinates.
    Returns: [N, 1] array of r coordinates.
    """
    dimg = h5["dimensions"]
    r = np.asarray(dimg["r_coords"][...], dtype=np.float64)
    # Return as [N, 1] for consistency with 2D case
    return r.reshape(-1, 1)

def _list_field_datasets(h5: h5py.File, group: str) -> List[str]:
    """List all dataset names in a group."""
    if group not in h5:
        return []
    return [k for k, v in h5[group].items() if isinstance(v, h5py.Dataset)]

def build_time_pairs(T: int, max_time_diff: Optional[int], time_step: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build time index pairs for training.
    For MagLIF: uses only timesteps 3, 12, 20 (optimal from autocorrelation analysis).
    Returns (input_indices, output_indices).
    """
    if T <= 1:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    
    # MagLIF-specific: use only optimal timesteps from autocorrelation analysis
    # Timesteps 3, 12, 20 correspond to optimal autocorrelation ~0.5 for different fields
    # Ignore max_time_diff and time_step config for MagLIF to ensure all optimal lags are used
    optimal_lags = [3, 12, 20]
    
    ti, to = [], []
    for lag in optimal_lags:
        # Only use lag if it fits within available timesteps
        if lag < T:
            for i in range(0, T - lag, 1):
                ti.append(i)
                to.append(i + lag)
    return np.asarray(ti, np.int64), np.asarray(to, np.int64)

def _as_TN_C(arr: np.ndarray, field: str, T_hint: Optional[int]) -> np.ndarray:
    """
    Normalize field data to (T, N, C) format.
    Expected input shapes:
    - (B, T, N) where B=n_trajectories (usually 1)
    - (B, T, N, C) where C is number of components
    """
    x = np.asarray(arr)
    
    if x.ndim == 3:  # (B, T, N) - scalar field
        return x[0, :, :, None]  # -> (T, N, 1)
    elif x.ndim == 4:  # (B, T, N, C) - vector field
        return x[0, :, :, :]  # -> (T, N, C)
    else:
        raise RuntimeError(f"Unsupported rank {x.ndim} for field '{field}'. Expected 3 or 4.")

def _read_field_TNC(h5: h5py.File, group: str, field: str, T_hint: Optional[int]) -> np.ndarray:
    """
    Read a field from HDF5 and convert to (T, N, C) format.
    """
    if group not in h5:
        raise KeyError(f"Group '{group}' not found. Have: {list(h5.keys())}")
    if field not in h5[group]:
        have = list(h5[group].keys())
        raise KeyError(f"Field '{field}' not in group '{group}'. Have: {have}")
    x = np.array(h5[f"{group}/{field}"][...], copy=False)
    return _as_TN_C(x, field=field, T_hint=T_hint).astype(np.float32)

# ---------------- iterable dataset ----------------
class WellH5PairIterableMagLIF(IterableDataset):
    """
    Streams (x_in, y) pairs for MagLIF 1D dataset.
    - Handles 12 channels from fields: Rho, rho_Be, rho_DT, T_elec, T_ion, Rad_Temp, Vel, P_ion, P_elec, n_elec, bmag, jz
    - Uses true time values for time features
    - 1D radial coordinates only
    """

    def __init__(
        self,
        split_dir: str,
        stats: Dict,
        time_step: int,
        max_time_diff: Optional[int],
        field_names: List[str],
        cache_samples: int = 1,
        normalization_mode: str = "standard",
        sample_ratio: Optional[float] = None,
        files_list: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.max_samples = int(max_samples) if max_samples is not None else None
        if files_list is not None:
            self.files = list(files_list)
        else:
            self.files = sorted(glob.glob(os.path.join(split_dir, "*.hdf5")))
        if not self.files:
            raise FileNotFoundError(f"No .hdf5 files in {split_dir}")
        self.stats = stats
        self.time_step = int(time_step)
        self.max_time_diff = None if (max_time_diff is None) else int(max_time_diff)
        self.field_names = list(field_names)
        self.cache_samples = int(cache_samples)
        self.normalization_mode = normalization_mode
        self.signed_fields = ["Vel", "jz"]  # Fields that use asinh normalization
        self.sample_ratio = sample_ratio  # Subsampling ratio for all2all training (None = no subsampling)

        self._cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._order: List[str] = []

    def _touch_cache(self, fp: str, payload: Dict[str, torch.Tensor]):
        if self.cache_samples <= 0:
            return
        if fp in self._cache:
            try: 
                self._order.remove(fp)
            except ValueError: 
                pass
        self._cache[fp] = payload
        self._order.append(fp)
        while len(self._order) > self.cache_samples:
            old = self._order.pop(0)
            self._cache.pop(old, None)

    def _read_all_TNC(self, fp: str) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Returns (uTNC, tvals) as CPU float tensors / numpy array.
        uTNC: (T, N, Cu) stacked from all field_names
        tvals: (T,) true times from file
        """
        if fp in self._cache:
            pack = self._cache[fp]
            return pack["uTNC"], pack["tvals"]

        with h5py.File(fp, "r") as h5:
            t_vals = _time_from_dimensions(h5)
            T_hint = int(t_vals.shape[0])

            # Read all fields and stack
            chunks = []
            for fname in self.field_names:
                field_data = _read_field_TNC(h5, "fields", fname, T_hint)  # (T, N, C)
                chunks.append(field_data)
            
            uTNC = np.concatenate(chunks, axis=-1).astype(np.float32)  # (T, N, Cu)

        uTNC_t = torch.from_numpy(uTNC)
        self._touch_cache(fp, {
            "uTNC": uTNC_t, 
            "tvals": torch.from_numpy(t_vals.astype(np.float32))
        })
        return uTNC_t, t_vals

    def __len__(self):
        total = 0
        for fp in self.files:
            with h5py.File(fp, "r") as h5:
                T = int(_time_from_dimensions(h5).shape[0])
            ti, to = build_time_pairs(T, self.max_time_diff, self.time_step)
            total += int(ti.shape[0])
        return total

    def __iter__(self):
        # Extract stats
        u_mean = self.stats["u"]["mean"].reshape(1, -1)  # [1, Cu]
        u_std  = self.stats["u"]["std"].reshape(1, -1)
        # if not hasattr(self, "_norm_debug_printed"):
        #     self._norm_debug_printed = True
        #     print(f"[DEBUG DATASET] Normalization in dataset __iter__:")
        #     print(f"  u_mean shape: {u_mean.shape}, first 3 values: {u_mean.flatten()[:3].tolist()}")
        #     print(f"  u_std shape: {u_std.shape}, first 3 values: {u_std.flatten()[:3].tolist()}")
        #     print(f"  Stats come from self.stats (computed from training data)")

        st_mu = float(self.stats["start_time"]["mean"])
        st_sd = float(self.stats["start_time"]["std"])
        dt_mu = float(self.stats["time_diffs"]["mean"])
        dt_sd = float(self.stats["time_diffs"]["std"])

        yielded = 0
        for fp in self.files:
            uTNC, t_vals = self._read_all_TNC(fp)  # (T, N, Cu), (T,)
            T, N, Cu = uTNC.shape
            ti, to = build_time_pairs(T, self.max_time_diff, self.time_step)
            
            # Apply subsampling if enabled
            if self.sample_ratio is not None and self.sample_ratio < 1.0:
                n_pairs = len(ti)
                n_sample = max(1, int(n_pairs * self.sample_ratio))
                if n_sample < n_pairs:
                    indices = np.random.choice(n_pairs, size=n_sample, replace=False)
                    indices = np.sort(indices)  # Keep temporal order
                    ti = ti[indices]
                    to = to[indices]

            for i, o in zip(ti, to):
                u_in  = uTNC[i]  # [N, Cu]
                u_out = uTNC[o]

                # Normalize u according to normalization_mode
                if self.normalization_mode == "log":
                    # Apply channel-specific normalization (log or asinh)
                    norm_params_1 = self.stats.get("norm_params_1", None)
                    if norm_params_1 is None:
                        raise ValueError("norm_params_1 not found in stats for log normalization")
                    norm_params_1 = norm_params_1.to(u_in.device) if isinstance(norm_params_1, torch.Tensor) else torch.tensor(norm_params_1, device=u_in.device, dtype=u_in.dtype)
                    u_in_norm = torch.zeros_like(u_in)
                    y = torch.zeros_like(u_out)
                    
                    for ch_idx, field_name in enumerate(self.field_names):
                        if field_name in self.signed_fields:
                            # asinh normalization for signed fields
                            scale = float(norm_params_1[ch_idx]) if len(norm_params_1) > ch_idx else 1.0
                            scale = max(scale, 1e-10)
                            u_in_norm[:, ch_idx] = torch.asinh(u_in[:, ch_idx] / scale)
                            y[:, ch_idx] = torch.asinh(u_out[:, ch_idx] / scale)
                        else:
                            # log normalization for other fields
                            offset = float(norm_params_1[ch_idx]) if len(norm_params_1) > ch_idx else 1e-6
                            u_in_norm[:, ch_idx] = torch.log(u_in[:, ch_idx] + offset)
                            y[:, ch_idx] = torch.log(u_out[:, ch_idx] + offset)
                    
                    # Standardize: (normalized - mean) / std
                    u_in_norm = (u_in_norm - u_mean) / u_std
                    y = (y - u_mean) / u_std
                elif self.normalization_mode == "quantile":
                    # Apply quantile normalization per channel
                    norm_params_1 = self.stats.get("norm_params_1", None)
                    norm_params_2 = self.stats.get("norm_params_2", None)
                    if norm_params_1 is None or norm_params_2 is None:
                        raise ValueError("norm_params_1 and norm_params_2 not found in stats for quantile normalization")
                    
                    num_quantiles = int(norm_params_2[0])
                    num_channels = int(norm_params_2[1])
                    quantiles_array = norm_params_1.reshape(num_channels, num_quantiles).cpu().numpy()
                    
                    u_in_norm = torch.zeros_like(u_in)
                    y = torch.zeros_like(u_out)
                    
                    from scipy import interpolate
                    from scipy.stats import norm
                    
                    for ch_idx in range(u_in.shape[-1]):
                        # Normalize input
                        ch_values_in = u_in[:, ch_idx].cpu().numpy()
                        ch_quantiles = np.sort(quantiles_array[ch_idx])
                        quantile_ranks = np.linspace(0, 1, len(ch_quantiles))
                        
                        interp_func = interpolate.interp1d(
                            ch_quantiles, quantile_ranks,
                            kind='linear',
                            bounds_error=False,
                            fill_value=(0.0, 1.0)
                        )
                        ranks_in = np.clip(interp_func(ch_values_in), 0.0, 1.0)
                        ranks_in_clipped = np.clip(ranks_in, 0.001, 0.999)
                        u_in_norm[:, ch_idx] = torch.from_numpy(norm.ppf(ranks_in_clipped)).to(u_in.device)
                        
                        # Normalize output
                        ch_values_out = u_out[:, ch_idx].cpu().numpy()
                        ranks_out = np.clip(interp_func(ch_values_out), 0.0, 1.0)
                        ranks_out_clipped = np.clip(ranks_out, 0.001, 0.999)
                        y[:, ch_idx] = torch.from_numpy(norm.ppf(ranks_out_clipped)).to(u_out.device)
                    
                    # Standardize: (normalized - mean) / std
                    u_in_norm = (u_in_norm - u_mean) / u_std
                    y = (y - u_mean) / u_std
                else:
                    # Standard normalization
                    u_in_norm = (u_in - u_mean) / u_std
                    y = (u_out - u_mean) / u_std
                # if not hasattr(self, "_first_sample_printed"):
                #     self._first_sample_printed = True
                #     print(f"[DEBUG DATASET] First sample normalization check:")
                #     print(f"  u_out (raw) first 3 values: {u_out[0, :3].tolist()}")
                #     print(f"  u_mean first 3: {u_mean[0, :3].tolist()}")
                #     print(f"  u_std first 3: {u_std[0, :3].tolist()}")
                #     print(f"  y (normalized) first 3 values: {y[0, :3].tolist()}")
                #     print(f"  y stats: min={y.min():.6f}, max={y.max():.6f}, mean={y.mean():.6f}, std={y.std():.6f}")

                # Compute time features from true time values
                start_t = float(t_vals[i])
                diff_t  = float(t_vals[o] - t_vals[i])
                
                start_norm = (start_t - st_mu) / (st_sd if st_sd > 0 else 1.0)
                diff_norm  = (diff_t  - dt_mu) / (dt_sd if dt_sd > 0 else 1.0)
                
                st_feat = torch.full((N, 1), start_norm, dtype=torch.float32)
                td_feat = torch.full((N, 1), diff_norm,  dtype=torch.float32)

                # Build input: [u_norm, start_time, time_diff]
                x_in = torch.cat([u_in_norm, st_feat, td_feat], dim=-1)  # [N, Cu+2]

                traj_end_time = float(t_vals[T - 1])
                lag_index = int(o - i)
                yield (x_in, y, traj_end_time, lag_index)
                yielded += 1
                if self.max_samples is not None and yielded >= self.max_samples:
                    return



