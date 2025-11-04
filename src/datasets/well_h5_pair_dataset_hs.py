# -*- coding: utf-8 -*-
# src/datasets/well_h5_pair_dataset.py
import os, glob, h5py, numpy as np, torch
from torch.utils.data import IterableDataset
from collections import deque
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

def _list_field_datasets(h5: h5py.File, group_name: str) -> List[str]:
    if group_name not in h5:
        return []
    g = h5[group_name]
    names = list(g.attrs.get("field_names", []))
    if names:
        return [n for n in names if n in g]
    return sorted([k for k, v in g.items() if isinstance(v, h5py.Dataset)])

def _mesh_from_dimensions(h5: h5py.File) -> np.ndarray:
    xs = h5["dimensions/x"][...]   # (W,)
    ys = h5["dimensions/y"][...]   # (H,)
    xv, yv = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([xv, yv], axis=-1).reshape(-1, 2).astype(np.float32)

def _time_from_dimensions(h5: h5py.File) -> np.ndarray:
    if "dimensions/time" in h5:
        return h5["dimensions/time"][...].astype(np.float32)
    # Fallback: infer from a field
    for grp in ("t0_fields", "t1_fields", "t2_fields"):
        names = _list_field_datasets(h5, grp)
        for nm in names:
            d = h5[f"{grp}/{nm}"]
            shp = d.shape
            if len(shp) >= 4:  # (B,T,*,*) or (T,*,*,C)
                return np.arange(shp[1] if shp[0] > 4 else shp[0], dtype=np.float32)
            if len(shp) == 3:  # (T,*,*) or (B,*,*)
                return np.arange(shp[0], dtype=np.float32)
    raise RuntimeError("Could not infer time dimension.")

class H5TrajectoryIndex:
    def __init__(self, files: List[str], fields_group="t0_fields"):
        self.files = files
        self.lookup = []
        for fi, fp in enumerate(files):
            with h5py.File(fp, "r") as h5:
                names = _list_field_datasets(h5, fields_group)
                if not names:
                    continue
                d = h5[f"{fields_group}/{names[0]}"]
                shp = d.shape
                B = shp[0] if (len(shp) >= 4 and shp[0] > 1) else 1
                for bi in range(B):
                    self.lookup.append((fi, bi))
    def __len__(self): return len(self.lookup)
    def __getitem__(self, i): return self.lookup[i]

class WellH5PairIterable(IterableDataset):
    """
    Streams (t_in, t_out) pairs from The Well HDF5 files.
    Only the required time slices are read for each pair.
    Assumptions:
      * Fixed grid (fx).
      * Stepper mode 'output'.
      * We exclude 'mask' from inputs (use 2 channels: pressure_im, pressure_re).
    """
    def __init__(
        self,
        split_dir: str,
        stats: Dict,
        t_pairs: Tuple[np.ndarray, np.ndarray],
        active_groups: Tuple[str, ...] = ("t0_fields",),
        select_fields: Optional[List[str]] = None,     # e.g. ["pressure_im","pressure_re"]
        cache_samples: int = 0,
    ):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(split_dir, "*.hdf5")))
        if not self.files:
            raise FileNotFoundError(f"No .hdf5 files found in {split_dir}")
        self.index = H5TrajectoryIndex(self.files, fields_group=active_groups[0])
        self.stats = stats
        self.t_in, self.t_out = t_pairs
        self.active_groups = active_groups
        self.select_fields = select_fields or ["pressure_im", "pressure_re"]
        # ensure mask is not included
        self.select_fields = [f for f in self.select_fields if f != "mask"]
        self.cache_samples = max(int(cache_samples), 0)
        self._cache, self._lru = {}, deque()

        self.k_max = int(self.t_out.max() - self.t_in.min()) if len(self.t_in) else 1
        self.k_step = int(self.t_in[1] - self.t_in[0]) if len(self.t_in) > 1 else 1
        # make them visible to __len__()
        self.stats["max_time_diff"] = torch.tensor(self.k_max)
        self.stats["time_step"]     = torch.tensor(self.k_step)

    @lru_cache(maxsize=None)
    def _traj_T(self, file_idx: int, traj_idx: int) -> int:
        """Infer number of time steps T for a given trajectory by inspecting shapes only."""
        fp = self.files[file_idx]
        with h5py.File(fp, "r") as f:
            # Prefer explicit time dimension if available
            if "dimensions/time" in f:
                return int(f["dimensions/time"].shape[0])

            # Otherwise, infer from a time-varying field under any active group
            for grp in self.active_groups:
                names = _list_field_datasets(f, grp)
                for nm in names:
                    d = f[f"{grp}/{nm}"]
                    shp = d.shape  # e.g., (B,T,H,W[,C]) or (T,H,W[,C]) or (B,H,W) for static
                    if len(shp) >= 4:
                        # (B,T,H,W[,C]) or (T,H,W,C)
                        if shp[0] > 4 and shp[1] > 1:
                            return int(shp[1])            # (B,T,*,*)
                        if shp[0] > 1 and shp[1] > 4:
                            return int(shp[0])            # (T,*,*,C)
                    if len(shp) == 3 and shp[0] > 4 and shp[1] > 4:
                        # (B,H,W) => static; fallback to dataset_config.time_len if present
                        return int(getattr(self, "default_T", 1))
                    if len(shp) == 3 and shp[0] > 1 and shp[1] > 1:
                        # (T,H,W)
                        return int(shp[0])
            # Fallback if everything was static
            return int(getattr(self, "default_T", 1))

    @staticmethod
    def _num_pairs_for_T(T: int, k_max: int, k_step: int) -> int:
        """How many (t_in, t_out) pairs would build_time_pairs() create for length T."""
        if T <= 1:
            return 0
        num_timesteps = min(T - 1, k_max)
        cnt = 0
        for lag in range(k_step, num_timesteps + 1, k_step):
            for i in range(0, num_timesteps - lag + 1, k_step):
                cnt += 1
        return cnt

    def __len__(self) -> int:
        """
        Total number of pairs in this split = sum over trajectories of P_i.
        Safe for DataLoader.__len__ and progress bars / epoch averaging.
        """
        k_max = int(self.stats.get("max_time_diff",  getattr(self, "k_max", 50)))
        k_step= int(self.stats.get("time_step",      getattr(self, "k_step", 2)))

        total = 0
        for gidx in range(len(self.index)):
            file_i, traj_i = self.index[gidx]
            T_i = self._traj_T(file_i, traj_i)
            total += self._num_pairs_for_T(T_i, k_max, k_step)
        return total

    def _touch_cache(self, key, val):
        if self.cache_samples == 0: return
        if key in self._cache:
            try: self._lru.remove(key)
            except ValueError: pass
        self._cache[key] = val
        self._lru.append(key)
        while len(self._lru) > self.cache_samples:
            old = self._lru.popleft()
            self._cache.pop(old, None)

    def _expand_to_T_HWC(self, dset, traj_i: int, T: int):
        shp = dset.shape
        def np_at(idx): return np.array(dset[idx], copy=False)
        if len(shp) == 2:  # (H,W) static
            H, W = shp
            return np.broadcast_to(np_at(...)[None, ...], (T, H, W))[..., None]
        if len(shp) == 3:
            if shp[0] == T:  # (T,H,W)
                return np_at((slice(None), slice(None), slice(None)))[..., None]
            else:            # (B,H,W)
                H, W = shp[1:]
                x0 = np_at((traj_i, slice(None), slice(None)))
                return np.broadcast_to(x0[None, ...], (T, H, W))[..., None]
        if len(shp) == 4:
            if shp[0] == T:         # (T,H,W,C)
                return np_at((slice(None), slice(None), slice(None), slice(None)))
            else:                   # (B,T,H,W)
                x = np_at((traj_i, slice(None), slice(None), slice(None)))
                return x[..., None] # (T,H,W,1)
        if len(shp) == 5:           # (B,T,H,W,C)
            return np_at((traj_i, slice(None), slice(None), slice(None), slice(None)))
        raise RuntimeError(f"Unsupported shape {shp} in HDF5 dataset")

    def _get_field_THWC(self, f: h5py.File, name_candidates: List[str], traj_i: int, T: int):
        for grp in self.active_groups:
            for nm in self.select_fields:
                # prioritize exact matches like "t0_fields/pressure_im"
                for cand in (f"{grp}/{nm}", nm, f"fields/{nm}"):
                    if cand in f:
                        return self._expand_to_T_HWC(f[cand], traj_i, T), nm
        # fallback over provided name candidates
        for cand in name_candidates:
            if cand in f:
                return self._expand_to_T_HWC(f[cand], traj_i, T), os.path.basename(cand)
        return None, None

    def _read_one_traj_TNC(self, file_path: str, traj_i: int) -> torch.Tensor:
        """
        Return pressures only as (T, N, 2) in order [pressure_im, pressure_re].
        """
        with h5py.File(file_path, "r") as f:
            # try to infer T,H,W from a pressure dataset
            # attempt pressure_im then pressure_re
            pim_arr, nm_im = self._get_field_THWC(f, ["t0_fields/pressure_im"], traj_i, T=50)  # 50 is typical but not enforced
            pre_arr, nm_re = self._get_field_THWC(f, ["t0_fields/pressure_re"], traj_i, T=50)

            # If either came back None, try stacked dataset with last channel≥2
            if pim_arr is None or pre_arr is None:
                # scan any 5D dataset (B,T,H,W,C>=2)
                stacked = None
                for k, ds in f.items():
                    if hasattr(ds, "shape") and len(ds.shape) == 5 and ds.shape[-1] >= 2:
                        stacked = self._expand_to_T_HWC(ds, traj_i, ds.shape[1])
                        break
                if stacked is None:
                    raise RuntimeError("Could not locate pressure fields in file.")
                pim_arr, pre_arr = stacked[..., :1], stacked[..., 1:2]

            # ensure same T,H,W and produce (T,H,W,2)
            T = pim_arr.shape[0]
            THW2 = np.concatenate([pim_arr, pre_arr], axis=-1).astype(np.float32)  # (T,H,W,2)
            THW2 = np.ascontiguousarray(THW2)
            T, H, W, _ = THW2.shape
            return torch.from_numpy(THW2.reshape(T, H * W, 2))  # (T,N,2)

    def __iter__(self):
        # normalization (reshape to [1,C] for broadcasting over [N,C])
        u_mean = self.stats["u"]["mean"].reshape(1, -1)  # [1,2]
        u_std  = self.stats["u"]["std"].reshape(1, -1)   # [1,2]
        start_vals = self.stats["start_time"]["norm_values"].cpu().numpy()
        diff_vals  = self.stats["time_diffs"]["norm_values"].cpu().numpy()

        P = len(self.t_in)
        for gidx in range(len(self.index)):
            file_i, traj_i = self.index[gidx]
            fp = self.files[file_i]
            key = (file_i, traj_i)

            if key in self._cache:
                uTNC = self._cache[key]
            else:
                uTNC = self._read_one_traj_TNC(fp, traj_i)  # [T,N,2]
                if self.cache_samples > 0:
                    self._touch_cache(key, uTNC)

            T, N, C = uTNC.shape

            # # Build per-trajectory pairs and time features on the fly
            # t_in_i, t_out_i = build_time_pairs(T, self.k_max, self.k_step)
            # # normalize using global stats (consistent across trajectories)
            # st = torch.from_numpy(t_in_i.astype(np.float32))
            # td = torch.from_numpy((t_out_i - t_in_i).astype(np.float32))
            # st = ((st - self.stats["start_time"]["mean"]) / self.stats["start_time"]["std"]).view(-1, 1).expand(-1, N).unsqueeze(-1)
            # td = ((td - self.stats["time_diffs"]["mean"]) / self.stats["time_diffs"]["std"]).view(-1, 1).expand(-1, N).unsqueeze(-1)

            for p in range(P):
                ti = int(self.t_in[p]); to = int(self.t_out[p])
                u_in  = uTNC[ti]                    # [N,2]
                u_out = uTNC[to]                    # [N,2]
                u_in_norm = (u_in - u_mean) / u_std
                st = torch.full((N, 1), float(start_vals[p]), dtype=torch.float32)
                td = torch.full((N, 1), float(diff_vals[p]),  dtype=torch.float32)
                x_in = torch.cat([u_in_norm, st, td], dim=-1)  # [N, 4]
                y    = (u_out - u_mean) / u_std                # [N, 2]
                yield (x_in, y)



def build_time_pairs(T: int, max_time_diff: int, time_step: int) -> Tuple[np.ndarray, np.ndarray]:
    num_timesteps = min(T - 1, max_time_diff)
    t_in, t_out = [], []
    for lag in range(time_step, max_time_diff + 1, time_step):
        for i in range(0, T - lag, time_step):
            t_in.append(i); t_out.append(i + lag)
    return np.asarray(t_in, dtype=np.int32), np.asarray(t_out, dtype=np.int32)
