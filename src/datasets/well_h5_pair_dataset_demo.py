# -*- coding: utf-8 -*-
# src/datasets/well_h5_pair_dataset_demo.py
import os, glob, h5py, numpy as np, torch
from torch.utils.data import IterableDataset
from typing import Dict, List, Tuple, Optional

# ---------------- basic helpers (self-contained) ----------------
def _time_from_dimensions(h5: h5py.File) -> np.ndarray:
    if "dimensions" not in h5 or "time" not in h5["dimensions"]:
        raise KeyError("dimensions/time not found in HDF5")
    return np.asarray(h5["dimensions/time"][...], dtype=np.float64)

def _mesh_from_dimensions(h5: h5py.File) -> np.ndarray:
    dimg = h5["dimensions"]
    r = np.asarray(dimg["r_coords"][...], dtype=np.float64)
    z = np.asarray(dimg["z_coords"][...], dtype=np.float64)
    Z, R = np.meshgrid(z, r, indexing="ij")  # (H,W)
    coords = np.stack([R, Z], axis=-1).reshape(-1, 2)  # [N,2]
    return coords

def _list_field_datasets(h5: h5py.File, group: str) -> List[str]:
    if group not in h5:
        return []
    return [k for k, v in h5[group].items() if isinstance(v, h5py.Dataset)]

def build_time_pairs(T: int, max_time_diff: Optional[int], time_step: int) -> Tuple[np.ndarray, np.ndarray]:
    if T <= 1:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    s = max(1, int(time_step))
    ti, to = [], []
    # NOTE: we still step in index space to choose pair indices,
    # but the *features* below use true time values t[i], t[o].
    for lag in range(s, max_time_diff + 1, s):
        for i in range(0, T - lag, 1):
            ti.append(i); to.append(i + lag)
    return np.asarray(ti, np.int64), np.asarray(to, np.int64)

def _as_THW_C(arr: np.ndarray, field: str, T_hint: Optional[int]) -> np.ndarray:
    """
    Normalize to (T,H,W,C).
    Accepts (T,H,W), (T,H,W,C), (B,T,H,W), (B,T,H,W,C) (use B=0).
    """
    x = np.asarray(arr)
    if x.ndim == 3:          # (T,H,W)
        return x[..., None]
    if x.ndim == 5:          # (B,T,H,W,C)
        return x[0]
    if x.ndim == 4:
        if x.shape[-1] in (1,2,3,4,8,16) and (T_hint is None or x.shape[0] == T_hint):
            return x
        if (T_hint is not None) and (x.shape[1] == T_hint) and (x.shape[0] != T_hint):
            return x[0][..., None]
        return x[0][..., None]
    raise RuntimeError(f"Unsupported rank {x.ndim} for field '{field}'")

def _read_field_TNC(h5: h5py.File, group: str, field: str, T_hint: Optional[int]) -> np.ndarray:
    if group not in h5:
        raise KeyError(f"Group '{group}' not found. Have: {list(h5.keys())}")
    if field not in h5[group]:
        have = list(h5[group].keys())
        raise KeyError(f"Field '{field}' not in group '{group}'. Have: {have}")
    x = np.array(h5[f"{group}/{field}"][...], copy=False)
    THWC = _as_THW_C(x, field=field, T_hint=T_hint).astype(np.float32)
    T, H, W, C = THWC.shape
    return THWC.reshape(T, H * W, C)  # (T,N,C)

# ---------------- iterable ----------------
class WellH5PairIterableDEMO(IterableDataset):
    """
    Streams (x_in, y) pairs for your DEMO dataset from a split directory of .hdf5 files.
    - u = all scalar/vector fields under t0_fields + t1_fields (order: t0 sorted, then t1 sorted).
    - c = forcing_fields/current_drive (1 channel), concatenated into x_in (conditioning).
    - x_in = [u_norm, c_norm, start_time_norm, time_diff_norm]; y = u_out_norm.
    - Uses TRUE TIME (t[i], t[o]-t[i]) for time features & stats.
    - Handles variable T across files.
    """

    def __init__(
        self,
        split_dir: str,
        stats: Dict,
        time_step: int,
        max_time_diff: Optional[int],
        u_fields_t0: List[str],
        u_fields_t1: List[str],
        cache_samples: int = 1,
    ):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(split_dir, "*.hdf5")))
        if not self.files:
            raise FileNotFoundError(f"No .hdf5 files in {split_dir}")
        self.stats = stats
        self.time_step = int(time_step)
        self.max_time_diff = None if (max_time_diff is None) else int(max_time_diff)
        self.u_fields_t0 = list(u_fields_t0)
        self.u_fields_t1 = list(u_fields_t1)
        self.cache_samples = int(cache_samples)

        self._cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._order: List[str] = []

    def _touch_cache(self, fp: str, payload: Dict[str, torch.Tensor]):
        if self.cache_samples <= 0:
            return
        if fp in self._cache:
            try: self._order.remove(fp)
            except ValueError: pass
        self._cache[fp] = payload
        self._order.append(fp)
        while len(self._order) > self.cache_samples:
            old = self._order.pop(0)
            self._cache.pop(old, None)

    def _read_all_TNC(self, fp: str) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Returns (uTNC, cTNC, tvals) as CPU float tensors / numpy array.
        uTNC: (T,N,Cu) stacked as [t0_fields..., t1_fields...]
        cTNC: (T,N,1) from forcing_fields/current_drive
        tvals: (T,) true times from file
        """
        if fp in self._cache:
            pack = self._cache[fp]
            return pack["uTNC"], pack["cTNC"], pack["tvals"]

        with h5py.File(fp, "r") as h5:
            t_vals = _time_from_dimensions(h5)                 # TRUE times
            T_hint = int(t_vals.shape[0])

            chunks = []
            for fname in sorted(self.u_fields_t0):
                chunks.append(_read_field_TNC(h5, "t0_fields", fname, T_hint))
            for fname in sorted(self.u_fields_t1):
                chunks.append(_read_field_TNC(h5, "t1_fields", fname, T_hint))
            uTNC = np.concatenate(chunks, axis=-1).astype(np.float32)  # (T,N,Cu)

            # conditioning (current_drive)
            cTNC = _read_field_TNC(h5, "forcing_fields", "current_drive", T_hint)  # (T,N,1)

        uTNC_t = torch.from_numpy(uTNC)
        cTNC_t = torch.from_numpy(cTNC)
        self._touch_cache(fp, {"uTNC": uTNC_t, "cTNC": cTNC_t, "tvals": torch.from_numpy(t_vals.astype(np.float32))})
        return uTNC_t, cTNC_t, t_vals

    def __len__(self):
        total = 0
        for fp in self.files:
            with h5py.File(fp, "r") as h5:
                T = int(_time_from_dimensions(h5).shape[0])
            ti, to = build_time_pairs(T, self.max_time_diff, self.time_step)
            total += int(ti.shape[0])
        return total

    def __iter__(self):
        # stats
        u_mean = self.stats["u"]["mean"].reshape(1, -1)  # [1,Cu]
        u_std  = self.stats["u"]["std"].reshape(1, -1)
        c_mean = self.stats["c"]["mean"].reshape(1, -1)  # [1,1]
        c_std  = self.stats["c"]["std"].reshape(1, -1)

        st_mu = float(self.stats["start_time"]["mean"])
        st_sd = float(self.stats["start_time"]["std"])
        dt_mu = float(self.stats["time_diffs"]["mean"])
        dt_sd = float(self.stats["time_diffs"]["std"])

        for fp in self.files:
            uTNC, cTNC, t_vals = self._read_all_TNC(fp)  # (T,N,Cu), (T,N,1), (T,)
            T, N, Cu = uTNC.shape
            ti, to = build_time_pairs(T, self.max_time_diff, self.time_step)

            for i, o in zip(ti, to):
                u_in  = uTNC[i]                      # [N,Cu]
                u_out = uTNC[o]
                c_in  = cTNC[i]                      # [N,1]

                u_in_norm = (u_in - u_mean) / u_std
                y         = (u_out - u_mean) / u_std
                c_norm    = (c_in - c_mean) / c_std

                # TRUE time features
                start_t = float(t_vals[i])
                diff_t  = float(t_vals[o] - t_vals[i])
                # if not hasattr(self, "_dbg"):
                #     self._dbg = True
                #     print(f"[SANITY] t[i]={start_t*1e9:.6f}, t[o]={start_t*1e9+diff_t*1e9:.6f}, Δt={diff_t*1e9:.6f}  | "
                #         f"z-st={((start_t - st_mu)/(st_sd if st_sd>0 else 1.0)):.4f}  "
                #         f"z-Δt={((diff_t  - dt_mu)/(dt_sd if dt_sd>0 else 1.0)):.4f}")

                start_norm = (start_t - st_mu) / (st_sd if st_sd > 0 else 1.0)
                diff_norm  = (diff_t  - dt_mu) / (dt_sd if dt_sd > 0 else 1.0)
                st = torch.full((N, 1), start_norm, dtype=torch.float32)
                td = torch.full((N, 1), diff_norm,  dtype=torch.float32)

                x_in = torch.cat([u_in_norm, c_norm, st, td], dim=-1)  # [N, Cu+1+2]
                yield (x_in, y)
