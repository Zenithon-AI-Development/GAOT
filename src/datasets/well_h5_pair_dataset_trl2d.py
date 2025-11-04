# -*- coding: utf-8 -*-
# TRL2D-specific: src/datasets/well_h5_pair_dataset_trl2d.py
import os, glob, h5py, numpy as np, torch
from torch.utils.data import IterableDataset
from typing import Dict, List, Tuple, Optional

# Reuse helpers that know The Well's 'dimensions' layout
from .well_h5_pair_dataset_hs import (
    _mesh_from_dimensions, _time_from_dimensions, _list_field_datasets
)

# ---------------- time pairing ----------------
def build_time_pairs(T: int, max_time_diff: Optional[int], time_step: int) -> Tuple[np.ndarray, np.ndarray]:
    if T <= 1:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    s = max(1, int(time_step))
    kmax = T - 1 if (max_time_diff is None) else max(1, min(int(max_time_diff), T - 1))
    ti, to = [], []
    # use all start indices, stride 1; lag stride = time_step
    for lag in range(s, kmax + 1, s):
        for i in range(0, T - lag, 1):
            ti.append(i); to.append(i + lag)
    return np.asarray(ti, np.int64), np.asarray(to, np.int64)

# ---------------- shape helpers ----------------
def _as_THW_C(arr: np.ndarray, field: str, T_hint: Optional[int], vector_fields: set) -> np.ndarray:
    """
    Return array as (T,H,W,C). Accepts shapes:
      (T,H,W)                   -> (T,H,W,1)
      (T,H,W,C)                 -> (T,H,W,C)
      (B,T,H,W)                 -> (T,H,W,1)   (take B=0)
      (B,T,H,W,C)               -> (T,H,W,C)   (take B=0)
    Heuristic to disambiguate (T,H,W,C) vs (B,T,H,W):
      - If field is in vector_fields and arr.ndim==4 and arr.shape[-1] <= 16 -> treat last dim as C.
      - Else if arr.ndim==4 and T_hint is not None and arr.shape[0] != T_hint and arr.shape[1] == T_hint:
            treat as (B,T,H,W) -> take B=0, add channel=1.
      - Else if arr.ndim==4 and arr.shape[-1] in (1,2,3,4) and (T_hint is None or arr.shape[0] == T_hint):
            treat as (T,H,W,C).
      - Else if arr.ndim==4:
            default to (B,T,H,W) -> take B=0, channel=1.
    """
    x = np.asarray(arr)
    if x.ndim == 3:
        T, H, W = x.shape
        return x[..., None]  # (T,H,W,1)

    if x.ndim == 5:
        # (B,T,H,W,C) is what TRL2D velocity often uses in some exports
        return x[0]  # (T,H,W,C)

    if x.ndim == 4:
        if field in vector_fields and x.shape[-1] <= 16:
            # (T,H,W,C)
            return x
        if (T_hint is not None) and (x.shape[0] != T_hint) and (x.shape[1] == T_hint):
            # likely (B,T,H,W) -> use B=0
            return x[0][..., None]  # (T,H,W,1)
        # try treating last dim as channels if small
        if x.shape[-1] in (1, 2, 3, 4) and (T_hint is None or x.shape[0] == T_hint):
            return x  # (T,H,W,C)
        # fallback: consider it's (B,T,H,W)
        return x[0][..., None]

    raise RuntimeError(f"Unsupported rank {x.ndim} for field '{field}'")

def _read_field_TNC(h5: h5py.File, group: str, field: str, T_hint: Optional[int], vector_fields: set) -> np.ndarray:
    if group not in h5:
        raise KeyError(f"Group '{group}' not found. Have: {list(h5.keys())}")
    if field not in h5[group]:
        have = list(h5[group].keys())
        raise KeyError(f"Field '{field}' not in {group}. Have: {have}")
    d = h5[f"{group}/{field}"]
    x = np.array(d[...], copy=False).astype(np.float32)
    THWC = _as_THW_C(x, field=field, T_hint=T_hint, vector_fields=vector_fields)  # (T,H,W,C)
    T, H, W, C = THWC.shape
    return THWC.reshape(T, H * W, C)  # (T,N,C)

# ---------------- iterable ----------------
class WellH5PairIterableTRL2D(IterableDataset):
    """
    Streams (x_in, y) pairs for TRL2D from a split directory of .hdf5 files.
    Handles variable T across files and vector fields (velocity) from t1_fields.
    Normalizes 'u' with stats from stats.yaml and computes time features per pair.
    """

    def __init__(
        self,
        split_dir: str,
        stats: Dict,
        time_step: int,
        max_time_diff: Optional[int],
        select_fields: List[str],
        field_groups: Optional[Dict[str, str]] = None,
        vector_fields: Optional[List[str]] = None,
        cache_samples: int = 1,
    ):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(split_dir, "*.hdf5")))
        if not self.files:
            raise FileNotFoundError(f"No .hdf5 files in {split_dir}")
        self.stats = stats
        self.time_step = int(time_step)
        self.max_time_diff = None if (max_time_diff is None) else int(max_time_diff)
        self.fields = list(select_fields)
        # default groups for TRL2D
        self.field_groups = dict(field_groups or {
            "density": "t0_fields",
            "pressure": "t0_fields",
            "velocity": "t1_fields",   # 2 channels
        })
        self.vector_fields = set(vector_fields or ["velocity"])
        self.cache_samples = int(cache_samples)

        # tiny per-file cache
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

    def _read_all_fields_TNC(self, fp: str) -> torch.Tensor:
        if fp in self._cache:
            return self._cache[fp]["TNC"]

        with h5py.File(fp, "r") as h5:
            # use declared time grid as hint for dimensional disambiguation
            t_vals = _time_from_dimensions(h5)
            T_hint = int(t_vals.shape[0])

            # build channel stack in the requested order
            chunks = []
            for fname in self.fields:
                grp = self.field_groups.get(fname, "t0_fields")
                TNC = _read_field_TNC(h5, grp, fname, T_hint=T_hint, vector_fields=self.vector_fields)
                chunks.append(TNC)  # (T,N,Cf)
            TNC_all = np.concatenate(chunks, axis=-1).astype(np.float32)  # (T,N,C_total)

        TNC_t = torch.from_numpy(TNC_all)  # CPU tensor
        self._touch_cache(fp, {"TNC": TNC_t})
        return TNC_t

    def __len__(self):
        # exact sum over files (each with its own T)
        total = 0
        for fp in self.files:
            with h5py.File(fp, "r") as h5:
                T = int(_time_from_dimensions(h5).shape[0])
            ti, to = build_time_pairs(T, self.max_time_diff, self.time_step)
            total += int(ti.shape[0])
        return total

    def __iter__(self):
        # u stats
        u_mean = self.stats["u"]["mean"].reshape(1, -1)  # [1,C]
        u_std  = self.stats["u"]["std"].reshape(1, -1)

        # time-index stats (index space, not physical time)
        st_mu = float(self.stats["start_time"]["mean"])
        st_sd = float(self.stats["start_time"]["std"])
        dt_mu = float(self.stats["time_diffs"]["mean"])
        dt_sd = float(self.stats["time_diffs"]["std"])

        for fp in self.files:
            uTNC = self._read_all_fields_TNC(fp)  # (T,N,C)
            T, N, C = uTNC.shape
            ti, to = build_time_pairs(T, self.max_time_diff, self.time_step)

            for i, o in zip(ti, to):
                u_in  = uTNC[i]                 # [N,C]
                u_out = uTNC[o]                 # [N,C]

                # normalize u
                u_in_norm  = (u_in - u_mean) / u_std
                y          = (u_out - u_mean) / u_std

                # per-pair time features (index-based)
                start_norm = (float(i) - st_mu) / st_sd
                diff_norm  = (float(o - i) - dt_mu) / dt_sd
                st = torch.full((N, 1), start_norm, dtype=torch.float32)
                td = torch.full((N, 1), diff_norm,  dtype=torch.float32)

                x_in = torch.cat([u_in_norm, st, td], dim=-1)  # [N, C+2]
                yield (x_in, y)
