# -*- coding: utf-8 -*-
# src/datasets/well_h5_pair_dataset_multires.py
import os, glob, h5py, numpy as np, torch
from torch.utils.data import IterableDataset
from typing import Dict, List, Tuple, Optional

# Reuse Well helpers for dims
from .well_h5_pair_dataset_demo import (
    _mesh_from_dimensions, _time_from_dimensions
)

# ---------------- time pairing (indices for pair selection; features use true time) ----------------
def build_time_pairs(T: int, max_time_diff: Optional[int], time_step: int) -> Tuple[np.ndarray, np.ndarray]:
    if T <= 1:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    s = max(1, int(time_step))
    ti, to = [], []
    for lag in range(s, max_time_diff + 1, s):
        for i in range(0, T - lag, s):
            ti.append(i); to.append(i + lag)
    return np.asarray(ti, np.int32), np.asarray(to, np.int32)

# ---------------- shape helpers ----------------
def _as_THW_C(arr: np.ndarray, field: str, T_hint: Optional[int], vector_fields: set) -> np.ndarray:
    """
    Normalize to (T,H,W,C).
    """
    x = np.asarray(arr)
    if x.ndim == 3:
        return x[..., None]
    if x.ndim == 5:
        return x[0]
    if x.ndim == 4:
        if field in vector_fields and x.shape[-1] <= 16:
            return x
        if (T_hint is not None) and (x.shape[0] != T_hint) and (x.shape[1] == T_hint):
            return x[0][..., None]
        if x.shape[-1] in (1, 2, 3, 4) and (T_hint is None or x.shape[0] == T_hint):
            return x
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
    THWC = _as_THW_C(x, field=field, T_hint=T_hint, vector_fields=vector_fields)
    T, H, W, C = THWC.shape
    return THWC.reshape(T, H * W, C)  # (T,N,C)

# ---------------- iterable ----------------
class MultiResWellH5PairIterable(IterableDataset):
    """
    Streams (x_in, y, coord_bucket) for *multiple resolution buckets*.

    Guarantees: items are yielded in contiguous blocks of size 'emit_granularity'
    *per bucket*, so a PyTorch DataLoader with the same batch_size will form
    batches that never mix resolutions.

    x_in = [ u_in_norm | c_in_norm | start_time_norm | time_diff_norm ]
    y    =              u_out_norm
    coord_bucket = [N,2] (shared coordinates for the bucket)
    """

    def __init__(
        self,
        files_by_bucket: List[List[str]],
        coords_by_bucket: List[torch.Tensor],
        stats: Dict,
        time_step: int,
        max_time_diff: Optional[int],
        emit_granularity: int,
        u_fields: List[Tuple[str, str]],
        c_fields: List[Tuple[str, str]],
        vector_fields: Optional[List[str]] = None,
        drop_last_on_bucket: bool = True,
    ):
        super().__init__()
        assert len(files_by_bucket) == len(coords_by_bucket), "buckets and coords mismatch"
        self.files_by_bucket = files_by_bucket
        self.coords_by_bucket = coords_by_bucket
        self.stats = stats
        self.time_step = int(time_step)
        self.max_time_diff = None if (max_time_diff is None) else int(max_time_diff)
        self.emit_granularity = max(1, int(emit_granularity))
        self.u_fields = list(u_fields)
        self.c_fields = list(c_fields)
        self.vector_fields = set(vector_fields or ["velocity", "b_field"])
        self.drop_last_on_bucket = bool(drop_last_on_bucket)

    def __len__(self):
        total = 0
        for files in self.files_by_bucket:
            for fp in files:
                with h5py.File(fp, "r") as h5:
                    T = int(_time_from_dimensions(h5).shape[0])
                ti, to = build_time_pairs(T, self.max_time_diff, self.time_step)
                total += int(ti.shape[0])
        if self.drop_last_on_bucket and self.emit_granularity > 1:
            blocks = total // self.emit_granularity
            return blocks * self.emit_granularity
        return total

    def _read_u_c_TNC(self, fp: str) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        with h5py.File(fp, "r") as h5:
            t_vals = _time_from_dimensions(h5)
            T_hint = int(t_vals.shape[0])

            u_chunks = []
            for grp, fname in self.u_fields:
                u_chunks.append(_read_field_TNC(h5, grp, fname, T_hint, self.vector_fields))
            u_TNC = np.concatenate(u_chunks, axis=-1).astype(np.float32)

            if len(self.c_fields) > 0:
                c_chunks = []
                for grp, fname in self.c_fields:
                    c_chunks.append(_read_field_TNC(h5, grp, fname, T_hint, self.vector_fields))
                c_TNC = np.concatenate(c_chunks, axis=-1).astype(np.float32)
            else:
                c_TNC = np.zeros((u_TNC.shape[0], u_TNC.shape[1], 0), dtype=np.float32)

        return torch.from_numpy(u_TNC), torch.from_numpy(c_TNC), t_vals  # CPU

    def _iter_bucket_emitting_blocks(self, files: List[str], coord_bucket: torch.Tensor):
        # stats
        u_mean = self.stats["u"]["mean"].reshape(1, -1)  # [1,Cu]
        u_std  = self.stats["u"]["std"].reshape(1, -1)
        Cc = int(self.stats.get("c", {}).get("mean", torch.zeros(1)).numel())
        if Cc > 0:
            c_mean = self.stats["c"]["mean"].reshape(1, -1)
            c_std  = self.stats["c"]["std"].reshape(1, -1)
        else:
            c_mean = c_std = None

        st_mu = float(self.stats["start_time"]["mean"])
        st_sd = float(self.stats["start_time"]["std"])
        dt_mu = float(self.stats["time_diffs"]["mean"])
        dt_sd = float(self.stats["time_diffs"]["std"])

        buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for fp in files:
            uTNC, cTNC, t_vals = self._read_u_c_TNC(fp)  # (T,N,Cu), (T,N,Cc), (T,)
            T, N, Cu = uTNC.shape
            ti, to = build_time_pairs(T, self.max_time_diff, self.time_step)

            for i, o in zip(ti, to):
                u_in  = uTNC[i]                # [N,Cu]
                u_out = uTNC[o]                # [N,Cu]
                u_in_norm = (u_in - u_mean) / u_std
                y         = (u_out - u_mean) / u_std

                if Cc > 0:
                    c_in_norm = (cTNC[i] - c_mean) / c_std
                    x_core = torch.cat([u_in_norm, c_in_norm], dim=-1)
                else:
                    x_core = u_in_norm

                # TRUE-time features
                start_t = float(t_vals[i])
                diff_t  = float(t_vals[o] - t_vals[i])
                start_norm = (start_t - st_mu) / (st_sd if st_sd > 0 else 1.0)
                diff_norm  = (diff_t  - dt_mu) / (dt_sd if dt_sd > 0 else 1.0)
                st = torch.full((N, 1), start_norm, dtype=torch.float32)
                td = torch.full((N, 1), diff_norm,  dtype=torch.float32)
                x_in = torch.cat([x_core, st, td], dim=-1)  # [N, Cu(+Cc)+2]

                buf.append((x_in, y, coord_bucket))
                if len(buf) == self.emit_granularity:
                    for item in buf:
                        yield item
                    buf.clear()

        if not self.drop_last_on_bucket and len(buf) > 0:
            for item in buf:
                yield item

    def __iter__(self):
        for files, coord in zip(self.files_by_bucket, self.coords_by_bucket):
            yield from self._iter_bucket_emitting_blocks(files, coord)

# ---------------- lightweight per-file slice reader used by multires processor ----------------
def _as_HW_C(arr, field):
    x = np.asarray(arr)
    if x.ndim == 2:
        return x[..., None]
    if x.ndim == 3:
        return x
    raise RuntimeError(f"Unsupported rank {x.ndim} for field {field}")

def _read_slice_HW_C(h5, group, field, t_idx):
    x = h5[f"{group}/{field}"][t_idx]
    return _as_HW_C(x, field).astype(np.float32)

class WellH5PairIterableDEMO_Lite(IterableDataset):
    """
    Streams (x_in, y) without loading full (T,...) tensors.
    Uses TRUE time for time features and stats.
    """
    def __init__(self, split_dir: str, stats: Dict, time_step: int, max_time_diff: Optional[int],
                 u_fields_t0: List[str], u_fields_t1: List[str], cache_samples: int = 0):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(split_dir, "*.hdf5")))
        if not self.files:
            raise FileNotFoundError(f"No .hdf5 in {split_dir}")
        self.stats = stats
        self.time_step = int(time_step)
        self.max_time_diff = None if (max_time_diff is None) else int(max_time_diff)
        self.u_fields_t0 = list(u_fields_t0)
        self.u_fields_t1 = list(u_fields_t1)
        self.cache_samples = int(cache_samples)

    def __iter__(self):
        u_mean = self.stats["u"]["mean"].reshape(1, -1)
        u_std  = self.stats["u"]["std"].reshape(1, -1)
        Cu = u_mean.shape[-1]
        c_mean = self.stats["c"]["mean"].reshape(1, -1) if "c" in self.stats else None
        c_std  = self.stats["c"]["std"].reshape(1, -1)  if "c" in self.stats else None
        Cc = 0 if c_mean is None else int(c_mean.shape[-1])

        st_mu = float(self.stats["start_time"]["mean"])
        st_sd = float(self.stats["start_time"]["std"])
        dt_mu = float(self.stats["time_diffs"]["mean"])
        dt_sd = float(self.stats["time_diffs"]["std"])

        for fp in self.files:
            with h5py.File(fp, "r") as h5:
                t_vals = _time_from_dimensions(h5).astype(np.float64)
                T = int(t_vals.shape[0])
                ti, to = build_time_pairs(T, self.max_time_diff, self.time_step)

                u_dsets = [h5[f"t0_fields/{nm}"] for nm in sorted(self.u_fields_t0)] + \
                          [h5[f"t1_fields/{nm}"] for nm in sorted(self.u_fields_t1)]
                c_dset  = h5["forcing_fields/current_drive"] if "forcing_fields" in h5 else None

                for i, o in zip(ti, to):
                    u_in_chunks  = [ _read_slice_HW_C(h5, *ds.name.split('/')[-2:], i) for ds in u_dsets ]
                    u_out_chunks = [ _read_slice_HW_C(h5, *ds.name.split('/')[-2:], o) for ds in u_dsets ]
                    u_in  = np.concatenate(u_in_chunks,  axis=-1)   # (H,W,Cu)
                    u_out = np.concatenate(u_out_chunks, axis=-1)
                    H, W, _ = u_in.shape
                    N = H * W
                    u_in  = torch.from_numpy(u_in.reshape(N, Cu))
                    u_out = torch.from_numpy(u_out.reshape(N, Cu))

                    u_in_norm = (u_in - u_mean) / u_std
                    y         = (u_out - u_mean) / u_std

                    if c_dset is not None:
                        c_in = _as_HW_C(c_dset[i], "current_drive").reshape(N, 1)
                        c_in = torch.from_numpy(c_in.astype(np.float32))
                        c_norm = (c_in - c_mean) / c_std
                        x_core = torch.cat([u_in_norm, c_norm], dim=-1)
                    else:
                        x_core = u_in_norm

                    # TRUE-time features
                    start_t = float(t_vals[i])
                    diff_t  = float(t_vals[o] - t_vals[i])
                    start_norm = (start_t - st_mu) / (st_sd if st_sd > 0 else 1.0)
                    diff_norm  = (diff_t  - dt_mu) / (dt_sd if dt_sd > 0 else 1.0)
                    st = torch.full((N, 1), start_norm, dtype=torch.float32)
                    td = torch.full((N, 1), diff_norm,  dtype=torch.float32)
                    x_in = torch.cat([x_core, st, td], dim=-1)  # [N, Cu(+Cc)+2]

                    yield (x_in, y)
