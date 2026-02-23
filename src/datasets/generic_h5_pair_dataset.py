# -*- coding: utf-8 -*-
# src/datasets/generic_h5_pair_dataset.py
import os, h5py, numpy as np, torch
from torch.utils.data import IterableDataset, Dataset
from typing import List, Tuple, Dict, Optional
from collections import deque

# Toggle all debug prints from here:
DEBUG = True
def dprint(*args, **kwargs):
    if DEBUG: print(*args, **kwargs)

# --- keep your exact function (unchanged) ---
def build_time_pairs(T: int, max_time_diff: int, time_step: int) -> Tuple[np.ndarray, np.ndarray]:
    if T <= 1:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    M = min(T - 1, int(max_time_diff))
    s = int(time_step)
    t_in, t_out = [], []
    for lag in range(s, max_time_diff + 1, s):
        for i in range(0, T - lag, s):
            t_in.append(i); t_out.append(i + lag)
    return np.asarray(t_in, np.int64), np.asarray(t_out, np.int64)

def count_pairs_for_T(T: int, max_time_diff: int, time_step: int) -> int:
    ti, to = build_time_pairs(T, max_time_diff, time_step)
    return int(len(ti))

class H5TrajIndex:
    """Map global sample index -> (file_idx, batch_idx). Accept 4D (T,W,H,C), 5D (B,T,W,H,C), or Group of (B,T,W,H)."""
    def __init__(self, files: List[str], dataset_key: str):
        self.files = files
        self.lookup: List[Tuple[int,int]] = []
        for fi, fp in enumerate(files):
            with h5py.File(fp, "r") as f:
                if dataset_key not in f:
                    continue
                item = f[dataset_key]
                if isinstance(item, h5py.Group):
                    names = sorted(item.keys())
                    d = item[names[0]]
                    B = int(d.shape[0]) if len(d.shape) == 5 else 1
                else:
                    d = item
                    B = int(d.shape[0]) if len(d.shape) == 5 else 1
                for bi in range(B):
                    self.lookup.append((fi, bi))
        if not self.lookup:
            raise FileNotFoundError("No trajectories found across split.")
        dprint(f"[H5PAIR] index built with {len(self.lookup)} trajectories")

    def __len__(self): return len(self.lookup)
    def __getitem__(self, i): return self.lookup[i]

# ---------- Lazy arrays for train/val/test ----------
class H5LazyArray:
    """
    Minimal lazy ND-array for GAOT compatibility:
      - shape: (num_samples, T_min, N, C)
      - __getitem__ supports [sample, time_idx] and [sample, time_idx_array]
    Only requested frames are read.
    """
    def __init__(self, files: List[str], dataset_key: Optional[str], resolution: Optional[Tuple[int, int]] = None):
        self.files = list(files)
        self.key = dataset_key
        if self.key is None:
            self.index = None
            self.shape = None
            return
        self.index = H5TrajIndex(self.files, self.key)

        self._resolution = resolution  # (W, H) when provided by processor (fx)
        if resolution is not None:
            W, H = int(resolution[0]), int(resolution[1])
            with h5py.File(self.files[self.index[0][0]], "r") as f0:
                item0 = f0[self.key]
                C = len(sorted(item0.keys())) if isinstance(item0, h5py.Group) else (int(item0.shape[4]) if len(item0.shape) == 5 else int(item0.shape[3]))
        else:
            with h5py.File(self.files[self.index[0][0]], "r") as f0:
                item0 = f0[self.key]
                if isinstance(item0, h5py.Group):
                    names = sorted(item0.keys())
                    d0 = item0[names[0]]
                    W = int(d0.shape[2]); H = int(d0.shape[3]); C = len(names)
                else:
                    d0 = item0
                    if len(d0.shape) == 5: W, H, C = int(d0.shape[2]), int(d0.shape[3]), int(d0.shape[4])
                    else:                   W, H, C = int(d0.shape[1]), int(d0.shape[2]), int(d0.shape[3])
            self._resolution = None
        self._N = W * H; self._C = C

        self._T_list = []
        for g in range(len(self.index)):
            fi, bi = self.index[g]
            with h5py.File(self.files[fi], "r") as f:
                item = f[self.key]
                if isinstance(item, h5py.Group):
                    d = item[sorted(item.keys())[0]]
                else:
                    d = item
                T = int(d.shape[1] if len(d.shape) >= 4 else d.shape[0])  # (B,T,W,H): T=axis 1
            self._T_list.append(T)
        self._T_min = int(min(self._T_list)) if self._T_list else 0
        self.shape = (len(self.index), self._T_min, self._N, self._C)
        dprint(f"[H5LAZY] key={self.key} shape={self.shape}")

    def _read_frame(self, fi: int, bi: int, t: int) -> torch.Tensor:
        with h5py.File(self.files[fi], "r") as f:
            item = f[self.key]
            if isinstance(item, h5py.Group):
                names = sorted(item.keys())
                parts = [np.array(item[n][0, t] if len(item[n].shape) == 4 else (item[n][bi, t] if len(item[n].shape) == 5 else item[n][t]), copy=False) for n in names]
                x = np.stack(parts, axis=-1)
            else:
                d = item
                if len(d.shape) == 5:
                    x = np.array(d[bi, t], copy=False)
                elif len(d.shape) == 4:
                    x = np.array(d[0, t], copy=False)  # (B,T,W,H)
                else:
                    x = np.array(d[t], copy=False)
            W, H, C = x.shape
            if self._resolution is not None and (W, H) != (self._resolution[0], self._resolution[1]):
                # Resize to processor grid (e.g. file 256x256, processor 51x256)
                from scipy.ndimage import zoom
                W0, H0 = self._resolution[0], self._resolution[1]
                zoom_factors = (W0 / W, H0 / H, 1.0)
                x = zoom(x, zoom_factors, order=1)
                W, H = W0, H0
            out = torch.from_numpy(np.ascontiguousarray(x).reshape(W*H, C).astype(np.float32))
            return out

    def __getitem__(self, key):
        if self.index is None:
            raise IndexError("H5LazyArray(None) cannot be indexed.")
        # expected use: u_data[sample_idx, times]
        if not isinstance(key, tuple) or len(key) < 2:
            raise IndexError("H5LazyArray expects indexing like [sample, times]")
        s, tsel = key[0], key[1]
        fi, bi = self.index[int(s)]

        if isinstance(tsel, (list, tuple, np.ndarray, torch.Tensor)):
            tlist = list(map(int, np.asarray(tsel)))
            frames = [self._read_frame(fi, bi, t) for t in tlist]
            return torch.stack(frames, dim=0)  # [K, N, C]
        elif isinstance(tsel, slice):
            rng = range(tsel.start or 0, tsel.stop or self._T_min, tsel.step or 1)
            frames = [self._read_frame(fi, bi, int(t)) for t in rng]
            return torch.stack(frames, dim=0) if frames else torch.empty(0, self._N, self._C)
        else:
            # single int -> [N, C]
            return self._read_frame(fi, bi, int(tsel))

class GenericH5PairIterable(IterableDataset):
    """
    Streams (x_in, y[, coords]) pairs for GAOT training from generic HDF5 files.
    If coords_per_traj is not None (vx mode), the iterator yields (x_in, y, coords).
    """
    def __init__(self,
                 split_files: List[str],
                 dataset_key: str,
                 cond_key: Optional[str],
                 x_fixed: Optional[np.ndarray],      # used in fx
                 stats: Dict,
                 time_step: int,
                 max_time_diff: int,
                 stepper_mode: str = "output",
                 cache_frames: int = 8,
                 coords_per_traj: Optional[List[torch.Tensor]] = None,  # -> vx
                 ensure_same_resolution_in_batch: bool = True,
                 resolution: Optional[Tuple[int, int]] = None):  # (W, H) from processor in fx
        super().__init__()
        if not split_files:
            raise FileNotFoundError("No HDF5 files provided for split.")

        self.files = list(split_files)
        self.dataset_key = dataset_key
        self.cond_key = cond_key
        self.stats = stats
        self.time_step = int(max(1, time_step))
        self.max_time_diff = int(max_time_diff)
        self.stepper_mode = str(stepper_mode)
        self.resolution = resolution
        self.index = H5TrajIndex(self.files, self.dataset_key)

        # small LRU cache for frames: {(fi,bi,key,t) -> Tensor[N,C]}
        self._cache: Dict[Tuple[int,int,str,int], torch.Tensor] = {}
        self._lru = deque()
        self.cache_frames = int(cache_frames)

        # coords handling
        self.coords_per_traj = coords_per_traj  # None (fx) or list of [N,2] tensors (vx)
        if self.coords_per_traj is None and x_fixed is not None:
            # fx
            self.x_data = torch.as_tensor(x_fixed, dtype=torch.float32)  # [N,2]
            dprint(f"[H5PAIR] fx x_data shape={tuple(self.x_data.shape)}")
        elif self.coords_per_traj is not None:
            # vx: stack for possible graph building or sanity checks
            try:
                self.x_data = torch.stack(self.coords_per_traj, dim=0)  # [B, N, 2]
            except Exception as e:
                raise RuntimeError(f"Could not stack coords_per_traj (non-uniform N?). "
                                   f"Bucket by resolution before loader. Err: {e}")
            dprint(f"[H5PAIR] vx x_data stacked shape={tuple(self.x_data.shape)}")
        else:
            self.x_data = None

        # lazy arrays so TestDataset / utilities can index from disk (use resolution so grid matches processor)
        self.u_data = H5LazyArray(self.files, self.dataset_key, resolution=self.resolution)
        self.c_data = H5LazyArray(self.files, self.cond_key, resolution=self.resolution) if self.cond_key else None

        # Reference time index grid (indices, constant Δt assumption)
        T_min = self.u_data.shape[1] if self.u_data and self.u_data.shape else 0
        self.t_values = torch.arange(T_min, dtype=torch.float32)

        self.ensure_same_resolution_in_batch = bool(ensure_same_resolution_in_batch)

        self._printed_first_pair = False

    def _read_frame(self, fp: str, bi: int, t: int, key: str) -> torch.Tensor:
        with h5py.File(fp, "r") as f:
            item = f[key]
            if isinstance(item, h5py.Group):
                names = sorted(item.keys())
                parts = [np.array(item[n][0, t] if len(item[n].shape) == 4 else (item[n][bi, t] if len(item[n].shape) == 5 else item[n][t]), copy=False) for n in names]
                x = np.stack(parts, axis=-1)
            else:
                d = item
                if len(d.shape) == 5:
                    x = np.array(d[bi, t], copy=False)
                elif len(d.shape) == 4:
                    x = np.array(d[0, t], copy=False)  # (B,T,W,H)
                else:
                    raise RuntimeError(f"Unsupported rank {len(d.shape)} in {fp}:{key}")
            W, H, C = x.shape
            if self.resolution is not None and (W, H) != (self.resolution[0], self.resolution[1]):
                from scipy.ndimage import zoom
                W0, H0 = self.resolution[0], self.resolution[1]
                x = zoom(x, (W0 / W, H0 / H, 1.0), order=1)
                W, H = W0, H0
            return torch.from_numpy(np.ascontiguousarray(x).reshape(W*H, C).astype(np.float32))

    def _frame_cached(self, fi: int, bi: int, key: str, t: int) -> torch.Tensor:
        ck = (fi, bi, key, t)
        if ck in self._cache:
            return self._cache[ck]
        out = self._read_frame(self.files[fi], bi, t, key)
        if self.cache_frames > 0:
            if ck in self._cache:
                try: self._lru.remove(ck)
                except ValueError: pass
            self._cache[ck] = out
            self._lru.append(ck)
            while len(self._lru) > self.cache_frames:
                old = self._lru.popleft()
                self._cache.pop(old, None)
        return out

    def _traj_T(self, fp: str) -> int:
        with h5py.File(fp, "r") as f:
            item = f[self.dataset_key]
            if isinstance(item, h5py.Group):
                d = item[sorted(item.keys())[0]]
            else:
                d = item
            return int(d.shape[1] if len(d.shape) >= 4 else d.shape[0])  # (B,T,W,H): T=axis 1

    # ---- IterableDataset API ----
    def __len__(self) -> int:
        total = 0
        for g in range(len(self.index)):
            fi, bi = self.index[g]
            T = self._traj_T(self.files[fi])
            total += count_pairs_for_T(T, self.max_time_diff, self.time_step)
        dprint(f"[H5PAIR] __len__ total pairs={total}")
        return total

    def __iter__(self):
        # preload stats for broadcasting
        u_mean = self.stats["u"]["mean"].reshape(1, -1)
        u_std  = self.stats["u"]["std"].reshape(1, -1)
        if self.cond_key:
            c_mean = self.stats["c"]["mean"].reshape(1, -1)
            c_std  = self.stats["c"]["std"].reshape(1, -1)

        st_mean = float(self.stats["start_time"]["mean"]); st_std = float(self.stats["start_time"]["std"])
        dt_mean = float(self.stats["time_diffs"]["mean"]); dt_std = float(self.stats["time_diffs"]["std"])

        for g in range(len(self.index)):
            fi, bi = self.index[g]
            fp = self.files[fi]
            T = self._traj_T(fp)
            if T <= 1:
                continue
            t_in, t_out = build_time_pairs(T, self.max_time_diff, self.time_step)
            if DEBUG:
                dprint(f"[H5PAIR] traj#{g} file={os.path.basename(fp)} T={T} pairs={len(t_in)}")

            coords_g = None
            if self.coords_per_traj is not None:
                coords_g = self.coords_per_traj[g]  # [N,2]

            for p in range(len(t_in)):
                ti, to = int(t_in[p]), int(t_out[p])
                lag = to - ti

                u_in  = self._frame_cached(fi, bi, self.dataset_key, ti)  # [N, Cu]
                u_out = self._frame_cached(fi, bi, self.dataset_key, to)  # [N, Cu]
                u_in_norm = (u_in - u_mean) / u_std

                feats = [u_in_norm]
                if self.cond_key:
                    c_in = self._frame_cached(fi, bi, self.cond_key, ti)  # [N, Cc]
                    feats.append((c_in - c_mean) / c_std)

                # time features (last two dims)
                N = u_in.shape[0]
                st_norm = (float(ti)  - st_mean) / (st_std if st_std > 0 else 1.0)
                dt_norm = (float(lag) - dt_mean) / (dt_std if dt_std > 0 else 1.0)
                feats.append(torch.full((N,1), st_norm, dtype=torch.float32))
                feats.append(torch.full((N,1), dt_norm, dtype=torch.float32))
                x_in = torch.cat(feats, dim=-1)            # [N, Cu(+Cc)+2]
                if DEBUG and not self._printed_first_pair:
                    dprint(f"[H5PAIR] first pair x_in shape={tuple(x_in.shape)} y shape={tuple(u_out.shape)}")
                    dprint(f"[H5PAIR] first pair u_in (raw) min/mean/max={u_in.min().item():.4f}/{u_in.mean().item():.4f}/{u_in.max().item():.4f}")
                    dprint(f"[H5PAIR] first pair u_out (raw) min/mean/max={u_out.min().item():.4f}/{u_out.mean().item():.4f}/{u_out.max().item():.4f}")
                    dprint(f"[H5PAIR] first pair y (norm) min/mean/max={y.min().item():.4f}/{y.mean().item():.4f}/{y.max().item():.4f} ti={ti} to={to} lag={lag}")
                    self._printed_first_pair = True

                # targets according to stepper_mode
                if self.stepper_mode == "output":
                    y = (u_out - u_mean) / u_std
                elif self.stepper_mode == "residual":
                    res = u_out - u_in
                    if "res" in self.stats:
                        rmean = self.stats["res"]["mean"].reshape(1, -1)
                        rstd  = self.stats["res"]["std"].reshape(1, -1)
                        y = (res - rmean) / rstd
                    else:
                        y = res
                elif self.stepper_mode == "time_der":
                    der = (u_out - u_in) / max(lag, 1)
                    if "der" in self.stats:
                        dmean = self.stats["der"]["mean"].reshape(1, -1)
                        dstd  = self.stats["der"]["std"].reshape(1, -1)
                        y = (der - dmean) / dstd
                    else:
                        y = der
                else:
                    raise ValueError(f"Unsupported stepper_mode: {self.stepper_mode}")

                if coords_g is None:
                    yield (x_in, y)                # fx
                else:
                    yield (x_in, y, coords_g)      # vx


# -------- NEW: autoregressive test dataset with t_values and x_data --------
class H5AutoregTestDataset(Dataset):
    """
    Returns (x0, target_sequence) for autoregressive evaluation.
    - x0: normalized u_t0 (+ optional c_t0) + two dummy time features (zeros)
    - target_sequence: raw u for time_indices[1:]
    Exposes .t_values and .x_data so SequentialTrainer can use them.
    """
    def __init__(self,
                 files: List[str],
                 dataset_key: str,
                 cond_key: Optional[str],
                 x_fixed: Optional[np.ndarray],     # fx: [N,2]
                 stats: Dict,
                 time_indices: Optional[np.ndarray] = None):
        super().__init__()
        if not files:
            raise FileNotFoundError("No HDF5 files provided for test split.")

        self.files = list(files)
        self.dataset_key = dataset_key
        self.cond_key = cond_key
        self.stats = stats

        self.index = H5TrajIndex(self.files, self.dataset_key)
        self.u_data = H5LazyArray(self.files, self.dataset_key)
        self.c_data = H5LazyArray(self.files, self.cond_key) if self.cond_key else None

        # expose t_values (index-based) and time_indices
        T_min = self.u_data.shape[1]
        self.t_values = torch.arange(T_min, dtype=torch.float32)
        self.time_indices = np.arange(T_min, dtype=np.int64) if time_indices is None \
            else np.asarray(time_indices, dtype=np.int64)

        # expose x_data for fx mode (vx not needed in your current setup)
        self.x_data = torch.as_tensor(x_fixed, dtype=torch.float32) if x_fixed is not None else None

        if DEBUG:
            N = self.u_data.shape[2]; Cu = self.u_data.shape[3]
            dprint(f"[H5TEST] lazy u_data shape={self.u_data.shape} -> N={N} Cu={Cu}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        # initial time index and subsequent targets (index space)
        t0 = int(self.time_indices[0])
        t_future = self.time_indices[1:]

        # read start frame (u), normalize
        u0 = self.u_data[idx, t0]                                           # [N, Cu]
        u_mean = self.stats["u"]["mean"].reshape(1, -1)
        u_std  = self.stats["u"]["std"].reshape(1, -1)
        u0n = (u0 - u_mean) / u_std

        feats = [u0n]
        if self.cond_key:
            c0 = self.c_data[idx, t0]
            c_mean = self.stats["c"]["mean"].reshape(1, -1)
            c_std  = self.stats["c"]["std"].reshape(1, -1)
            feats.append((c0 - c_mean) / c_std)

        # dummy time features (zeros) – model computes real t features internally per step
        N = u0.shape[0]
        feats.append(torch.zeros((N,1), dtype=torch.float32))
        feats.append(torch.zeros((N,1), dtype=torch.float32))
        x0 = torch.cat(feats, dim=-1)                                       # [N, Cu(+Cc)+2]

        # target sequence is raw u for future indices (GAOT will denorm predictions before comparing)
        y_seq = self.u_data[idx, t_future]                                  # [K, N, Cu]
        return x0, y_seq
