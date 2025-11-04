# -*- coding: utf-8 -*-
# src/datasets/generic_h5_sequential_data_processor.py
import os, glob, h5py, numpy as np, torch
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List, Optional, DefaultDict
from collections import defaultdict, Counter

from .sequential_data_processor import SequentialDataProcessor

# Toggle all debug prints from here:
DEBUG = True
def dprint(*args, **kwargs):
    if DEBUG: print(*args, **kwargs)

# ------ small HDF5 helpers ----------------------------------------------------
def _probe_WH_C_T(fp: str, key: str) -> Tuple[int, int, int, int]:
    with h5py.File(fp, "r") as f:
        d = f[key]
        shp = d.shape
        if len(shp) == 5:   # (B,T,W,H,C)
            return int(shp[2]), int(shp[3]), int(shp[4]), int(shp[1])
        elif len(shp) == 4: # (T,W,H,C)
            return int(shp[1]), int(shp[2]), int(shp[3]), int(shp[0])
        else:
            raise RuntimeError(f"Unsupported rank {len(shp)} for {key} in {fp}")

def _traj_T(fp: str, key: str) -> int:
    with h5py.File(fp, "r") as f:
        d = f[key]
        return int(d.shape[1] if len(d.shape) == 5 else d.shape[0])

# ------ streamed stats (u/c) --------------------------------------------------
def _stream_mean_std_over_train(train_files: List[str], dataset_key: str,
                                max_samples: Optional[int] = None,
                                chunk_T: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    files = train_files if not max_samples else train_files[:max_samples]
    c_sum = None; c_sumsq = None; total = 0
    for fp in files:
        with h5py.File(fp, "r") as f:
            d = f[dataset_key]
            if len(d.shape) == 5:   # (B,T,W,H,C) -> take first B dim
                T = int(d.shape[1])
                for t0 in range(0, T, chunk_T):
                    t1 = min(t0 + chunk_T, T)
                    x = np.array(d[0, t0:t1], copy=False)        # (chunk,W,H,C)
                    X = x.reshape(-1, x.shape[-1]).astype(np.float32)
            elif len(d.shape) == 4: # (T,W,H,C)
                T = int(d.shape[0])
                for t0 in range(0, T, chunk_T):
                    t1 = min(t0 + chunk_T, T)
                    x = np.array(d[t0:t1], copy=False)           # (chunk,W,H,C)
                    X = x.reshape(-1, x.shape[-1]).astype(np.float32)
            else:
                raise RuntimeError(f"Unsupported rank {len(d.shape)} in {fp}:{dataset_key}")

            if c_sum is None:
                c_sum   = np.zeros(X.shape[-1], dtype=np.float64)
                c_sumsq = np.zeros(X.shape[-1], dtype=np.float64)
            c_sum   += X.sum(axis=0, dtype=np.float64)
            c_sumsq += (X.astype(np.float64) ** 2).sum(axis=0)
            total   += X.shape[0]

    mean = (c_sum / max(total, 1)).astype(np.float32)
    var  = (c_sumsq / max(total, 1) - mean.astype(np.float64)**2).astype(np.float32)
    var  = np.maximum(var, 1e-12); std = np.sqrt(var).astype(np.float32)
    return mean, std

# ------ streamed time-index stats (for the 2 time features) -------------------
def _stream_time_index_stats(files: List[str], dataset_key: str,
                             max_time_diff: int, time_step: int) -> Tuple[float,float,float,float]:
    n_st=0; st_sum=0.0; st_sumsq=0.0
    n_dt=0; dt_sum=0.0; dt_sumsq=0.0
    for fp in files:
        T = _traj_T(fp, dataset_key)
        if T <= 1: 
            continue
        s = max(1, int(time_step))
        M = min(T-1, int(max_time_diff))
        for lag in range(s, M+1, s):            # mirror your build_time_pairs()
            for i in range(0, T - lag, s):
                n_st += 1; st_sum += i; st_sumsq += i*i
                d = lag
                n_dt += 1; dt_sum += d; dt_sumsq += d*d
    if n_st == 0:
        return 0.0, 1.0, 1.0, 1.0
    st_mean = st_sum / n_st
    st_std  = float(np.sqrt(max(st_sumsq / n_st - st_mean**2, 1e-12)))
    if n_dt == 0:
        return float(st_mean), float(st_std), 1.0, 1.0
    dt_mean = dt_sum / n_dt
    dt_std  = float(np.sqrt(max(dt_sumsq / n_dt - dt_mean**2, 1e-12)))
    # BUGFIX: return (st_mean, st_std, dt_mean, dt_std)
    return float(st_mean), float(st_std), float(dt_mean), float(dt_std)

# ------ NEW: streamed residual / derivative stats over *training pairs* -------
def _stream_pair_stats_over_train(train_files: List[str], dataset_key: str,
                                  max_time_diff: int, time_step: int,
                                  kind: str = "der",
                                  chunk_T: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """
    kind: 'res' for (u_{t+lag} - u_t), 'der' for (u_{t+lag}-u_t)/lag
    Uses the same pairing as build_time_pairs to avoid train/test scale mismatch.
    Streams over time in small chunks to keep memory low.
    Returns (mean[C], std[C]).
    """
    s = max(1, int(time_step))
    # Accumulators in float64 for numerical stability
    sum_c   = None
    sumsq_c = None
    count   = 0

    for fp in train_files:
        with h5py.File(fp, "r") as f:
            d = f[dataset_key]
            # collapse to (T,W,H,C)
            if len(d.shape) == 5:
                arr = d[0]   # (T,W,H,C)
            elif len(d.shape) == 4:
                arr = d      # (T,W,H,C)
            else:
                raise RuntimeError(f"Unsupported rank {len(d.shape)} in {fp}:{dataset_key}")

            T = int(arr.shape[0])
            if T <= 1:
                continue
            M = min(T-1, int(max_time_diff))
            if M < s:
                continue

            # stream in temporal chunks, but include a lookahead of M
            for t0 in range(0, T, chunk_T):
                t1 = min(T, t0 + chunk_T + M)
                x = np.array(arr[t0:t1], copy=False).astype(np.float32)  # (Tc,W,H,C)
                Tc, W, H, C = x.shape
                if sum_c is None:
                    sum_c   = np.zeros(C, dtype=np.float64)
                    sumsq_c = np.zeros(C, dtype=np.float64)

                for lag in range(s, M+1, s):
                    # for this chunk, usable pairs start at indices [0 .. Tc-lag-1]
                    end = Tc - lag
                    if end <= 0: 
                        continue
                    u_in  = x[:end]           # (end, W, H, C)
                    u_out = x[lag:lag+end]    # (end, W, H, C)
                    if kind == "res":
                        vals = (u_out - u_in).reshape(-1, C)            # (end*W*H, C)
                    elif kind == "der":
                        vals = ((u_out - u_in) / float(lag)).reshape(-1, C)
                    else:
                        raise ValueError("kind must be 'res' or 'der'")

                    # accumulate
                    sum_c   += vals.sum(axis=0, dtype=np.float64)
                    sumsq_c += (vals.astype(np.float64)**2).sum(axis=0)
                    count   += vals.shape[0]

    if count == 0:
        # Fallback: avoid division by zero; produce safe scales
        C_guess = 1 if sum_c is None else sum_c.shape[0]
        mean = np.zeros(C_guess, dtype=np.float32)
        std  = np.ones(C_guess,  dtype=np.float32)
        return mean, std

    mean = (sum_c / float(count)).astype(np.float32)
    var  = (sumsq_c / float(count) - mean.astype(np.float64)**2).astype(np.float32)
    var  = np.maximum(var, 1e-12)
    std  = np.sqrt(var).astype(np.float32)
    return mean, std

# ------ the main processor ----------------------------------------------------
class GenericH5SequentialDataProcessor(SequentialDataProcessor):
    """
    Streaming loader for generic HDF5 datasets.
    - Folder of .hdf5 files (one trajectory per file, or B>1 supported).
    - fx mode: fixed coords tensor [N,2].
    - vx mode: variable coords per trajectory; we bucket by resolution so coords can stack.
    - Streams only needed frames for u/c.
    - Optional 'cond_key' for exogenous channels c.
    - Supports variable T per trajectory (assumes constant Δt per traj).
    """

    def _load_raw_sequential_data(self) -> dict:
        dprint("[H5SEQ] Loading and preprocessing sequential data...")
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

        dataset_key = getattr(self.dataset_config, "dataset_key", "input_fields")
        cond_key    = getattr(self.dataset_config, "cond_key", None)

        train_files = sorted(glob.glob(os.path.join(split_dirs["train"], "*.hdf5")))
        val_files   = sorted(glob.glob(os.path.join(split_dirs["val"],   "*.hdf5")))
        test_files  = sorted(glob.glob(os.path.join(split_dirs["test"],  "*.hdf5")))
        if not train_files:
            raise FileNotFoundError(f"No .hdf5 files in {split_dirs['train']}")

        dprint(f"[H5SEQ] Train file count: {len(train_files)}")
        # probe a few and summarize resolutions
        def summarize(files):
            res= []
            for i, fp in enumerate(files[:min(3, len(files))]):
                W,H,C,T = _probe_WH_C_T(fp, dataset_key)
                res.append((T, W, H, C))
            return res
        dprint(f"[H5SEQ] First 3 train T,W,H,C: {summarize(train_files)}")

        # reference t-grid for metadata compatibility only (index-based)
        W0, H0, C0, T0 = _probe_WH_C_T(train_files[0], dataset_key)
        t_vals = np.arange(T0, dtype=np.float32)

        # simple grid builder for a given (W,H)
        def make_grid(W, H):
            xv, yv = np.meshgrid(np.arange(W, dtype=np.float32),
                                 np.arange(H, dtype=np.float32), indexing="ij")
            return np.stack([xv, yv], axis=-1).reshape(-1, 2)

        return {
            "_split_dirs": split_dirs,
            "_dataset_key": dataset_key,
            "_cond_key": cond_key,
            "_train_files": train_files,
            "_val_files":   val_files,
            "_test_files":  test_files,
            "_t_ref": t_vals,        # reference only
            "_C_ref": C0,
            "_make_grid": make_grid
        }

    def _bucket_by_resolution(self, files: List[str], key: str) -> Dict[Tuple[int,int], List[str]]:
        buckets: DefaultDict[Tuple[int,int], List[str]] = defaultdict(list)
        for fp in files:
            W,H,C,T = _probe_WH_C_T(fp, key)
            buckets[(W,H)].append(fp)
        return dict(buckets)

    def _split_and_normalize_sequential_data(self, raw: dict, is_variable_coords: bool) -> dict:
        dataset_key = raw["_dataset_key"]; cond_key = raw["_cond_key"]
        train_files = raw["_train_files"]; val_files = raw["_val_files"]; test_files = raw["_test_files"]
        t_vals_ref  = raw["_t_ref"]; C_ref = raw["_C_ref"]; make_grid = raw["_make_grid"]

        # --- detect resolutions present in train ---
        train_buckets = self._bucket_by_resolution(train_files, dataset_key)
        uniq_train_res = list(train_buckets.keys())
        force_vx = bool(getattr(self.dataset_config, "force_vx", False))
        is_vx = force_vx or (len(uniq_train_res) > 1)
        if DEBUG:
            cnt = {res: len(v) for res, v in train_buckets.items()}
            dprint(f"[H5SEQ] train resolutions -> counts: {cnt}")
            dprint(f"[H5SEQ] Detected coord mode (before user override): {'vx' if is_vx else 'fx'}")

        # --- if vx, filter to one (W,H) bucket for this run (most frequent or user-specified) ---
        chosen_res = None
        if is_vx:
            user_res = getattr(self.dataset_config, "vx_target_wh", None)  # e.g., [256,256]
            if user_res is not None:
                chosen_res = tuple(int(x) for x in user_res)
                if chosen_res not in train_buckets:
                    raise ValueError(f"vx_target_wh={chosen_res} not found in train resolutions {list(train_buckets.keys())}")
            else:
                chosen_res = Counter({res: len(v) for res, v in train_buckets.items()}).most_common(1)[0][0]
            dprint(f"[H5SEQ] vx chosen resolution bucket: {chosen_res}")

            def filter_to_res(files):
                out = []
                for fp in files:
                    W,H,C,T = _probe_WH_C_T(fp, dataset_key)
                    if (W,H) == chosen_res:
                        out.append(fp)
                return out
            train_files = filter_to_res(train_files)
            val_files   = filter_to_res(val_files)
            test_files  = filter_to_res(test_files)
            dprint(f"[H5SEQ] files after filtering to {chosen_res}: train={len(train_files)} val={len(val_files)} test={len(test_files)}")

        # --- 1) stats for u (and c) from TRAIN only ---
        u_mean_np, u_std_np = _stream_mean_std_over_train(
            train_files, dataset_key,
            max_samples=getattr(self.dataset_config, "stats_max_samples", None),
            chunk_T=getattr(self.dataset_config, "stats_chunk_T", 32),
        )
        stats: Dict = {
            "u": {
                "mean": torch.from_numpy(u_mean_np).view(1,-1).to(self.dtype),
                "std":  torch.from_numpy(np.maximum(u_std_np,1e-8)).view(1,-1).to(self.dtype),
            }
        }
        if cond_key:
            c_mean_np, c_std_np = _stream_mean_std_over_train(
                train_files, cond_key,
                max_samples=getattr(self.dataset_config, "stats_max_samples", None),
                chunk_T=getattr(self.dataset_config, "stats_chunk_T", 32),
            )
            stats["c"] = {
                "mean": torch.from_numpy(c_mean_np).view(1,-1).to(self.dtype),
                "std":  torch.from_numpy(np.maximum(c_std_np,1e-8)).view(1,-1).to(self.dtype),
            }
        if DEBUG:
            dprint(f"[H5SEQ] stats[u]: mean/std shapes -> {tuple(stats['u']['mean'].shape)}, {tuple(stats['u']['std'].shape)}")
            if "c" in stats:
                dprint(f"[H5SEQ] stats[c]: mean/std shapes -> {tuple(stats['c']['mean'].shape)}, {tuple(stats['c']['std'].shape)}")

        # --- 2) global time-index stats over TRAIN (for the two time features) ---
        kmax = int(self.max_time_diff if self.max_time_diff is not None else 10**9)
        step = int(self.time_step if self.time_step is not None else 1)
        st_m, st_s, dt_m, dt_s = _stream_time_index_stats(train_files, dataset_key, kmax, step)
        stats["start_time"] = {"mean": torch.tensor(st_m, dtype=self.dtype),
                               "std":  torch.tensor(st_s + 1e-8, dtype=self.dtype)}
        stats["time_diffs"] = {"mean": torch.tensor(dt_m, dtype=self.dtype),
                               "std":  torch.tensor(dt_s + 1e-8, dtype=self.dtype)}
        if DEBUG:
            dprint(f"[H5SEQ] time stats: start(mean={st_m:.3f}, std={st_s:.3f})  diff(mean={dt_m:.3f}, std={dt_s:.3f})")

        # --- 3) NEW: residual/derivative stats over TRAIN pairs (to match stepper_mode) ---
        # we compute both to be robust; GAOT will use the one it needs
        res_mean_np, res_std_np = _stream_pair_stats_over_train(train_files, dataset_key, kmax, step, kind="res")
        der_mean_np, der_std_np = _stream_pair_stats_over_train(train_files, dataset_key, kmax, step, kind="der")
        stats["res"] = {
            "mean": torch.from_numpy(res_mean_np).view(1,-1).to(self.dtype),
            "std":  torch.from_numpy(np.maximum(res_std_np,1e-8)).view(1,-1).to(self.dtype),
        }
        stats["der"] = {
            "mean": torch.from_numpy(der_mean_np).view(1,-1).to(self.dtype),
            "std":  torch.from_numpy(np.maximum(der_std_np,1e-8)).view(1,-1).to(self.dtype),
        }
        if DEBUG:
            dprint(f"[H5SEQ] stats[res]: mean/std shapes -> {tuple(stats['res']['mean'].shape)}, {tuple(stats['res']['std'].shape)}")
            dprint(f"[H5SEQ] stats[der]: mean/std shapes -> {tuple(stats['der']['mean'].shape)}, {tuple(stats['der']['std'].shape)}")

        # expose stats on self (used by iterables + trainer)
        self.stats = stats
        self.t_values = t_vals_ref

        # --- 4) Build split dicts (fx or vx) for trainer compatibility ----------
        if not is_vx:
            # fx: single fixed grid from first train file
            W0,H0,C0,T0 = _probe_WH_C_T(train_files[0], dataset_key)
            x_fixed = make_grid(W0, H0).astype(np.float32)
            if DEBUG:
                dprint(f"[H5SEQ] fx mode: grid {W0}x{H0} -> N={x_fixed.shape[0]}")
            # dummy_u just for channel inference by trainer; not used to read data
            dummy_u = torch.zeros((1, 2, x_fixed.shape[0], C_ref), dtype=self.dtype)
            self.runtime_hints = {"use_trainer_autoreg": True}
            return {
                "train": {"u": dummy_u, "c": None, "x": torch.tensor(x_fixed, dtype=self.dtype), "t": torch.tensor(t_vals_ref, dtype=self.dtype)},
                "val":   {"u": dummy_u, "c": None, "x": torch.tensor(x_fixed, dtype=self.dtype), "t": torch.tensor(t_vals_ref, dtype=self.dtype)},
                "test":  {"u": dummy_u, "c": None, "x": torch.tensor(x_fixed, dtype=self.dtype), "t": torch.tensor(t_vals_ref, dtype=self.dtype)},
                "_stream_meta": {
                    "files": {"train": train_files, "val": val_files, "test": test_files},
                    "dataset_key": dataset_key,
                    "cond_key": cond_key,
                    "x_fixed": x_fixed.astype(np.float32),
                    "time_step": step,
                    "max_time_diff": kmax,
                    "stepper_mode": getattr(self, "stepper_mode", "output"),
                    "coords": {"train": None, "val": None, "test": None}  # fx
                }
            }

        # vx: build per-traj coords for each split (all same resolution by filtering above)
        def coords_for_split(files):
            coords_list = []
            for i, fp in enumerate(files):
                W,H,C,T = _probe_WH_C_T(fp, dataset_key)
                coords = make_grid(W, H).astype(np.float32)  # [N,2]
                coords_list.append(torch.from_numpy(coords))
                if DEBUG and i < 3:
                    dprint(f"[H5SEQ] vx coords {os.path.basename(fp)} -> shape={coords.shape}")
            return coords_list

        coords_train = coords_for_split(train_files)
        coords_val   = coords_for_split(val_files)
        coords_test  = coords_for_split(test_files)

        def stack_coords(lst, tag):
            if not lst:
                return torch.empty(0, 0, 2)
            out = torch.stack(lst, dim=0)  # [B, N, 2]
            if DEBUG:
                dprint(f"[H5SEQ] vx stacked {tag} coords: {tuple(out.shape)}")
            return out

        x_train = stack_coords(coords_train, "train")
        x_val   = stack_coords(coords_val, "val")
        x_test  = stack_coords(coords_test, "test")

        N_ref = x_train.shape[1] if x_train.numel() else (x_val.shape[1] if x_val.numel() else x_test.shape[1])
        dummy_u = torch.zeros((1, 2, int(N_ref), C_ref), dtype=self.dtype)

        return {
            "train": {"u": dummy_u, "c": None, "x": x_train.to(self.dtype), "t": torch.tensor(t_vals_ref, dtype=self.dtype)},
            "val":   {"u": dummy_u, "c": None, "x": x_val.to(self.dtype),   "t": torch.tensor(t_vals_ref, dtype=self.dtype)},
            "test":  {"u": dummy_u, "c": None, "x": x_test.to(self.dtype),  "t": torch.tensor(t_vals_ref, dtype=self.dtype)},
            "_stream_meta": {
                "files": {"train": train_files, "val": val_files, "test": test_files},
                "dataset_key": dataset_key,
                "cond_key": cond_key,
                "x_fixed": None,
                "time_step": step,
                "max_time_diff": kmax,
                "stepper_mode": getattr(self, "stepper_mode", "output"),
                "coords": {  # list per split for the iterable to pick per-trajectory coords
                    "train": coords_train,
                    "val":   coords_val,
                    "test":  coords_test
                }
            }
        }

    def create_sequential_data_loaders(self, data_splits: Dict, is_variable_coords: bool, **kwargs):
        # Late import to avoid circular
        from .generic_h5_pair_dataset import GenericH5PairIterable

        meta = data_splits["_stream_meta"]

        def mk(split):
            return GenericH5PairIterable(
                split_files=meta["files"][split],
                dataset_key=meta["dataset_key"],
                cond_key=meta["cond_key"],
                x_fixed=meta["x_fixed"],                 # fx uses this; vx will ignore
                stats=self.stats,
                time_step=meta["time_step"],
                max_time_diff=meta["max_time_diff"],
                stepper_mode=meta["stepper_mode"],
                cache_frames=getattr(self.dataset_config, "stream_cache_frames", 8),
                coords_per_traj=meta["coords"][split],   # None in fx, list[Tensors] in vx
                ensure_same_resolution_in_batch=True
            )

        loaders = {}
        if getattr(self.dataset_config, "train", True):
            loaders["train"] = DataLoader(
                mk("train"),
                batch_size=self.dataset_config.batch_size,
                shuffle=False,  # keep sequential so batches don't mix resolutions/files
                num_workers=self.dataset_config.num_workers,
                pin_memory=True,
            )
            loaders["val"] = DataLoader(
                mk("val"),
                batch_size=self.dataset_config.batch_size,
                shuffle=False,
                num_workers=self.dataset_config.num_workers,
                pin_memory=True,
            )
        else:
            loaders["train"] = loaders["val"] = None

        loaders["test"] = DataLoader(
            mk("test"),
            batch_size=self.dataset_config.batch_size,
            shuffle=False,
            num_workers=self.dataset_config.num_workers,
            pin_memory=True,
        )
        return loaders
