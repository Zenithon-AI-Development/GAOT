# -*- coding: utf-8 -*-
# Multires Well-style streaming processor for DEMO z-pinch
import os, glob, h5py, numpy as np, torch
from torch.utils.data import DataLoader, IterableDataset
from typing import Dict, List, Optional, Tuple

from .sequential_data_processor import SequentialDataProcessor
from .well_h5_pair_dataset_demo import (
    _time_from_dimensions, _mesh_from_dimensions, _list_field_datasets,
    _read_field_TNC, _as_THW_C, build_time_pairs, WellH5PairIterableDEMO
)
from .well_h5_pair_dataset_multires import WellH5PairIterableDEMO_Lite  # FIX: make sure class exists and import matches
from ..utils.scaling import CoordinateScaler

# -------- u/c stats across TRAIN files of all buckets ----------
def _stack_u_for_file(h5: h5py.File, u0: List[str], u1: List[str]):
    T_hint = int(_time_from_dimensions(h5).shape[0])
    arrs = []
    for nm in sorted(u0):
        x = np.asarray(h5[f"t0_fields/{nm}"][...])
        THWC = _as_THW_C(x, field=nm, T_hint=T_hint).astype(np.float32)
        T,H,W,C = THWC.shape
        arrs.append(THWC.reshape(T, H*W, C))
    for nm in sorted(u1):
        x = np.asarray(h5[f"t1_fields/{nm}"][...])
        THWC = _as_THW_C(x, field=nm, T_hint=T_hint).astype(np.float32)
        T,H,W,C = THWC.shape
        arrs.append(THWC.reshape(T, H*W, C))
    return np.concatenate(arrs, axis=-1)

def _u_c_mean_std_across_buckets(train_files: List[str], u0: List[str], u1: List[str]):
    u_sum=u_sumsq=None; u_count=0
    c_sum=0.0; c_sumsq=0.0; c_count=0
    for fp in train_files:
        with h5py.File(fp, "r") as h5:
            uTNC = _stack_u_for_file(h5, u0, u1)
            Cu = uTNC.shape[-1]
            if u_sum is None:
                u_sum   = np.zeros((Cu,), dtype=np.float64)
                u_sumsq = np.zeros((Cu,), dtype=np.float64)
            x = uTNC.reshape(-1, Cu).astype(np.float64)
            u_sum   += x.sum(0)
            u_sumsq += (x*x).sum(0)
            u_count += x.shape[0]

            c = np.asarray(h5["forcing_fields/current_drive"][...])
            T_hint = int(_time_from_dimensions(h5).shape[0])
            cTHWC = _as_THW_C(c, field="current_drive", T_hint=T_hint).astype(np.float32)
            v = cTHWC.reshape(-1).astype(np.float64)
            c_sum   += v.sum()
            c_sumsq += (v*v).sum()
            c_count += v.size

    u_mean = u_sum / max(1,u_count)
    u_var  = np.maximum(u_sumsq / max(1,u_count) - u_mean*u_mean, 1e-12)
    u_std  = np.sqrt(u_var)

    c_mean = np.array([c_sum / max(1,c_count)], dtype=np.float64)
    c_var  = np.maximum(c_sumsq / max(1,c_count) - c_mean*c_mean, 1e-12)
    c_std  = np.sqrt(c_var)
    return u_mean.astype(np.float32), u_std.astype(np.float32), c_mean.astype(np.float32), c_std.astype(np.float32)

# TRUE-time stats across all buckets
def _stream_true_time_stats(files: List[str], time_step: int, max_time_diff: Optional[int]):
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

# -------- dataset that yields READY-MADE batches per bucket (no cross-bucket mixing) ----------
class _MultiResBatchIterableDEMO(IterableDataset):
    """
    Yields (xB, yB, coordB) where every item in the batch comes from the same bucket.
    coordB is [N,2] for that bucket; xB is [B,N, Cu+1+2]; yB is [B,N, Cu].
    """
    def __init__(
        self,
        bucket_split_dirs: List[str],
        stats: Dict,
        time_step: int,
        max_time_diff: Optional[int],
        u_fields_t0: List[str],
        u_fields_t1: List[str],
        coords_by_bucket: List[torch.Tensor],
        batch_size: int,
        cache_samples: int = 1,
        shuffle_buckets: bool = True,
    ):
        super().__init__()
        assert len(bucket_split_dirs) == len(coords_by_bucket)
        self.bucket_dirs = list(bucket_split_dirs)
        self.stats = stats
        self.time_step = int(time_step)
        self.max_time_diff = max_time_diff
        self.u_fields_t0 = list(u_fields_t0)
        self.u_fields_t1 = list(u_fields_t1)
        self.coords = [c.detach().clone() for c in coords_by_bucket]
        self.batch_size = int(batch_size)
        self.cache_samples = int(cache_samples)
        self.shuffle_buckets = bool(shuffle_buckets)

    def __iter__(self):
        inner = []
        for split_dir in self.bucket_dirs:
            inner.append(
                WellH5PairIterableDEMO_Lite(
                    split_dir=split_dir,
                    stats=self.stats,
                    time_step=self.time_step,
                    max_time_diff=self.max_time_diff,
                    u_fields_t0=self.u_fields_t0,
                    u_fields_t1=self.u_fields_t1,
                    cache_samples=self.cache_samples,
                )
            )
        order = np.random.permutation(len(inner)).tolist() if self.shuffle_buckets else list(range(len(inner)))
        for bi in order:
            coordB = self.coords[bi]            # [N,2] (already scaled by processor)
            buf_x, buf_y = [], []
            for x, y in inner[bi]:             # x:[N,F], y:[N,Cu]
                buf_x.append(x)
                buf_y.append(y)
                if len(buf_x) == self.batch_size:
                    yield {"x": torch.stack(buf_x, 0), "y": torch.stack(buf_y, 0), "coord": coordB}
                    buf_x.clear(); buf_y.clear()
            if buf_x:
                yield {"x": torch.stack(buf_x, 0), "y": torch.stack(buf_y, 0), "coord": coordB}
                buf_x.clear(); buf_y.clear()

    # def __len__(self):
    #     import glob, h5py
    #     total_pairs = 0
    #     for split_dir in self.bucket_dirs:
    #         for fp in glob.glob(os.path.join(split_dir, "*.hdf5")):
    #             with h5py.File(fp, "r") as h5:
    #                 T = int(_time_from_dimensions(h5).shape[0])
    #             ti, to = build_time_pairs(T, self.max_time_diff, self.time_step)
    #             total_pairs += int(len(ti))
    #     batches = max(1, total_pairs // self.batch_size + (1 if total_pairs % self.batch_size else 0))
    #     return batches

    def __len__(self):
        import glob, h5py, math
        total_batches = 0
        for split_dir in self.bucket_dirs:
            pairs_in_bucket = 0
            for fp in glob.glob(os.path.join(split_dir, "*.hdf5")):
                with h5py.File(fp, "r") as h5:
                    T = int(_time_from_dimensions(h5).shape[0])
                ti, to = build_time_pairs(T, self.max_time_diff, self.time_step)
                pairs_in_bucket += int(len(ti))
            # you flush the residual buffer at the end of each bucket → use ceil
            if pairs_in_bucket > 0:
                total_batches += math.ceil(pairs_in_bucket / max(1, self.batch_size))
        return total_batches

class WellH5SequentialDataProcessorMultiResDEMO(SequentialDataProcessor):
    """
    Scans all resolution folders under dataset_config.base_path (e.g. 32x32, 64x32, 128x128, 128x32, 256x256),
    builds per-bucket streamers, and returns DataLoaders whose items are already batches (xB, yB, coordB).

    Coordinates are scaled to [-1,1] (per config). Time stats & features use TRUE TIME.
    """

    def _load_raw_sequential_data(self) -> Dict:
        base = self.dataset_config.base_path
        bucket_roots = sorted([
            d for d in glob.glob(os.path.join(base, "*"))
            if os.path.isdir(d) and os.path.isdir(os.path.join(d, "data", "train"))
        ])
        if not bucket_roots:
            raise FileNotFoundError(f"No bucket folders with data/train under: {base}")

        split_dirs_by_bucket = {"train": [], "val": [], "test": []}
        train_files_all = []
        coords_by_bucket_phys = []
        u_fields_t0 = None; u_fields_t1 = None

        for br in bucket_roots:
            sd_train = os.path.join(br, "data", "train")
            sd_val   = os.path.join(br, "data", "valid")
            sd_test  = os.path.join(br, "data", "test")
            for sd in (sd_train, sd_val, sd_test):
                if not os.path.isdir(sd):
                    raise FileNotFoundError(f"Missing split directory: {sd}")

            split_dirs_by_bucket["train"].append(sd_train)
            split_dirs_by_bucket["val"].append(sd_val)
            split_dirs_by_bucket["test"].append(sd_test)

            tfiles = sorted(glob.glob(os.path.join(sd_train, "*.hdf5")))
            if not tfiles:
                raise FileNotFoundError(f"No .hdf5 files under {sd_train}")

            with h5py.File(tfiles[0], "r") as h5:
                coords = _mesh_from_dimensions(h5).astype(np.float32)   # [N,2] physical coords
                coords_by_bucket_phys.append(torch.tensor(coords, dtype=self.dtype))
                if u_fields_t0 is None or u_fields_t1 is None:
                    u_fields_t0 = _list_field_datasets(h5, "t0_fields")
                    u_fields_t1 = _list_field_datasets(h5, "t1_fields")

            train_files_all.extend(tfiles)

        assert u_fields_t0 is not None and u_fields_t1 is not None

        # === Coordinate scaling to [-1,1] across buckets ===
        # scaler = CoordinateScaler(target_range=(-1, 1),
        #                           mode=self.dataset_config.coord_scaling)
        # if getattr(self.metadata, "domain_x", None) is not None:
        #     (x0, z0), (x1, z1) = self.metadata.domain_x
        #     scaler.fit(torch.tensor([[x0, z0], [x1, z1]], dtype=self.dtype))
        # else:
        #     # Fit on all physical coords concatenated (rare)
        #     all_coords = torch.cat(coords_by_bucket_phys, dim=0)
        #     scaler.fit(all_coords)
        # coords_by_bucket = [scaler.transform(c) for c in coords_by_bucket_phys]

        # === Coordinate domain determination for consistent scaling ===
        # When testing on a subset of resolutions (e.g., only 32x32), we need to use
        # the SAME domain that was used during training (which may have included all resolutions).
        # Otherwise, coordinates get scaled differently and the model sees shifted inputs.
        #
        # Example: Training on [32x32, 64x64, 128x128, 256x256] uses domain [0.000977, 0.499023]
        #          Testing on [32x32 only] would compute domain [0.007812, 0.492188]
        #          This causes coordinate shift and poor predictions!
        #
        # Solution: Auto-detect training directory (e.g., ZEN_WELL_test -> ZEN_WELL_train)
        #           and use its domain for consistent scaling.
        from ..utils.helpers_true_domain import true_domain_from_dir
        
        # Try to find corresponding training directory for consistent domain
        # This ensures that when testing on a subset of resolutions, we use the same
        # domain that was used during training
        train_dirs_for_domain = None
        base_path_str = str(self.dataset_config.base_path).rstrip('/')
        
        # Only attempt this if the path suggests it's test/validation data
        if any(keyword in base_path_str.lower() for keyword in ['test', 'val', 'valid', 'eval']):
            potential_train_paths = []
            
            # Strategy 1: Replace "_test" with "_train" (and variants)
            if "_test" in base_path_str.lower():
                potential_train_paths.append(base_path_str.replace("_test", "_train").replace("_Test", "_Train").replace("_TEST", "_TRAIN"))
            
            # Strategy 2: Replace "test" suffix with "train"
            if base_path_str.lower().endswith("test"):
                potential_train_paths.append(base_path_str[:-4] + "train")
            
            # Strategy 3: Replace any path component containing "test" with "train"
            path_parts = base_path_str.split('/')
            for i, part in enumerate(path_parts):
                if 'test' in part.lower():
                    new_parts = path_parts[:]
                    new_parts[i] = part.replace('test', 'train').replace('Test', 'Train').replace('TEST', 'TRAIN')
                    potential_train_paths.append('/'.join(new_parts))
            
            # Check if any of these paths exist and have the right bucket structure
            for train_path in potential_train_paths:
                if os.path.isdir(train_path):
                    # Check if it has bucket structure with training data
                    potential_buckets = [
                        os.path.join(d, "data", "train")
                        for d in glob.glob(os.path.join(train_path, "*"))
                        if os.path.isdir(d) and os.path.isdir(os.path.join(d, "data", "train"))
                    ]
                    if potential_buckets:
                        train_dirs_for_domain = potential_buckets
                        print(f"[DOMAIN] Auto-detected training data at: {train_path}")
                        break
        
        # Compute domain from training data (either found training dir or current split)
        if train_dirs_for_domain is not None:
            # Use domain from corresponding training directory
            rmin = zmin = +float("inf")
            rmax = zmax = -float("inf")
            for td in train_dirs_for_domain:
                (rmn, zmn), (rmx, zmx) = true_domain_from_dir(td)
                rmin, zmin = min(rmin, rmn), min(zmin, zmn)
                rmax, zmax = max(rmax, rmx), max(zmax, zmx)
            true_domain = ((rmin, zmin), (rmax, zmax))
            print(f"[DOMAIN] Using domain from training directory: R=[{rmin:.6f}, {rmax:.6f}], Z=[{zmin:.6f}, {zmax:.6f}]")
        else:
            # Fall back to computing from available data in current split
            train_dirs = split_dirs_by_bucket["train"]
            rmin = zmin = +float("inf")
            rmax = zmax = -float("inf")
            for td in train_dirs:
                (rmn, zmn), (rmx, zmx) = true_domain_from_dir(td)
                rmin, zmin = min(rmin, rmn), min(zmin, zmn)
                rmax, zmax = max(rmax, rmx), max(zmax, zmx)
            true_domain = ((rmin, zmin), (rmax, zmax))
            print(f"[DOMAIN] Computed domain from current data: R=[{rmin:.6f}, {rmax:.6f}], Z=[{zmin:.6f}, {zmax:.6f}]")
        
        self.coord_scaler = CoordinateScaler(target_range=(-1, 1),
                                     mode=getattr(self.dataset_config, "coord_scaling", "domain"))
        (self.coord_scaler
            .fit(torch.tensor([[rmin, zmin], [rmax, zmax]], dtype=self.dtype)))
        self.metadata.domain_x = true_domain

        # Keep coords *physical* here; trainer will scale once
        coords_by_bucket = coords_by_bucket_phys

        coords_by_bucket_scaled = [self.coord_scaler.transform(c) for c in coords_by_bucket_phys]

        # === Compute statistics (u, c, time) ===
        # Use files from the training domain directory if we auto-detected it
        # Otherwise use the current base_path training files
        files_for_stats = train_files_all
        if train_dirs_for_domain is not None:
            # Collect all training files from the auto-detected training directory
            print(f"[STATS] Using training directory for statistics computation")
            files_for_stats = []
            for train_dir in train_dirs_for_domain:
                files_for_stats.extend(sorted(glob.glob(os.path.join(train_dir, "*.hdf5"))))
            print(f"[STATS] Found {len(files_for_stats)} training files across {len(train_dirs_for_domain)} buckets")
        else:
            print(f"[STATS] Using current base_path for statistics: {len(files_for_stats)} files")
        
        # global stats (u,c)
        u_mean, u_std, c_mean, c_std = _u_c_mean_std_across_buckets(files_for_stats, u_fields_t0, u_fields_t1)

        time_step = int(self.time_step if self.time_step is not None else 1)
        max_diff  = None if (self.max_time_diff is None) else int(self.max_time_diff)
        st_m, st_s, dt_m, dt_s = _stream_true_time_stats(files_for_stats, time_step, max_diff)

        self.stats = {
            "u": {"mean": torch.tensor(u_mean, dtype=self.dtype).view(1,-1),
                  "std":  torch.tensor(u_std,  dtype=self.dtype).clamp_min(1e-8).view(1,-1)},
            "c": {"mean": torch.tensor(c_mean, dtype=self.dtype).view(1,-1),
                  "std":  torch.tensor(c_std,  dtype=self.dtype).clamp_min(1e-8).view(1,-1)},
            "start_time": {"mean": torch.tensor(st_m, dtype=self.dtype),
                           "std":  torch.tensor(st_s + 1e-8, dtype=self.dtype)},
            "time_diffs": {"mean": torch.tensor(dt_m, dtype=self.dtype),
                           "std":  torch.tensor(dt_s + 1e-8, dtype=self.dtype)},
        }
        
        # Log computed statistics for verification
        print(f"[STATS] Computed statistics from {len(files_for_stats)} files:")
        print(f"  u channels: {u_mean.shape[0]}")
        print(f"  u_mean: {[f'{x:.3e}' for x in u_mean]}")
        print(f"  u_std: {[f'{x:.3e}' for x in u_std]}")
        print(f"  c_mean: {c_mean[0]:.3e}, c_std: {c_std[0]:.3e}")
        print(f"  start_time: {st_m:.6e} ± {st_s:.6e}")
        print(f"  time_diffs: {dt_m:.6e} ± {dt_s:.6e}")
        
        # Save stats to file for inspection
        import json
        stats_save = {
            "u_mean": u_mean.tolist(),
            "u_std": u_std.tolist(),
            "c_mean": c_mean.tolist(),
            "c_std": c_std.tolist(),
            "start_time_mean": float(st_m),
            "start_time_std": float(st_s),
            "time_diffs_mean": float(dt_m),
            "time_diffs_std": float(dt_s),
            "num_files": len(files_for_stats),
            "source": "auto-detected" if train_dirs_for_domain is not None else "current_data"
        }
        stats_path = "/tmp/gaot_test_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats_save, f, indent=2)
        print(f"[STATS] Saved to {stats_path} for inspection")

        # reference x/t just for metadata & plotting labels
        x_ref = coords_by_bucket[0]
        with h5py.File(train_files_all[0], "r") as h5:
            t_vals = _time_from_dimensions(h5).astype(np.float32)

        Cu = int(u_mean.shape[0]); Cc = 1
        Nref = x_ref.shape[0]
        dummy_u = torch.zeros((1, 2, Nref, Cu), dtype=self.dtype)
        dummy_c = torch.zeros((1, 2, Nref, Cc), dtype=self.dtype)

        # Store field names for trainer access (needed for visualization)
        self.u_fields_t0 = u_fields_t0
        self.u_fields_t1 = u_fields_t1

        # stash
        return {
            "train": {"u": dummy_u, "c": dummy_c, "x": x_ref, "t": torch.tensor(t_vals, dtype=self.dtype)},
            "val":   {"u": dummy_u, "c": dummy_c, "x": x_ref, "t": torch.tensor(t_vals, dtype=self.dtype)},
            "test":  {"u": dummy_u, "c": dummy_c, "x": x_ref, "t": torch.tensor(t_vals, dtype=self.dtype)},
            "_multi_meta": {
                "buckets": split_dirs_by_bucket,
                "coords_by_bucket": coords_by_bucket_scaled,   # SCALED coords persisted here
                "u_fields_t0": u_fields_t0,
                "u_fields_t1": u_fields_t1,
                "time_step": time_step,
                "max_time_diff": max_diff,
            }
        }

    def _split_and_normalize_sequential_data(self, raw: Dict, is_variable_coords: bool) -> Dict:
        # Nothing else to do — coords are pre-scaled, stats are set; keep t_values for labels
        self.t_values = raw["train"]["t"].cpu().numpy()
        return raw

    def create_sequential_data_loaders(self, data_splits: Dict, is_variable_coords: bool, **kwargs):
        meta = data_splits["_multi_meta"]

        def mk(split):
            return _MultiResBatchIterableDEMO(
                bucket_split_dirs=meta["buckets"][split],
                stats=self.stats,
                time_step=meta["time_step"],
                max_time_diff=meta["max_time_diff"],
                u_fields_t0=meta["u_fields_t0"],
                u_fields_t1=meta["u_fields_t1"],
                coords_by_bucket=meta["coords_by_bucket"],   # already scaled
                batch_size=self.dataset_config.batch_size,
                cache_samples=getattr(self.dataset_config, "stream_cache_samples", 1),
                shuffle_buckets=getattr(self.dataset_config, "shuffle", True),
            )

        def make_loader(dataset):
            return DataLoader(
                dataset,
                batch_size=None,              # dataset already emits full batches
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
        self.runtime_hints = {"use_trainer_autoreg": True}
        return loaders
