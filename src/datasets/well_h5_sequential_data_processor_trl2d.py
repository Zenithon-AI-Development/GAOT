# -*- coding: utf-8 -*-
# src/datasets/well_h5_sequential_data_processor_trl2d.py
import os, glob, h5py, numpy as np, torch, yaml
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List

from .sequential_data_processor import SequentialDataProcessor
from .well_h5_pair_dataset_trl2d import (
    WellH5PairIterableTRL2D, build_time_pairs, _mesh_from_dimensions, _time_from_dimensions, _list_field_datasets
)

# ---------------- stats.yaml ----------------
def _load_stats_yaml(base_path: str, dataset_name: str) -> Dict[str, Dict[str, float]]:
    stats_path = os.path.join(base_path, dataset_name, "stats.yaml")
    if not os.path.isfile(stats_path):
        raise FileNotFoundError(f"stats.yaml not found: {stats_path}")
    with open(stats_path, "r") as f:
        stats = yaml.safe_load(f)
    if "mean" not in stats or "std" not in stats:
        raise ValueError(f"stats.yaml missing required keys 'mean'/'std' at {stats_path}")
    return {"mean": dict(stats["mean"]), "std": dict(stats["std"])}

# ---------------- time-index stats over train ----------------
def _stream_time_index_stats(files: List[str], time_step: int, max_time_diff: Optional[int]) -> Tuple[float,float,float,float]:
    n_st=0; st_sum=0.0; st_sumsq=0.0
    n_dt=0; dt_sum=0.0; dt_sumsq=0.0
    for fp in files:
        with h5py.File(fp, "r") as f:
            T = int(_time_from_dimensions(f).shape[0])
        ti, to = build_time_pairs(T, max_time_diff, time_step)
        if ti.size == 0:
            continue
        # accumulate in index-space
        dts = (to - ti).astype(np.float64)
        n_st += int(ti.size)
        st_sum += ti.sum()
        st_sumsq += (ti.astype(np.float64)**2).sum()
        n_dt += int(dts.size)
        dt_sum += dts.sum()
        dt_sumsq += (dts**2).sum()
    if n_st == 0:
        return 0.0, 1.0, 1.0, 1.0
    st_mean = st_sum / n_st
    st_std  = float(np.sqrt(max(st_sumsq / n_st - st_mean**2, 1e-12)))
    if n_dt == 0:
        return st_mean, st_std, 1.0, 1.0
    dt_mean = dt_sum / n_dt
    dt_std  = float(np.sqrt(max(dt_sumsq / n_dt - dt_mean**2, 1e-12)))
    return float(st_mean), float(st_std), float(dt_mean), float(dt_std)

class WellH5SequentialDataProcessorTRL2D(SequentialDataProcessor):
    """
    TRL2D streaming processor (fx only).
    - Reads density & pressure from t0_fields, velocity (2 channels) from t1_fields.
    - Supports variable T across files (50 and 101 in your set).
    - Uses stats.yaml for mean/std; time features standardized from train split.
    """

    def _load_raw_sequential_data(self) -> Dict:
        base = self.dataset_config.base_path
        name = self.dataset_config.name

        # user-selectable fields; default to everything in stats.yaml except 'mask'
        stats_yaml = _load_stats_yaml(base, name)
        if getattr(self.dataset_config, "well_fields", None):
            select_fields = [f for f in self.dataset_config.well_fields if f != "mask"]
        else:
            select_fields = [k for k in stats_yaml["mean"].keys() if k != "mask"]

        # default groups + allow override
        default_groups = {"density": "t0_fields", "pressure": "t0_fields", "velocity": "t1_fields"}
        field_groups = dict(getattr(self.dataset_config, "field_groups", default_groups))
        # ensure mapping exists for chosen fields
        for f in select_fields:
            field_groups.setdefault(f, default_groups.get(f, "t0_fields"))

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

        # probe coords/time from the first train file
        with h5py.File(train_files[0], "r") as h5:
            x_fixed = _mesh_from_dimensions(h5).astype(np.float32)   # [N,2]
            t_vals  = _time_from_dimensions(h5).astype(np.float32)   # [T_ref]
        C_fields = 0
        # expand stats into per-channel lists in field order
        means_expanded: List[float] = []
        stds_expanded:  List[float] = []

        for f in select_fields:
            m_val = stats_yaml["mean"][f]
            s_val = stats_yaml["std"][f]
            if isinstance(m_val, (list, tuple)):
                # e.g. velocity: [mu_x, mu_y]
                if not isinstance(s_val, (list, tuple)) or len(s_val) != len(m_val):
                    raise ValueError(f"stats.yaml std for '{f}' must have same length as mean")
                means_expanded.extend([float(x) for x in m_val])
                stds_expanded.extend([float(x) for x in s_val])
                C_fields += len(m_val)
            else:
                # scalar field
                means_expanded.append(float(m_val))
                stds_expanded.append(float(s_val))
                C_fields += 1

        return {
            "_split_dirs": split_dirs,
            "_train_files": train_files,
            "_val_files": val_files,
            "_test_files": test_files,
            "_select_fields": select_fields,
            "_field_groups": field_groups,
            "x_fixed": x_fixed,  # [N,2]
            "t": t_vals,         # reference timeline (for metadata only)
            "C": C_fields,
            "_u_mean": np.asarray(means_expanded, dtype=np.float32),
            "_u_std":  np.asarray(stds_expanded,  dtype=np.float32),
        }

    def _split_and_normalize_sequential_data(self, raw: Dict, is_variable_coords: bool) -> Dict:
        train_files = raw["_train_files"]; val_files = raw["_val_files"]; test_files = raw["_test_files"]
        select_fields = raw["_select_fields"]; field_groups = raw["_field_groups"]

        # u stats from stats.yaml (already expanded per channel)
        u_mean = torch.tensor(raw["_u_mean"], dtype=self.dtype).view(1, -1)
        u_std  = torch.tensor(raw["_u_std"],  dtype=self.dtype).clamp_min(1e-8).view(1, -1)

        # time-index stats from train split (respecting your max_time_diff / time_step)
        time_step = int(self.time_step if self.time_step is not None else 1)
        max_diff  = None if (self.max_time_diff is None) else int(self.max_time_diff)
        st_m, st_s, dt_m, dt_s = _stream_time_index_stats(train_files, time_step, max_diff)

        self.stats = {
            "u": {"mean": u_mean.to(self.dtype), "std": u_std.to(self.dtype)},
            "start_time": {"mean": torch.tensor(st_m, dtype=self.dtype),
                           "std":  torch.tensor(st_s + 1e-8, dtype=self.dtype)},
            "time_diffs": {"mean": torch.tensor(dt_m, dtype=self.dtype),
                           "std":  torch.tensor(dt_s + 1e-8, dtype=self.dtype)},
        }

        # coords/t for trainer (fx)
        x_fixed = torch.tensor(raw["x_fixed"], dtype=self.dtype)
        t_vals  = torch.tensor(raw["t"], dtype=self.dtype)
        N = x_fixed.shape[0]; C = int(raw["C"])
        dummy_u = torch.zeros((1, 2, N, C), dtype=self.dtype)

        # Stash everything needed to build streaming loaders
        return {
            "train": {"u": dummy_u, "c": None, "x": x_fixed, "t": t_vals},
            "val":   {"u": dummy_u, "c": None, "x": x_fixed, "t": t_vals},
            "test":  {"u": dummy_u, "c": None, "x": x_fixed, "t": t_vals},
            "_stream_meta": {
                "split_dirs": raw["_split_dirs"],
                "select_fields": select_fields,
                "field_groups": field_groups,
                "time_step": time_step,
                "max_time_diff": max_diff,
            }
        }

    def create_sequential_data_loaders(self, data_splits: Dict, is_variable_coords: bool, **kwargs):
        meta = data_splits["_stream_meta"]

        def mk(split):
            return WellH5PairIterableTRL2D(
                split_dir=meta["split_dirs"][split],
                stats=self.stats,
                time_step=meta["time_step"],
                max_time_diff=meta["max_time_diff"],
                select_fields=meta["select_fields"],
                field_groups=meta["field_groups"],
                cache_samples=getattr(self.dataset_config, "stream_cache_samples", 1)
            )

        loaders = {}
        if getattr(self.dataset_config, "train", True):
            loaders["train"] = DataLoader(
                mk("train"),
                batch_size=self.dataset_config.batch_size,
                shuffle=False,
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
        # expose a reference t-grid (used only for axis labels/plots)
        self.t_values = data_splits["train"]["t"].cpu().numpy()
        return loaders
