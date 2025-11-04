# -*- coding: utf-8 -*-
# src/datasets/well_h5_sequential_data_processor.py
import os, glob, h5py, numpy as np, torch, yaml
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List

from .sequential_data_processor import SequentialDataProcessor
from .well_h5_pair_dataset_hs import (
    WellH5PairIterable, build_time_pairs, _mesh_from_dimensions, _time_from_dimensions, _list_field_datasets
)

def _load_stats_yaml(base_path: str, dataset_name: str) -> Dict[str, Dict[str, float]]:
    """
    Read The Well stats.yaml and return a dict with 'mean' and 'std' maps.
    Expected structure (as in the Well export):
      mean: {mask: ..., pressure_im: ..., pressure_re: ...}
      std:  {mask: ..., pressure_im: ..., pressure_re: ...}
    """
    stats_path = os.path.join(base_path, dataset_name, "stats.yaml")
    if not os.path.isfile(stats_path):
        raise FileNotFoundError(f"stats.yaml not found: {stats_path}")
    with open(stats_path, "r") as f:
        stats = yaml.safe_load(f)
    if "mean" not in stats or "std" not in stats:
        raise ValueError(f"stats.yaml missing required keys 'mean'/'std' at {stats_path}")
    return {"mean": dict(stats["mean"]), "std": dict(stats["std"])}

class WellH5SequentialDataProcessor(SequentialDataProcessor):
    """
    HDF5-backed streaming data processor for The Well (no WellDataset dependency).
    Fixed-coordinate (fx) only; builds pair-streaming loaders that never hold
    whole trajectories in RAM/GPU. Normalization taken from stats.yaml.
    """

    def _probe_file(self, path: str, active_groups: Tuple[str, ...], select_fields: Optional[List[str]]):
        with h5py.File(path, "r") as h5:
            x_fixed = _mesh_from_dimensions(h5)  # [N,2]
            t_vals  = _time_from_dimensions(h5)  # [T]
            # count channels across selected fields
            C = 0
            field_names = []
            for grp in active_groups:
                names = _list_field_datasets(h5, grp)
                if select_fields:
                    names = [n for n in names if n in select_fields]
                for nm in names:
                    d = h5[f"{grp}/{nm}"]
                    shp = d.shape
                    # treat (..,C) as C channels else scalar
                    Ck = shp[-1] if (len(shp) >= 5) else 1
                    C += Ck
                    field_names.append((grp, nm, Ck))
            return x_fixed.astype(np.float32), t_vals.astype(np.float32), int(C), field_names

    def _load_raw_sequential_data(self) -> Dict:
        base = self.dataset_config.base_path   # e.g. /.../datasets/
        name = self.dataset_config.name        # e.g. helmholtz_staircase

        # We explicitly exclude 'mask' from inputs; default to only pressure channels:
        user_fields = getattr(self.dataset_config, "well_fields", None)
        select_fields = [f for f in (user_fields or ["pressure_im", "pressure_re"]) if f != "mask"]

        # choose groups (most Helmholtz fields are under t0_fields)
        active_groups = tuple(getattr(self.dataset_config, "well_groups", ("t0_fields",)))

        split_dirs = {
            "train": os.path.join(base, name, "data", "train"),
            "val":   os.path.join(base, name, "data", "valid"),
            "test":  os.path.join(base, name, "data", "test"),
        }
        for k, d in split_dirs.items():
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Missing split directory: {d}")

        # probe shapes from first train file
        first = sorted(glob.glob(os.path.join(split_dirs["train"], "*.hdf5")))
        if not first:
            raise FileNotFoundError(f"No .hdf5 files found under {split_dirs['train']}")
        x_fixed, t_vals, C, field_names = self._probe_file(first[0], active_groups, select_fields)

        # load stats.yaml and build per-channel arrays (order = select_fields)
        stats_yaml = _load_stats_yaml(base, name)
        try:
            u_mean_list = [float(stats_yaml["mean"][f]) for f in select_fields]
            u_std_list  = [float(stats_yaml["std"][f])  for f in select_fields]
        except KeyError as e:
            raise KeyError(f"Field {e} not present in stats.yaml. Have keys: {list(stats_yaml['mean'].keys())}")

        return {
            "_split_dirs": split_dirs,
            "_active_groups": active_groups,
            "_select_fields": select_fields,
            "x_fixed": x_fixed,     # [N,2]
            "t": t_vals,            # [T]
            "C": len(select_fields),
            "field_names": field_names,
            "_u_mean": np.asarray(u_mean_list, dtype=np.float32),
            "_u_std":  np.asarray(u_std_list,  dtype=np.float32),
        }

    def _split_and_normalize_sequential_data(self, raw: Dict, is_variable_coords: bool) -> Dict:
        T = len(raw["t"])
        t_vals = raw["t"]
        x_fixed = raw["x_fixed"]

        # time stats in index units (robust to differing physical times between files)
        idx_time = torch.arange(T, dtype=self.dtype)
        start_time_mean = (idx_time[:-1].mean() if T > 1 else torch.tensor(0.0, dtype=self.dtype))
        start_time_std  = (idx_time[:-1].std()  if T > 1 else torch.tensor(1.0, dtype=self.dtype)) + 1e-8
        dt = idx_time[1:] - idx_time[:-1] if T > 1 else torch.tensor([1.0], dtype=self.dtype)
        time_diff_mean = dt.mean()
        time_diff_std  = dt.std() + 1e-8

        # build pair indices + precomputed normalized scalars for iterable
        t_in, t_out = build_time_pairs(T, self.max_time_diff, self.time_step)
        start_times_norm = (idx_time[t_in] - start_time_mean) / start_time_std
        time_diffs_norm  = ((idx_time[t_out] - idx_time[t_in]) - time_diff_mean) / time_diff_std

        # stats: use stats.yaml for u, time stats from indices
        u_mean = torch.tensor(raw["_u_mean"], dtype=self.dtype).view(1, -1)  # [1,C]
        u_std  = torch.tensor(raw["_u_std"],  dtype=self.dtype).clamp_min(1e-8).view(1, -1)

        self.stats = {
            "u": {"mean": u_mean, "std": u_std},
            "start_time": {"mean": start_time_mean, "std": start_time_std,
                           "norm_values": start_times_norm.to(self.dtype)},
            "time_diffs": {"mean": time_diff_mean, "std": time_diff_std,
                           "norm_values": time_diffs_norm.to(self.dtype)},
        }

        # minimal dummy u so the trainer can infer channel counts
        N = x_fixed.shape[0]
        C = raw["C"]
        dummy_u = torch.zeros((1, 2, N, C), dtype=self.dtype)

        data_splits = {
            "train": {"u": dummy_u, "c": None, "x": torch.tensor(x_fixed, dtype=self.dtype), "t": torch.tensor(t_vals, dtype=self.dtype)},
            "val":   {"u": dummy_u, "c": None, "x": torch.tensor(x_fixed, dtype=self.dtype), "t": torch.tensor(t_vals, dtype=self.dtype)},
            "test":  {"u": dummy_u, "c": None, "x": torch.tensor(x_fixed, dtype=self.dtype), "t": torch.tensor(t_vals, dtype=self.dtype)},
            "_stream_meta": {
                "split_dirs": raw["_split_dirs"],
                "active_groups": raw["_active_groups"],
                "select_fields": raw["_select_fields"],
                "t_pairs": (t_in, t_out),
            }
        }
        self.t_values = t_vals
        return data_splits

    def create_sequential_data_loaders(self, data_splits: Dict, is_variable_coords: bool, **kwargs):
        meta = data_splits["_stream_meta"]

        def mk(split):
            return WellH5PairIterable(
                split_dir=meta["split_dirs"][split],
                stats=self.stats,
                t_pairs=meta["t_pairs"],
                active_groups=meta["active_groups"],
                select_fields=meta["select_fields"],
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
        return loaders
