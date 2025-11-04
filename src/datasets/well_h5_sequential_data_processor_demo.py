# -*- coding: utf-8 -*-
# src/datasets/well_h5_sequential_data_processor_demo.py
import os, glob, h5py, numpy as np, torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from ..utils.scaling import CoordinateScaler

from .sequential_data_processor import SequentialDataProcessor
from .well_h5_pair_dataset_demo import (
    WellH5PairIterableDEMO,
    build_time_pairs,
    _mesh_from_dimensions, _time_from_dimensions, _list_field_datasets,
)

# ---------- helpers to compute TRUE-TIME stats across TRAIN ----------
def _stream_true_time_stats(files: List[str], time_step: int, max_time_diff: Optional[int]) -> Tuple[float,float,float,float]:
    n_st=0; st_sum=0.0; st_sumsq=0.0
    n_dt=0; dt_sum=0.0; dt_sumsq=0.0
    for fp in files:
        with h5py.File(fp, "r") as f:
            t = _time_from_dimensions(f).astype(np.float64)
            T = int(t.shape[0])
        ti, to = build_time_pairs(T, max_time_diff, time_step)
        if ti.size == 0: 
            continue
        # TRUE times
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

# ---------- streaming mean/std over TRAIN for u and c ----------
def _stack_u_for_file(h5: h5py.File, u_fields_t0: List[str], u_fields_t1: List[str]) -> np.ndarray:
    t_vals = _time_from_dimensions(h5); T_hint = int(t_vals.shape[0])
    chunks = []
    for nm in sorted(u_fields_t0):
        chunks.append(np.asarray(h5[f"t0_fields/{nm}"][...]))
    for nm in sorted(u_fields_t1):
        chunks.append(np.asarray(h5[f"t1_fields/{nm}"][...]))
    # normalize all to (T,H,W,C) then to (T,N,C)
    arrs = []
    from .well_h5_pair_dataset_demo import _as_THW_C
    for nm, x in zip(list(sorted(u_fields_t0))+list(sorted(u_fields_t1)), chunks):
        THWC = _as_THW_C(x, field=nm, T_hint=T_hint).astype(np.float32)
        T,H,W,C = THWC.shape
        arrs.append(THWC.reshape(T, H*W, C))
    return np.concatenate(arrs, axis=-1)  # (T,N,Cu)

def _u_c_mean_std(train_files: List[str], u_fields_t0: List[str], u_fields_t1: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u_sum=None; u_sumsq=None; u_count=0
    c_sum=0.0; c_sumsq=0.0; c_count=0

    for fp in train_files:
        with h5py.File(fp, "r") as h5:
            uTNC = _stack_u_for_file(h5, u_fields_t0, u_fields_t1)  # (T,N,Cu)
            Cu = uTNC.shape[-1]
            if u_sum is None:
                u_sum   = np.zeros((Cu,), dtype=np.float64)
                u_sumsq = np.zeros((Cu,), dtype=np.float64)
            x = uTNC.reshape(-1, Cu).astype(np.float64)  # (T*N,Cu)
            u_sum   += x.sum(axis=0)
            u_sumsq += (x * x).sum(axis=0)
            u_count += x.shape[0]

            c = np.asarray(h5["forcing_fields/current_drive"][...])
            from .well_h5_pair_dataset_demo import _as_THW_C
            T_hint = int(_time_from_dimensions(h5).shape[0])
            cTHWC = _as_THW_C(c, field="current_drive", T_hint=T_hint).astype(np.float32)
            cTN1  = cTHWC.reshape(cTHWC.shape[0], -1, cTHWC.shape[-1])  # (T,N,1)
            v = cTN1.reshape(-1).astype(np.float64)
            c_sum   += v.sum()
            c_sumsq += (v * v).sum()
            c_count += v.size

    u_mean = u_sum / max(1, u_count)
    u_var  = np.maximum(u_sumsq / max(1, u_count) - u_mean*u_mean, 1e-12)
    u_std  = np.sqrt(u_var)

    c_mean = np.array([c_sum / max(1, c_count)], dtype=np.float64)
    c_var  = np.maximum(c_sumsq / max(1, c_count) - c_mean*c_mean, 1e-12)
    c_std  = np.sqrt(c_var)
    return u_mean.astype(np.float32), u_std.astype(np.float32), c_mean.astype(np.float32), c_std.astype(np.float32)

class WellH5SequentialDataProcessorDEMO(SequentialDataProcessor):
    """
    Well-style streaming processor (fx only) for your DEMO z-pinch dataset.
    - Discovers all u fields under t0_fields + t1_fields.
    - Uses forcing_fields/current_drive as conditioning (c).
    - Computes per-channel mean/std from TRAIN split (streaming).
    - Builds pair-streaming loaders; no whole-trajectory tensors in RAM/GPU.
    - **Coordinates are scaled to [-1,1]** using CoordinateScaler (per your config).
    - **Time stats and time features use true time** t[i], t[o]-t[i].
    """

    def _load_raw_sequential_data(self) -> Dict:
        base = self.dataset_config.base_path
        name = self.dataset_config.name

        split_dirs = {
            "train": os.path.join(base, name, "data", "train"),
            "val":   os.path.join(base, name, "data", "valid"),
            "test":  os.path.join(base, name, "data", "test"),
        }
        for k,d in split_dirs.items():
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Missing split directory: {d}")

        train_files = sorted(glob.glob(os.path.join(split_dirs["train"], "*.hdf5")))
        val_files   = sorted(glob.glob(os.path.join(split_dirs["val"],   "*.hdf5")))
        test_files  = sorted(glob.glob(os.path.join(split_dirs["test"],  "*.hdf5")))
        if not train_files:
            raise FileNotFoundError(f"No .hdf5 files found under {split_dirs['train']}")

        # Probe coords/time + discover fields from first train file
        with h5py.File(train_files[0], "r") as h5:
            x_fixed_phys = _mesh_from_dimensions(h5).astype(np.float32)  # [N,2] physical coords
            t_vals  = _time_from_dimensions(h5).astype(np.float32)       # [T_ref]
            u_fields_t0 = _list_field_datasets(h5, "t0_fields")
            u_fields_t1 = _list_field_datasets(h5, "t1_fields")

        # # === Coordinate scaling to [-1,1] (per GAOT style) ===
        # scaler = CoordinateScaler(target_range=(-1, 1),
        #                           mode=self.dataset_config.coord_scaling)
        # # Prefer domain corners if provided; else fit on actual coords
        # if getattr(self.metadata, "domain_x", None) is not None:
        #     (x0, z0), (x1, z1) = self.metadata.domain_x
        #     scaler.fit(torch.tensor([[x0, z0], [x1, z1]], dtype=self.dtype))
        # else:
        #     scaler.fit(torch.tensor(x_fixed_phys, dtype=self.dtype))
        # x_fixed = scaler.transform(torch.tensor(x_fixed_phys, dtype=self.dtype))
        # === Coordinate scaler: fit here, trainer applies transform once ===
        # === Coordinate scaler (fit once) ===

        from ..utils.helpers_true_domain import true_domain_from_dir
        train_dir = os.path.join(base, name, "data", "train")
        true_domain = true_domain_from_dir(train_dir)           # ((rmin,zmin),(rmax,zmax))

        self.coord_scaler = CoordinateScaler(target_range=(-1, 1),
                                            mode=getattr(self.dataset_config, "coord_scaling", "domain"))
        # Fit on *true* corners, not metadata
        (r0, z0), (r1, z1) = true_domain
        self.coord_scaler.fit(torch.tensor([[r0, z0], [r1, z1]], dtype=self.dtype))

        # Make this visible to anything that still reads metadata.domain_x (e.g. generate_latent_queries)
        self.metadata.domain_x = ((r0, z0), (r1, z1))

        # IMPORTANT: return UNscaled coords; trainer will apply self.coord_scaler exactly once
        x_fixed = torch.tensor(x_fixed_phys, dtype=self.dtype)

        # scaler = CoordinateScaler(target_range=(-1, 1),
        #                           mode=getattr(self.dataset_config, "coord_scaling", "global_scaling"))
        # use_domain = (getattr(self.dataset_config, "coord_scaling", "global_scaling") == "domain") \
        #              and (getattr(self.metadata, "domain_x", None) is not None)
        # if use_domain:
        #     (x0, z0), (x1, z1) = self.metadata.domain_x
        #     scaler.fit(torch.tensor([[x0, z0], [x1, z1]], dtype=self.dtype))
        # else:
        #     scaler.fit(torch.tensor(x_fixed_phys, dtype=self.dtype))
        # self.coord_scaler = scaler
        # # IMPORTANT: return UNscaled coords here; trainer will apply scaling exactly once
        # x_fixed = torch.tensor(x_fixed_phys, dtype=self.dtype)



        # Mean/std over TRAIN (u and c)
        u_mean, u_std, c_mean, c_std = _u_c_mean_std(train_files, u_fields_t0, u_fields_t1)

        # Count channels for u dummy
        with h5py.File(train_files[0], "r") as h5:
            from .well_h5_pair_dataset_demo import _as_THW_C
            T_hint = int(_time_from_dimensions(h5).shape[0])
            chunks = []
            for nm in sorted(u_fields_t0):
                x = np.asarray(h5[f"t0_fields/{nm}"][...])
                THWC = _as_THW_C(x, field=nm, T_hint=T_hint).astype(np.float32)
                T,H,W,C = THWC.shape
                chunks.append(THWC.reshape(T, H*W, C))
            for nm in sorted(u_fields_t1):
                x = np.asarray(h5[f"t1_fields/{nm}"][...])
                THWC = _as_THW_C(x, field=nm, T_hint=T_hint).astype(np.float32)
                T,H,W,C = THWC.shape
                chunks.append(THWC.reshape(T, H*W, C))
            Cu = int(np.concatenate(chunks, axis=-1).shape[-1])
        Cc = 1  # current_drive

        # TRUE-time stats
        time_step = int(self.time_step if self.time_step is not None else 1)
        max_diff  = None if (self.max_time_diff is None) else int(self.max_time_diff)
        st_m, st_s, dt_m, dt_s = _stream_true_time_stats(train_files, time_step, max_diff)

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

        t_vals_t  = torch.tensor(t_vals, dtype=self.dtype)
        N = x_fixed.shape[0]

        # dummies only signal channel counts to the trainer
        dummy_u = torch.zeros((1, 2, N, Cu), dtype=self.dtype)
        dummy_c = torch.zeros((1, 2, N, Cc), dtype=self.dtype)
        
        # Store field names for trainer access
        self.u_fields_t0 = u_fields_t0
        self.u_fields_t1 = u_fields_t1

        return {
            "train": {"u": dummy_u, "c": dummy_c, "x": x_fixed, "t": t_vals_t},
            "val":   {"u": dummy_u, "c": dummy_c, "x": x_fixed, "t": t_vals_t},
            "test":  {"u": dummy_u, "c": dummy_c, "x": x_fixed, "t": t_vals_t},
            "_stream_meta": {
                "split_dirs": split_dirs,
                "u_fields_t0": u_fields_t0,
                "u_fields_t1": u_fields_t1,
                "time_step": time_step,
                "max_time_diff": max_diff,
            }
        }

    def _split_and_normalize_sequential_data(self, raw: Dict, is_variable_coords: bool) -> Dict:
        # Already normalized via streaming in _load_raw_sequential_data; just set t_values
        self.t_values = raw["train"]["t"].cpu().numpy()
        return raw

    def create_sequential_data_loaders(self, data_splits: Dict, is_variable_coords: bool, **kwargs):
        meta = data_splits["_stream_meta"]

        def mk(split):
            return WellH5PairIterableDEMO(
                split_dir=meta["split_dirs"][split],
                stats=self.stats,
                time_step=meta["time_step"],
                max_time_diff=meta["max_time_diff"],
                u_fields_t0=meta["u_fields_t0"],
                u_fields_t1=meta["u_fields_t1"],
                cache_samples=getattr(self.dataset_config, "stream_cache_samples", 1),
            )

        def make_loader(dataset):
            return DataLoader(
                dataset,
                batch_size=self.dataset_config.batch_size,
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
        # ensure t_values for plotting/labels
        self.t_values = data_splits["train"]["t"].cpu().numpy()
        # small hint to trainer AR path
        self.runtime_hints = {"use_trainer_autoreg": True}
        return loaders
