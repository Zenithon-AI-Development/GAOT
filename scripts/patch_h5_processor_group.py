"""Patch generic_h5_sequential_data_processor.py for HDF5 group support (t0_fields). Run from repo root."""
import sys
f = "src/datasets/generic_h5_sequential_data_processor.py"
s = open(f).read()
need_write = False
if "_read_u_chunk" not in s:
    insert = '''def _read_u_chunk(f, key, bi, t_slice):
    item = f[key]
    if isinstance(item, h5py.Group):
        names = sorted(item.keys())
        parts = []
        for n in names:
            d = item[n]
            if len(d.shape) == 5:
                x = np.array(d[bi, t_slice], copy=False)
            elif len(d.shape) == 4:
                # (B,T,W,H): take first batch index then time slice
                x = np.array(d[0, t_slice], copy=False)
            else:
                x = np.array(d[t_slice], copy=False)
            parts.append(x)
        out = np.stack(parts, axis=-1)
        if out.ndim == 5 and out.shape[0] == 1:
            out = out.squeeze(0)
        return out
    d = item
    if len(d.shape) == 5:
        x = np.array(d[bi, t_slice], copy=False)
        return x.squeeze(0) if x.ndim == 5 and x.shape[0] == 1 else x
    return np.array(d[t_slice], copy=False)

'''
    idx = s.find("def _stream_mean_std_over_train")
    if idx != -1:
        s = s[:idx] + insert + s[idx:]
        need_write = True
old = """    for fp in files:
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
            total   += X.shape[0]"""
new = """    for fp in files:
        with h5py.File(fp, "r") as f:
            item = f[dataset_key]
            if isinstance(item, h5py.Group):
                names = sorted(item.keys())
                first = item[names[0]]
                T = int(first.shape[1])  # time axis for (B,T,W,H) or (B,T,W,H,C)
                bi = 0
            else:
                T = int(item.shape[1] if len(item.shape) == 5 else item.shape[0])
                bi = 0 if len(item.shape) == 5 else None
            for t0 in range(0, T, chunk_T):
                t1 = min(t0 + chunk_T, T)
                x = _read_u_chunk(f, dataset_key, bi, slice(t0, t1))
                X = x.reshape(-1, x.shape[-1]).astype(np.float32)
                if c_sum is None:
                    c_sum   = np.zeros(X.shape[-1], dtype=np.float64)
                    c_sumsq = np.zeros(X.shape[-1], dtype=np.float64)
                c_sum   += X.sum(axis=0, dtype=np.float64)
                c_sumsq += (X.astype(np.float64) ** 2).sum(axis=0)
                total   += X.shape[0]"""
if old in s:
    s = s.replace(old, new, 1)
    need_write = True

# Patch _stream_pair_stats_over_train for group support (remote: d = f[dataset_key], arr = d[0] or d)
old2 = """        with h5py.File(fp, "r") as f:
            d = f[dataset_key]
            # collapse to (T,W,H,C)
            if len(d.shape) == 5:
                arr = d[0]   # (T,W,H,C)
            elif len(d.shape) == 4:
                arr = d      # (T,W,H,C)
            else:
                raise RuntimeError(f"Unsupported rank {len(d.shape)} in {fp}:{dataset_key}")

            T = int(arr.shape[0])"""
new2 = """        with h5py.File(fp, "r") as f:
            item = f[dataset_key]
            if isinstance(item, h5py.Group):
                first = item[sorted(item.keys())[0]]
                # Time is axis 1 for both (B,T,W,H) and (B,T,W,H,C)
                T = int(first.shape[1])
                bi = 0
            else:
                d = item
                if len(d.shape) == 5:
                    T = int(d.shape[1])
                    bi = 0
                elif len(d.shape) == 4:
                    T = int(d.shape[0])
                    bi = None
                else:
                    raise RuntimeError(f"Unsupported rank {len(d.shape)} in {fp}:{dataset_key}")"""
if old2 in s:
    s = s.replace(old2, new2, 1)
# Replace array read in _stream_pair_stats (remote uses arr[t0:t1])
s = s.replace(
    "x = np.array(arr[t0:t1], copy=False).astype(np.float32)  # (Tc,W,H,C)",
    "x = _read_u_chunk(f, dataset_key, bi, slice(t0, t1)).astype(np.float32)  # (Tc,W,H,C)",
    1,
)
# If _read_u_chunk already exists but lacks 4D/squeeze (from earlier patch), fix the Group branch
old_group = """        for n in names:
            d = item[n]
            if len(d.shape) == 5:
                x = np.array(d[bi, t_slice], copy=False)
            else:
                x = np.array(d[t_slice], copy=False)
            parts.append(x)
        return np.stack(parts, axis=-1)"""
new_group = """        for n in names:
            d = item[n]
            if len(d.shape) == 5:
                x = np.array(d[bi, t_slice], copy=False)
            elif len(d.shape) == 4:
                x = np.array(d[0, t_slice], copy=False)
            else:
                x = np.array(d[t_slice], copy=False)
            parts.append(x)
        out = np.stack(parts, axis=-1)
        if out.ndim == 5 and out.shape[0] == 1:
            out = out.squeeze(0)
        return out"""
if old_group in s and new_group not in s:
    s = s.replace(old_group, new_group, 1)
open(f, "w").write(s)
