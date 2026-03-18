"""Patch generic_h5_pair_dataset.py for HDF5 group support (t0_fields). Run from repo root."""
f = "src/datasets/generic_h5_pair_dataset.py"
s = open(f).read()

# 1. H5TrajIndex
old1 = """                d = f[dataset_key]
                if len(d.shape) == 5:
                    B = int(d.shape[0])
                elif len(d.shape) == 4:
                    B = 1
                else:
                    continue"""
new1 = """                item = f[dataset_key]
                if isinstance(item, h5py.Group):
                    names = sorted(item.keys())
                    d = item[names[0]]
                    B = int(d.shape[0]) if len(d.shape) == 5 else 1
                else:
                    d = item
                    if len(d.shape) == 5:
                        B = int(d.shape[0])
                    elif len(d.shape) == 4:
                        B = 1
                    else:
                        continue"""
if old1 in s:
    s = s.replace(old1, new1, 1)

# 2. H5LazyArray probe W,H,C
old2 = """        with h5py.File(self.files[self.index[0][0]], "r") as f0:
            d0 = f0[self.key]
            if len(d0.shape) == 5: W, H, C = int(d0.shape[2]), int(d0.shape[3]), int(d0.shape[4])
            else:                   W, H, C = int(d0.shape[1]), int(d0.shape[2]), int(d0.shape[3])"""
new2 = """        with h5py.File(self.files[self.index[0][0]], "r") as f0:
            item0 = f0[self.key]
            if isinstance(item0, h5py.Group):
                names = sorted(item0.keys())
                d0 = item0[names[0]]
                W = int(d0.shape[2]); H = int(d0.shape[3]); C = len(names)
            else:
                d0 = item0
                if len(d0.shape) == 5: W, H, C = int(d0.shape[2]), int(d0.shape[3]), int(d0.shape[4])
                else:                   W, H, C = int(d0.shape[1]), int(d0.shape[2]), int(d0.shape[3])"""
if old2 in s:
    s = s.replace(old2, new2, 1)

# 3. H5LazyArray _T_list loop
old3 = """            with h5py.File(self.files[fi], "r") as f:
                d = f[self.key]
                T = int(d.shape[1] if len(d.shape) == 5 else d.shape[0])"""
new3 = """            with h5py.File(self.files[fi], "r") as f:
                item = f[self.key]
                if isinstance(item, h5py.Group):
                    d = item[sorted(item.keys())[0]]
                else:
                    d = item
                T = int(d.shape[1] if len(d.shape) >= 4 else d.shape[0])"""
if old3 in s:
    s = s.replace(old3, new3, 1)

# 4. H5LazyArray _read_frame
old4 = """    def _read_frame(self, fi: int, bi: int, t: int) -> torch.Tensor:
        with h5py.File(self.files[fi], "r") as f:
            d = f[self.key]
            if len(d.shape) == 5:   # (B,T,W,H,C)
                x = np.array(d[bi, t], copy=False)
            else:                   # (T,W,H,C)
                x = np.array(d[t], copy=False)
            W, H, C = x.shape"""
new4 = """    def _read_frame(self, fi: int, bi: int, t: int) -> torch.Tensor:
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
            W, H, C = x.shape"""
if old4 in s:
    s = s.replace(old4, new4, 1)

# 5. GenericH5PairIterable _read_frame
old5 = """    def _read_frame(self, fp: str, bi: int, t: int, key: str) -> torch.Tensor:
        with h5py.File(fp, "r") as f:
            d = f[key]
            if len(d.shape) == 5:
                x = np.array(d[bi, t], copy=False)  # (W,H,C)
            elif len(d.shape) == 4:
                x = np.array(d[t], copy=False)
            else:
                raise RuntimeError(f"Unsupported rank {len(d.shape)} in {fp}:{key}")
            W, H, C = x.shape"""
new5 = """    def _read_frame(self, fp: str, bi: int, t: int, key: str) -> torch.Tensor:
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
            W, H, C = x.shape"""
if old5 in s:
    s = s.replace(old5, new5, 1)

# 6. _traj_T
old6 = """    def _traj_T(self, fp: str) -> int:
        with h5py.File(fp, "r") as f:
            d = f[self.dataset_key]
            return int(d.shape[1] if len(d.shape) == 5 else d.shape[0])"""
new6 = """    def _traj_T(self, fp: str) -> int:
        with h5py.File(fp, "r") as f:
            item = f[self.dataset_key]
            if isinstance(item, h5py.Group):
                d = item[sorted(item.keys())[0]]
            else:
                d = item
            return int(d.shape[1] if len(d.shape) >= 4 else d.shape[0])"""
if old6 in s:
    s = s.replace(old6, new6, 1)

open(f, "w").write(s)
