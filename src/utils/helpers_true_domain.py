# utils/helpers_true_domain.py
import h5py, numpy as np, glob, os

def true_domain_from_file(h5: h5py.File):
    r = np.asarray(h5["dimensions/r_coords"][...], dtype=np.float64)
    z = np.asarray(h5["dimensions/z_coords"][...], dtype=np.float64)
    rmin, rmax = float(np.nanmin(r)), float(np.nanmax(r))
    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    return (rmin, zmin), (rmax, zmax)

def true_domain_from_dir(split_dir: str):
    rmin = zmin = +float("inf")
    rmax = zmax = -float("inf")
    for fp in glob.glob(os.path.join(split_dir, "*.hdf5")):
        with h5py.File(fp, "r") as h5:
            (rmn, zmn), (rmx, zmx) = true_domain_from_file(h5)
        rmin, zmin = min(rmin, rmn), min(zmin, zmn)
        rmax, zmax = max(rmax, rmx), max(zmax, zmx)
    return (rmin, zmin), (rmax, zmax)

