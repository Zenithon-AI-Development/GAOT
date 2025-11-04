# utils/helpers_true_domain_1d.py
"""
Helper functions for extracting true domain from 1D HDF5 datasets.
"""
import h5py, numpy as np, glob, os

def true_domain_from_file_1d(h5: h5py.File):
    """
    Extract domain bounds from a 1D HDF5 file.
    Returns: ((rmin,), (rmax,)) - tuple of 1-tuples for consistency
    """
    r = np.asarray(h5["dimensions/r_coords"][...], dtype=np.float64)
    rmin, rmax = float(np.nanmin(r)), float(np.nanmax(r))
    return (rmin,), (rmax,)

def true_domain_from_dir_1d(split_dir: str):
    """
    Compute domain bounds across all HDF5 files in a directory.
    Returns: ((rmin,), (rmax,))
    """
    rmin = +float("inf")
    rmax = -float("inf")
    for fp in glob.glob(os.path.join(split_dir, "*.hdf5")):
        with h5py.File(fp, "r") as h5:
            (rmn,), (rmx,) = true_domain_from_file_1d(h5)
        rmin = min(rmin, rmn)
        rmax = max(rmax, rmx)
    return (rmin,), (rmax,)



