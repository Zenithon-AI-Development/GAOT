"""Utility functions for reading the datasets."""
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Sequence, NamedTuple, Literal
from copy import deepcopy

@dataclass
class Metadata:
  periodic: bool
  group_u: str
  group_c: str
  group_x: str
  type: Literal['poseidon', 'rigno', 'gaot']
  fix_x: bool
  domain_x: tuple[Sequence[int], Sequence[int]]
  domain_t: tuple[int, int]
  active_variables: Sequence[int]  # Index of variables in input/output
  chunked_variables: Sequence[int]  # Index of variable groups
  num_variable_chunks: int  # Number of variable chunks
  signed: dict[str, Union[bool, Sequence[bool]]]
  names: dict[str, Sequence[str]]
  global_mean: Sequence[float]
  global_std: Sequence[float]

"""
Reference: https://github.com/camlab-ethz/rigno/blob/main/rigno/dataset.py
"""

ACTIVE_VARS_NS = [0, 1]
ACTIVE_VARS_CE = [0, 1, 2, 3]
ACTIVE_VARS_GCE = [0, 1, 2, 3, 5]
ACTIVE_VARS_RD = [0]
ACTIVE_VARS_WE = [0]
ACTIVE_VARS_PE = [0]

CHUNKED_VARS_NS = [0, 0]
CHUNKED_VARS_CE = [0, 1, 1, 2, 3]
CHUNKED_VARS_GCE = [0, 1, 1, 2, 3, 4]
CHUNKED_VARS_RD = [0]
CHUNKED_VARS_WE = [0]
CHUNKED_VARS_PE = [0]

SIGNED_NS = {'u': [True, True], 'c': None}
SIGNED_CE = {'u': [False, True, True, False, False], 'c': None}
SIGNED_GCE = {'u': [False, True, True, False, False, False], 'c': None}
SIGNED_RD = {'u': [True], 'c': None}
SIGNED_WE = {'u': [True], 'c': [False]}
SIGNED_PE = {'u': [True], 'c': [True]}

NAMES_NS = {'u': ['$v_x$', '$v_y$'], 'c': None}
NAMES_CE = {'u': ['$\\rho$', '$v_x$', '$v_y$', '$p$'], 'c': None}
NAMES_GCE = {'u': ['$\\rho$', '$v_x$', '$v_y$', '$p$', 'E', '$\\phi$'], 'c': None}
NAMES_RD = {'u': ['$u$'], 'c': None}
NAMES_WE = {'u': ['$u$'], 'c': ['$c$']}
NAMES_PE = {'u': ['$u$'], 'c': ['$f$']}

DATASET_METADATA = {
  'benchmarking/maglif': Metadata(
    periodic=False,
    group_u=None,
    group_c=None,
    group_x='dimensions',
    type='gaot',
    domain_x=([0,], [3.5e-05,]), # r (radial coordinate in meters)
    domain_t=None,
    fix_x=True,
    active_variables=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    chunked_variables=[0, 0, 0, 0, 0, 0, 0, 0, 0],
    num_variable_chunks=1,
    signed={'u': [False, False, False, False, True, False, False, False, False], 'c': None},
    names={'u': ['Rho', 'mat(1)%zeff', 'T_elec', 'T_ion', 'Vel', 'P_ion', 'P_elec', 'n_elec', 'bmag'], 'c': None},
    global_mean=[0, 0, 0, 0, 0, 0, 0, 0, 0],
    global_std=[1, 1, 1, 1, 1, 1, 1, 1, 1],
  ),
  'benchmarking/demo': Metadata(
    periodic=False,
    group_u=None, # t0_fields/density, t0_fields/pressure, t1_fields/bfield, t1_fields/velocity
    group_c=None,
    group_x='dimensions',
    type='gaot',
    domain_x=([0.0078125, 0.117188], [0.0078125, 0.117188]), # r, z
    domain_t=None,
    fix_x=True,
    active_variables=[0, 1, 2, 3, 4, 5],
    chunked_variables=[0, 0, 0, 0, 0, 0],
    num_variable_chunks=1,
    signed={'u': [False, False, True, True, True, True], 'c': True},
    names={'u': ['density', 'pressure', 'bfieldz', 'bfieldp', 'velx', 'vely'], 'c': 'current'},
    global_mean=[0.386953, 1.82954e+13, 3.65433e+12, 824881, 1.63379e+06, -305902],
    global_std=[1.07441, 4.70374e+13, 1.28989e+13, 2.57453e+06, 4.98058e+06, 815341],
  ),
  'benchmarking/demo_nc': Metadata(
    periodic=False,
    group_u='u', # t0_fields/density, t0_fields/pressure, t1_fields/bfield, t1_fields/velocity
    group_c='c',
    group_x='x',
    type='gaot',
    domain_x=([0.0078125, 0.117188], [0.0078125, 0.117188]), # r, z
    domain_t=(0, 1.40006e-07),
    fix_x=True,
    active_variables=[0, 1, 2, 3, 4, 5],
    chunked_variables=[0, 0, 0, 0, 0, 0],
    num_variable_chunks=1,
    signed={'u': [False, False, True, True, True, True], 'c': True},
    names={'u': ['density', 'pressure', 'bfieldz', 'bfieldp', 'velx', 'vely'], 'c': 'current'},
    global_mean=[0.386953, 1.82954e+13, 3.65433e+12, 824881, 1.63379e+06, -305902],
    global_std=[1.07441, 4.70374e+13, 1.28989e+13, 2.57453e+06, 4.98058e+06, 815341],
  ),
  # ---- Helmholtz staircase (2D acoustic) ----
  'benchmarking/test': Metadata(
    periodic=False,
    group_u='t0_fields',
    group_c=None,
    group_x='dimensions',
    type='gaot',
    # spatial domain (x1 horizontal, x2 vertical) from dataset description:
    # x1 ∈ [-8.0, 8.0], x2 ∈ [-0.5, 3.5]
    domain_x=([0, 1], [0, 1]), ### CHECK
    # NOTE: time range depends on omega for each simulation: t in [0, 2π/ω].
    # We set a generic placeholder here; if all files in a single run share the same ω, set the exact t_max.
    # Use (0.0, 2*pi) here as a generic placeholder; adjust to your per-file ω if needed.
    domain_t=None,
    fix_x=True,    # mask is stationary, structured grid (1024 x 256 image)
    # channel order chosen from stats.yaml keys: mask, pressure_im, pressure_re
    active_variables=[0, 1, 2, 3, 4, 5],
    chunked_variables=[0, 0, 0, 0, 0, 0],
    num_variable_chunks=1,
    # mask (non-negative), pressures signed (real/imag can be negative)
    signed={'u': [False, True, True, True, True, True, True], 'c': None},
    names={'u': ['density', 'pressure', 'velx', 'vely', 'magz', 'magp'], 'c': None},
    global_mean=[0, 0, 0, 0, 0, 0],
    global_std=[1, 1, 1, 1, 1, 1], #### DUMMY
  ),
  # ---- MHD_64 (3D MHD dataset) ----
  'benchmarking/mhd64': Metadata(
    periodic=True,
    group_u=None,   # variables in t0_fields (density) and t1_fields (magnetic_field, velocity)
    group_c=None,
    group_x='dimensions',
    type='gaot',
    # domain_x: use file dimensions arrays (64 points) but metadata can stay as pixel extents
    domain_x=([0.0, 0.0, 0.0], [63.0, 63.0, 63.0]),
    domain_t=None,  # per-file dimensions/time (files hold t)
    fix_x=True,
    # channels: density, magnetic_x,y,z, velocity_x,y,z
    active_variables=[0, 1, 2, 3, 4, 5, 6],
    chunked_variables=[0, 0, 0, 0, 0, 0, 0],
    num_variable_chunks=1,
    signed={'u': [False, True, True, True, True, True, True], 'c': None},
    names={'u': [
        't0_fields/density',
        't1_fields/magnetic_field_0', 't1_fields/magnetic_field_1', 't1_fields/magnetic_field_2',
        't1_fields/velocity'
    ], 'c': None},
    global_mean=[
        1.0015E+00,
        5.3329E-01, 8.4965E-10, 1.9938E-08,
        9.6152E-03, 6.7833E-02, 1.9315E-02
    ],
    global_std=[
        8.9228E-01,
        5.2867E-01, 3.1709E-01, 3.2008E-01,
        4.8515E-01, 4.3814E-01, 4.7858E-01
    ],
  ),
  # ---- Rayleigh–Taylor instability (3D) ----
  'benchmarking/rti': Metadata(
    periodic=False,
    group_u=None,
    group_c=None,
    group_x='dimensions',
    type='gaot',
    domain_x=([0.0, 0.0, 0.0], [127.0, 127.0, 127.0]),  # 128^3 -> indices 0..127
    domain_t=None,
    fix_x=True,
    # t0_fields/density, t1_fields/velocity (vector 3 comps)
    active_variables=[0, 1, 2, 3],
    chunked_variables=[0, 0, 0, 0],
    num_variable_chunks=1,
    signed={'u': [False, True, True, True], 'c': None},
    names={'u': [
        't0_fields/density',
        't1_fields/velocity'
    ], 'c': None},
    global_mean=[7.7363E-01, -6.0036E-06, -1.6880E-05, -4.4674E-06],
    global_std=[2.6884E-01, 8.2252E-03, 8.1858E-03, 1.3937E-02],
  ),
  # ---- Helmholtz staircase (2D acoustic) ----
  'benchmarking/zpinch': Metadata(
    periodic=False,
    group_u='t0_fields',
    group_c=None,
    group_x='dimensions',
    type='gaot',
    # spatial domain (x1 horizontal, x2 vertical) from dataset description:
    # x1 ∈ [-8.0, 8.0], x2 ∈ [-0.5, 3.5]
    domain_x=([0, 1], [0, 1]), ### CHECK
    # NOTE: time range depends on omega for each simulation: t in [0, 2π/ω].
    # We set a generic placeholder here; if all files in a single run share the same ω, set the exact t_max.
    # Use (0.0, 2*pi) here as a generic placeholder; adjust to your per-file ω if needed.
    domain_t=None,
    fix_x=True,    # mask is stationary, structured grid (1024 x 256 image)
    # channel order chosen from stats.yaml keys: mask, pressure_im, pressure_re
    active_variables=[0, 1, 2, 3, 4, 5],
    chunked_variables=[0, 0, 0, 0, 0, 0],
    num_variable_chunks=1,
    # mask (non-negative), pressures signed (real/imag can be negative)
    signed={'u': [False, True, True, True, True, True], 'c': None},
    names={'u': ['density', 'pressure', 'velx', 'vely', 'magz', 'magp'], 'c': None},
    global_mean=[0, 0, 0, 0, 0, 0],
    global_std=[1, 1, 1, 1, 1, 1], #### DUMMY
  ),
  # ---- Helmholtz staircase (2D acoustic) ----
  'benchmarking/hs': Metadata(
    periodic=False,
    group_u='t0_fields',
    group_c=None,
    group_x='dimensions',
    type='gaot',
    # spatial domain (x1 horizontal, x2 vertical) from dataset description:
    # x1 ∈ [-8.0, 8.0], x2 ∈ [-0.5, 3.5]
    domain_x=([-8.0, -0.5], [8.0, 3.5]),
    # NOTE: time range depends on omega for each simulation: t in [0, 2π/ω].
    # We set a generic placeholder here; if all files in a single run share the same ω, set the exact t_max.
    # Use (0.0, 2*pi) here as a generic placeholder; adjust to your per-file ω if needed.
    domain_t=None,
    fix_x=True,    # mask is stationary, structured grid (1024 x 256 image)
    # channel order chosen from stats.yaml keys: mask, pressure_im, pressure_re
    active_variables=[0, 1],
    chunked_variables=[0, 0],
    num_variable_chunks=1,
    # mask (non-negative), pressures signed (real/imag can be negative)
    signed={'u': [True, True], 'c': None},
    names={'u': ['pressure_im', 'pressure_re'], 'c': None},
    global_mean=[1.0585E-03, 7.4248E-05],
    global_std=[1.9992E-01, 1.9963E-01],
  ),
  # ---- Helmholtz staircase (2D acoustic) ----
  'benchmarking/hs_nc': Metadata(
    periodic=False,
    group_u='u',
    group_c=None,
    group_x='x',
    type='gaot',
    # spatial domain (x1 horizontal, x2 vertical) from dataset description:
    # x1 ∈ [-8.0, 8.0], x2 ∈ [-0.5, 3.5]
    domain_x=([-8.0, -0.5], [8.0, 3.5]),
    # NOTE: time range depends on omega for each simulation: t in [0, 2π/ω].
    # We set a generic placeholder here; if all files in a single run share the same ω, set the exact t_max.
    # Use (0.0, 2*pi) here as a generic placeholder; adjust to your per-file ω if needed.
    domain_t=(0, 2*3.141592653589793),
    fix_x=True,    # mask is stationary, structured grid (1024 x 256 image)
    # channel order chosen from stats.yaml keys: mask, pressure_im, pressure_re
    active_variables=[0, 1, 2],
    chunked_variables=[0, 0, 0],
    num_variable_chunks=1,
    # mask (non-negative), pressures signed (real/imag can be negative)
    signed={'u': [False, True, True], 'c': None},
    names={'u': ['mask', 'pressure_im', 'pressure_re'], 'c': None},
    global_mean=[6.4869E-02, 1.0585E-03, 7.4248E-05],
    global_std=[2.4629E-01, 1.9992E-01, 1.9963E-01],
  ),
  # ---- Turbulent Radiative Layer - 2D (NetCDF export) ----
  'benchmarking/trl2d_nc': Metadata(
      periodic=False,                 # periodic in x, open in y; keep False like your existing entry
      group_u='u',                    # matches the variable name you wrote in the .nc
      group_c=None,
      group_x='x',                    # matches the variable name you wrote in the .nc
      type='gaot',                    # netcdf “gaot-style” (u/x live at top level)
      # It’s best to set domain_x to the actual min/max of your x variable (see snippet below).
      # If you don’t want to compute now, you can reuse your previous box; but computing is safer.
      domain_x=[[-0.5, -1.0], [0.5, 2.0]],  # TEMP: replace with true mins/maxes from the file
      domain_t=(0, 159.7033),
      fix_x=True,                     # fixed grid for TRL-2D
      active_variables=[0, 1, 2, 3],  # density, pressure, vx, vy
      chunked_variables=[0, 0, 0, 0],
      num_variable_chunks=1,
      signed={'u': [False, False, True, True], 'c': None},
      names={'u': ['density', 'pressure', 'velocity_x', 'velocity_y'], 'c': None},
      # These are ignored since your config sets "use_metadata_stats": false.
      global_mean=[3.4847E+01, 9.4475E-01, 6.1707E-03, -2.4651E-02],
      global_std=[4.4284E+01, 6.0970E-02, 4.1764E-02, 4.0095E-02],
  ),
  # ---- Turbulent Radiative Layer - 2D (NetCDF export) ----
  'benchmarking/trl2d_nc_halfres': Metadata(
      periodic=False,                 # periodic in x, open in y; keep False like your existing entry
      group_u='u',                    # matches the variable name you wrote in the .nc
      group_c=None,
      group_x='x',                    # matches the variable name you wrote in the .nc
      type='gaot',                    # netcdf “gaot-style” (u/x live at top level)
      # It’s best to set domain_x to the actual min/max of your x variable (see snippet below).
      # If you don’t want to compute now, you can reuse your previous box; but computing is safer.
      domain_x=[[-0.5, -1.0], [0.5, 2.0]],  # TEMP: replace with true mins/maxes from the file
      domain_t=(0, 159.7033),
      fix_x=True,                     # fixed grid for TRL-2D
      active_variables=[0, 1, 2, 3],  # density, pressure, vx, vy
      chunked_variables=[0, 0, 0, 0],
      num_variable_chunks=1,
      signed={'u': [False, False, True, True], 'c': None},
      names={'u': ['density', 'pressure', 'velocity_x', 'velocity_y'], 'c': None},
      # These are ignored since your config sets "use_metadata_stats": false.
      global_mean=[3.4847E+01, 9.4475E-01, 6.1707E-03, -2.4651E-02],
      global_std=[4.4284E+01, 6.0970E-02, 4.1764E-02, 4.0095E-02],
  ),
  # ---- Turbulent Radiative Layer - 2D ----
  'benchmarking/trl2d': Metadata(
    periodic=False,  # periodic in x, open in y
    group_u=None,
    group_c=None,
    group_x='dimensions',
    type='gaot',
    domain_x=[[-0.5, -1.0], [0.5, 2.0]],
    domain_t=None,  # per-file time
    fix_x=True,
    # t0_fields/density, t0_fields/pressure, t1_fields/velocity (2 comps)
    active_variables=[0, 1, 2, 3],
    chunked_variables=[0, 0, 0, 0],
    num_variable_chunks=1,
    signed={'u': [False, False, True, True], 'c': None},
    names={'u': [
        'density',
        'pressure',
        'velocity_x',
        'velocity_y'
    ], 'c': None},
    global_mean=[3.4847E+01, 9.4475E-01, 6.1707E-03, -2.4651E-02],
    global_std=[4.4284E+01, 6.0970E-02, 4.1764E-02, 4.0095E-02],
  ),
  # ---- Shear Flow (2D) ----
  'benchmarking/sf': Metadata(
    periodic=True,   # description: periodic BCs
    group_u=None,    # variables live under t0_fields and t1_fields -> use full paths in names['u']
    group_c=None,
    group_x='dimensions',
    type='gaot',
    domain_x=([0.0, -1.0], [1.0, 1.0]),  # as earlier
    domain_t=None,   # use per-file dimensions/time
    fix_x=True,
    # variable ordering: t0_fields/pressure, t0_fields/tracer, t1_fields/velocity (vector last dim=2)
    active_variables=[0, 1, 2, 3],
    chunked_variables=[0, 0, 0, 0],
    num_variable_chunks=1,
    signed={'u': [True, False, True, True], 'c': None},
    names={'u': ['t0_fields/pressure', 't0_fields/tracer', 't1_fields/velocity'], 'c': None},
    # Means/stds: pressure, tracer, velocity_x, velocity_y
    global_mean=[-1.2310E-09, 2.3845E-03, 2.3845E-03, -1.7308E-07],
    global_std=[8.2052E-02, 3.5828E-01, 4.0194E-01, 1.0421E-01],
  ),
  # steady Euler
  'compressible_flow/naca2412': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='gaot',
    domain_x=([-1, -1.5], [2.5, 2]),
    domain_t=None,
    fix_x=False,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [False], 'c': [False, False, False]},
    names={'u': ['$\\rho$'], 'c': ['Mach', 'AOA', 'SDF']},
    global_mean=[0.96086993],
    global_std=[0.18490477],
  ),
  'compressible_flow/naca0012': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='gaot',
    domain_x=([-1, -1.5], [2.5, 2]),
    domain_t=None,
    fix_x=False,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [False], 'c': [False, False, False]},
    names={'u': ['$\\rho$'], 'c': ['Mach', 'AOA', 'SDF']},
    global_mean=[0.96999054],
    global_std=[0.17089098],
  ),
  'compressible_flow/rae2822': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='gaot',
    domain_x=([-1, -1.5], [2.5, 2]),
    domain_t=None,
    fix_x=False,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [False], 'c': [False, False, False]},
    names={'u': ['$\\rho$'], 'c': ['Mach', 'AOA', 'SDF']},
    global_mean=[0.96746538],
    global_std=[0.17268029],
  ),
  'compressible_flow/bluff': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='gaot',
    domain_x=([-9.0, -9.0], [9.0, 9.0]),
    domain_t=None,
    fix_x=False,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [False], 'c': [False, False, False]},
    names={'u': ['$\\rho$'], 'c': ['Mach', 'AOA', 'SDF']},
    global_mean=[0.95306754],
    global_std=[0.3144897],
  ),
 
  # compressible_flow: [density, velocity, velocity, pressure, energy]
  'compressible_flow/CE-Gauss': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_CE,
    chunked_variables=CHUNKED_VARS_CE,
    num_variable_chunks=len(set(CHUNKED_VARS_CE)),
    signed=SIGNED_CE,
    names=NAMES_CE,
    global_mean=[0.80, 0., 0., 2.513],
    global_std=[0.31, 0.391, 0.356, 0.185],
  ),
  'compressible_flow/CE-RP': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_CE,
    chunked_variables=CHUNKED_VARS_CE,
    num_variable_chunks=len(set(CHUNKED_VARS_CE)),
    signed=SIGNED_CE,
    names=NAMES_CE,
    global_mean=[0.80, 0., 0., 0.215],
    global_std=[0.31, 0.391, 0.356, 0.185],
  ),
  'compressible_flow/CE-CRP': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='gaot',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_CE,
    chunked_variables=CHUNKED_VARS_CE,
    num_variable_chunks=len(set(CHUNKED_VARS_CE)),
    signed=SIGNED_CE,
    names=NAMES_CE,
    global_mean=[0.80, 0., 0., 0.553],
    global_std=[0.31, 0.391, 0.356, 0.185],
  ),
  'compressible_flow/CE-KH': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='gaot',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_CE,
    chunked_variables=CHUNKED_VARS_CE,
    num_variable_chunks=len(set(CHUNKED_VARS_CE)),
    signed=SIGNED_CE,
    names=NAMES_CE,
    global_mean=[0.80, 0., 0., 1.0],
    global_std=[0.31, 0.391, 0.356, 0.185],
  ),
  'compressible_flow/CE-RPUI': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='gaot',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_CE,
    chunked_variables=CHUNKED_VARS_CE,
    num_variable_chunks=len(set(CHUNKED_VARS_CE)),
    signed=SIGNED_CE,
    names=NAMES_CE,
    global_mean=[0.80, 0., 0., 1.33],
    global_std=[0.31, 0.391, 0.356, 0.185],
  ),
 
  # incompressible_fluids: [velocity, velocity]
  'incompressible_fluids/NS-Gauss': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  'incompressible_fluids/NS-PwC': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  'incompressible_fluids/NS-SL': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  'incompressible_fluids/NS-SVS': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  'incompressible_fluids/NS-Sines': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='gaot',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
 
  # elliptic PDEs
  'elliptic_pdes/Elasticity': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=None,
    fix_x=False,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [False], 'c': [False]},
    names={'u': ['$\\sigma$'], 'c': ['$d$']},
    global_mean=[187.477],
    global_std=[127.046],
  ),
  'elliptic_pdes/Poisson-C-Sines': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='rigno',
    domain_x=([-.5, -.5], [1.5, 1.5]),
    domain_t=None,
    fix_x=True,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [True], 'c': [True]},
    names={'u': ['$u$'], 'c': ['$f$']},
    global_mean=[0.],
    global_std=[0.00064911455],
  ),
  'elliptic_pdes/Poisson-Gauss': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=None,
    fix_x=True,
    active_variables=ACTIVE_VARS_PE,
    chunked_variables=CHUNKED_VARS_PE,
    num_variable_chunks=len(set(CHUNKED_VARS_PE)),
    signed=SIGNED_PE,
    names=NAMES_PE,
    global_mean=[0.0005603458434937093],
    global_std=[0.02401226126952699],
  ),

  # Parabolic PDEs
  'parabolic_pdes/Heat-L-Sines': Metadata(
    periodic=False,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0., 0.], [1., 1.]),
    domain_t=(0, 0.002),
    fix_x=True,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [True], 'c': None},
    names={'u': ['$u$'], 'c': None},
    global_mean=[-0.009399102],
    global_std=[0.020079814],
  ),
  'parabolic_pdes/ACE': Metadata(
    periodic=False,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 0.0002),
    fix_x=True,
    active_variables=ACTIVE_VARS_RD,
    chunked_variables=CHUNKED_VARS_RD,
    num_variable_chunks=len(set(CHUNKED_VARS_RD)),
    signed=SIGNED_RD,
    names=NAMES_RD,
    global_mean=[0.002484262],
    global_std=[0.65351176],
  ),
  
  # Hyperbolic PDEs
  'hyperbolic_pdes/Wave-C-Sines': Metadata(
    periodic=False,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([-.5, -.5], [1.5, 1.5]),
    domain_t=(0, 0.1),
    fix_x=True,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [True], 'c': None},
    names={'u': ['$u$'], 'c': None},
    global_mean=[0.],
    global_std=[0.011314605],
  ),
  'hyperbolic_pdes/Wave-Layer': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_WE,
    chunked_variables=CHUNKED_VARS_WE,
    num_variable_chunks=len(set(CHUNKED_VARS_WE)),
    signed=SIGNED_WE,
    names=NAMES_WE,
    global_mean=[0.03467443221585092],
    global_std=[0.10442421752963911],
  ),
  'hyperbolic_pdes/Wave-Gauss': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_WE,
    chunked_variables=CHUNKED_VARS_WE,
    num_variable_chunks=len(set(CHUNKED_VARS_WE)),
    signed=SIGNED_WE,
    names=NAMES_WE,
    global_mean=[0.0334376316],
    global_std=[0.1171879068],
  ),
  'hyperbolic_pdes/Wave-L-Sines': Metadata(
    periodic=False,
    group_u='u',
    group_c=None,
    group_x='x',
    type='gaot',
    domain_x=([0.5, 0.], [1.5, 1.]),
    domain_t=(0, 0.1),
    fix_x=True,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [True], 'c': None},
    names={'u': ['$u$'], 'c': None},
    global_mean=[0.],
    global_std=[0.01080257],
  ),

}

