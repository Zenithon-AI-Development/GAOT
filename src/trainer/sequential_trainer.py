"""
Sequential Trainer for GAOT.
Handles time-dependent datasets with autoregressive prediction capabilities.
"""
import torch
import numpy as np
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm

from ..core.base_trainer import BaseTrainer
from ..core.trainer_utils import move_to_device, denormalize_data
from ..datasets.sequential_data_processor import SequentialDataProcessor
from ..datasets.graph_builder import GraphBuilder
from ..datasets.data_utils import TestDataset, collate_sequential_batch
from ..model.gaot import GAOT
from ..utils.metrics import compute_batch_errors, compute_final_metric
from ..utils.plotting import plot_estimates, create_sequential_animation, plot_estimates_1d, create_sequential_animation_1d
from ..datasets.well_h5_sequential_data_processor_hs import WellH5SequentialDataProcessor
from ..datasets.well_h5_sequential_data_processor_trl2d import WellH5SequentialDataProcessorTRL2D
from ..datasets.well_h5_sequential_data_processor_demo import WellH5SequentialDataProcessorDEMO
from ..datasets.well_h5_sequential_data_processor_maglif import WellH5SequentialDataProcessorMagLIF
from ..datasets.generic_h5_sequential_data_processor import GenericH5SequentialDataProcessor
from ..datasets.well_h5_sequential_data_processor_multires import WellH5SequentialDataProcessorMultiResDEMO, _MultiResBatchIterableDEMO
from ..utils.timer import IterProfiler

import time
from statistics import mean





class SequentialTrainer(BaseTrainer):
    """
    Sequential trainer for sequential (time-dependent) problems.
    Automatically handles both fixed and variable coordinate modes.
    Supports autoregressive prediction and multiple stepper modes.
    """
    
    def __init__(self, config):
        # Initialize data processor
        self.data_processor = None
        self.graph_builder = None
        
        # Coordinate mode and data info
        self.coord_mode = None  # Will be determined from data
        self.coord_dim = None
        self.latent_tokens_coord = None
        self.coord = None       # For fx mode
        
        # Sequential-specific attributes
        self.stats = None
        self.max_time_diff = None
        self.stepper_mode = None
        self.t_values = None
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        super().__init__(config)

    def _sync(self):
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
    
    def init_dataset(self, dataset_config):
        """Initialize dataset and data loaders for sequential data."""
        print("Initializing sequential dataset...")
        
        if getattr(dataset_config, "backend", "netcdf").lower() == "generic_h5":
            self.data_processor = GenericH5SequentialDataProcessor(
                dataset_config=dataset_config, metadata=self.metadata, dtype=self.dtype
            )
        elif getattr(dataset_config, "backend", "netcdf").lower() == "well_multires":
            # self.data_processor = _MultiResBatchIterableDEMO(
            #     dataset_config=dataset_config, metadata=self.metadata, dtype=self.dtype
            # )
            self.data_processor = WellH5SequentialDataProcessorMultiResDEMO(
                dataset_config=dataset_config, metadata=self.metadata, dtype=self.dtype
            )
        elif getattr(dataset_config, "backend", "netcdf").lower() == "well_maglif":
            self.data_processor = WellH5SequentialDataProcessorMagLIF(
                dataset_config=dataset_config, metadata=self.metadata, dtype=self.dtype
            )
        elif getattr(dataset_config, "backend", "netcdf").lower() == "well":
            self.data_processor = WellH5SequentialDataProcessorDEMO(
                dataset_config=dataset_config, metadata=self.metadata, dtype=self.dtype
            )
            # self.data_processor = WellH5SequentialDataProcessorTRL2D(
            #     dataset_config=dataset_config, metadata=self.metadata, dtype=self.dtype
            # )
        elif getattr(dataset_config, "backend", "netcdf").lower() == "well_hs":
            self.data_processor = WellH5SequentialDataProcessor(
                dataset_config=dataset_config, metadata=self.metadata, dtype=self.dtype
            )
        else:
            self.data_processor = SequentialDataProcessor(
                dataset_config=dataset_config, metadata=self.metadata, dtype=self.dtype
            )

        data_splits, is_variable_coords = self.data_processor.load_and_process_data()

        self.coord_mode = 'vx' if is_variable_coords else 'fx'
        print(f"Detected coordinate mode: {self.coord_mode}")
        
        self.max_time_diff = self.data_processor.max_time_diff
        self.time_step = self.data_processor.time_step
        self.stepper_mode = self.data_processor.stepper_mode
        self.t_values = self.data_processor.t_values
        self.stats = self.data_processor.stats
        # If we're using a streaming HDF5 backend, align metadata stats with loader stats
        _backend = getattr(self.dataset_config, "backend", "netcdf").lower()
        if _backend in ("well", "well_multires", "well_maglif", "generic_h5"):
            if self.stats is not None and "u" in self.stats:
                mu = self.stats["u"]["mean"].view(-1).detach().cpu().numpy()
                sd = self.stats["u"]["std"].view(-1).detach().cpu().numpy()
                # These two are used by utils/metrics.py
                self.metadata.global_mean = mu.tolist()
                self.metadata.global_std  = sd.tolist()
                # (chunk mapping already handled by Metadata.active_variables/chunked_variables)

        
        self.bucket_coords = getattr(self.data_processor, "bucket_coords_scaled", None)  # dict or None

        
        latent_queries = self.data_processor.generate_latent_queries(
            self.model_config.latent_tokens_size
        )
        self.latent_tokens_coord = latent_queries
        
        coord_sample = (data_splits['train']['x'] if is_variable_coords 
                       else data_splits['train']['x'])
        self.coord_dim = coord_sample.shape[-1]
        
        u_sample = data_splits['train']['u']
        c_sample = data_splits['train']['c']
        self.num_output_channels = u_sample.shape[-1]
        
        # Compute input channels: u + time(2) + optional c + optional conditional_norm(-1)
        self.num_input_channels = u_sample.shape[-1] + 2  # u + start_time + time_diff
        if c_sample is not None:
            self.num_input_channels += c_sample.shape[-1]  # add c channels
        
        # Account for conditional normalization
        if getattr(self.model_config, 'use_conditional_norm', False):
            self.num_input_channels -= 1  # one less due to conditional norm
        
        if is_variable_coords:
            # Variable coordinates mode - need to build graphs
            self._init_variable_coords_mode(data_splits)
        else:
            # Fixed coordinates mode - simpler setup
            self._init_fixed_coords_mode(data_splits)

        print("Sequential dataset initialization complete.")
    
    def _init_variable_coords_mode(self, data_splits):
        """Initialize for variable coordinates mode."""
        print("Setting up variable coordinates mode for sequential data...")
        
        # Create graph builder
        neighbor_search_method = self.model_config.args.magno.neighbor_search_method
        self.graph_builder = GraphBuilder(neighbor_search_method=neighbor_search_method)
        
        # Get graph building parameters
        gno_radius = getattr(self.model_config.args.magno, 'radius', 0.033)
        scales = getattr(self.model_config.args.magno, 'scales', [1.0])
        
        # Build graphs for all splits
        all_graphs = self.graph_builder.build_all_graphs(
            data_splits=data_splits,
            latent_queries=self.latent_tokens_coord,
            gno_radius=gno_radius,
            scales=scales,
            build_train=self.setup_config.train
        )
        
        # Create data loaders with graphs
        loader_kwargs = {
            'encoder_graphs': {
                'train': all_graphs['train']['encoder'] if all_graphs['train'] else None,
                'val': all_graphs['val']['encoder'] if all_graphs['val'] else None,
                'test': all_graphs['test']['encoder']
            },
            'decoder_graphs': {
                'train': all_graphs['train']['decoder'] if all_graphs['train'] else None,
                'val': all_graphs['val']['decoder'] if all_graphs['val'] else None,
                'test': all_graphs['test']['decoder']
            }
        }
        
        loaders = self.data_processor.create_sequential_data_loaders(
            data_splits=data_splits,
            is_variable_coords=True,
            **loader_kwargs
        )
        
        self.train_loader = loaders['train']
        self.val_loader = loaders['val']
        self.test_loader = loaders['test']
    
    def _init_fixed_coords_mode(self, data_splits):
        """Initialize for fixed coordinates mode."""
        print("Setting up fixed coordinates mode for sequential data...")
        
        self.coord = self.data_processor.coord_scaler(data_splits['train']['x'])
        # print("[SANITY] coord min/max per dim after scaling:",
        #     self.coord.min(0).values.tolist(),
        #     self.coord.max(0).values.tolist())
        # print("[SANITY] latent tokens shape:", tuple(self.latent_tokens_coord.shape))
        with torch.no_grad():
            coord = self.coord.float()
            lt    = self.latent_tokens_coord.float()
            r     = float(self.model_config.args.magno.radius)

            # distances from each spatial node to nearest latent
            dmin = torch.cdist(coord, lt).min(dim=1).values
            covered = (dmin <= r).float().mean().item()
            d95 = dmin.quantile(torch.tensor(0.95)).item()
            dmax = dmin.max().item()

            print(f"[CHECK] coverage={covered*100:.2f}%  r={r:.4f}  dmin_95={d95:.4f}  dmin_max={dmax:.4f}")
            # Optional: flag if we’re missing >1% of nodes
            if covered < 0.99:
                print("[WARN] Latent coverage < 99%. Increase radius OR fix latent placement.")



        # ###
        # P = int(self.coord.shape[0])
        # N_lat = int(self.latent_tokens_coord.shape[0])
        # r = float(self.model_config.args.magno.radius)
        # print(f"[SHAPES] P={P:,}  N_lat={N_lat:,}  radius={r}")
        # # ballpark edges per pass (fx mode)
        # import math
        # enc = N_lat * (P/4.0) * math.pi * r*r
        # dec = P     * (N_lat/4.0) * math.pi * r*r
        # print(f"[EST] ~enc_edges={enc:,.0f}  ~dec_edges={dec:,.0f}  total≈{enc+dec:,.0f}")
        # ###
        # print(self.coord.min(0).values, self.coord.max(0).values)  # should be ~[-1,1]
        # print(self.latent_tokens_coord.shape)                       # e.g., (4096, 2)
        # ###
        
        loaders = self.data_processor.create_sequential_data_loaders(
            data_splits=data_splits,
            is_variable_coords=False
        )
        
        self.train_loader = loaders['train']
        self.val_loader = loaders['val']
        self.test_loader = loaders['test']
        # quick sanity: dataset length (number of pairs) if available
        try:
            n_pairs = len(self.train_loader.dataset) if self.train_loader is not None else 0
            print(f"[DATA] train pairs (dataset __len__): {n_pairs:,}")
        except Exception:
            pass

    def _unpack_batch_fx_any(self, batch):
        """
        Robustly unpack batches from:
        - (x, y, coord)
        - (x, y)
        - [{'x':..., 'y':..., 'coord':...}]  (rare)
        - {'x':..., 'y':..., 'coord':...}
        - [ (x,y,coord) ]  (when batch_size=None + default collate wraps once)
        Always returns (x, y, coord_or_fixed)
        """
        # List wrapper from DataLoader
        if isinstance(batch, list):
            if len(batch) == 0:
                raise RuntimeError("Empty batch from DataLoader")
            if len(batch) == 1:
                batch = batch[0]    # unwrap singleton
            else:
                # If ever a list of multiple samples slipped through, stack them
                if all(isinstance(b, (tuple, list)) and len(b) in (2,3) for b in batch):
                    xs, ys, cs = [], [], []
                    for b in batch:
                        if len(b) == 3:
                            x, y, c = b
                        else:
                            x, y = b
                            c = self.coord
                        xs.append(x); ys.append(y); cs.append(c)
                    # coords might be [N,2] per sample; stack to [B,N,2]
                    x = torch.stack(xs, 0); y = torch.stack(ys, 0)
                    c = torch.stack(cs, 0) if torch.is_tensor(cs[0]) else self.coord
                    return x, y, c
                # else fall through and let errors surface so we can see the shape

        # Dict batch (some collates or future datasets)
        if isinstance(batch, dict):
            x = batch['x']
            y = batch['y']
            c = batch.get('coord', self.coord)
            return x, y, c

        # Tuple / list
        if isinstance(batch, (tuple, list)):
            if len(batch) == 3:
                return batch[0], batch[1], batch[2]
            if len(batch) == 2:
                return batch[0], batch[1], self.coord
            if len(batch) == 1:
                # sometimes we get ((x,y,coord),)
                inner = batch[0]
                if isinstance(inner, (tuple, list)) and len(inner) in (2,3):
                    return self._unpack_batch_fx_any(inner)
            raise ValueError(f"Unexpected batch tuple/list structure: len={len(batch)} type0={type(batch[0])}")

        # Raw tensor is not valid (we need x and y)
        raise ValueError(f"Unexpected batch type: {type(batch)}")

    
    def init_model(self, model_config):
        """Initialize the GAOT model for sequential data."""
        model_config.args.magno.coord_dim = self.coord_dim
        
        self.model = GAOT(
            input_size=self.num_input_channels,
            output_size=self.num_output_channels,
            config=model_config
        )

        # attach profiler handle so GAOT.forward can time encode/process/decode
        try:
            self.model._profiler = IterProfiler(enabled=bool(getattr(self.setup_config, "profile", True)))
        except Exception:
            pass
        
        print(f"Initialized {model_config.name} model for sequential data with {self.coord_dim}D coordinates")
    
    def train_step(self, batch):
        """Perform one training step."""
        if self.coord_mode == 'fx':
            return self._train_step_fixed_coords(batch)
        else:
            return self._train_step_variable_coords(batch)
    
    def _train_step_fixed_coords(self, batch):
        """Training step for fixed coordinates mode."""
        # allow (x, y) or (x, y, coord)
        x_batch, y_batch, coord = self._unpack_batch_fx_any(batch)
        if not hasattr(self, "_dbg_once"):
            self._dbg_once = True
            print(f"[DBG] x {tuple(x_batch.shape)}  y {tuple(y_batch.shape)}  coord {tuple(coord.shape)}")

        # x_batch, y_batch = batch
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        coord   = coord.to(self.device)
        latent_tokens_coord = self.latent_tokens_coord.to(self.device)

        r = float(self.model_config.args.magno.radius)
        lt = self.latent_tokens_coord
        # print(f"[SANITY] coord range {coord.min().item():.3f}..{coord.max().item():.3f} | "
        #     f"latent range {lt.min().item():.3f}..{lt.max().item():.3f} | radius={r}")

        # Quick connectivity probe: min distance from a sample of coords to any latent
        # with torch.no_grad():
        #     idx = torch.linspace(0, coord.shape[0]-1, steps=min(256, coord.shape[0])).long()
        #     dmin = torch.cdist(coord[idx], lt).min().item()
        #     print(f"[SANITY] min dist(coord→latent) = {dmin:.4f}  (should be < radius)")


        # Handle conditional normalization
        if getattr(self.model_config, 'use_conditional_norm', False):
            
            pred = self.model(
                latent_tokens_coord=latent_tokens_coord,
                xcoord=coord,
                pndata=x_batch[..., :-1],         # exclude last time feature
                timer=getattr(self, "timer", None),
                condition=x_batch[..., 0, -2:-1]  # condition from time features
            )
        else:
            pred = self.model(
                latent_tokens_coord=latent_tokens_coord,
                xcoord=coord,
                pndata=x_batch,
                timer=getattr(self, "timer", None)
            )
        
        # print(f"[SANITY] pred μσ: {pred.mean().item():.5f} {pred.std().item():.5f} | "
        #     f"y μσ: {y_batch.mean().item():.5f} {y_batch.std().item():.5f}")
        
        # with torch.no_grad():
        #     print("[SANITY] pred μσ:", float(pred.mean()), float(pred.std()),
        #         "| y μσ:", float(y_batch.mean()), float(y_batch.std()))
        #     # time features from this batch (assumes last two dims are [st, dt])
        #     st = x_batch[..., -2].mean().item()
        #     dt = x_batch[..., -1].mean().item()
        #     print(f"[SANITY] start_time (norm) ~{st:.4f}  time_diff (norm) ~{dt:.4f}")

        
        return self.loss_fn(pred, y_batch)

    def _train_step_variable_coords(self, batch):
        """Training step for variable coordinates mode."""
        if len(batch) > 2:
            x_batch, y_batch, coord_batch = batch
        else:
            x_batch, y_batch, coord_batch, encoder_graph_batch, decoder_graph_batch = batch
            encoder_graph_batch = move_to_device(encoder_graph_batch, self.device)
            decoder_graph_batch = move_to_device(decoder_graph_batch, self.device)
        
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        coord_batch = coord_batch.to(self.device)
        latent_tokens_coord = self.latent_tokens_coord.to(self.device)
        
        # Handle conditional normalization
        if getattr(self.model_config, 'use_conditional_norm', False):
            pred = self.model(
                latent_tokens_coord=latent_tokens_coord,
                xcoord=coord_batch,
                pndata=x_batch[..., :-1],          # exclude last time feature
                timer=getattr(self, "timer", None),
                condition=x_batch[..., 0, -2:-1],  # condition from time features
                encoder_nbrs=encoder_graph_batch if len(batch) > 3 else None,
                decoder_nbrs=decoder_graph_batch if len(batch) > 3 else None
            )
        else:
            pred = self.model(
                latent_tokens_coord=latent_tokens_coord,
                xcoord=coord_batch,
                pndata=x_batch,
                timer=getattr(self, "timer", None),
                encoder_nbrs=encoder_graph_batch if len(batch) > 3 else None,
                decoder_nbrs=decoder_graph_batch if len(batch) > 3 else None
            )
        
        return self.loss_fn(pred, y_batch)
    
    def validate(self, loader):
        """Validate the model on validation set."""
        if loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in loader:
                if self.coord_mode == 'fx':
                    loss = self._validate_fixed_coords(batch)
                else:
                    loss = self._validate_variable_coords(batch)
                
                total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def _validate_fixed_coords(self, batch):
        """Validation step for fixed coordinates."""
        
        x_batch, y_batch, coord = self._unpack_batch_fx_any(batch)

        # x_batch, y_batch = batch
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        coord   = coord.to(self.device)
        latent_tokens_coord = self.latent_tokens_coord.to(self.device)
        # coord = self.coord.to(self.device)
        
        if getattr(self.model_config, 'use_conditional_norm', False):
            pred = self.model(
                latent_tokens_coord=latent_tokens_coord,
                xcoord=coord,
                pndata=x_batch[..., :-1],
                timer=getattr(self, "timer", None),
                condition=x_batch[..., 0, -2:-1]
            )
        else:
            pred = self.model(
                latent_tokens_coord=latent_tokens_coord,
                xcoord=coord,
                pndata=x_batch,
                timer=getattr(self, "timer", None)
            )
        # After computing pred and y in any loop:
        # print(f"[SANITY] pred μσ: {pred.mean().item():.5f} {pred.std().item():.5f} | "
        #     f"y μσ: {y_batch.mean().item():.5f} {y_batch.std().item():.5f}")

        
        return self.loss_fn(pred, y_batch)
    
    def _validate_variable_coords(self, batch):
        """Validation step for variable coordinates."""
        if len(batch) > 2:
            x_batch, y_batch, coord_batch = batch
            encoder_graph_batch = decoder_graph_batch = None
        else:
            x_batch, y_batch, coord_batch, encoder_graph_batch, decoder_graph_batch = batch
            encoder_graph_batch = move_to_device(encoder_graph_batch, self.device)
            decoder_graph_batch = move_to_device(decoder_graph_batch, self.device)
        
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        coord_batch = coord_batch.to(self.device)
        latent_tokens_coord = self.latent_tokens_coord.to(self.device)
        
        if getattr(self.model_config, 'use_conditional_norm', False):
            pred = self.model(
                latent_tokens_coord=latent_tokens_coord,
                xcoord=coord_batch,
                pndata=x_batch[..., :-1],
                timer=getattr(self, "timer", None),
                condition=x_batch[..., 0, -2:-1],
                encoder_nbrs=encoder_graph_batch,
                decoder_nbrs=decoder_graph_batch
            )
        else:
            pred = self.model(
                latent_tokens_coord=latent_tokens_coord,
                xcoord=coord_batch,
                pndata=x_batch,
                timer=getattr(self, "timer", None),
                encoder_nbrs=encoder_graph_batch,
                decoder_nbrs=decoder_graph_batch
            )
        
        return self.loss_fn(pred, y_batch)
    
    def _call_model_autoregressive_predict(self, x_batch, time_indices, coord_batch=None):
        """
        Call the model's autoregressive_predict method with appropriate parameters.
        
        Args:
            x_batch: Initial input batch at time t=0
            time_indices: Array of time indices for prediction
            coord_batch: Coordinate batch for variable coords mode
            
        Returns:
            Predicted outputs over time
        """
        if getattr(self.data_processor, "runtime_hints", {}).get("use_trainer_autoreg", False):
            return self._autoregressive_predict_trainer_side(x_batch, time_indices)
        latent_tokens_coord = self.latent_tokens_coord.to(self.device)
        
        if self.coord_mode == 'fx':
            fixed_coord = self.coord.to(self.device)
            encoder_nbrs = None
            decoder_nbrs = None
        else:
            # For variable coordinates mode - not yet fully implemented
            # And time stepper mode like residual, time_der didn't support variable coordinates mode
            # This is where we would pass variable coordinates and graphs
            fixed_coord = None
            encoder_nbrs = None  # TODO: Pass actual encoder graphs
            decoder_nbrs = None  # TODO: Pass actual decoder graphs
            raise NotImplementedError("Variable coordinates autoregressive prediction not yet implemented")
        
        return self.model.autoregressive_predict(
            x_batch=x_batch,
            time_indices=time_indices,
            t_values=self.t_values,
            stats=self.stats,
            stepper_mode=self.stepper_mode,
            latent_tokens_coord=latent_tokens_coord,
            fixed_coord=fixed_coord,
            encoder_nbrs=encoder_nbrs,
            decoder_nbrs=decoder_nbrs,
            use_conditional_norm=getattr(self.model_config, 'use_conditional_norm', False)
        )

    def _autoregressive_predict_trainer_side(self, x0, time_indices):
        """
        Trainer-side AR rollout for generic_h5 & well backends.
        - x0: [B, N, Cu(+Cc)+2] (two dummy time features in the last dims)
        - returns denormalized u predictions [B, K, N, Cu], where K=len(time_indices)-1
        """
        self.model.eval()
        device = self.device

        # Current state (normalized features coming from TestDataset)
        x_curr = x0.to(device)                    # [B, N, F]
        B, N, F = x_curr.shape

        # Stats
        u_mean = self.stats["u"]["mean"].to(device)   # [1, Cu]
        u_std  = self.stats["u"]["std"].to(device)    # [1, Cu]
        Cu = u_mean.shape[-1]
        Cc = int(self.stats["c"]["mean"].shape[-1]) if "c" in self.stats else 0

        res_m = self.stats.get("res", {}).get("mean", None)
        res_s = self.stats.get("res", {}).get("std",  None)
        der_m = self.stats.get("der", {}).get("mean", None)
        der_s = self.stats.get("der", {}).get("std",  None)
        if res_m is not None: res_m = res_m.to(device)
        if res_s is not None: res_s = res_s.to(device)
        if der_m is not None: der_m = der_m.to(device)
        if der_s is not None: der_s = der_s.to(device)

        st_mu = float(self.stats["start_time"]["mean"])
        st_sd = float(self.stats["start_time"]["std"])
        dt_mu = float(self.stats["time_diffs"]["mean"])
        dt_sd = float(self.stats["time_diffs"]["std"])

        latent_tokens_coord = self.latent_tokens_coord.to(device)
        coord = self.coord.to(device)

        # def time_feats(i, d):
        #     st = (float(i) - st_mu) / (st_sd if st_sd > 0 else 1.0)
        #     dt = (float(d) - dt_mu) / (dt_sd if dt_sd > 0 else 1.0)
        #     tf = torch.stack([
        #         torch.full((N,), st, dtype=torch.float32, device=device),
        #         torch.full((N,), dt, dtype=torch.float32, device=device)
        #     ], dim=-1)                 # [N, 2]
        #     return tf.unsqueeze(0).expand(B, N, 2)   # [B, N, 2]
        def time_feats(i_prev, i_curr):
            t_prev = float(self.t_values[i_prev])
            t_diff = float(self.t_values[i_curr] - self.t_values[i_prev])
            st = (t_prev - st_mu) / (st_sd if st_sd > 0 else 1.0)
            dt = (t_diff - dt_mu) / (dt_sd if dt_sd > 0 else 1.0)
            tf = torch.stack([
                torch.full((N,), st, dtype=torch.float32, device=device),
                torch.full((N,), dt, dtype=torch.float32, device=device)
            ], dim=-1)
            return tf.unsqueeze(0).expand(B, N, 2)

        preds_denorm = []

        # time_indices like [t0, t1, t2, ...]; we predict t1..end autoregressively
        # for k in range(1, len(time_indices)):
        #     i_prev = int(time_indices[k-1])
        #     i_curr = int(time_indices[k])
        #     lag = max(1, i_curr - i_prev)

        #     tfb = time_feats(i_prev, lag)                 # [B, N, 2]
        for k in range(1, len(time_indices)):
            i_prev = int(time_indices[k-1])
            i_curr = int(time_indices[k])
            tfb = time_feats(i_prev, i_curr)

            x_step = torch.cat([x_curr[..., :Cu+Cc], tfb], dim=-1)   # [B, N, Cu(+Cc)+2]

            # Same call pattern as training
            if getattr(self.model_config, 'use_conditional_norm', False):
                pred_norm = self.model(
                    latent_tokens_coord=latent_tokens_coord,
                    xcoord=coord,
                    pndata=x_step[..., :-1],             # drop last time feature (matches your train/val)
                    timer=getattr(self, "timer", None),
                    condition=x_step[..., 0, -2:-1]      # per-sample condition from the time features
                )
            else:
                pred_norm = self.model(
                    latent_tokens_coord=latent_tokens_coord,
                    xcoord=coord,
                    pndata=x_step,
                    timer=getattr(self, "timer", None)
                )
            # pred_norm: [B, N, Cu]

            # De-normalize to u_next for metrics and for rolling the state
            if self.stepper_mode == "output":
                u_next = pred_norm * u_std + u_mean
            elif self.stepper_mode == "residual":
                res_den = pred_norm * (res_s if res_s is not None else 1.0) + (res_m if res_m is not None else 0.0)
                u_prev  = x_curr[..., :Cu] * u_std + u_mean
                u_next  = u_prev + res_den
            # elif self.stepper_mode == "time_der":
            #     der_den = pred_norm * (der_s if der_s is not None else 1.0) + (der_m if der_m is not None else 0.0)
            #     u_prev  = x_curr[..., :Cu] * u_std + u_mean
            #     u_next  = u_prev + der_den * float(lag)
            elif self.stepper_mode == "time_der":
                der_den = pred_norm * (der_s if der_s is not None else 1.0) + (der_m if der_m is not None else 0.0)
                u_prev  = x_curr[..., :Cu] * u_std + u_mean
                dt_sec  = float(self.t_values[i_curr] - self.t_values[i_prev])  # true Δt
                u_next  = u_prev + der_den * dt_sec

            else:
                raise ValueError(f"Unsupported stepper_mode: {self.stepper_mode}")

            preds_denorm.append(u_next)                   # [B, N, Cu]

            # Rebuild normalized features to feed the next step
            u_next_norm = (u_next - u_mean) / u_std
            if Cc > 0:
                x_curr = torch.cat([u_next_norm, x_curr[..., Cu:Cu+Cc], tfb], dim=-1)
            else:
                x_curr = torch.cat([u_next_norm, tfb], dim=-1)

        return torch.stack(preds_denorm, dim=1) if preds_denorm else torch.empty(B, 0, N, Cu, device=device)

    def _load_full_trajectory_from_h5(self, file_path: str) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Load a complete trajectory from a single HDF5 file for visualization.
        Returns (u_data, c_data, t_values) as tensors.
        u_data: [T, N, Cu] - denormalized
        c_data: [T, N, Cc] - denormalized (or None if no conditioning)
        t_values: [T] - time values
        """
        import h5py
        
        # Check which type of processor we have
        processor_type = type(self.data_processor).__name__
        
        if 'MagLIF' in processor_type:
            # MagLIF uses unified field_names
            from ..datasets.well_h5_pair_dataset_maglif import _time_from_dimensions, _read_field_TNC
            
            field_names = self.data_processor.field_names if hasattr(self.data_processor, 'field_names') else []
            
            with h5py.File(file_path, "r") as h5:
                t_vals = _time_from_dimensions(h5)
                T_hint = int(t_vals.shape[0])
                
                # Load u fields from 'fields' group
                chunks = []
                for fname in field_names:
                    chunks.append(_read_field_TNC(h5, "fields", fname, T_hint))
                uTNC = np.concatenate(chunks, axis=-1).astype(np.float32) if chunks else np.zeros((T_hint, 1, 0), dtype=np.float32)
                
                # No conditioning for MagLIF
                cTNC = None
        else:
            # Demo/other processors use t0_fields/t1_fields + forcing_fields
            from ..datasets.well_h5_pair_dataset_demo import _time_from_dimensions, _read_field_TNC
            
            u_fields_t0 = self.data_processor.u_fields_t0 if hasattr(self.data_processor, 'u_fields_t0') else []
            u_fields_t1 = self.data_processor.u_fields_t1 if hasattr(self.data_processor, 'u_fields_t1') else []
            
            with h5py.File(file_path, "r") as h5:
                t_vals = _time_from_dimensions(h5)
                T_hint = int(t_vals.shape[0])
                
                # Load u fields
                chunks = []
                for fname in sorted(u_fields_t0):
                    chunks.append(_read_field_TNC(h5, "t0_fields", fname, T_hint))
                for fname in sorted(u_fields_t1):
                    chunks.append(_read_field_TNC(h5, "t1_fields", fname, T_hint))
                uTNC = np.concatenate(chunks, axis=-1).astype(np.float32) if chunks else np.zeros((T_hint, 1, 0), dtype=np.float32)
                
                # Load conditioning (current_drive)
                try:
                    cTNC = _read_field_TNC(h5, "forcing_fields", "current_drive", T_hint)  # (T,N,1)
                except (KeyError, ValueError):
                    cTNC = None
        
        u_tensor = torch.from_numpy(uTNC)
        c_tensor = torch.from_numpy(cTNC) if cTNC is not None else None
        return u_tensor, c_tensor, t_vals
    
    def _predict_full_trajectory_streaming(self, file_path: str) -> Dict:
        """
        Load full trajectory from HDF5 and create predictions using time pairs.
        
        Prediction strategy:
        - Snapshots 0 to (lag-1): Use ground truth
        - Snapshot lag: Predict from snapshot 0 with lag
        - Snapshot lag+1: Predict from snapshot 1 with lag
        - etc.
        
        This matches how the model was trained (on pairs with fixed lag).
        """
        # Load full trajectory
        u_full, c_full, t_vals = self._load_full_trajectory_from_h5(file_path)  # [T,N,Cu], [T,N,Cc] or None, [T]
        T, N, Cu = u_full.shape
        
        # Get time step (lag) from config
        lag = int(self.time_step) if self.time_step is not None else 4
        
        # print(f"[TRAJECTORY] Creating predictions using pair-based approach (lag={lag})")
        # print(f"  Snapshots 0-{lag-1}: Ground truth")
        # print(f"  Snapshots {lag}-{T-1}: Predictions from GT with lag={lag}")
        
        # Get normalization stats
        u_mean = self.stats["u"]["mean"].to(self.device)  # [1,Cu]
        u_std  = self.stats["u"]["std"].to(self.device)
        
        # Conditioning stats (if available)
        if "c" in self.stats and c_full is not None:
            c_mean = self.stats["c"]["mean"].to(self.device)  # [1,Cc]
            c_std  = self.stats["c"]["std"].to(self.device)
            has_conditioning = True
        else:
            c_mean = c_std = None
            has_conditioning = False
        
        st_mu = float(self.stats["start_time"]["mean"])
        st_sd = float(self.stats["start_time"]["std"])
        dt_mu = float(self.stats["time_diffs"]["mean"])
        dt_sd = float(self.stats["time_diffs"]["std"])
        
        # Build prediction sequence
        pred_sequence = []
        
        with torch.no_grad():
            # For each timestep from lag onwards
            for t_out in range(lag, T):
                t_in = t_out - lag  # Input timestep
                
                # Get input state (ground truth)
                u_in = u_full[t_in:t_in+1].to(self.device)  # [1,N,Cu]
                
                # Normalize u
                u_in_norm = (u_in - u_mean) / u_std
                
                # Normalize c (if available)
                if has_conditioning:
                    c_in = c_full[t_in:t_in+1].to(self.device)  # [1,N,Cc]
                    c_in_norm = (c_in - c_mean) / c_std
                else:
                    c_in_norm = None
                
                # Compute time features
                t_start = float(t_vals[t_in])
                t_diff = float(t_vals[t_out] - t_vals[t_in])
                start_norm = (t_start - st_mu) / (st_sd if st_sd > 0 else 1.0)
                diff_norm = (t_diff - dt_mu) / (dt_sd if dt_sd > 0 else 1.0)
                
                # Build time feature tensors
                st = torch.full((1, N, 1), start_norm, dtype=torch.float32, device=self.device)
                td = torch.full((1, N, 1), diff_norm, dtype=torch.float32, device=self.device)
                
                # Build input (with or without conditioning)
                if has_conditioning:
                    x_in = torch.cat([u_in_norm, c_in_norm, st, td], dim=-1)  # [1,N,Cu+Cc+2]
                else:
                    x_in = torch.cat([u_in_norm, st, td], dim=-1)  # [1,N,Cu+2]
                
                # Model prediction
                latent_tokens_coord = self.latent_tokens_coord.to(self.device)
                coord = self.coord.to(self.device)
                
                if getattr(self.model_config, 'use_conditional_norm', False):
                    pred_norm = self.model(
                        latent_tokens_coord=latent_tokens_coord,
                        xcoord=coord,
                        pndata=x_in[..., :-1],
                        timer=getattr(self, "timer", None),
                        condition=x_in[..., 0, -2:-1]
                    )
                else:
                    pred_norm = self.model(
                        latent_tokens_coord=latent_tokens_coord,
                        xcoord=coord,
                        pndata=x_in,
                        timer=getattr(self, "timer", None)
                    )
                
                # Denormalize based on stepper_mode
                if self.stepper_mode == "output":
                    pred_denorm = pred_norm * u_std + u_mean
                elif self.stepper_mode == "residual":
                    res_m = self.stats.get("res", {}).get("mean", None)
                    res_s = self.stats.get("res", {}).get("std", None)
                    if res_m is not None: res_m = res_m.to(self.device)
                    if res_s is not None: res_s = res_s.to(self.device)
                    res_denorm = pred_norm * (res_s if res_s is not None else 1.0) + (res_m if res_m is not None else 0.0)
                    u_prev = u_in  # Already denormalized (from u_full)
                    pred_denorm = u_prev + res_denorm
                elif self.stepper_mode == "time_der":
                    der_m = self.stats.get("der", {}).get("mean", None)
                    der_s = self.stats.get("der", {}).get("std", None)
                    if der_m is not None: der_m = der_m.to(self.device)
                    if der_s is not None: der_s = der_s.to(self.device)
                    der_denorm = pred_norm * (der_s if der_s is not None else 1.0) + (der_m if der_m is not None else 0.0)
                    u_prev = u_in  # Already denormalized
                    dt_sec = float(t_vals[t_out] - t_vals[t_in])
                    pred_denorm = u_prev + der_denorm * dt_sec
                else:
                    raise ValueError(f"Unsupported stepper_mode: {self.stepper_mode}")
                
                pred_sequence.append(pred_denorm[0])  # [N,Cu]
        
        # Stack predictions
        pred_seq = torch.stack(pred_sequence, dim=0)  # [T-lag, N, Cu]
        
        # Ground truth for comparison (from lag onwards)
        gt_seq = u_full[lag:].to(self.device)  # [T-lag, N, Cu]
        
        # Get coordinates (physical for plotting)
        if self.coord_mode == 'fx':
            coords_scaled = self.coord.cpu()
            coords_phys = self.data_processor.coord_scaler.inverse_transform(coords_scaled)
            coords = coords_phys.numpy()
        else:
            coords = None
        
        # Build complete trajectory: first `lag` snapshots are GT, rest are predictions
        # For animation, we want the COMPLETE sequence [T, N, Cu]
        complete_gt = u_full.cpu().numpy()  # [T, N, Cu] - full GT (raw from HDF5)
        complete_pred = u_full.cpu().numpy().copy()  # Start with GT
        complete_pred[lag:] = pred_seq.cpu().numpy()  # Replace from lag onwards with predictions
        
        # CRITICAL DEBUG: Check if data is in correct scale
        # print(f"\n[CRITICAL DEBUG] Data scale verification:")
        # print(f"  Number of channels: {Cu}")
        # print(f"  u_full (from HDF5) at t={lag}:")
        # for ch in range(min(Cu, 6)):
        #     print(f"    Ch{ch}: min={u_full[lag, :, ch].min():.3e}, max={u_full[lag, :, ch].max():.3e}, mean={u_full[lag, :, ch].mean():.3e}")
        # print(f"  pred_seq (model output after denorm) at t={lag}:")
        # for ch in range(min(Cu, 6)):
        #     print(f"    Ch{ch}: min={pred_seq[0, :, ch].min():.3e}, max={pred_seq[0, :, ch].max():.3e}, mean={pred_seq[0, :, ch].mean():.3e}")
        # print(f"  Normalization stats:")
        # for ch in range(min(Cu, 6)):
        #     print(f"    Ch{ch}: u_mean={u_mean[0, ch].item():.3e}, u_std={u_std[0, ch].item():.3e}")
        # print(f"  Stepper mode: {self.stepper_mode}")
        # 
        # # Compute error at first prediction using compute_batch_errors approach
        # # CRITICAL: Use metadata.global_mean/std (like compute_batch_errors does)
        # gt_at_lag = u_full[lag:lag+1].to(self.device)  # [1, N, Cu]
        # pred_at_lag = pred_seq[0:1].to(self.device)    # [1, N, Cu]
        # 
        # global_mean_torch = torch.tensor(self.metadata.global_mean, device=self.device, dtype=self.dtype).reshape(1, 1, -1)
        # global_std_torch = torch.tensor(self.metadata.global_std, device=self.device, dtype=self.dtype).reshape(1, 1, -1)
        # 
        # print(f"  Metadata normalization stats (used for error computation):")
        # for ch in range(min(Cu, 6)):
        #     print(f"    Ch{ch}: global_mean={global_mean_torch[0, 0, ch].item():.3e}, global_std={global_std_torch[0, 0, ch].item():.3e}")
        # 
        # # Normalize both using METADATA stats (not self.stats)
        # gt_norm = (gt_at_lag - global_mean_torch) / global_std_torch
        # pred_norm = (pred_at_lag - global_mean_torch) / global_std_torch
        # 
        # # Compute relative error in normalized space
        # abs_error_norm = torch.abs(pred_norm - gt_norm)
        # error_sum = abs_error_norm.sum()
        # gt_sum = torch.abs(gt_norm).sum()
        # rel_error_first = (error_sum / (gt_sum + 1e-10)).item()
        # 
        # print(f"  First prediction (t={lag}) relative error (with metadata stats): {rel_error_first:.6f}")
        # print(f"  Expected: ~0.16 (16%) based on pair eval")
        
        # Prepare example data for plotting
        # Convention: gt_sequence and pred_sequence should have matching shapes
        # CRITICAL: Use metadata.global_mean/std for error computation (like compute_batch_errors does)
        global_mean_np = np.array(self.metadata.global_mean, dtype=np.float32)
        global_std_np = np.array(self.metadata.global_std, dtype=np.float32)
        
        example_data = {
            'input': u_full[0:1].cpu().numpy(),  # [1, N, Cu] initial state
            'coords': coords,  # [N, 2] physical coordinates
            'gt_sequence': complete_gt,  # [T, N, Cu] complete ground truth
            'pred_sequence': complete_pred,  # [T, N, Cu] GT[0:lag] + predictions[lag:]
            'time_indices': np.arange(T),  # [T]
            't_values': torch.tensor(t_vals, dtype=self.dtype),
            # Use METADATA stats (not self.stats) to match compute_batch_errors!
            'u_mean': global_mean_np.reshape(1, -1),  # [1, Cu]
            'u_std': global_std_np.reshape(1, -1)     # [1, Cu]
        }
        
        # Debug output
        # print(f"\n[DEBUG] Trajectory shapes:")
        # print(f"  GT (complete): {complete_gt.shape}")
        # print(f"  Pred (GT[0:{lag}] + model[{lag}:]): {complete_pred.shape}")
        # print(f"  Predictions created: {T-lag} (from timestep {lag} to {T-1})")
        # 
        # # Check variation
        # gt_var = np.std(complete_gt[lag:], axis=0).mean()  # Variation in predicted region
        # pred_var = np.std(complete_pred[lag:], axis=0).mean()
        # print(f"[DEBUG] Temporal variation (in predicted region {lag}-{T-1}):")
        # print(f"  GT std: {gt_var:.6e}")
        # print(f"  Pred std: {pred_var:.6e}")
        # print(f"  Ratio: {pred_var/gt_var*100:.1f}%")
        
        # Save predictions
        import os
        output_dir = os.path.dirname(self.path_config.result_path)
        os.makedirs(output_dir, exist_ok=True)
        npz_path = os.path.join(output_dir, 'full_trajectory_predictions.npz')
        np.savez(
            npz_path,
            gt_full=complete_gt,
            pred_full=complete_pred,
            coords=coords,
            lag=lag,
            t_values=t_vals
        )
        # print(f"[DEBUG] Saved to: {npz_path}")
        
        return example_data

    def _test_streaming_pairs(self, mode: str = "pairs"):
        """Pair-based evaluation path for streaming datasets (IterableDataset)."""
        from ..utils.metrics import compute_batch_errors, compute_final_metric
        self.model.eval()
        all_relative_errors = []
        all_mse_losses = []  # Track MSE loss (same as training metric)
        all_l1_losses = []   # Track relative L1 loss
        all_l2_losses = []   # Track relative L2 loss
        example_data = None
        overall = {
            "batches": 0, "n_traj": 0, "n_steps": 0,
            "ms_per_traj": [], "ms_per_sample_step": []
        }

        # Dynamic progress bar (no hardcoded total)
        seen = 0
        first_batch_time = None
        batch_times = []
        
        print(f"\n[EVALUATION] Starting pair-based evaluation for metrics...")
        
        # Debug: Print model and data info
        print(f"[DEBUG] Model info:")
        print(f"  Model training mode: {self.model.training}")
        print(f"  Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Input channels: {self.num_input_channels}")
        print(f"  Output channels: {self.num_output_channels}")
        print(f"[DEBUG] Data stats:")
        print(f"  u_mean: {self.stats['u']['mean'].flatten()[:6].tolist()}")
        print(f"  u_std: {self.stats['u']['std'].flatten()[:6].tolist()}")
        if 'c' in self.stats:
            print(f"  c_mean: {self.stats['c']['mean'].flatten().tolist()}")
            print(f"  c_std: {self.stats['c']['std'].flatten().tolist()}")
        if 'start_time' in self.stats:
            print(f"  start_time mean/std: {self.stats['start_time']['mean'].item():.6e} / {self.stats['start_time']['std'].item():.6e}")
            print(f"  time_diffs mean/std: {self.stats['time_diffs']['mean'].item():.6e} / {self.stats['time_diffs']['std'].item():.6e}")
        print(f"[DEBUG] Dataset info:")
        print(f"  Stepper mode: {self.stepper_mode}")
        print(f"  Max time diff: {self.max_time_diff}")
        print(f"  Time step: {self.time_step}")
        if hasattr(self, 'coord_scaler'):
            print(f"  Coordinate scaler: {type(self.coord_scaler).__name__}")

        with torch.no_grad():
            pbar = tqdm(
                desc=f"Testing ({mode}, streaming)",
                unit="sample",
                dynamic_ncols=True,
                colour="blue",
                leave=True
            )
            batch_idx = 0
            for batch in self.test_loader:
                if self.coord_mode == 'fx':
                    x_batch, y_batch, coord = self._unpack_batch_fx_any(batch)
                    
                    # CRITICAL DEBUG - First batch only
                    if batch_idx == 0:
                        print(f"\n{'='*70}")
                        print(f"FIRST BATCH DEBUG (batch_idx=0)")
                        print(f"{'='*70}")
                        print(f"x_batch shape: {x_batch.shape}")
                        print(f"y_batch shape: {y_batch.shape}")
                        print(f"coord shape: {coord.shape}")
                        print(f"x_batch stats: min={x_batch.min():.6f}, max={x_batch.max():.6f}, mean={x_batch.mean():.6f}")
                        print(f"y_batch stats: min={y_batch.min():.6f}, max={y_batch.max():.6f}, mean={y_batch.mean():.6f}")
                        print(f"coord stats: min={coord.min():.6f}, max={coord.max():.6f}")
                        print(f"{'='*70}\n")
                    batch_idx += 1

                    # x_batch, y_batch = batch
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    coord   = coord.to(self.device)
                    latent_tokens_coord = self.latent_tokens_coord.to(self.device)
                    # coord = self.coord.to(self.device)

                    self._sync()
                    t0 = time.perf_counter()
                    batch_idx = len(batch_times) + (1 if first_batch_time is not None else 0)

                    if getattr(self.model_config, 'use_conditional_norm', False):
                        pred = self.model(
                            latent_tokens_coord=latent_tokens_coord,
                            xcoord=coord,
                            pndata=x_batch[..., :-1],
                            timer=getattr(self, "timer", None),
                            condition=x_batch[..., 0, -2:-1]
                        )
                    else:
                        pred = self.model(
                            latent_tokens_coord=latent_tokens_coord,
                            xcoord=coord,
                            pndata=x_batch,
                            timer=getattr(self, "timer", None)
                        )

                    self._sync()
                    dt = time.perf_counter() - t0  # seconds
                    
                    # Track timing separately for first batch vs rest
                    if batch_idx == 0:
                        first_batch_time = dt
                    else:
                        batch_times.append(dt)
                    
                    B = x_batch.shape[0]  # samples in this batch
                    seen += B
                    pbar.update(B)  
                    K = 1  # one step per pair in streaming eval

                    ms_total = dt * 1000.0
                    ms_per_pair = ms_total / max(1, B)
                    ms_per_sample_step = ms_total / max(1, B * K)  # K=1 for pairs, so same as ms_per_pair
                    pbar.set_postfix({"ms/sample": f"{ms_per_pair:7.2f}", "seen": seen})

                    overall["batches"] += 1
                    overall["n_traj"]  += B
                    overall["n_steps"] += (B * K)
                    overall["ms_per_traj"].append(ms_per_pair)
                    overall["ms_per_sample_step"].append(ms_per_sample_step)
                    # pbar.set_postfix({
                    #     "ms/batch": f"{ms_total:7.2f}",
                    #     "ms/traj": f"{ms_per_traj:7.2f}",
                    #     "ms/sample-step": f"{ms_per_sample_step:7.3f}"
                    # })  # tqdm supports `set_postfix` for dynamic stats. :contentReference[oaicite:1]{index=1}


                    
                    # print(f"[SANITY] pred μσ: {pred.mean().item():.5f} {pred.std().item():.5f} | "
                    #     f"y μσ: {y_batch.mean().item():.5f} {y_batch.std().item():.5f}")
                else:
                    # vx: same idea—unpack graphs if present (mirrors your _validate_variable_coords)
                    if len(batch) == 3:
                        x_batch, y_batch, coord_batch = batch
                        encoder_graph_batch = decoder_graph_batch = None
                    else:
                        x_batch, y_batch, coord_batch, encoder_graph_batch, decoder_graph_batch = batch
                        from ..core.trainer_utils import move_to_device
                        encoder_graph_batch = move_to_device(encoder_graph_batch, self.device)
                        decoder_graph_batch = move_to_device(decoder_graph_batch, self.device)
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    coord_batch = coord_batch.to(self.device)
                    latent_tokens_coord = self.latent_tokens_coord.to(self.device)

                    if getattr(self.model_config, 'use_conditional_norm', False):
                        pred = self.model(
                            latent_tokens_coord=latent_tokens_coord,
                            xcoord=coord_batch,
                            pndata=x_batch[..., :-1],
                            timer=getattr(self, "timer", None),
                            condition=x_batch[..., 0, -2:-1],
                            encoder_nbrs=encoder_graph_batch,
                            decoder_nbrs=decoder_graph_batch
                        )
                    else:
                        pred = self.model(
                            latent_tokens_coord=latent_tokens_coord,
                            xcoord=coord_batch,
                            pndata=x_batch,
                            timer=getattr(self, "timer", None),
                            encoder_nbrs=encoder_graph_batch,
                            decoder_nbrs=decoder_graph_batch
                        )

                # Debug first batch - Check for input issues
                if overall["batches"] == 0:
                    print(f"\n[DEBUG] First batch analysis:")
                    print(f"  Shapes: x_batch{tuple(x_batch.shape)}, y_batch{tuple(y_batch.shape)}, coord{tuple(coord.shape)}, pred{tuple(pred.shape)}")
                    print(f"  Coord (scaled) range: [{coord.min():.6f}, {coord.max():.6f}]")
                    
                    # Check if using conditional norm
                    uses_cond_norm = getattr(self.model_config, 'use_conditional_norm', False)
                    print(f"  Conditional norm: {uses_cond_norm}")
                    
                    # Analyze input channels
                    print(f"  x_batch statistics:")
                    print(f"    Overall: min={x_batch.min():.4f}, max={x_batch.max():.4f}, mean={x_batch.mean():.4f}, std={x_batch.std():.4f}")
                    print(f"    Per channel (sample 0, all nodes averaged):")
                    for ch in range(x_batch.shape[-1]):
                        ch_data = x_batch[0, :, ch]
                        print(f"      Ch{ch}: mean={ch_data.mean():.4f}, std={ch_data.std():.4f}, range=[{ch_data.min():.4f}, {ch_data.max():.4f}]")
                    
                    # Check normalized outputs
                    print(f"  y_batch (normalized target): min={y_batch.min():.4f}, max={y_batch.max():.4f}, mean={y_batch.mean():.4f}")
                    print(f"  pred (normalized output): min={pred.min():.4f}, max={pred.max():.4f}, mean={pred.mean():.4f}")
                    print(f"  Prediction error: min={(pred-y_batch).min():.4f}, max={(pred-y_batch).max():.4f}, mean={(pred-y_batch).mean():.4f}, abs_mean={(pred-y_batch).abs().mean():.4f}")
                
                # Compute MSE loss (same metric as training, in normalized space)
                mse_loss = torch.nn.functional.mse_loss(pred, y_batch)
                all_mse_losses.append(mse_loss.item())
                
                # Compute relative L1 and L2 losses on NORMALIZED data (like compute_batch_errors does)
                # This ensures consistency with the standard metric computation
                abs_error_norm = torch.abs(pred - y_batch)
                abs_truth_norm = torch.abs(y_batch)
                
                rel_l1 = (abs_error_norm.sum() / (abs_truth_norm.sum() + 1e-10)).item()
                rel_l2 = (torch.sqrt((abs_error_norm**2).sum()) / (torch.sqrt((abs_truth_norm**2).sum()) + 1e-10)).item()
                
                if overall["batches"] == 0:
                    print(f"  First batch rel_l1: {rel_l1:.6f}, rel_l2: {rel_l2:.6f}, mse: {mse_loss.item():.6f}\n")
                
                all_l1_losses.append(rel_l1)
                all_l2_losses.append(rel_l2)
                
                # Denormalize for compute_batch_errors (expects denormalized data)
                u_mean = self.stats["u"]["mean"].to(self.device)  # [1, Cu]
                u_std  = self.stats["u"]["std"].to(self.device)   # [1, Cu]
                y_den  = y_batch * u_std + u_mean
                p_den  = pred    * u_std + u_mean

                # compute metric per pair (wrap to [B,1,N,C] so it matches your metric utils)
                rel = compute_batch_errors(y_den[:, None, :, :],
                                        p_den[:, None, :, :],
                                        self.metadata)
                all_relative_errors.append(rel)

            # Close progress bar after loop completes
            pbar.close()
        
        # Compute final metrics from all pairs
        if overall["batches"] > 0:
            avg_traj = mean([x for x in overall["ms_per_traj"] if not np.isnan(x)])*702/4/1000
            avg_step = mean([x for x in overall["ms_per_sample_step"] if not np.isnan(x)])
            print(
                f"\n[SPEED] Averaged over {overall['batches']} batches "
                f"({overall['n_traj']} trajectories, {overall['n_steps']} sample-steps): "
                f"{avg_traj:.2f} s/trajectory, {avg_step:.3f} ms/sample-step"
            )

        all_relative_errors = torch.cat(all_relative_errors, dim=0)
        final_metric = compute_final_metric(all_relative_errors)
        
        # Compute average losses
        avg_mse_loss = np.mean(all_mse_losses) if all_mse_losses else 0.0
        avg_l1_loss = np.mean(all_l1_losses) if all_l1_losses else 0.0
        avg_l2_loss = np.mean(all_l2_losses) if all_l2_losses else 0.0
        
        # Print all metrics
        print(f"[METRICS] {mode} losses:")
        print(f"  MSE loss:    {avg_mse_loss:.6f}")
        print(f"  Rel L1 loss: {avg_l1_loss:.6f}")
        print(f"  Rel L2 loss: {avg_l2_loss:.6f}")
        
        # Print timing statistics (normalized by batch size where relevant)
        if first_batch_time is not None and overall["n_traj"] > 0:
            # Estimate first batch size (roughly total/num_batches)
            est_first_batch_size = overall["n_traj"] / max(1, overall["batches"])
            ms_per_sample_first = (first_batch_time * 1000.0) / max(1, est_first_batch_size)
            print(f"\n[Timing] First batch (warmup): {first_batch_time*1000:.2f} ms total, ~{ms_per_sample_first:.2f} ms/sample")
        if batch_times:
            mean_time = np.mean(batch_times)
            std_time = np.std(batch_times)
            # Get average batch size from subsequent batches
            avg_batch_size = (overall["n_traj"] - est_first_batch_size) / max(1, len(batch_times))
            ms_per_sample_mean = (mean_time * 1000.0) / max(1, avg_batch_size)
            print(f"[Timing] Subsequent batches: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms total (mean ± std)")
            print(f"[Timing]                     {ms_per_sample_mean:.2f} ms/sample (averaged)")
            print(f"[Timing] Total batches: {len(batch_times) + 1} (1 warmup + {len(batch_times)} measured)")
            print(f"[Timing] Total samples: {overall['n_traj']}")
        
        # NOW generate full trajectory visualization AFTER pair evaluation completes
        # Mirror behavior of original .nc pipeline
        try:
            # Get a test file for full trajectory visualization
            import glob
            import os
            
            # Check if we're using multires backend (bucket-based structure)
            backend = getattr(self.dataset_config, "backend", "netcdf").lower()
            if backend == "well_multires":
                # For multires: get test files from bucket directories
                base = self.dataset_config.base_path
                # Find all bucket test directories
                test_dirs = sorted([
                    os.path.join(d, "data", "test")
                    for d in glob.glob(os.path.join(base, "*"))
                    if os.path.isdir(d) and os.path.isdir(os.path.join(d, "data", "test"))
                ])
                test_files = []
                for test_dir in test_dirs:
                    test_files.extend(sorted(glob.glob(os.path.join(test_dir, "*.hdf5"))))
            else:
                # For single-resolution: use name-based path
                base = self.dataset_config.base_path
                name = self.dataset_config.name
                test_dir = os.path.join(base, name, "data", "test")
                test_files = sorted(glob.glob(os.path.join(test_dir, "*.hdf5")))
            
            if test_files:
                # Use first test file for visualization (mirrors .nc pipeline behavior)
                test_file = test_files[0]
                example_data = self._predict_full_trajectory_streaming(test_file)
        except Exception as e:
            print(f"\n[WARNING] Could not generate full trajectory visualization: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: don't crash, just skip visualization
            example_data = None
        
        return final_metric, example_data

    
    def test(self):
        """Test the model with different prediction modes."""
        print("Starting sequential model testing...")
        
        self.model.eval()
        self.model.to(self.device)
        
        if hasattr(self.dataset_config, 'predict_mode') and self.dataset_config.predict_mode == "all":
            modes = ["autoregressive", "direct", "star"]
        else:
            modes = [getattr(self.dataset_config, 'predict_mode', 'autoregressive')]
        
        errors_dict = {}
        example_data = None

        # overall = {
        #     "batches": 0, "n_traj": 0, "n_steps": 0,
        #     "ms_per_traj": [], "ms_per_sample_step": []
        # }
        
        for mode in modes:
            print(f"Testing in {mode} mode...")
            overall = {
                "batches": 0, "n_traj": 0, "n_steps": 0,
                "ms_per_traj": [], "ms_per_sample_step": []
            }
            all_relative_errors = []
        #     if mode == "autoregressive":
        #         time_indices = np.arange(0, 6, 1)  # [0, 2, 4, ..., 14]
        #         # time_indices = np.arange(0, 15, 2)  # [0, 2, 4, ..., 14]
        #     elif mode == "direct":
        #         time_indices = np.array([0, 14])
        #     elif mode == "star":
        #         time_indices = np.array([0, 4, 8, 12, 14])
        #     else:
        #         time_indices = np.arange(0, 15, 2)  # Default

            # # If our test loader is streaming (IterableDataset), it won’t have .u_data:
            # if not hasattr(self.test_loader.dataset, "u_data"):
            #     # Use pair-based evaluation instead of AR-in-memory
            #     _ = self._test_streaming_pairs(mode)
            #     continue  # go to next mode

            if not hasattr(self.test_loader.dataset, "u_data"):
                # Use pair-based evaluation instead of AR-in-memory (streaming)
                _, example_pairs = self._test_streaming_pairs(mode)
                if example_pairs is not None:
                    self._plot_test_results(example_pairs)
                    if self.coord_mode == 'fx':
                        self._create_animation(example_pairs)
                continue  # go to next mode

            if not hasattr(self.test_loader.dataset, "u_data"):
                # Use pair-based evaluation instead of AR-in-memory
                T_avail = int(len(self.t_values))
            else:
                T_avail = int(self.test_loader.dataset.u_data.shape[1])
            step    = int(getattr(self.dataset_config, 'time_step', 1))     # e.g. 1
            maxdiff = int(getattr(self.dataset_config, 'max_time_diff', T_avail-1))

            # the largest valid index is bounded by both
            max_idx = min(T_avail - 1, maxdiff)

            if mode == "autoregressive":
                time_indices = np.arange(0, max_idx + 1, step, dtype=int)
            elif mode == "direct":
                time_indices = np.array([0, max_idx], dtype=int)
            elif mode == "star":
                # 5 anchors spaced across [0, max_idx]
                time_indices = np.unique(np.linspace(0, max_idx, num=5, dtype=int))
            else:
                time_indices = np.arange(0, max_idx + 1, step, dtype=int)

            
            test_data_splits = {
                'test': {
                    'u': self.test_loader.dataset.u_data,
                    'c': self.test_loader.dataset.c_data,
                    'x': getattr(self.test_loader.dataset, 'x_data', None),
                    't': self.t_values
                }
            }

            test_dataset = TestDataset(
                u_data=self.test_loader.dataset.u_data,
                c_data=self.test_loader.dataset.c_data,
                t_values=self.test_loader.dataset.t_values,
                metadata=self.metadata,
                time_indices=time_indices,
                stats=self.stats,
                x_data=self.test_loader.dataset.x_data,
                is_variable_coords=(self.coord_mode == 'vx')
            )
            
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.dataset_config.batch_size,
                shuffle=False,
                num_workers=self.dataset_config.num_workers,
                collate_fn=collate_sequential_batch
            )
            
            # Timing tracking
            first_batch_time = None
            batch_times = []
            
            with torch.no_grad():
                # Count actual number of batches dynamically
                pbar = tqdm(desc=f"Testing ({mode})", colour="blue")
                
                for i, batch in enumerate(test_loader):
                    if self.coord_mode == 'fx':
                        x_batch, y_batch = batch
                    else:
                        x_batch, y_batch, coord_batch = batch
                    
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    # Timing: sync and measure
                    self._sync()
                    t0 = time.perf_counter()
                    pred = self._call_model_autoregressive_predict(x_batch, time_indices, coord_batch if len(batch) == 3 else None)
                    self._sync()
                    dt = time.perf_counter() - t0  # seconds
                    
                    # Store timings
                    if i == 0:
                        first_batch_time = dt
                    else:
                        batch_times.append(dt)

                    # B = x_batch.shape[0]                       # trajectories in this batch
                    # K = pred.shape[1] if pred.ndim >= 4 else 0 # predicted steps per trajectory

                    # ms_total = dt * 1000.0
                    # ms_per_traj = (ms_total / B) if B > 0 else float("nan")
                    # ms_per_sample_step = (ms_total / (B * max(1, K))) if (B > 0 and K > 0) else float("nan")

                    # live display on progress bar
                    # pbar.set_postfix({
                    #     "ms/batch": f"{ms_total:7.2f}",
                    #     "ms/traj": f"{ms_per_traj:7.2f}",
                    #     "ms/sample-step": f"{ms_per_sample_step:7.3f}"
                    # })  # tqdm supports `set_postfix` for dynamic stats. :contentReference[oaicite:1]{index=1}

                    # accumulate
                    # overall["batches"] += 1
                    # overall["n_traj"]  += B
                    # overall["n_steps"] += (B * max(1, K))
                    # overall["ms_per_traj"].append(ms_per_traj)
                    # overall["ms_per_sample_step"].append(ms_per_sample_step)
                    
                    metric_type = getattr(self.dataset_config, 'metric', 'final_step')
                    if metric_type == "final_step":
                        relative_errors = compute_batch_errors(
                            y_batch[:, -1:, :, :], pred[:, -1:, :, :], self.metadata)
                    elif metric_type == "all_step":
                        relative_errors = compute_batch_errors(y_batch, pred, self.metadata)
                    else:
                        raise ValueError(f"Unknown metric: {metric_type}")
                    
                    all_relative_errors.append(relative_errors)
                    pbar.update(1)

                    if example_data is None:
                        coord_batch_for_plot = coord_batch if self.coord_mode == 'vx' and len(batch) == 3 else None
                        example_data = self._prepare_example_data(x_batch, y_batch, pred, time_indices, coord_batch_for_plot)
                
                pbar.close()
            
            all_relative_errors = torch.cat(all_relative_errors, dim=0)
            final_metric = compute_final_metric(all_relative_errors)
            errors_dict[mode] = final_metric
            print(f"{mode} mode error: {final_metric}")
            
            # Print timing statistics (normalized by batch size)
            total_samples = len(all_relative_errors)
            total_batches = len(batch_times) + (1 if first_batch_time is not None else 0)
            
            if first_batch_time is not None and total_samples > 0:
                est_first_batch_size = total_samples / max(1, total_batches)
                ms_per_traj_first = (first_batch_time * 1000.0) / max(1, est_first_batch_size)
                print(f"\n[Timing] First batch (warmup): {first_batch_time*1000:.2f} ms total, ~{ms_per_traj_first:.2f} ms/trajectory")
            if batch_times and total_samples > 0:
                mean_time = np.mean(batch_times)
                std_time = np.std(batch_times)
                avg_batch_size = (total_samples - est_first_batch_size) / max(1, len(batch_times)) if first_batch_time is not None else total_samples / len(batch_times)
                ms_per_traj_mean = (mean_time * 1000.0) / max(1, avg_batch_size)
                print(f"[Timing] Subsequent batches: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms total (mean ± std)")
                print(f"[Timing]                     {ms_per_traj_mean:.2f} ms/trajectory (averaged)")
                print(f"[Timing] Total: {total_batches} batches, {total_samples} trajectories")

            # if overall["batches"] > 0:
            #     avg_traj = mean([x for x in overall["ms_per_traj"] if not np.isnan(x)])
            #     avg_step = mean([x for x in overall["ms_per_sample_step"] if not np.isnan(x)])
            #     print(
            #         f"[SPEED] Averaged over {overall['batches']} batches "
            #         f"({overall['n_traj']} trajectories, {overall['n_steps']} sample-steps): "
            #         f"{avg_traj:.2f} ms/trajectory, {avg_step:.3f} ms/sample-step"
            #     )
        
        self._store_test_results(errors_dict, modes)
        
        if example_data is not None:
            self._plot_test_results(example_data)
            
            # TODO: Add animation for vx mode, currently only support fx mode
            if self.coord_mode == 'fx':
                self._create_animation(example_data)
        
        print("Sequential model testing complete.")
    
    def _prepare_example_data(self, x_batch, y_batch, pred, time_indices, coord_batch=None):
        """Prepare data for plotting."""
        u_dim = self.stats["u"]["mean"].shape[0]
        c_dim = self.stats["c"]["mean"].shape[0] if "c" in self.stats else 0
        if c_dim > 0:
            x_u_part = x_batch[..., :u_dim].cpu() * self.stats["u"]["std"] + self.stats["u"]["mean"]
            x_c_part = x_batch[..., u_dim:u_dim+c_dim].cpu() * self.stats["c"]["std"] + self.stats["c"]["mean"]
            x_input = np.stack([x_u_part.numpy(), x_c_part.numpy()], axis=-1)
        else:
            x_u_part = x_batch[..., :u_dim].cpu() * self.stats["u"]["std"] + self.stats["u"]["mean"]
            x_input = x_u_part.numpy()
        
        if self.coord_mode == 'fx':
            original_coords = self.data_processor.coord_scaler.inverse_transform(self.coord.cpu())
            coord_data = original_coords.numpy()
        else:
            if coord_batch is not None:
                original_coords = self.data_processor.coord_scaler.inverse_transform(coord_batch[-1].cpu())
                coord_data = original_coords.numpy()
            else:
                coord_data = None
        
        return {
            'input': x_input[-1],
            'coords': coord_data,
            'gt_sequence': y_batch[-1].cpu().numpy(),
            'pred_sequence': pred[-1].cpu().numpy(),
            'time_indices': time_indices,
            't_values': self.t_values
        }
    
    def _store_test_results(self, errors_dict, modes):
        """Store test results in config datarow."""
        if len(modes) > 1:  
            self.config.datarow["relative error (direct)"] = errors_dict.get("direct", 0.0)
            self.config.datarow["relative error (auto2)"] = errors_dict.get("autoregressive", 0.0)
            self.config.datarow["relative error (auto4)"] = errors_dict.get("star", 0.0)
        else:  
            mode = modes[0]
            self.config.datarow[f"relative error ({mode})"] = errors_dict.get(mode, 0.0)
    
    def _plot_test_results(self, example_data):
        """Create and save test result plots."""
        try:
            # Get input data
            input_data = example_data['input']
            if input_data.ndim == 3 and input_data.shape[0] == 1:
                input_data = input_data[0]
            
            # Get final timestep for comparison
            gt_final = example_data['gt_sequence'][-1]  # [N, C]
            pred_final = example_data['pred_sequence'][-1]  # [N, C]
            
            # Check if this is 1D or 2D+
            if self.coord_dim == 1:
                # Use 1D line plots
                fig = plot_estimates_1d(
                    u_inp=input_data,
                    u_gtr=gt_final,
                    u_prd=pred_final,
                    x_inp=example_data['coords'],
                    x_out=example_data['coords'],
                    names=self.metadata.names['u'],
                    domain=self.metadata.domain_x,
                    show_error=True
                )
            else:
                # Use 2D scatter plots
                if self.metadata.names['c'] and 'c' in self.stats:
                    # Plot with condition data
                    # Note: For now, plotting u fields only; c can be added if needed
                    fig = plot_estimates(
                        u_inp=input_data,
                        u_gtr=gt_final,
                        u_prd=pred_final,
                        x_inp=example_data['coords'],
                        x_out=example_data['coords'],
                        names=self.metadata.names['u'],
                        symmetric=self.metadata.signed['u'],
                        domain=self.metadata.domain_x
                    )
                else:
                    # Plot without condition data
                    fig = plot_estimates(
                        u_inp=input_data,
                        u_gtr=gt_final,
                        u_prd=pred_final,
                        x_inp=example_data['coords'],
                        x_out=example_data['coords'],
                        names=self.metadata.names['u'],
                        symmetric=self.metadata.signed['u'],
                        domain=self.metadata.domain_x
                    )
            
            fig.savefig(self.path_config.result_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
            print(f"Plot saved to {self.path_config.result_path}")
            
            import matplotlib.pyplot as plt
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Could not create plot: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_animation(self, example_data):
        """Create and save animation for sequential data."""
        import gc
        try:
            animation_path = self.path_config.result_path.replace('.png', '.gif')
            gt_sequence = example_data['gt_sequence']      # [T, N, C] complete GT
            pred_sequence = example_data['pred_sequence']  # [T, N, C] GT[0:lag] + predictions[lag:]
            coords = example_data['coords']                # [N, 2] or [N, 1]
            time_indices = example_data['time_indices']
            t_values_full = example_data['t_values']
            
            # Get input data
            input_data = example_data['input']
            if input_data.ndim == 3 and input_data.shape[0] == 1:
                input_data = input_data[0]
            
            # Create time values for labels
            t_vals_np = t_values_full.cpu().numpy() if torch.is_tensor(t_values_full) else np.asarray(t_values_full)
            time_values = [float(t_vals_np[idx]) for idx in time_indices]
            
            # Check if this is 1D or 2D+
            if self.coord_dim == 1:
                # Use 1D line animation with frame limit to avoid memory issues
                # For ~20GB GPU instances, safe limit is ~200-300 frames
                max_frames = 300  # Conservative limit for memory safety
                create_sequential_animation_1d(
                    gt_sequence=gt_sequence,
                    pred_sequence=pred_sequence,
                    coords=coords,
                    save_path=animation_path,
                    input_data=input_data,
                    time_values=time_values,
                    interval=50,  # Slower animation (50ms per frame)
                    names=self.metadata.names['u'] if self.metadata.names.get('u') else None,
                    domain=self.metadata.domain_x if hasattr(self.metadata, 'domain_x') else None,
                    show_error=True,
                    u_mean=example_data.get('u_mean'),  # Pass normalization stats
                    u_std=example_data.get('u_std'),     # for correct error computation
                    max_frames=max_frames
                )
            else:
                # Use 2D scatter animation
                create_sequential_animation(
                    gt_sequence=gt_sequence,
                    pred_sequence=pred_sequence,
                    coords=coords,
                    save_path=animation_path,
                    input_data=input_data,  
                    time_values=time_values,
                    interval=50,  # Slower animation (50ms per frame)
                    symmetric=self.metadata.signed['u'] if self.metadata.signed.get('u') else [True],
                    domain=self.metadata.domain_x if hasattr(self.metadata, 'domain_x') else None,
                    names=self.metadata.names['u'] if self.metadata.names.get('u') else None,
                    colorbar_type="light",
                    show_error=True,
                    dynamic_colorscale=True,  # Adaptive colorscale for better contrast
                    u_mean=example_data.get('u_mean'),  # Pass normalization stats
                    u_std=example_data.get('u_std')     # for correct error computation
                )
            
            print(f"Animation saved to {animation_path}")
            
        except MemoryError as e:
            print(f"\n{'='*60}")
            print(f"MEMORY ERROR: Animation creation failed due to insufficient memory")
            print(f"{'='*60}")
            print(f"Your dataset has {len(example_data['time_indices'])} timesteps.")
            print(f"Try one of these solutions:")
            print(f"  1. Reduce max_frames parameter (currently set in code)")
            print(f"  2. Use a smaller time_step in config (currently {self.time_step})")
            print(f"  3. Reduce max_time_diff in config (currently {self.max_time_diff})")
            print(f"  4. Skip animation by setting predict_mode to single mode instead of 'all'")
            print(f"{'='*60}\n")
            # Clean up to prevent crash
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Could not create animation: {e}")
            import traceback
            traceback.print_exc()
            # Clean up
            gc.collect()
