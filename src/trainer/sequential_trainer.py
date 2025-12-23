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
        # This happens regardless of train/test flag - stats are always computed from training data
        _backend = getattr(self.dataset_config, "backend", "netcdf").lower()
        if _backend in ("well", "well_multires", "well_maglif", "generic_h5"):
            if self.stats is not None and "u" in self.stats:
                mu = self.stats["u"]["mean"].view(-1).detach().cpu().numpy()
                sd = self.stats["u"]["std"].view(-1).detach().cpu().numpy()
                # These two are used by utils/metrics.py
                self.metadata.global_mean = mu.tolist()
                self.metadata.global_std  = sd.tolist()
                # (chunk mapping already handled by Metadata.active_variables/chunked_variables)
                train_flag = getattr(self.setup_config, "train", True)
                # print(f"[DEBUG NORM] Overwriting metadata stats with real stats from data processor (train={train_flag})")
                # print(f"[DEBUG NORM] self.stats['u']['mean'] shape: {self.stats['u']['mean'].shape}, first 3 values: {self.stats['u']['mean'].flatten()[:3].tolist()}")
                # print(f"[DEBUG NORM] self.stats['u']['std'] shape: {self.stats['u']['std'].shape}, first 3 values: {self.stats['u']['std'].flatten()[:3].tolist()}")
                # print(f"[DEBUG NORM] metadata.global_mean length: {len(self.metadata.global_mean)}, first 3 values: {self.metadata.global_mean[:3]}")
                # print(f"[DEBUG NORM] metadata.global_std length: {len(self.metadata.global_std)}, first 3 values: {self.metadata.global_std[:3]}")
                # Verify stats were computed (should always be available, even when train=False)
                if self.stats is None:
                    # print(f"[DEBUG NORM] WARNING: self.stats is None! This should not happen.")
                    pass
                elif "u" not in self.stats:
                    # print(f"[DEBUG NORM] WARNING: self.stats['u'] is missing! Available keys: {list(self.stats.keys())}")
                    pass
            else:
                # print(f"[DEBUG NORM] WARNING: Cannot overwrite metadata stats - self.stats is None or missing 'u' key")
                pass
                # print(f"  self.stats is None: {self.stats is None}")
                # if self.stats is not None:
                #     print(f"  Available keys in self.stats: {list(self.stats.keys())}")

        
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
        # if not hasattr(self, "_batch_structure_debugged"):
        #     self._batch_structure_debugged = True
        #     print(f"\n[DEBUG BATCH STRUCTURE] Raw batch type: {type(batch)}")
        #     if isinstance(batch, (tuple, list)):
        #         print(f"  Batch length: {len(batch)}")
        #         for i, item in enumerate(batch[:3]):  # First 3 items
        #             print(f"  Item {i} type: {type(item)}, shape: {item.shape if hasattr(item, 'shape') else 'N/A'}")
        
        x_batch, y_batch, coord = self._unpack_batch_fx_any(batch)
        # if not hasattr(self, "_dbg_once"):
        #     self._dbg_once = True
        #     print(f"[DBG] After unpack: x {tuple(x_batch.shape)}  y {tuple(y_batch.shape)}  coord {tuple(coord.shape)}")
        #     print(f"[DBG] x_batch dtype: {x_batch.dtype}, device: {x_batch.device}")
        #     print(f"[DBG] y_batch dtype: {y_batch.dtype}, device: {y_batch.device}")
        #     print(f"[DBG] x_batch stats: min={x_batch.min().item():.6f}, max={x_batch.max().item():.6f}, mean={x_batch.mean().item():.6f}, std={x_batch.std().item():.6f}")
        #     print(f"[DBG] y_batch stats: min={y_batch.min().item():.6f}, max={y_batch.max().item():.6f}, mean={y_batch.mean().item():.6f}, std={y_batch.std().item():.6f}")

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

        loss_train = self.loss_fn(pred, y_batch)
        # Print training loss every 10 batches to track learning
        if not hasattr(self, "_train_batch_count"):
            self._train_batch_count = 0
        self._train_batch_count += 1
        
        # if self._train_batch_count <= 3 or self._train_batch_count % 50 == 0:
        #     print(f"[TRAIN BATCH {self._train_batch_count}] loss={loss_train.item():.6f}, pred_mean={pred.mean().item():.6f}, y_mean={y_batch.mean().item():.6f}")
        
        # if not hasattr(self, "_train_loss_debug_printed"):
        #     self._train_loss_debug_printed = True
        #     print(f"\n[DEBUG TRAIN] Training loss computation (first batch):")
        #     print(f"  pred shape: {pred.shape}, y_batch shape: {y_batch.shape}")
        #     print(f"  pred stats (normalized): min={pred.min().item():.6f}, max={pred.max().item():.6f}, mean={pred.mean().item():.6f}, std={pred.std().item():.6f}")
        #     print(f"  y_batch stats (normalized): min={y_batch.min().item():.6f}, max={y_batch.max().item():.6f}, mean={y_batch.mean().item():.6f}, std={y_batch.std().item():.6f}")
        #     print(f"  loss_train: {loss_train.item():.6f}")
        #     print(f"  self.loss_fn type: {type(self.loss_fn)}, reduction: {getattr(self.loss_fn, 'reduction', 'N/A')}")
        #     # Check if data is actually normalized (should have mean ~0, std ~1)
        #     print(f"  y_batch normalized check: mean={y_batch.mean().item():.6f}, std={y_batch.std().item():.6f} (should be ~0 and ~1)")
        #     print(f"  pred normalized check: mean={pred.mean().item():.6f}, std={pred.std().item():.6f}")
        #     # Compute MSE manually to verify
        #     mse_manual = ((pred - y_batch) ** 2).mean().item()
        #     print(f"  MSE manual computation: {mse_manual:.6f} (should match loss_train)")
        #     print(f"  Using stats from self.stats: mean shape={self.stats['u']['mean'].shape}, first 3={self.stats['u']['mean'].flatten()[:3].tolist()}")
        #     print(f"  metadata.global_mean (should match): first 3={self.metadata.global_mean[:3]}")
        #     print(f"  metadata.global_std (should match): first 3={self.metadata.global_std[:3]}")
        #     # Check if stats match
        #     stats_mean_tensor = self.stats['u']['mean'].flatten().cpu()
        #     stats_std_tensor = self.stats['u']['std'].flatten().cpu()
        #     metadata_mean_tensor = torch.tensor(self.metadata.global_mean, dtype=self.dtype)
        #     metadata_std_tensor = torch.tensor(self.metadata.global_std, dtype=self.dtype)
        #     mean_match = torch.allclose(stats_mean_tensor, metadata_mean_tensor, atol=1e-5)
        #     std_match = torch.allclose(stats_std_tensor, metadata_std_tensor, atol=1e-5)
        #     print(f"  Stats match check: mean={mean_match}, std={std_match}")
        #     if not mean_match or not std_match:
        #         print(f"    WARNING: Stats mismatch detected during training!")
        
        return loss_train

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

        loss_val = self.loss_fn(pred, y_batch)
        # if not hasattr(self, "_val_loss_debug_printed"):
        #     self._val_loss_debug_printed = True
        #     print(f"\n[DEBUG VAL] Validation loss computation (first batch):")
        #     print(f"  pred shape: {pred.shape}, y_batch shape: {y_batch.shape}")
        #     print(f"  pred stats (normalized): min={pred.min().item():.6f}, max={pred.max().item():.6f}, mean={pred.mean().item():.6f}, std={pred.std().item():.6f}")
        #     print(f"  y_batch stats (normalized): min={y_batch.min().item():.6f}, max={y_batch.max().item():.6f}, mean={y_batch.mean().item():.6f}, std={y_batch.std().item():.6f}")
        #     print(f"  loss_val: {loss_val.item():.6f}")
        #     print(f"  self.loss_fn type: {type(self.loss_fn)}")
        #     # Check if data is actually normalized (should have mean ~0, std ~1)
        #     print(f"  y_batch normalized check: mean={y_batch.mean().item():.6f}, std={y_batch.std().item():.6f} (should be ~0 and ~1)")
        #     print(f"  pred normalized check: mean={pred.mean().item():.6f}, std={pred.std().item():.6f}")
        #     # Compute MSE manually to verify
        #     mse_manual = ((pred - y_batch) ** 2).mean().item()
        #     print(f"  MSE manual computation: {mse_manual:.6f} (should match loss_val)")
        #     print(f"  Using stats from self.stats: mean shape={self.stats['u']['mean'].shape}, first 3={self.stats['u']['mean'].flatten()[:3].tolist()}")
        #     print(f"  metadata.global_mean (should match): first 3={self.metadata.global_mean[:3]}")
        #     print(f"  metadata.global_std (should match): first 3={self.metadata.global_std[:3]}")
        return loss_val
    
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
    
    def _autoregressive_predict_trainer_side_with_coord(self, x0, time_indices, coord):
        """
        Trainer-side AR rollout with custom coordinates (for per-resolution evaluation).
        - x0: [B, N, Cu(+Cc)+2] (two dummy time features in the last dims)
        - coord: [N, 2] (resolution-specific coordinates)
        - returns denormalized u predictions [B, K, N, Cu], where K=len(time_indices)-1
        """
        self.model.eval()
        device = self.device

        # Current state (normalized features coming from TestDataset)
        x_curr = x0.to(device)                    # [B, N, F]
        B, N, F = x_curr.shape
        
        # Ensure coord is on device and correct shape
        coord = coord.to(device)  # [N, 2]

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

        for k in range(1, len(time_indices)):
            i_prev = int(time_indices[k-1])
            i_curr = int(time_indices[k])
            tfb = time_feats(i_prev, i_curr)

            x_step = torch.cat([x_curr[..., :Cu+Cc], tfb], dim=-1)   # [B, N, Cu(+Cc)+2]

            # Same call pattern as training, but with custom coordinates
            if getattr(self.model_config, 'use_conditional_norm', False):
                pred_norm = self.model(
                    latent_tokens_coord=latent_tokens_coord,
                    xcoord=coord,  # Use the provided coordinates
                    pndata=x_step[..., :-1],             # drop last time feature (matches your train/val)
                    timer=getattr(self, "timer", None),
                    condition=x_step[..., 0, -2:-1]      # per-sample condition from the time features
                )
            else:
                pred_norm = self.model(
                    latent_tokens_coord=latent_tokens_coord,
                    xcoord=coord,  # Use the provided coordinates
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
            elif self.stepper_mode == "time_der":
                der_den = pred_norm * (der_s if der_s is not None else 1.0) + (der_m if der_m is not None else 0.0)
                u_prev  = x_curr[..., :Cu] * u_std + u_mean
                dt_sec  = float(self.t_values[i_curr] - self.t_values[i_prev])  # true Δt
                u_next  = u_prev + der_den * dt_sec
            else:
                raise ValueError(f"Unsupported stepper_mode: {self.stepper_mode}")

            preds_denorm.append(u_next)  # [B, N, Cu]

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
        # print(f"[DEBUG] Model info:")
        # print(f"  Model training mode: {self.model.training}")
        # print(f"  Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        # print(f"  Input channels: {self.num_input_channels}")
        # print(f"  Output channels: {self.num_output_channels}")
        # print(f"[DEBUG] Data stats:")
        # print(f"  u_mean: {self.stats['u']['mean'].flatten()[:6].tolist()}")
        # print(f"  u_std: {self.stats['u']['std'].flatten()[:6].tolist()}")
        # if 'c' in self.stats:
        #     print(f"  c_mean: {self.stats['c']['mean'].flatten().tolist()}")
        #     print(f"  c_std: {self.stats['c']['std'].flatten().tolist()}")
        # if 'start_time' in self.stats:
        #     print(f"  start_time mean/std: {self.stats['start_time']['mean'].item():.6e} / {self.stats['start_time']['std'].item():.6e}")
        #     print(f"  time_diffs mean/std: {self.stats['time_diffs']['mean'].item():.6e} / {self.stats['time_diffs']['std'].item():.6e}")
        # print(f"[DEBUG] Dataset info:")
        # print(f"  Stepper mode: {self.stepper_mode}")
        # print(f"  Max time diff: {self.max_time_diff}")
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
                    # if batch_idx == 0:
                    #     print(f"\n{'='*70}")
                    #     print(f"FIRST BATCH DEBUG (batch_idx=0)")
                    #     print(f"{'='*70}")
                    #     print(f"x_batch shape: {x_batch.shape}")
                    #     print(f"y_batch shape: {y_batch.shape}")
                    #     print(f"coord shape: {coord.shape}")
                    #     print(f"x_batch stats: min={x_batch.min():.6f}, max={x_batch.max():.6f}, mean={x_batch.mean():.6f}")
                    #     print(f"y_batch stats: min={y_batch.min():.6f}, max={y_batch.max():.6f}, mean={y_batch.mean():.6f}")
                    #     print(f"coord stats: min={coord.min():.6f}, max={coord.max():.6f}")
                    #     print(f"{'='*70}\n")
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
                # if overall["batches"] == 0:
                #     print(f"\n[DEBUG] First batch analysis:")
                #     print(f"  Shapes: x_batch{tuple(x_batch.shape)}, y_batch{tuple(y_batch.shape)}, coord{tuple(coord.shape)}, pred{tuple(pred.shape)}")
                #     print(f"  Coord (scaled) range: [{coord.min():.6f}, {coord.max():.6f}]")
                #     
                #     # Check if using conditional norm
                #     uses_cond_norm = getattr(self.model_config, 'use_conditional_norm', False)
                #     print(f"  Conditional norm: {uses_cond_norm}")
                #     
                #     # Analyze input channels
                #     print(f"  x_batch statistics:")
                #     print(f"    Overall: min={x_batch.min():.4f}, max={x_batch.max():.4f}, mean={x_batch.mean():.4f}, std={x_batch.std():.4f}")
                #     print(f"    Per channel (sample 0, all nodes averaged):")
                #     for ch in range(x_batch.shape[-1]):
                #         ch_data = x_batch[0, :, ch]
                #         print(f"      Ch{ch}: mean={ch_data.mean():.4f}, std={ch_data.std():.4f}, range=[{ch_data.min():.4f}, {ch_data.max():.4f}]")
                #     
                #     # Check normalized outputs
                #     print(f"  y_batch (normalized target): min={y_batch.min():.4f}, max={y_batch.max():.4f}, mean={y_batch.mean():.4f}")
                #     print(f"  pred (normalized output): min={pred.min():.4f}, max={pred.max():.4f}, mean={pred.mean():.4f}")
                #     print(f"  Prediction error: min={(pred-y_batch).min():.4f}, max={(pred-y_batch).max():.4f}, mean={(pred-y_batch).mean():.4f}, abs_mean={(pred-y_batch).abs().mean():.4f}")
                
                # Compute MSE loss using same method as validation (for 1D/maglif problems)
                # Use self.loss_fn (nn.MSELoss()) instead of functional.mse_loss to match validation exactly
                if self.coord_dim == 1 or getattr(self.dataset_config, "backend", "").lower() == "well_maglif":
                    # For 1D/maglif: use same loss computation as validation
                    mse_loss = self.loss_fn(pred, y_batch)
                else:
                    # For other problems: keep original functional.mse_loss
                    mse_loss = torch.nn.functional.mse_loss(pred, y_batch)
                # if overall["batches"] == 0:
                #     print(f"\n[DEBUG TEST] Test loss computation (batch 0):")
                #     print(f"  pred shape: {pred.shape}, y_batch shape: {y_batch.shape}")
                #     print(f"  pred stats (normalized): min={pred.min().item():.6f}, max={pred.max().item():.6f}, mean={pred.mean().item():.6f}, std={pred.std().item():.6f}")
                #     print(f"  y_batch stats (normalized): min={y_batch.min().item():.6f}, max={y_batch.max().item():.6f}, mean={y_batch.mean().item():.6f}, std={y_batch.std().item():.6f}")
                #     print(f"  mse_loss value: {mse_loss.item():.6f}")
                #     print(f"  Using loss function: {'self.loss_fn (same as validation)' if (self.coord_dim == 1 or getattr(self.dataset_config, 'backend', '').lower() == 'well_maglif') else 'torch.nn.functional.mse_loss'}")
                #     print(f"  self.loss_fn type: {type(self.loss_fn)}")
                #     print(f"  Using stats from self.stats: mean shape={self.stats['u']['mean'].shape}, first 3={self.stats['u']['mean'].flatten()[:3].tolist()}")
                #     print(f"  metadata.global_mean (should match): first 3={self.metadata.global_mean[:3]}")
                #     print(f"  metadata.global_std (should match): first 3={self.metadata.global_std[:3]}")
                #     # Compute MSE manually to verify
                #     mse_manual = ((pred - y_batch) ** 2).mean().item()
                #     print(f"  MSE manual computation: {mse_manual:.6f} (should match mse_loss)")
                #     # Check if data is actually normalized (should have mean ~0, std ~1)
                #     print(f"  y_batch normalized check: mean={y_batch.mean().item():.6f}, std={y_batch.std().item():.6f} (should be ~0 and ~1)")
                #     print(f"  pred normalized check: mean={pred.mean().item():.6f}, std={pred.std().item():.6f}")
                # Store per-batch MSE loss (this is already averaged over batch elements by nn.MSELoss)
                all_mse_losses.append(mse_loss.item())
                
                # Debug: Check if batch size affects averaging
                # if overall["batches"] == 0:
                #     batch_size = pred.shape[0]
                #     num_elements = pred.numel()
                #     print(f"  Batch size: {batch_size}, Total elements: {num_elements}")
                #     print(f"  MSE per element: {mse_loss.item():.6f} (already averaged by nn.MSELoss)")
                #     print(f"  Sum of squared errors: {((pred - y_batch) ** 2).sum().item():.6f}")
                #     print(f"  Sum / num_elements: {((pred - y_batch) ** 2).sum().item() / num_elements:.6f} (should match mse_loss)")
                
                # Debug: Print stats comparison (first batch only)
                # if overall["batches"] == 0:
                #     print(f"\n[DEBUG METRICS] Stats comparison:")
                #     print(f"  self.stats['u']['mean'] shape: {self.stats['u']['mean'].shape}, first 3: {self.stats['u']['mean'].flatten()[:3].tolist()}")
                #     print(f"  self.stats['u']['std'] shape: {self.stats['u']['std'].shape}, first 3: {self.stats['u']['std'].flatten()[:3].tolist()}")
                #     print(f"  metadata.global_mean length: {len(self.metadata.global_mean)}, first 3: {self.metadata.global_mean[:3]}")
                #     print(f"  metadata.global_std length: {len(self.metadata.global_std)}, first 3: {self.metadata.global_std[:3]}")
                #     # Check if they match
                #     stats_mean_tensor = self.stats['u']['mean'].flatten().cpu()
                #     stats_std_tensor = self.stats['u']['std'].flatten().cpu()
                #     metadata_mean_tensor = torch.tensor(self.metadata.global_mean, dtype=self.dtype)
                #     metadata_std_tensor = torch.tensor(self.metadata.global_std, dtype=self.dtype)
                #     mean_match = torch.allclose(stats_mean_tensor, metadata_mean_tensor, atol=1e-5)
                #     std_match = torch.allclose(stats_std_tensor, metadata_std_tensor, atol=1e-5)
                #     print(f"  Are they equal? mean: {mean_match}, std: {std_match}")
                #     if not mean_match:
                #         diff = (stats_mean_tensor - metadata_mean_tensor).abs().max().item()
                #         print(f"    Mean max diff: {diff:.6e}")
                #         print(f"    WARNING: metadata.global_mean does NOT match self.stats['u']['mean']!")
                #         print(f"    This could cause incorrect GAOT relative error computation!")
                #     if not std_match:
                #         diff = (stats_std_tensor - metadata_std_tensor).abs().max().item()
                #         print(f"    Std max diff: {diff:.6e}")
                #         print(f"    WARNING: metadata.global_std does NOT match self.stats['u']['std']!")
                #         print(f"    This could cause incorrect GAOT relative error computation!")
                #     print(f"  Normalized pred stats: mean={pred.mean().item():.6f}, std={pred.std().item():.6f}")
                #     print(f"  Normalized y_batch stats: mean={y_batch.mean().item():.6f}, std={y_batch.std().item():.6f}")
                #     print(f"  MSE loss (on normalized): {mse_loss.item():.6f}")
                #     print(f"  self.loss_fn reduction: {getattr(self.loss_fn, 'reduction', 'N/A')}")
                
                # For 1D/maglif: compute rel L1 and L2 on normalized data directly (same as training/validation)
                # Training/validation compute loss on normalized data directly, so we should do the same
                if self.coord_dim == 1 or getattr(self.dataset_config, "backend", "").lower() == "well_maglif":
                    # Use normalized data directly (same as training/validation)
                    abs_error_norm = torch.abs(pred - y_batch)
                    abs_truth_norm = torch.abs(y_batch)
                    rel_l1 = (abs_error_norm.sum() / (abs_truth_norm.sum() + 1e-10)).item()
                    rel_l2 = (torch.sqrt((abs_error_norm**2).sum()) / (torch.sqrt((abs_truth_norm**2).sum()) + 1e-10)).item()
                    
                    # if overall["batches"] == 0:
                    #     print(f"  [DEBUG METRICS] Rel L1/L2 computed on normalized data (like training):")
                    #     print(f"    Rel L1: {rel_l1:.6f}, Rel L2: {rel_l2:.6f}")
                else:
                    # For other problems: compute on normalized data (original method)
                    abs_error_norm = torch.abs(pred - y_batch)
                    abs_truth_norm = torch.abs(y_batch)
                    rel_l1 = (abs_error_norm.sum() / (abs_truth_norm.sum() + 1e-10)).item()
                    rel_l2 = (torch.sqrt((abs_error_norm**2).sum()) / (torch.sqrt((abs_truth_norm**2).sum()) + 1e-10)).item()
                
                # For GAOT relative error: denormalize first (compute_batch_errors expects denormalized)
                # Then compute_batch_errors will re-normalize using metadata.global_mean/std
                # But we need to ensure metadata stats match self.stats (they should after init_dataset)
                u_mean = self.stats["u"]["mean"].to(self.device)  # [1, Cu]
                u_std  = self.stats["u"]["std"].to(self.device)   # [1, Cu]
                y_den  = y_batch * u_std + u_mean
                p_den  = pred    * u_std + u_mean
                
                # if overall["batches"] == 0:
                #     print(f"  [DEBUG METRICS] For GAOT relative error:")
                #     print(f"    Denormalized y_den stats: mean={y_den.mean().item():.6f}, std={y_den.std().item():.6f}")
                #     print(f"    Denormalized p_den stats: mean={p_den.mean().item():.6f}, std={p_den.std().item():.6f}")
                #     print(f"    compute_batch_errors will re-normalize using metadata.global_mean/std")
                
                # if overall["batches"] == 0:
                #     print(f"  First batch rel_l1: {rel_l1:.6f}, rel_l2: {rel_l2:.6f}, mse: {mse_loss.item():.6f}\n")
                
                all_l1_losses.append(rel_l1)
                all_l2_losses.append(rel_l2)

                # compute GAOT relative error metric per pair (wrap to [B,1,N,C] so it matches your metric utils)
                # For 1D/maglif: use real_stats (self.stats) instead of metadata to match training
                if self.coord_dim == 1 or getattr(self.dataset_config, "backend", "").lower() == "well_maglif":
                    # Use self.stats directly to match training normalization
                    real_mean = self.stats["u"]["mean"].to(self.device).flatten()  # [Cu]
                    real_std = self.stats["u"]["std"].to(self.device).flatten()   # [Cu]
                    rel = compute_batch_errors(y_den[:, None, :, :],
                                            p_den[:, None, :, :],
                                            self.metadata,
                                            real_stats_mean=real_mean,
                                            real_stats_std=real_std)
                else:
                    # For other problems: use metadata (original behavior)
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
        # Note: During training, loss is computed as total_loss / len(trainer.train_loader)
        # which averages per-batch losses. We do the same here for consistency.
        avg_mse_loss = np.mean(all_mse_losses) if all_mse_losses else 0.0
        # if len(all_mse_losses) > 0:
        #     print(f"[DEBUG METRICS FINAL] Computed average MSE from {len(all_mse_losses)} batches")
        #     print(f"  Individual batch MSE values (first 5): {all_mse_losses[:5]}")
        #     print(f"  Average MSE: {avg_mse_loss:.6f}")
        #     print(f"  Min batch MSE: {min(all_mse_losses):.6f}, Max batch MSE: {max(all_mse_losses):.6f}")
        avg_l1_loss = np.mean(all_l1_losses) if all_l1_losses else 0.0
        avg_l2_loss = np.mean(all_l2_losses) if all_l2_losses else 0.0
        
        # Print all metrics
        print(f"[METRICS] {mode} losses:")
        print(f"  MSE loss:    {avg_mse_loss:.6f}")
        print(f"  Rel L1 loss: {avg_l1_loss:.6f}")
        print(f"  Rel L2 loss: {avg_l2_loss:.6f}")
        print(f"  GAOT relative error: {final_metric:.6f}")
        
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
                metric_value, example_pairs = self._test_streaming_pairs(mode)
                errors_dict[mode] = metric_value
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
        
        # For multires backend, also compute per-resolution metrics and animations
        backend = getattr(self.dataset_config, "backend", "netcdf").lower()
        if backend == "well_multires":
            print("\n" + "="*60)
            print("Computing per-resolution metrics and generating animations...")
            print("="*60)
            try:
                # Evaluate test split (default)
                self._test_per_resolution_multires("test")
                # Also evaluate train and valid splits if available
                if hasattr(self, 'train_loader') and self.train_loader is not None:
                    self._test_per_resolution_multires("train")
                if hasattr(self, 'val_loader') and self.val_loader is not None:
                    self._test_per_resolution_multires("valid")
            except Exception as e:
                print(f"\n[WARNING] Per-resolution evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                print("Continuing without per-resolution metrics...")
        
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
                    max_frames=max_frames,
                    metadata=self.metadata  # Pass metadata for GAOT relative error computation
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
    
    def _test_per_resolution_multires(self, split="test"):
        """
        Evaluate model per resolution for multires datasets.
        Groups batches by resolution (coord.shape[0]) and computes metrics separately.
        Also generates one animation per resolution and saves results to CSV.
        
        Args:
            split: "train", "valid", or "test" - which split to evaluate
        """
        from collections import defaultdict
        import pandas as pd
        
        print(f"\n[PER-RESOLUTION] Starting per-resolution evaluation for {split} split...")
        
        # Get the appropriate loader
        if split == "train":
            loader = getattr(self, 'train_loader', None)
        elif split == "valid":
            loader = getattr(self, 'val_loader', None)
        elif split == "test":
            loader = getattr(self, 'test_loader', None)
        else:
            print(f"WARNING: Unknown split '{split}'. Skipping per-resolution evaluation.")
            return
        
        # Check if loader is available
        if loader is None:
            print(f"WARNING: {split} loader is None. Skipping per-resolution evaluation for {split} split.")
            return
        
        # Group batches by resolution (using grid dimensions H,W as key, not just N)
        resolution_stats = defaultdict(lambda: {
            'count': 0,
            'sum_mse_norm': 0.0,
            'sum_rel1': 0.0,
            'sum_rel2': 0.0,
            'chunk_errors': [],
            'coord': None,
            'sample_input': None,
            'sample_target': None,
            'sample_pred': None,
            'N_points': 0,  # Store actual N for reference
            'res_key': None  # Store (H, W) tuple
        })
        
        self.model.eval()
        
        with torch.no_grad():
            pbar = tqdm(desc=f"Collecting per-resolution data ({split})", unit="batch", colour="cyan")
            try:
                for batch in loader:
                    # Multires batches come as dicts: {"x": ..., "y": ..., "coord": ...}
                    if not isinstance(batch, dict):
                        print("\nWARNING: Expected dict batch format for multires. Skipping per-resolution eval.")
                        return
                    
                    x_batch = batch["x"].to(self.device)
                    y_batch = batch["y"].to(self.device)
                    coord = batch["coord"].to(self.device)
                    
                    # Use coordinate grid dimensions to uniquely identify resolution
                    # This handles cases where different resolutions have the same N (e.g., 64x64 and 128x32 both have 4096 points)
                    coord_np = coord.detach().cpu().numpy()
                    xs = np.round(coord_np[:, 0], 6)
                    ys = np.round(coord_np[:, 1], 6)
                    ux = np.unique(xs)
                    uy = np.unique(ys)
                    H, W = len(ux), len(uy)
                    
                    # Use tuple (H, W) as unique resolution identifier
                    res_key = (H, W)
                    N = coord.shape[0]  # Number of points (kept for backward compatibility)
                    B = x_batch.shape[0]
                    
                    # Model prediction
                    latent_tokens_coord = self.latent_tokens_coord.to(self.device)
                    if getattr(self.model_config, 'use_conditional_norm', False):
                        x_input = x_batch[..., :-1]
                        condition = x_batch[..., 0, -2:-1] if x_batch.shape[-1] > 1 else None
                    else:
                        x_input = x_batch
                        condition = None
                    
                    if condition is not None:
                        pred = self.model(
                            latent_tokens_coord=latent_tokens_coord,
                            xcoord=coord,
                            pndata=x_input,
                            timer=getattr(self, "timer", None),
                            condition=condition
                        )
                    else:
                        pred = self.model(
                            latent_tokens_coord=latent_tokens_coord,
                            xcoord=coord,
                            pndata=x_input,
                            timer=getattr(self, "timer", None)
                        )
                    
                    # Accumulate metrics using res_key (H, W) instead of N
                    resolution_stats[res_key]['count'] += B
                    resolution_stats[res_key]['N_points'] = N  # Store N for reference
                    resolution_stats[res_key]['res_key'] = res_key
                    
                    # MSE on normalized data
                    mse_norm_per_sample = torch.mean((pred - y_batch) ** 2, dim=(-2, -1))
                    resolution_stats[res_key]['sum_mse_norm'] += mse_norm_per_sample.sum().item()
                    
                    # Denormalize for relative errors
                    u_mean = self.stats["u"]["mean"].to(self.device)
                    u_std = self.stats["u"]["std"].to(self.device)
                    pred_denorm = pred * u_std + u_mean
                    target_denorm = y_batch * u_std + u_mean
                    diff_denorm = pred_denorm - target_denorm
                    
                    # Relative L1
                    abs_diff = diff_denorm.abs().sum(dim=(-2, -1))
                    abs_target = target_denorm.abs().sum(dim=(-2, -1)).clamp_min(1e-12)
                    rel_l1_per_sample = abs_diff / abs_target
                    resolution_stats[res_key]['sum_rel1'] += rel_l1_per_sample.sum().item()
                    
                    # Relative L2
                    l2_diff = torch.sqrt((diff_denorm ** 2).sum(dim=(-2, -1)))
                    l2_target = torch.sqrt((target_denorm ** 2).sum(dim=(-2, -1))).clamp_min(1e-12)
                    rel_l2_per_sample = l2_diff / l2_target
                    resolution_stats[res_key]['sum_rel2'] += rel_l2_per_sample.sum().item()
                    
                    # GAOT relative L1 metric
                    rel_errors = compute_batch_errors(
                        target_denorm[:, None, :, :],
                        pred_denorm[:, None, :, :],
                        self.metadata
                    ).cpu()
                    resolution_stats[res_key]['chunk_errors'].append(rel_errors)
                    
                    # Store sample data for animation (first sample only)
                    if resolution_stats[res_key]['sample_input'] is None:
                        resolution_stats[res_key]['sample_input'] = x_batch[0:1].cpu()
                        resolution_stats[res_key]['sample_target'] = y_batch[0:1].cpu()
                        resolution_stats[res_key]['sample_pred'] = pred[0:1].detach().cpu()
                    if resolution_stats[res_key]['coord'] is None:
                        resolution_stats[res_key]['coord'] = coord.cpu()
                    
                    pbar.update(1)
            except (StopIteration, FileNotFoundError, RuntimeError) as e:
                print(f"\nWARNING: Error iterating test loader: {e}")
                print("This may happen if test data is not available for all resolutions.")
                print("Resolutions without test files will be automatically skipped.")
                print("Continuing with per-resolution evaluation for available resolutions...")
                # Don't return - continue with what we have collected so far
            except Exception as e:
                print(f"\nWARNING: Unexpected error during per-resolution evaluation: {e}")
                import traceback
                traceback.print_exc()
                pbar.close()
                return
            
            pbar.close()
        
        if not resolution_stats:
            print("WARNING: No data collected for per-resolution evaluation.")
            print("This may happen if test data is not available for any resolution.")
            print("Resolutions without test files (e.g., 128x32, 256x256) are automatically skipped.")
            return
        
        # Print per-resolution results and prepare CSV data
        print("\n" + "="*60)
        print("PER-RESOLUTION METRICS:")
        print("="*60)
        print("Note: Resolutions without test data (e.g., 128x32, 256x256) are automatically skipped.")
        
        def get_resolution_label(res_key, coord, N_points):
            """Get human-readable resolution label from (H, W) tuple"""
            if isinstance(res_key, tuple) and len(res_key) == 2:
                H, W = res_key
                return f"{H}x{W}"
            # Fallback: try to infer from coordinates
            if coord is not None:
                xy = coord.detach().cpu().numpy()
                xs = np.round(xy[:, 0], 6)
                ys = np.round(xy[:, 1], 6)
                ux = np.unique(xs)
                uy = np.unique(ys)
                H, W = len(ux), len(uy)
                if H * W == coord.shape[0]:
                    return f"{H}x{W}"
            return f"N={N_points}"
        
        csv_rows = []
        # Sort by resolution key (H, W) for consistent ordering
        for res_key in sorted(resolution_stats.keys(), key=lambda x: (x[0], x[1]) if isinstance(x, tuple) else (0, x)):
            data = resolution_stats[res_key]
            num_samples = data['count']
            if num_samples == 0:
                continue
            
            N_points = data.get('N_points', res_key if isinstance(res_key, int) else 0)
            
            mse_norm = data['sum_mse_norm'] / num_samples
            rel_l1 = data['sum_rel1'] / num_samples
            rel_l2 = data['sum_rel2'] / num_samples
            
            if data['chunk_errors']:
                chunk_errors = torch.cat(data['chunk_errors'], dim=0)
                gaot_rel = compute_final_metric(chunk_errors)
            else:
                gaot_rel = float('nan')
            
            coord = data['coord']
            res_label = get_resolution_label(res_key, coord, N_points)
            
            print(f"\nResolution: {res_label} (N={N_points})")
            print(f"  Samples: {num_samples}")
            print(f"  MSE (normalized): {mse_norm:.6e}")
            print(f"  Relative L1: {rel_l1:.6f}")
            print(f"  Relative L2: {rel_l2:.6f}")
            print(f"  GAOT relative L1: {gaot_rel:.6f}")
            
            csv_rows.append({
                'dataset': split,
                'resolution': res_label,
                'N_points': N_points,
                'num_samples': num_samples,
                'mse_normalized': mse_norm,
                'rel_l1': rel_l1,
                'rel_l2': rel_l2,
                'gaot_rel_l1': gaot_rel
            })
        
        # Save CSV results
        import os
        result_dir = os.path.dirname(self.path_config.result_path)
        csv_path = os.path.join(result_dir, f"per_resolution_metrics_{split}.csv")
        if csv_rows:
            df = pd.DataFrame(csv_rows)
            df.to_csv(csv_path, index=False)
            print(f"\nSaved per-resolution metrics to: {csv_path}")
        
        # Generate animations per resolution only for test split (to avoid too many files)
        if split != "test":
            print(f"\nSkipping animation generation for {split} split (only generating for test split).")
            return
        
        # Generate animations per resolution
        print("\n" + "="*60)
        print("Generating animations per resolution...")
        print("="*60)
        
        # Create animation directory
        animation_dir = os.path.join(result_dir, "per_resolution_animations")
        os.makedirs(animation_dir, exist_ok=True)
        
        # Build time indices for autoregressive prediction
        max_rollout_steps = 50
        time_step = int(getattr(self.dataset_config, 'time_step', 1))
        time_indices = np.arange(0, min(max_rollout_steps, 101), time_step, dtype=int)
        time_indices_anim = time_indices[::10]  # Every 10th frame for animation
        
        # Sort by resolution key for consistent ordering
        for res_key in sorted(resolution_stats.keys(), key=lambda x: (x[0], x[1]) if isinstance(x, tuple) else (0, x)):
            data = resolution_stats[res_key]
            if data['sample_input'] is None:
                continue
            
            coord = data['coord'].to(self.device)
            N_points = data.get('N_points', res_key if isinstance(res_key, int) else 0)
            res_label = get_resolution_label(res_key, coord, N_points)
            
            print(f"\nCreating animation for resolution: {res_label}")
            
            try:
                first_input_full = data['sample_input'].to(self.device)  # [1, N, C_full]
                u_channels = self.stats["u"]["mean"].shape[-1]
                c_channels = self.stats["c"]["mean"].shape[-1] if "c" in self.stats else 0
                
                # Extract u and c from input (input has u + c + time features)
                first_input_u_c = first_input_full[..., :u_channels + c_channels]  # [1, N, Cu+Cc]
                
                # Use trainer-side autoregressive prediction for better compatibility
                if getattr(self.data_processor, "runtime_hints", {}).get("use_trainer_autoreg", False):
                    # Build time features for initial state
                    st_mu = float(self.stats["start_time"]["mean"])
                    st_sd = float(self.stats["start_time"]["std"])
                    dt_mu = float(self.stats["time_diffs"]["mean"])
                    dt_sd = float(self.stats["time_diffs"]["std"])
                    
                    # Create initial x_batch with time features
                    N_nodes = coord.shape[0]
                    t0_idx = 0
                    t1_idx = time_indices[1] if len(time_indices) > 1 else time_indices[0] + time_step
                    t_start = float(self.t_values[t0_idx])
                    t_diff = float(self.t_values[t1_idx] - self.t_values[t0_idx])
                    start_norm = (t_start - st_mu) / (st_sd if st_sd > 0 else 1.0)
                    diff_norm = (t_diff - dt_mu) / (dt_sd if dt_sd > 0 else 1.0)
                    
                    st_feat = torch.full((1, N_nodes, 1), start_norm, dtype=torch.float32, device=self.device)
                    dt_feat = torch.full((1, N_nodes, 1), diff_norm, dtype=torch.float32, device=self.device)
                    x0_with_time = torch.cat([first_input_u_c, st_feat, dt_feat], dim=-1)  # [1, N, Cu+Cc+2]
                    
                    # Use resolution-specific coordinates for autoregressive prediction
                    pred_sequence = self._autoregressive_predict_trainer_side_with_coord(x0_with_time, time_indices, coord)
                    # pred_sequence: [1, K, N, Cu] where K = len(time_indices) - 1
                    pred_denorm = pred_sequence[0].detach().cpu()  # [K, N, Cu] - detach to avoid gradient issues
                else:
                    # Fallback to model's autoregressive_predict
                    with torch.no_grad():
                        pred_sequence = self.model.autoregressive_predict(
                            x_batch=first_input_u_c,
                            time_indices=time_indices,
                            t_values=self.t_values if hasattr(self, 't_values') else np.arange(len(time_indices)),
                            stats=self.stats,
                            stepper_mode=getattr(self.dataset_config, 'stepper_mode', 'output'),
                            latent_tokens_coord=self.latent_tokens_coord.to(self.device),
                            fixed_coord=coord,
                            encoder_nbrs=None,
                            decoder_nbrs=None,
                            use_conditional_norm=getattr(self.model_config, 'use_conditional_norm', False)
                        )  # [1, T-1, N, C]
                        
                        u_mean = self.stats["u"]["mean"].cpu()
                        u_std = self.stats["u"]["std"].cpu()
                        pred_denorm = pred_sequence[0].cpu() * u_std + u_mean  # [T-1, N, C]
                
                # For animation, we need GT sequence too (use predictions as placeholder for now)
                gt_denorm = pred_denorm.clone().detach()
                
                u_mean = self.stats["u"]["mean"].cpu()
                u_std = self.stats["u"]["std"].cpu()
                input_denorm = (first_input_u_c[0, :, :u_channels].detach().cpu() * u_std + u_mean).numpy()
                coord_phys = self.data_processor.coord_scaler.inverse_transform(coord.detach().cpu()).numpy()
                
                gt_anim = gt_denorm[::10].numpy()
                pred_anim = pred_denorm[::10].numpy()
                
                if hasattr(self, 't_values'):
                    t_vals = self.t_values
                    time_values = [float(t_vals[idx]) for idx in time_indices_anim[1:]]  # Skip first (input)
                else:
                    time_values = [float(idx) for idx in time_indices_anim[1:]]
                
                animation_path = os.path.join(animation_dir, f"animation_{res_label}.gif")
                
                create_sequential_animation(
                    gt_sequence=gt_anim,
                    pred_sequence=pred_anim,
                    coords=coord_phys,
                    save_path=animation_path,
                    input_data=input_denorm,
                    time_values=time_values,
                    interval=100,
                    symmetric=self.metadata.signed['u'] if hasattr(self.metadata, 'signed') and self.metadata.signed.get('u') else [True],
                    domain=self.metadata.domain_x if hasattr(self.metadata, 'domain_x') else None,
                    names=self.metadata.names.get('u', None) if hasattr(self.metadata, 'names') else None,
                    colorbar_type="light",
                    show_error=True,
                    dynamic_colorscale=True,
                    u_mean=u_mean.numpy(),
                    u_std=u_std.numpy()
                )
                
                print(f"  Saved animation to: {animation_path}")
                print(f"  Animation frames: {gt_anim.shape[0]}")
                    
            except Exception as e:
                print(f"  WARNING: Could not create animation for {res_label}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n[PER-RESOLUTION] Evaluation complete for {split} split. Animations saved to: {animation_dir}")
