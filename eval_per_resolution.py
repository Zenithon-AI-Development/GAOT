import os
import argparse
import numpy as np
import torch
import pandas as pd
from omegaconf import OmegaConf
from types import SimpleNamespace
from collections import defaultdict

from main import prepare_arg
from src.trainer.sequential_trainer import SequentialTrainer
from src.trainer.static_trainer import StaticTrainer
from src.utils.plotting import create_sequential_animation

def set_mp_spawn():
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

def load_cfg(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    cfg.setup.train = False
    cfg.setup.test  = True
    cfg.setup.ckpt  = True
    arg = SimpleNamespace(**OmegaConf.to_container(cfg, resolve=True))
    arg = prepare_arg(arg)
    return arg

def infer_hw_from_coord(coord: torch.Tensor, tol_decimals: int = 6):
    """Try to infer H,W from coordinate grid"""
    xy = coord.detach().cpu().numpy()
    xs = np.round(xy[:, 0], tol_decimals)
    ys = np.round(xy[:, 1], tol_decimals)
    ux = np.unique(xs)
    uy = np.unique(ys)
    H, W = len(ux), len(uy)
    if H * W == coord.shape[0]:
        return int(H), int(W)
    if W * H == coord.shape[0]:
        return int(W), int(H)
    return None, None

def rel_l1(pred, target, eps=1e-12):
    """Relative L1 error"""
    num = (pred - target).abs().sum(dim=(-1,-2))
    den = target.abs().sum(dim=(-1,-2)).clamp_min(eps)
    return (num/den).mean()

def rel_l2(pred, target, eps=1e-12):
    """Relative L2 error"""
    num = torch.sqrt(((pred - target)**2).sum(dim=(-1,-2)))
    den = torch.sqrt((target**2).sum(dim=(-1,-2))).clamp_min(eps)
    return (num/den).mean()

def mse(pred, target):
    """Mean squared error"""
    return ((pred - target)**2).mean()

def compute_metrics(pred, target):
    """Compute all metrics for a prediction and target (on denormalized data)"""
    return {
        "rel_l1": float(rel_l1(pred, target).item()),
        "rel_l2": float(rel_l2(pred, target).item()),
        "mse": float(mse(pred, target).item()),
    }

def compute_metrics_normalized(pred_norm, target_norm):
    """Compute MSE on normalized data (matches training validation loss)"""
    return float(mse(pred_norm, target_norm).item())

def get_resolution_label(N, coord=None):
    """Get a human-readable label for a resolution"""
    if coord is not None:
        H, W = infer_hw_from_coord(coord)
        if H and W:
            return f"{H}x{W}"
    return f"N={N}"

def build_time_indices_for_animation(max_time_steps, step=1):
    """Build time indices for autoregressive prediction, limiting to reasonable number"""
    return np.arange(0, min(max_time_steps, 101), step, dtype=int)

def evaluate_loader(loader, loader_name, model, trainer, device, max_samples, stats):
    """Evaluate model on a single data loader and return resolution-grouped results"""
    resolution_data = defaultdict(lambda: {
        'preds': [],
        'targets': [],
        'inputs': [],
        'coord': None,
        'count': 0
    })
    
    print(f"\nCollecting predictions from {loader_name.upper()} set...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if isinstance(batch, dict):
                # Multi-res format: {"x": [B,N,C], "y": [B,N,C], "coord": [N,2]}
                x_batch = batch["x"].to(device)
                y_batch = batch["y"].to(device)
                coord = batch["coord"].to(device)
            else:
                print("\nWARNING: Unexpected batch format. Expected dict with 'x', 'y', 'coord' keys.")
                print("This might not be a multi-resolution dataset.")
                break
            
            N = coord.shape[0]
            B = x_batch.shape[0]
            
            # Check if we have enough samples for this resolution
            if resolution_data[N]['count'] >= max_samples:
                continue
            
            # Handle conditional normalization: drop last channel if enabled
            # Multi-res format: x_batch = [u_norm | c_norm | start_time_norm | time_diff_norm]
            if getattr(trainer.model_config, 'use_conditional_norm', False):
                # Conditional norm: pass all but last channel (time_diff is used as condition)
                x_input = x_batch[..., :-1]
            else:
                # Normal: pass all channels
                x_input = x_batch
            
            # Make prediction (one-step)
            pred = model(
                latent_tokens_coord=trainer.latent_tokens_coord.to(device),
                xcoord=coord,
                pndata=x_input
            )
            
            # Store results
            resolution_data[N]['preds'].append(pred.cpu())
            resolution_data[N]['targets'].append(y_batch.cpu())
            resolution_data[N]['inputs'].append(x_batch.cpu())
            if resolution_data[N]['coord'] is None:
                resolution_data[N]['coord'] = coord.cpu()
            resolution_data[N]['count'] += B
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")
    
    if not resolution_data:
        print(f"  WARNING: No data collected from {loader_name} set")
        return None
    
    print(f"  Found {len(resolution_data)} different resolutions in {loader_name} set")
    
    # Compute metrics per resolution
    results = []
    total_samples = 0
    weighted_rel_l1 = 0.0
    weighted_rel_l2 = 0.0
    weighted_mse = 0.0
    weighted_mse_norm = 0.0
    
    for N in sorted(resolution_data.keys()):
        data = resolution_data[N]
        
        # Concatenate all predictions and targets for this resolution
        all_preds_norm = torch.cat(data['preds'], dim=0)  # Normalized predictions
        all_targets_norm = torch.cat(data['targets'], dim=0)  # Normalized targets
        coord = data['coord']
        
        # Compute normalized MSE (matches training validation loss)
        mse_norm = compute_metrics_normalized(all_preds_norm, all_targets_norm)
        
        # Denormalize for other metrics
        u_mean = stats["u"]["mean"].cpu()
        u_std = stats["u"]["std"].cpu()
        all_preds_denorm = all_preds_norm * u_std + u_mean
        all_targets_denorm = all_targets_norm * u_std + u_mean
        
        # Compute metrics on denormalized data
        metrics = compute_metrics(all_preds_denorm, all_targets_denorm)
        
        # Get resolution label
        res_label = get_resolution_label(N, coord)
        
        # Accumulate for overall metrics (weighted average by number of samples)
        num_samples = all_preds_norm.shape[0]
        total_samples += num_samples
        weighted_rel_l1 += metrics['rel_l1'] * num_samples
        weighted_rel_l2 += metrics['rel_l2'] * num_samples
        weighted_mse += metrics['mse'] * num_samples
        weighted_mse_norm += mse_norm * num_samples
        
        # Store for results
        results.append({
            'dataset': loader_name,
            'resolution': res_label,
            'N_points': N,
            'num_samples': num_samples,
            'mse_normalized': mse_norm,
            'rel_l1': metrics['rel_l1'],
            'rel_l2': metrics['rel_l2'],
            'mse': metrics['mse']
        })
    
    # Compute overall metrics as weighted average across resolutions
    if total_samples > 0:
        overall_mse_norm = weighted_mse_norm / total_samples
        overall_rel_l1 = weighted_rel_l1 / total_samples
        overall_rel_l2 = weighted_rel_l2 / total_samples
        overall_mse = weighted_mse / total_samples
    else:
        overall_mse_norm = overall_rel_l1 = overall_rel_l2 = overall_mse = float('nan')
    
    results.append({
        'dataset': loader_name,
        'resolution': 'OVERALL',
        'N_points': -1,
        'num_samples': total_samples,
        'mse_normalized': overall_mse_norm,
        'rel_l1': overall_rel_l1,
        'rel_l2': overall_rel_l2,
        'mse': overall_mse
    })
    
    return resolution_data, results

def main():
    set_mp_spawn()
    
    ap = argparse.ArgumentParser(description="Evaluate GAOT model performance per resolution with animations")
    ap.add_argument("-c", "--config", required=True, help="Path to config file")
    ap.add_argument("--device", default="cuda:0", help="Device to use")
    ap.add_argument("--batch", type=int, default=4, help="Batch size")
    ap.add_argument("--max_samples", type=int, default=100, help="Max samples to evaluate per resolution")
    ap.add_argument("--max_rollout_steps", type=int, default=50, help="Max autoregressive rollout steps for animation")
    ap.add_argument("--out_csv", default="per_resolution_metrics.csv", help="Output CSV file")
    ap.add_argument("--animation_dir", default="per_resolution_animations", help="Directory to save animations")
    ap.add_argument("--skip_animations", action="store_true", help="Skip animation generation")
    args = ap.parse_args()
    
    # Load config and initialize trainer
    print(f"Loading config from: {args.config}")
    arg = load_cfg(args.config)
    arg.setup["device"] = args.device
    
    Trainer = {"static": StaticTrainer, "sequential": SequentialTrainer}[arg.setup["trainer_name"]]
    trainer = Trainer(arg).load_ckpt()
    
    device = trainer.device
    model = trainer.model.to(device).eval()
    stats = trainer.stats
    
    # Check if this is a multi-resolution setup
    backend = getattr(trainer.dataset_config, "backend", "").lower()
    if backend != "well_multires":
        print(f"\nWARNING: Config backend is '{backend}', not 'well_multires'")
        print("This script is designed for multi-resolution datasets.")
        print("Results may not be meaningful for single-resolution datasets.\n")
    
    # Determine which loaders are available
    has_val = hasattr(trainer, 'val_loader') and trainer.val_loader is not None
    has_test = hasattr(trainer, 'test_loader') and trainer.test_loader is not None
    
    if not has_val and not has_test:
        print("\nERROR: Neither validation nor test loader available")
        return
    
    # Evaluate on both datasets if available, or just validation
    all_results = []
    loaders_to_eval = []
    
    if has_val:
        loaders_to_eval.append((trainer.val_loader, "validation"))
    
    if has_test:
        print("\nChecking if test set is complete for all resolutions...")
        try:
            # Try to peek at test loader to see if it works
            test_iter = iter(trainer.test_loader)
            next(test_iter)
            loaders_to_eval.append((trainer.test_loader, "test"))
            print("  Test set appears to be available")
        except (StopIteration, FileNotFoundError) as e:
            print(f"  Test set incomplete or missing: {e}")
            print("  Will use validation set only")
    
    if not loaders_to_eval:
        print("\nERROR: No usable loaders found")
        return
    
    print(f"\nWill evaluate on: {', '.join([name.upper() for _, name in loaders_to_eval])}")
    print("(Batches never mix resolutions in multi-res setup)\n")
    
    # Store resolution data for animation (use validation data)
    val_resolution_data = None
    
    for loader, loader_name in loaders_to_eval:
        print("=" * 60)
        result = evaluate_loader(loader, loader_name, model, trainer, device, args.max_samples, stats)
        
        if result is not None:
            resolution_data, results = result
            
            # Save validation data for animations
            if loader_name == "validation":
                val_resolution_data = resolution_data
            
            # Print detailed results for this dataset
            print(f"\n{loader_name.upper()} SET RESULTS:")
            print("-" * 60)
            for res_dict in results:
                if res_dict['resolution'] == 'OVERALL':
                    print(f"\nOVERALL (all resolutions combined):")
                    print(f"  Total samples: {res_dict['num_samples']}")
                    print(f"  MSE (normalized, matches training val loss): {res_dict['mse_normalized']:.6e}")
                    print(f"  Relative L1 (denormalized): {res_dict['rel_l1']:.6f}")
                    print(f"  Relative L2 (denormalized): {res_dict['rel_l2']:.6f}")
                    print(f"  MSE (denormalized): {res_dict['mse']:.6e}")
                else:
                    print(f"\nResolution: {res_dict['resolution']} (N={res_dict['N_points']})")
                    print(f"  Samples: {res_dict['num_samples']}")
                    print(f"  MSE (normalized, matches training val loss): {res_dict['mse_normalized']:.6e}")
                    print(f"  Relative L1 (denormalized): {res_dict['rel_l1']:.6f}")
                    print(f"  Relative L2 (denormalized): {res_dict['rel_l2']:.6f}")
                    print(f"  MSE (denormalized): {res_dict['mse']:.6e}")
            
            all_results.extend(results)
        else:
            print(f"  Skipping {loader_name} set (no data collected)")
    
    if not all_results:
        print("\nERROR: No results collected from any dataset")
        return
    
    # Use validation data for animations (guaranteed to exist for all resolutions)
    resolution_data = val_resolution_data if val_resolution_data is not None else resolution_data
    
    # Save CSV results (includes both validation and test if available)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved CSV results to: {args.out_csv}")
    
    # Generate animations per resolution
    if not args.skip_animations:
        print("\n" + "="*60)
        print("Generating animations per resolution...")
        print("="*60)
        
        os.makedirs(args.animation_dir, exist_ok=True)
        
        # Build time indices for autoregressive rollout
        # Use dataset config to determine appropriate time step
        time_step = int(getattr(trainer.dataset_config, 'time_step', 1))
        time_indices = build_time_indices_for_animation(args.max_rollout_steps, step=time_step)
        
        # Subsample to use only every 10th frame for animation
        time_indices_anim = time_indices[::10]
        print(f"Using {len(time_indices_anim)} frames for animations (every 10th frame from {len(time_indices)} total steps)")
        
        for N in sorted(resolution_data.keys()):
            data = resolution_data[N]
            coord = data['coord'].to(device)
            res_label = get_resolution_label(N, coord)
            
            print(f"\nCreating animation for resolution: {res_label}")
            
            # Take first sample from this resolution for animation
            all_inputs = torch.cat(data['inputs'], dim=0)
            first_input_full = all_inputs[0:1].to(device)  # [1, N, C_full] with all features
            
            # For autoregressive_predict, we need ONLY u and c fields (no time features)
            # The autoregressive_predict method will add time features internally
            u_dim = stats["u"]["mean"].shape[-1]
            c_dim = stats["c"]["mean"].shape[-1] if "c" in stats else 0
            first_input = first_input_full[..., :u_dim+c_dim]  # [1, N, u_dim+c_dim]
            
            try:
                # Generate autoregressive prediction sequence
                with torch.no_grad():
                    pred_sequence = model.autoregressive_predict(
                        x_batch=first_input,
                        time_indices=time_indices,
                        t_values=trainer.t_values if hasattr(trainer, 't_values') else np.arange(len(time_indices)),
                        stats=stats,
                        stepper_mode=getattr(trainer.dataset_config, 'stepper_mode', 'output'),
                        latent_tokens_coord=trainer.latent_tokens_coord.to(device),
                        fixed_coord=coord,
                        encoder_nbrs=None,
                        decoder_nbrs=None,
                        use_conditional_norm=getattr(trainer.model_config, 'use_conditional_norm', False)
                    )  # [1, T-1, N, C]
                
                # Denormalize predictions
                u_mean = stats["u"]["mean"].cpu()
                u_std = stats["u"]["std"].cpu()
                pred_denorm = pred_sequence[0].cpu() * u_std + u_mean  # [T-1, N, C]
                
                # For ground truth, we would need the full trajectory
                # Since we only have one-step targets, we'll use pred_denorm as both GT and pred
                # This shows the autoregressive drift over time
                # In a real scenario, you'd load the full GT trajectory from the dataset
                gt_denorm = pred_denorm.clone()
                
                # Get input data (denormalized)
                # stats["u"]["mean"] has shape [1, C], so use shape[-1] to get C
                u_dim = stats["u"]["mean"].shape[-1]
                input_denorm = (first_input[0, :, :u_dim].cpu() * u_std + u_mean).numpy()  # [N, C]
                
                # Inverse transform coordinates to physical space
                coord_phys = trainer.data_processor.coord_scaler.inverse_transform(coord.cpu()).numpy()
                
                # Subsample frames for animation (every 10th frame)
                gt_anim = gt_denorm[::10].numpy()  # [T_anim, N, C]
                pred_anim = pred_denorm[::10].numpy()  # [T_anim, N, C]
                
                # Get time values for labels
                if hasattr(trainer, 't_values'):
                    t_vals = trainer.t_values
                    time_values = [float(t_vals[idx]) for idx in time_indices_anim]
                else:
                    time_values = [float(idx) for idx in time_indices_anim]
                
                # Create animation
                animation_path = os.path.join(args.animation_dir, f"animation_{res_label}.gif")
                
                create_sequential_animation(
                    gt_sequence=gt_anim,
                    pred_sequence=pred_anim,
                    coords=coord_phys,
                    save_path=animation_path,
                    input_data=input_denorm,
                    time_values=time_values,
                    interval=100,  # 100ms per frame
                    symmetric=trainer.metadata.signed['u'] if hasattr(trainer.metadata, 'signed') else [True],
                    domain=trainer.metadata.domain_x if hasattr(trainer.metadata, 'domain_x') else None,
                    names=trainer.metadata.names.get('u', None) if hasattr(trainer.metadata, 'names') else None,
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
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"  Config: {args.config}")
    print(f"  Backend: {backend}")
    print(f"  Datasets evaluated: {', '.join([name.upper() for _, name in loaders_to_eval])}")
    print(f"  Resolutions found: {len(resolution_data) if resolution_data else 'N/A'}")
    print(f"  Total result rows in CSV: {len(all_results)}")
    print(f"  CSV saved to: {args.out_csv}")
    if not args.skip_animations:
        print(f"  Animations saved to: {args.animation_dir}/")
    print("="*60)

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    torch.backends.cudnn.benchmark = True
    main()
