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
from src.utils.metrics import compute_batch_errors, compute_final_metric

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
    """Evaluate model on a single data loader and return resolution-grouped results."""
    resolution_stats = defaultdict(lambda: {
        'count': 0,
        'sum_mse_norm': 0.0,
        'sum_mse_den': 0.0,
        'sum_rel1': 0.0,
        'sum_rel2': 0.0,
        'chunk_errors': [],
        'coord': None,
        'sample_input': None,
        'sample_target': None,
        'sample_pred': None
    })
    
    print(f"\nCollecting predictions from {loader_name.upper()} set...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if not isinstance(batch, dict):
                print("\nWARNING: Unexpected batch format. Expected dict with 'x', 'y', 'coord' keys.")
                print("This might not be a multi-resolution dataset.")
                break
            
            x_batch = batch["x"].to(device)
            y_batch = batch["y"].to(device)
            coord = batch["coord"].to(device)
            
            N = coord.shape[0]
            B = x_batch.shape[0]
            
            if max_samples > 0 and resolution_stats[N]['count'] >= max_samples:
                continue
            
            if getattr(trainer.model_config, 'use_conditional_norm', False):
                x_input = x_batch[..., :-1]
            else:
                x_input = x_batch
            
            pred = model(
                latent_tokens_coord=trainer.latent_tokens_coord.to(device),
                xcoord=coord,
                pndata=x_input
            )
            
            resolution_stats[N]['count'] += B
            
            mse_norm_per_sample = torch.mean((pred - y_batch) ** 2, dim=(-2, -1))
            resolution_stats[N]['sum_mse_norm'] += mse_norm_per_sample.sum().item()
            
            u_mean = stats["u"]["mean"].to(device)
            u_std = stats["u"]["std"].to(device)
            pred_denorm = pred * u_std + u_mean
            target_denorm = y_batch * u_std + u_mean
            diff_denorm = pred_denorm - target_denorm
            
            mse_den_per_sample = torch.mean(diff_denorm ** 2, dim=(-2, -1))
            resolution_stats[N]['sum_mse_den'] += mse_den_per_sample.sum().item()
            
            abs_diff = diff_denorm.abs().sum(dim=(-2, -1))
            abs_target = target_denorm.abs().sum(dim=(-2, -1)).clamp_min(1e-12)
            rel_l1_per_sample = abs_diff / abs_target
            resolution_stats[N]['sum_rel1'] += rel_l1_per_sample.sum().item()
            
            l2_diff = torch.sqrt((diff_denorm ** 2).sum(dim=(-2, -1)))
            l2_target = torch.sqrt((target_denorm ** 2).sum(dim=(-2, -1))).clamp_min(1e-12)
            rel_l2_per_sample = l2_diff / l2_target
            resolution_stats[N]['sum_rel2'] += rel_l2_per_sample.sum().item()
            
            rel_errors = compute_batch_errors(
                target_denorm[:, None, :, :],
                pred_denorm[:, None, :, :],
                trainer.metadata
            ).cpu()
            resolution_stats[N]['chunk_errors'].append(rel_errors)
            
            if resolution_stats[N]['sample_input'] is None:
                resolution_stats[N]['sample_input'] = x_batch[0:1].cpu()
                resolution_stats[N]['sample_target'] = y_batch[0:1].cpu()
                resolution_stats[N]['sample_pred'] = pred[0:1].detach().cpu()
            if resolution_stats[N]['coord'] is None:
                resolution_stats[N]['coord'] = coord.cpu()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")
    
    if not resolution_stats:
        print(f"  WARNING: No data collected from {loader_name} set")
        return None
    
    print(f"  Found {len(resolution_stats)} different resolutions in {loader_name} set")
    
    results = []
    total_samples = 0
    weighted_rel_l1 = 0.0
    weighted_rel_l2 = 0.0
    weighted_mse = 0.0
    weighted_mse_norm = 0.0
    overall_chunk_errors = []
    
    for N in sorted(resolution_stats.keys()):
        data = resolution_stats[N]
        num_samples = data['count']
        if num_samples == 0:
            continue
        
        mse_norm = data['sum_mse_norm'] / num_samples
        mse_den = data['sum_mse_den'] / num_samples
        rel_l1 = data['sum_rel1'] / num_samples
        rel_l2 = data['sum_rel2'] / num_samples
        
        if data['chunk_errors']:
            chunk_errors = torch.cat(data['chunk_errors'], dim=0)
            overall_chunk_errors.append(chunk_errors)
            gaot_rel = compute_final_metric(chunk_errors)
        else:
            gaot_rel = float('nan')
        
        coord = data['coord']
        res_label = get_resolution_label(N, coord)
        
        total_samples += num_samples
        weighted_rel_l1 += rel_l1 * num_samples
        weighted_rel_l2 += rel_l2 * num_samples
        weighted_mse += mse_den * num_samples
        weighted_mse_norm += mse_norm * num_samples
        
        results.append({
            'dataset': loader_name,
            'resolution': res_label,
            'N_points': N,
            'num_samples': num_samples,
            'mse_normalized': mse_norm,
            'rel_l1': rel_l1,
            'rel_l2': rel_l2,
            'mse': mse_den,
            'gaot_rel_l1': gaot_rel
        })
    
    if total_samples > 0:
        overall_mse_norm = weighted_mse_norm / total_samples
        overall_rel_l1 = weighted_rel_l1 / total_samples
        overall_rel_l2 = weighted_rel_l2 / total_samples
        overall_mse = weighted_mse / total_samples
        if overall_chunk_errors:
            overall_chunk_errors = torch.cat(overall_chunk_errors, dim=0)
            overall_gaot_rel = compute_final_metric(overall_chunk_errors)
        else:
            overall_gaot_rel = float('nan')
    else:
        overall_mse_norm = overall_rel_l1 = overall_rel_l2 = overall_mse = overall_gaot_rel = float('nan')
    
    results.append({
        'dataset': loader_name,
        'resolution': 'OVERALL',
        'N_points': -1,
        'num_samples': total_samples,
        'mse_normalized': overall_mse_norm,
        'rel_l1': overall_rel_l1,
        'rel_l2': overall_rel_l2,
        'mse': overall_mse,
        'gaot_rel_l1': overall_gaot_rel
    })
    
    return resolution_stats, results

def main():
    set_mp_spawn()
    
    ap = argparse.ArgumentParser(description="Evaluate GAOT model performance per resolution with animations")
    ap.add_argument("-c", "--config", required=True, help="Path to config file")
    ap.add_argument("--device", default="cuda:0", help="Device to use")
    ap.add_argument("--batch", type=int, default=4, help="Batch size")
    ap.add_argument("--max_samples", type=int, default=0, help="Cap samples per resolution (0 = use all)")
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
    
    # Store resolution data for animation (prefer validation set)
    val_resolution_data = None
    last_resolution_data = None
    
    for loader, loader_name in loaders_to_eval:
        print("=" * 60)
        result = evaluate_loader(loader, loader_name, model, trainer, device, args.max_samples, stats)
        
        if result is not None:
            resolution_data, results = result
            last_resolution_data = resolution_data
            
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
                    print(f"  GAOT relative L1 (trainer metric): {res_dict.get('gaot_rel_l1', float('nan')):.6f}")
                else:
                    print(f"\nResolution: {res_dict['resolution']} (N={res_dict['N_points']})")
                    print(f"  Samples: {res_dict['num_samples']}")
                    print(f"  MSE (normalized, matches training val loss): {res_dict['mse_normalized']:.6e}")
                    print(f"  Relative L1 (denormalized): {res_dict['rel_l1']:.6f}")
                    print(f"  Relative L2 (denormalized): {res_dict['rel_l2']:.6f}")
                    print(f"  MSE (denormalized): {res_dict['mse']:.6e}")
                    print(f"  GAOT relative L1 (trainer metric): {res_dict.get('gaot_rel_l1', float('nan')):.6f}")
            
            all_results.extend(results)
        else:
            print(f"  Skipping {loader_name} set (no data collected)")
    
    if not all_results:
        print("\nERROR: No results collected from any dataset")
        return
    
    # Use validation data for animations when available (matches training visuals)
    animation_data = val_resolution_data if val_resolution_data is not None else last_resolution_data
    
    # Save CSV results (includes both validation and test if available)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved CSV results to: {args.out_csv}")
    
    # Generate animations per resolution
    if not args.skip_animations:
        if animation_data is None:
            print("\nNo resolution data available for animations; skipping.")
        else:
            print("\n" + "="*60)
            print("Generating animations per resolution...")
            print("="*60)
            
            os.makedirs(args.animation_dir, exist_ok=True)
            
            time_step = int(getattr(trainer.dataset_config, 'time_step', 1))
            time_indices = build_time_indices_for_animation(args.max_rollout_steps, step=time_step)
            time_indices_anim = time_indices[::10]
            print(f"Using {len(time_indices_anim)} frames for animations (every 10th frame from {len(time_indices)} total steps)")
            
            for N in sorted(animation_data.keys()):
                data = animation_data[N]
                coord = data['coord'].to(device)
                res_label = get_resolution_label(N, coord)
                
                print(f"\nCreating animation for resolution: {res_label}")
                
                first_input_full = data['sample_input'].to(device)  # [1, N, C_full]
                u_channels = stats["u"]["mean"].shape[-1]
                c_channels = stats["c"]["mean"].shape[-1] if "c" in stats else 0
                first_input = first_input_full[..., :u_channels + c_channels]
                
                try:
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
                    
                    u_mean = stats["u"]["mean"].cpu()
                    u_std = stats["u"]["std"].cpu()
                    pred_denorm = pred_sequence[0].cpu() * u_std + u_mean  # [T-1, N, C]
                    gt_denorm = pred_denorm.clone()
                    
                    input_denorm = (first_input[0, :, :u_channels].cpu() * u_std + u_mean).numpy()
                    coord_phys = trainer.data_processor.coord_scaler.inverse_transform(coord.cpu()).numpy()
                    
                    gt_anim = gt_denorm[::10].numpy()
                    pred_anim = pred_denorm[::10].numpy()
                    
                    if hasattr(trainer, 't_values'):
                        t_vals = trainer.t_values
                        time_values = [float(t_vals[idx]) for idx in time_indices_anim]
                    else:
                        time_values = [float(idx) for idx in time_indices_anim]
                    
                    animation_path = os.path.join(args.animation_dir, f"animation_{res_label}.gif")
                    
                    create_sequential_animation(
                        gt_sequence=gt_anim,
                        pred_sequence=pred_anim,
                        coords=coord_phys,
                        save_path=animation_path,
                        input_data=input_denorm,
                        time_values=time_values,
                        interval=100,
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
