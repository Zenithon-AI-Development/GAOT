import os, argparse, math, json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from omegaconf import OmegaConf
from types import SimpleNamespace
from contextlib import nullcontext

from main import prepare_arg
from src.trainer.sequential_trainer import SequentialTrainer
from src.trainer.static_trainer import StaticTrainer
from src.datasets.data_utils import TestDataset, collate_sequential_batch

# ---------- tiny helpers ----------

def set_mp_spawn():
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

def pick_amp_dtype():
    major, minor = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16

def load_cfg(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    cfg.setup.train = False
    cfg.setup.test  = True
    cfg.setup.ckpt  = True
    arg = SimpleNamespace(**OmegaConf.to_container(cfg, resolve=True))
    arg = prepare_arg(arg)
    return arg

def build_time_indices(trainer, mode:str):
    T_avail = int(trainer.test_loader.dataset.u_data.shape[1])
    step    = int(getattr(trainer.dataset_config, 'time_step', 1))
    maxdiff = int(getattr(trainer.dataset_config, 'max_time_diff', T_avail-1))
    # print(T_avail, step, maxdiff)
    maxdiff = 101
    max_idx = min(T_avail - 1, maxdiff)
    if mode == "autoregressive":
        return np.arange(0, max_idx + 1, step, dtype=int)
    if mode == "direct":
        return np.array([0, max_idx], dtype=int)
    if mode == "star":
        return np.unique(np.linspace(0, max_idx, num=5, dtype=int))
    return np.arange(0, max_idx + 1, step, dtype=int)

def infer_hw_from_coord(coord: torch.Tensor, tol_decimals: int = 6):
    xy = coord.detach().cpu().numpy()
    xs = np.round(xy[:, 0], tol_decimals)
    ys = np.round(xy[:, 1], tol_decimals)
    ux = np.unique(xs); uy = np.unique(ys)
    H, W = len(ux), len(uy)
    if H * W == coord.shape[0]: return int(H), int(W)
    if W * H == coord.shape[0]: return int(W), int(H)
    return None, None

def make_square_grid_like(trainer, S:int):
    dom = trainer.metadata.domain_x  # [[xmin,ymin],[xmax,ymax]]
    xlin = torch.linspace(dom[0][0], dom[1][0], S, dtype=trainer.dtype, device=trainer.device)
    ylin = torch.linspace(dom[0][1], dom[1][1], S, dtype=trainer.dtype, device=trainer.device)
    xv, yv = torch.meshgrid(xlin, ylin, indexing='ij')
    coords = torch.stack([xv, yv], dim=-1).reshape(-1, 2)
    return trainer.data_processor.coord_scaler(coords)

# ---------- metrics you asked for ----------

def rel_l1(pred, target, eps=1e-12):
    num = (pred - target).abs().sum(dim=(-1,-2))  # sum over N,C
    den = target.abs().sum(dim=(-1,-2)).clamp_min(eps)
    return (num/den).mean()  # mean over batch and time window if present

def rel_l2(pred, target, eps=1e-12):
    num = torch.sqrt(((pred - target)**2).sum(dim=(-1,-2)))
    den = torch.sqrt((target**2).sum(dim=(-1,-2))).clamp_min(eps)
    return (num/den).mean()

def mse(pred, target):
    return ((pred - target)**2).mean()

# ---------- timing helpers (AMP + CUDA events) ----------

def cuda_forward_timing_fx(model, pndata, latent_tokens_coord, coord,
                           n_warmup=10, n_iters=50, amp_dtype=None):
    torch.cuda.synchronize()
    autocast_ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else nullcontext()
    with torch.no_grad(), autocast_ctx:
        for _ in range(n_warmup):
            _ = model(latent_tokens_coord=latent_tokens_coord, xcoord=coord, pndata=pndata)
    start_ev = torch.cuda.Event(enable_timing=True); end_ev = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(n_iters):
        start_ev.record()
        with torch.no_grad(), autocast_ctx:
            _ = model(latent_tokens_coord=latent_tokens_coord, xcoord=coord, pndata=pndata)
        end_ev.record()
        torch.cuda.synchronize()
        times.append(start_ev.elapsed_time(end_ev))
    arr = np.array(times, dtype=float)
    return {"mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p90_ms": float(np.percentile(arr, 90)),
            "p99_ms": float(np.percentile(arr, 99)),
            "iters": int(n_iters)}

def cuda_train_step_timing_fx(model, pndata, target, latent_tokens_coord, coord,
                              n_warmup=3, n_iters=10, lr=1e-4, amp_dtype=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    torch.cuda.synchronize()

    latq   = latent_tokens_coord.detach().clone()
    xcoord = coord.detach().clone()
    inp    = pndata.detach().clone()
    tgt    = target.detach().clone()

    autocast_ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    # warmup
    for _ in range(n_warmup):
        opt.zero_grad(set_to_none=True)
        with torch.enable_grad(), autocast_ctx:
            pred = model(latent_tokens_coord=latq, xcoord=xcoord, pndata=inp)
            loss = loss_fn(pred, tgt)
        if scaler.is_enabled():
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()

    start_ev = torch.cuda.Event(enable_timing=True); end_ev = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(n_iters):
        opt.zero_grad(set_to_none=True)
        start_ev.record()
        with torch.enable_grad(), autocast_ctx:
            pred = model(latent_tokens_coord=latq, xcoord=xcoord, pndata=inp)
            loss = loss_fn(pred, tgt)
        if scaler.is_enabled():
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
        end_ev.record()
        torch.cuda.synchronize()
        times.append(start_ev.elapsed_time(end_ev))

    model.eval()
    step_ms = float(np.mean(times))
    bsz = int(pndata.shape[0])
    return {"step_ms_mean": step_ms,
            "step_ms_p90": float(np.percentile(times, 90)),
            "samples_per_s": float(bsz * 1000.0 / step_ms),
            "iters": int(n_iters)}

# ---------- main ----------

def main():
    set_mp_spawn()

    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--config", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=4, help="batch size for AR/test part & for speed sweep target")
    ap.add_argument("--max_rollout_T", type=int, default=50)
    ap.add_argument("--test_samples", type=int, default=9)
    ap.add_argument("--grids", default="64,128,256,512", help="square grid sides for speed sweep")
    ap.add_argument("--out_csv", default=".results/examples/time_dep/gaot_eval_metrics.csv")
    ap.add_argument("--no_density_comp", action="store_true",
                    help="disable density compensation for synthetic square grids")
    args = ap.parse_args()

    # init
    arg = load_cfg(args.config)
    arg.setup["device"] = args.device
    Trainer = {"static": StaticTrainer, "sequential": SequentialTrainer}[arg.setup["trainer_name"]]
    trainer = Trainer(arg).load_ckpt()
    assert trainer.coord_mode == "fx", "This evaluator supports fixed-coords (fx) mode."

    device = trainer.device
    model  = trainer.model.to(device).eval()
    latq   = trainer.latent_tokens_coord.to(device)
    coord  = trainer.coord.to(device)
    stats  = trainer.stats
    tvals  = trainer.t_values
    # print(tvals)
    amp_dtype = pick_amp_dtype()

    # input/output channels GAOT expects ([B,N,Cin] -> [B,N,Cout])
    Cin = int(trainer.num_input_channels)
    Cout= int(trainer.num_output_channels)

    # ===== (A) AR + metrics on real test data =====
    time_indices = build_time_indices(trainer, mode="autoregressive")
    t_out = min(args.max_rollout_T, len(time_indices)-1)
    time_indices = time_indices[:(t_out+1)]

    test_ds = TestDataset(
        u_data=trainer.test_loader.dataset.u_data,
        c_data=trainer.test_loader.dataset.c_data,
        t_values=trainer.test_loader.dataset.t_values,
        metadata=trainer.metadata,
        time_indices=time_indices,
        stats=stats,
        x_data=getattr(trainer.test_loader.dataset, "x_data", None),
        is_variable_coords=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_sequential_batch
    )

    need = args.test_samples
    X_list, Y_list = [], []
    for xin, yseq in test_loader:
        X_list.append(xin); Y_list.append(yseq)
        if sum(t.shape[0] for t in X_list) >= need:
            break
    X_full = torch.cat(X_list, dim=0)[:need].to(device)   # [B*,N,Cin]
    Y_full = torch.cat(Y_list, dim=0)[:need].to(device)   # [B*,T-1,N,Cout]

    with torch.no_grad():
        pred_full = model.autoregressive_predict(
            x_batch=X_full, time_indices=time_indices, t_values=tvals, stats=stats,
            stepper_mode=trainer.stepper_mode, latent_tokens_coord=latq,
            fixed_coord=coord, encoder_nbrs=None, decoder_nbrs=None,
            use_conditional_norm=getattr(trainer.model_config, 'use_conditional_norm', False)
        )  # [B*, T-1, N, Cout]

    # stepwise metrics (what you asked for)
    per_step_metrics = []
    Tm1 = pred_full.shape[1]
    for k in range(Tm1):
        yk = Y_full[:, k]     # [B*,N,Cout]
        pk = pred_full[:, k]
        per_step_metrics.append({
            "k": int(k+1),  # 1..Tm1
            "rel_l1": float(rel_l1(pk, yk).item()),
            "rel_l2": float(rel_l2(pk, yk).item()),
            "mse":    float(mse(pk, yk).item()),
        })

    # overall aggregates over all rollout steps
    overall = {
        "rel_l1": float(rel_l1(pred_full.reshape(-1, pred_full.shape[-2], pred_full.shape[-1]),
                               Y_full.reshape(-1, Y_full.shape[-2], Y_full.shape[-1])).item()),
        "rel_l2": float(rel_l2(pred_full.reshape(-1, pred_full.shape[-2], pred_full.shape[-1]),
                               Y_full.reshape(-1, Y_full.shape[-2], Y_full.shape[-1])).item()),
        "mse":    float(mse(pred_full, Y_full).item()),
    }

    # also time native grid forward/train at B=4 (as requested)
    native_B = min(args.batch, X_full.shape[0])
    X_native = X_full[:native_B]
    Y_native1 = Y_full[:native_B, 0]  # one-step target for train timing

    lat_native = cuda_forward_timing_fx(model, X_native, latq, coord,
                                        n_warmup=10, n_iters=50, amp_dtype=amp_dtype)
    tr_native  = cuda_train_step_timing_fx(model, X_native, Y_native1, latq, coord,
                                           n_warmup=3, n_iters=10, lr=1e-4, amp_dtype=amp_dtype)

    H0, W0 = infer_hw_from_coord(coord)
    native_label = f"{H0}x{W0}" if (H0 and W0) else f"N={coord.shape[0]}"

    rows = []
    rows.append({
        "section": "native_eval",
        "grid": native_label,
        "B": int(native_B),
        "rollout_steps_available": int(Tm1),
        "rollout_steps_used": int(Tm1),
        "per_step_metrics_json": json.dumps(per_step_metrics),  # array of {k, rel_l1, rel_l2, mse}
        "overall_rel_l1": overall["rel_l1"],
        "overall_rel_l2": overall["rel_l2"],
        "overall_mse":    overall["mse"],
        "infer_ms_mean": lat_native["mean_ms"],
        "infer_p50_ms":  lat_native["p50_ms"],
        "infer_p90_ms":  lat_native["p90_ms"],
        "infer_p99_ms":  lat_native["p99_ms"],
        "train_step_ms_mean": tr_native["step_ms_mean"],
        "train_step_ms_p90":  tr_native["step_ms_p90"],
        "train_samples_per_s": tr_native["samples_per_s"],
    })

    print(f"[AR] steps_avail={Tm1} (requested up to {args.max_rollout_T}), "
          f"overall rel_l1={overall['rel_l1']:.6f}, rel_l2={overall['rel_l2']:.6f}, mse={overall['mse']:.6e}")
    print(f"[native {native_label} @B={native_B}] infer p50={lat_native['p50_ms']:.3f} ms, "
          f"train {tr_native['samples_per_s']:.2f} samples/s")


    # # ===== (B) SPEED SCALING on square dummy grids, B=4 target =====
    # sizes = [int(s) for s in args.grids.split(",")]
    # target_B = int(args.batch)
    # base_dtype = next(model.parameters()).dtype  # <- FIX for your dtype error

    # for S in sizes:
    #     # Build square coords
    #     coord_sq = make_square_grid_like(trainer, S).to(device)

    #     # Density compensation only for synthetic grids (prevents OOM in MAGNO attention)
    #     if not args.no_density_comp and (H0 and W0):
    #         density = (S * S) / float(H0 * W0)
    #         if density > 1.0:
    #             coord_sq = coord_sq * math.sqrt(density)

    #     N = coord_sq.shape[0]

    #     # Try B=4; if OOM, fall back to B=2 then B=1; extrapolate throughput back to B=4
    #     for try_B in [target_B, max(1, target_B//2), 1]:
    #         try:
    #             Xg = torch.zeros((try_B, N, Cin), dtype=base_dtype, device=device)
    #             Yg = torch.zeros((try_B, N, Cout), dtype=base_dtype, device=device)

    #             lat_g = cuda_forward_timing_fx(model, Xg, latq, coord_sq,
    #                                            n_warmup=10, n_iters=50, amp_dtype=amp_dtype)
    #             tr_g  = cuda_train_step_timing_fx(model, Xg, Yg, latq, coord_sq,
    #                                               n_warmup=3, n_iters=10, lr=1e-4, amp_dtype=amp_dtype)

    #             # If we had to reduce B, scale throughput to the requested B=4 for reporting.
    #             scaled_sps = tr_g["samples_per_s"] * (target_B / try_B)

    #             rows.append({
    #                 "section": "square_speed",
    #                 "grid": f"{S}x{S}",
    #                 "B_measured": int(try_B),
    #                 "B_reported": int(target_B),
    #                 "rollout_steps_available": 0,
    #                 "rollout_steps_used": 0,
    #                 "per_step_metrics_json": "[]",
    #                 "overall_rel_l1": float("nan"),
    #                 "overall_rel_l2": float("nan"),
    #                 "overall_mse":    float("nan"),
    #                 "infer_ms_mean": lat_g["mean_ms"],
    #                 "infer_p50_ms":  lat_g["p50_ms"],
    #                 "infer_p90_ms":  lat_g["p90_ms"],
    #                 "infer_p99_ms":  lat_g["p99_ms"],
    #                 "train_step_ms_mean": tr_g["step_ms_mean"],
    #                 "train_step_ms_p90":  tr_g["step_ms_p90"],
    #                 "train_samples_per_s_measured": tr_g["samples_per_s"],
    #                 "train_samples_per_s_reported_B4": scaled_sps,
    #             })
    #             print(f"[speed {S}x{S}] B{try_B}→report B{target_B}: "
    #                   f"infer p50={lat_g['p50_ms']:.3f} ms, "
    #                   f"train {scaled_sps:.2f} samples/s (reported)")
    #             break  # done for this grid
    #         except torch.cuda.OutOfMemoryError:
    #             torch.cuda.empty_cache()
    #             if try_B == 1:
    #                 print(f"[skip] {S}x{S}: OOM even with AMP & B=1. Skipping.")
    #             else:
    #                 print(f"[retry] {S}x{S}: OOM at B={try_B}, retrying with smaller B.")
    #             continue

    # save
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"\nSaved metrics -> {args.out_csv}")

if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    main()
