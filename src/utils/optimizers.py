import torch
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import numpy as np
import time
from copy import deepcopy
from dataclasses import dataclass

from ..utils.timer import IterProfiler

# --- W&B helpers (safe no-op if wandb disabled) ---
def _wb_run(trainer):
    return getattr(trainer, "wandb_run", None)

def _wb_log(trainer, payload: dict, step: int | None = None):
    run = _wb_run(trainer)
    if run is None:
        return
    import wandb
    wandb.log(payload, step=step)

def _current_lr(optimizer, scheduler):
    try:
        if scheduler is not None:
            return float(scheduler.get_last_lr()[0])
    except Exception:
        pass
    return float(optimizer.param_groups[0]["lr"])

###############
# Config
###############
@dataclass
class OptimizerArgsConfig:
    lr: float = 1e-3                    # Initial learning rate
    weight_decay: float = 1e-3          # L2 regularization (weight decay) coefficient
    epoch: int = 100                    # Total number of training epochs
    loss_scale: float = 1.0             # Loss scaling factor 
    eval_every_eps: int = 1             # Evaluate every n epochs
    scheduler: str = "mix"              # Learning rate scheduler type, support ['step', 'cos', 'exp', 'mix']
    early_save_metric: str = 'val'      # Metric for early stopping, support ['train', 'val']
    # for mix scheduler
    max_lr: float = 1e-2                # Maximum learning rate for the cosine annealing phase
    min_lr: float = 1e-5                # Minimum learning rate for the cosine annealing phase
    final_lr: float = 1e-5              # Final learning rate for the exponential decay phase
    # for step scheduler
    scheduler_step_size: int = 100      # Step size (number of epochs) for StepLR scheduler
    scheduler_gamma: float = 0.8        # Multiplicative factor for learning rate decay in StepLR scheduler
    scheduler_T_max: int = 100          # Maximum number of iterations (usually total epochs) for CosineAnnealingLR scheduler
    scheduler_eta_min: float = 1e-4     # Minimum learning rate for CosineAnnealingLR scheduler

###############
# Scheduler
###############
class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epochs, warmup_epochs, cosine_epochs, exp_decay_epochs,
                 initial_lr, max_lr, min_lr, final_lr, last_epoch=-1):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.cosine_epochs = cosine_epochs
        self.exp_decay_epochs = exp_decay_epochs
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.final_lr = final_lr
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # first phase (warm up): initial_lr to max_lr
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * (self.last_epoch / max(1, self.warmup_epochs - 1))
        elif self.last_epoch < self.warmup_epochs + self.cosine_epochs:
            # second stage (cosine): max_lr to min_lr
            epoch = self.last_epoch - self.warmup_epochs
            cosine_ratio = (1 + np.cos(np.pi * epoch / self.cosine_epochs)) / 2
            lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_ratio
        else:
            # third stage (expontential)： min_lr to final_lr
            epoch = self.last_epoch - self.warmup_epochs - self.cosine_epochs
            decay_steps = max(1, self.exp_decay_epochs - 1)
            lr = self.min_lr * ((self.final_lr / self.min_lr) ** (epoch / decay_steps))
        return [lr for _ in self.optimizer.param_groups]

###############
# Optimizer
###############
class AdamOptimizer:
    optimizer: torch.optim.Adam
    scheduler: torch.optim.lr_scheduler.StepLR
    epoch: int
    batch_size: int
    lr: float   
    loss_scale: float
    eval_every_eps: int

    def __init__(self, params, config):
        self.optimizer = torch.optim.Adam(params, lr=config.lr)
        self.epoch = config.epoch
        self.lr = config.lr  
        self.loss_scale = config.loss_scale
        self.eval_every_eps = config.eval_every_eps
        self.early_save_metric = config.early_save_metric.lower()

        if self.early_save_metric not in ['train', 'val']:
            raise ValueError("`early_save_metric` must be either 'train' or 'val'.")

        if config.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
        elif config.scheduler == 'cos':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.scheduler_T_max, eta_min=config.scheduler_eta_min)
        elif config.scheduler == 'exp':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config.scheduler_gamma)
        elif config.scheduler == 'mix':
            warmup_epochs = int(0.02 * self.epoch)
            cosine_epochs = int(0.96 * self.epoch)
            exp_decay_epochs = self.epoch - warmup_epochs - cosine_epochs
            if warmup_epochs == 0:
                warmup_epochs = 1
                cosine_epochs -= 1
            if exp_decay_epochs == 0:
                exp_decay_epochs = 1
                cosine_epochs -= 1
            self.scheduler = CustomLRScheduler(
                optimizer=self.optimizer,
                total_epochs=self.epoch,
                warmup_epochs=warmup_epochs,
                cosine_epochs=cosine_epochs,
                exp_decay_epochs=exp_decay_epochs,
                initial_lr=self.lr,
                max_lr=config.max_lr,
                min_lr=config.min_lr,
                final_lr=config.final_lr
            )
        else:
            self.scheduler = None

    def optimize(self, trainer: 'BaseTrainer',
                 description: str = "AdamWOptimizer",
                 color: str = "blue"):
        time_total = 0.0
        best_loss, best_epoch, best_state = np.inf, -1, None
        losses, epochs, val_epochs, val_losses = [], [], [], []

        # global step to make W&B charts pretty
        global_step = getattr(trainer, "global_step", 0)
        log_every = int(getattr(trainer.config, "wandb", {}).get("log_interval", 0) or 0)

        pbar = tqdm(total=self.epoch, desc=description, colour=color)
        for epoch in range(self.epoch):
            trainer.model.train()
            total_loss = 0.0

            data_t0 = time.time()
            for ib, batch in enumerate(trainer.train_loader):
                data_dt = time.time() - data_t0
                # if data_dt > 10:  # adjust threshold
                #     print(f"[DEBUG] slow data fetch: {data_dt:.2f}s (batch {ib})")

                t0 = time.time()
                self.optimizer.zero_grad(set_to_none=True)

                # FWD
                torch.cuda.synchronize(trainer.device) if trainer.device.type == "cuda" else None
                fwd_t0 = time.time()
                train_loss = trainer.train_step(batch)
                torch.cuda.synchronize(trainer.device) if trainer.device.type == "cuda" else None
                fwd_dt = time.time() - fwd_t0

                # BWD
                bwd_t0 = time.time()
                train_loss.backward()
                torch.cuda.synchronize(trainer.device) if trainer.device.type == "cuda" else None
                bwd_dt = time.time() - bwd_t0

                # STEP
                step_t0 = time.time()
                self.optimizer.step()
                torch.cuda.synchronize(trainer.device) if trainer.device.type == "cuda" else None
                step_dt = time.time() - step_t0

                total_loss += train_loss.detach()
                
                # Debug: Print first few batch losses to understand the scale
                # if epoch == 0 and ib < 3:
                #     print(f"[DEBUG TRAIN BATCH] Epoch {epoch}, batch {ib}: loss={train_loss.detach().cpu().item():.6f}, accumulated_total={total_loss.cpu().item():.6f}")

                # log every batch while debugging
                _wb_log(trainer, {
                    "timing/data_sec": data_dt,
                    "timing/fwd_sec": fwd_dt,
                    "timing/bwd_sec": bwd_dt,
                    "timing/step_sec": step_dt,
                })

                data_t0 = time.time()  # start timing the *next* data fetch
                # if ib == 0:
                #     print(f"[DEBUG] first batch timings: data={data_dt:.2f}s fwd={fwd_dt:.2f}s bwd={bwd_dt:.2f}s step={step_dt:.2f}s")

            if self.scheduler is not None:
                self.scheduler.step()

            # epoch-end eval
            num_batches_actual = ib + 1  # Actual number of batches processed
            num_batches_from_len = len(trainer.train_loader) if hasattr(trainer.train_loader, '__len__') else num_batches_actual
            # if epoch == 0:
            #     print(f"[DEBUG OPTIMIZER] Epoch {epoch} end: batches_processed={num_batches_actual}, len(loader)={num_batches_from_len}, total_loss={total_loss.cpu().item():.6f}")
            train_loss = total_loss.cpu().item() / num_batches_from_len
            # if epoch == 0:
            #     print(f"[DEBUG OPTIMIZER] Epoch {epoch} end: computed train_loss={train_loss:.6f} (total_loss/{num_batches_from_len})")
            val_loss = None
            if (epoch + 1) % self.eval_every_eps == 0:
                val_result = trainer.validate(trainer.val_loader)
                # Handle both dict return (new) and float return (old) for backward compatibility
                if isinstance(val_result, dict):
                    val_loss = val_result.get("loss", 0.0)
                else:
                    val_loss = val_result
                pbar.set_postfix({"loss": train_loss, "val_loss": val_loss})
                val_losses.append(val_loss)
                val_epochs.append(epoch)
            else:
                pbar.set_postfix({"loss": train_loss})

            losses.append(train_loss)
            epochs.append(epoch)
            pbar.update(1)

            # choose which metric to early-save on
            current_loss = val_loss if (self.early_save_metric == 'val' and val_loss is not None) else train_loss
            if current_loss < best_loss:
                best_loss, best_epoch = current_loss, epoch
                best_state = deepcopy(trainer.model.state_dict())

            # W&B epoch logging
            _wb_log(
                trainer,
                {
                    "train/epoch": epoch,
                    "train/loss": float(train_loss),
                    **({"valid/loss": float(val_loss)} if val_loss is not None else {}),
                    "train/lr": _current_lr(self.optimizer, self.scheduler),
                },
                step=global_step,
            )

        if best_state is not None:
            trainer.model.load_state_dict(best_state)

        # Push summary so Sweeps/Dashboards can snapshot bests
        run = _wb_run(trainer)
        if run is not None:
            run.summary["best/epoch"] = int(best_epoch)
            run.summary["best/loss"] = float(best_loss)

        pbar.close()
        trainer.global_step = global_step  # persist across calls if needed

        return {
            "train": {"loss": losses, "epoch": epochs},
            "valid": {"loss": val_losses, "epoch": val_epochs},
            "best": {"epoch": best_epoch, "loss": best_loss},
            "time": time_total,
        }

class AdamWOptimizer:
    optimizer: torch.optim.AdamW
    scheduler: torch.optim.lr_scheduler._LRScheduler
    epoch: int
    batch_size: int
    lr: float
    loss_scale: float
    eval_every_eps: int

    def __init__(self, params, config):
        self.optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
        self.epoch = config.epoch
        self.lr = config.lr  
        self.loss_scale = config.loss_scale
        self.eval_every_eps = config.eval_every_eps
        self.early_save_metric = config.early_save_metric.lower()

        if self.early_save_metric not in ['train', 'val']:
            raise ValueError("`early_save_metric` must be either 'train' or 'val'.")
        
        if config.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
        elif config.scheduler == 'cos':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.scheduler_T_max, eta_min=config.scheduler_eta_min)
        elif config.scheduler == 'exp':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config.scheduler_gamma)
        elif config.scheduler == 'mix':
            warmup_epochs = int(0.02 * self.epoch)
            cosine_epochs = int(0.90 * self.epoch)
            exp_decay_epochs = self.epoch - warmup_epochs - cosine_epochs
            if warmup_epochs == 0:
                warmup_epochs = 1
                cosine_epochs -= 1
            if exp_decay_epochs == 0:
                exp_decay_epochs = 1
                cosine_epochs -= 1
            self.scheduler = CustomLRScheduler(
                optimizer=self.optimizer,
                total_epochs=self.epoch,
                warmup_epochs=warmup_epochs,
                cosine_epochs=cosine_epochs,
                exp_decay_epochs=exp_decay_epochs,
                initial_lr=self.lr,
                max_lr=config.max_lr,
                min_lr=config.min_lr,
                final_lr=config.final_lr
            )
        else:
            self.scheduler = None

    def optimize(self, trainer: 'BaseTrainer',
                 description: str = "AdamWOptimizer",
                 color: str = "blue"):
        time_total = 0.0
        best_loss, best_epoch, best_state = np.inf, -1, None
        losses, epochs, val_epochs, val_losses = [], [], [], []

        ckpt_every_steps  = int(getattr(trainer.setup_config, "ckpt_every_steps", 0))     # 0 = disabled
        ckpt_every_epochs = int(getattr(trainer.setup_config, "ckpt_every_epochs", 10))    # save each epoch by default
        save_best         = bool(getattr(trainer.setup_config, "save_best", True))

        # global step to make W&B charts pretty
        global_step = getattr(trainer, "global_step", 0)
        log_every = int(getattr(trainer.config, "wandb", {}).get("log_interval", 0) or 0)

    #     ### With time measurement per batch
    #     # pbar = tqdm(total=self.epoch, desc=description, colour=color)
    #     # for epoch in range(self.epoch):
    #     #     trainer.model.train()
    #     #     total_loss = 0.0

    #     #     data_t0 = time.time()
    #     #     for ib, batch in enumerate(trainer.train_loader):
    #     #         data_dt = time.time() - data_t0
    #     #         if data_dt > 10:  # adjust threshold
    #     #             print(f"[DEBUG] slow data fetch: {data_dt:.2f}s (batch {ib})")

    #     #         t0 = time.time()
    #     #         self.optimizer.zero_grad(set_to_none=True)

    #     #         # FWD
    #     #         torch.cuda.synchronize(trainer.device) if trainer.device.type == "cuda" else None
    #     #         fwd_t0 = time.time()
    #     #         train_loss = trainer.train_step(batch)
    #     #         torch.cuda.synchronize(trainer.device) if trainer.device.type == "cuda" else None
    #     #         fwd_dt = time.time() - fwd_t0

    #     #         # BWD
    #     #         bwd_t0 = time.time()
    #     #         train_loss.backward()
    #     #         torch.cuda.synchronize(trainer.device) if trainer.device.type == "cuda" else None
    #     #         bwd_dt = time.time() - bwd_t0

    #     #         # STEP
    #     #         step_t0 = time.time()
    #     #         self.optimizer.step()
    #     #         torch.cuda.synchronize(trainer.device) if trainer.device.type == "cuda" else None
    #     #         step_dt = time.time() - step_t0

    #     #         total_loss += train_loss.detach()

    #     #         # log every batch while debugging
    #     #         _wb_log(trainer, {
    #     #             "timing/data_sec": data_dt,
    #     #             "timing/fwd_sec": fwd_dt,
    #     #             "timing/bwd_sec": bwd_dt,
    #     #             "timing/step_sec": step_dt,
    #     #         })

    #     #         data_t0 = time.time()  # start timing the *next* data fetch
    #     #         if ib in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:  # first 10 batches
    #     #             print(f"[DEBUG] {ib}th b  gs: data={data_dt:.2f}s fwd={fwd_dt:.2f}s bwd={bwd_dt:.2f}s step={step_dt:.2f}s")

    #     ### With progress bar per epoch
    #     # pbar_epoch = tqdm(total=self.epoch, desc=description, colour=color)
    #     # for epoch in range(self.epoch):
    #     #     trainer.model.train()
    #     #     total_loss = 0.0

    #     #     # inner bar: batches
    #     #     num_batches = len(trainer.train_loader)
    #     #     bar = tqdm(
    #     #         total=num_batches,
    #     #         desc=f"Epoch {epoch+1}/{self.epoch}",
    #     #         leave=False
    #     #     )

    #     #     epoch_t0 = time.time()
    #     #     for ib, batch in enumerate(trainer.train_loader):
    #     #         batch_t0 = time.time()
    #     #         self.optimizer.zero_grad(set_to_none=True)

    #     #         # forward / backward / step
    #     #         train_loss = trainer.train_step(batch)
    #     #         train_loss.backward()
    #     #         self.optimizer.step()

    #     #         total_loss += train_loss.detach()
    #     #         # update inner bar
    #     #         bar.set_postfix(
    #     #             loss=float(train_loss.detach().cpu()),
    #     #             lr=_current_lr(self.optimizer, self.scheduler)
    #     #         )
    #     #         bar.update(1)

    #     #     bar.close()

    #     #     # scheduler step
    #     #     if self.scheduler is not None:
    #     #         self.scheduler.step()

    #     #     # epoch-end eval
    #     #     train_loss = total_loss.cpu().item() / num_batches
    #     #     val_loss = None
    #     #     if (epoch + 1) % self.eval_every_eps == 0:
    #     #         val_loss = trainer.validate(trainer.val_loader)

    #     #     # show on epoch bar
    #     #     pbar_epoch.set_postfix({
    #     #         "loss": f"{train_loss:.4e}",
    #     #         **({"val": f"{val_loss:.4e}"} if val_loss is not None else {})
    #     #     })
    #     #     pbar_epoch.update(1)

    #     #     # wandb epoch log
    #     #     _wb_log(
    #     #         trainer,
    #     #         {
    #     #             "train/epoch": epoch,
    #     #             "train/loss": float(train_loss),
    #     #             **({"valid/loss": float(val_loss)} if val_loss is not None else {}),
    #     #             "train/lr": _current_lr(self.optimizer, self.scheduler),
    #     #             "epoch/seconds": time.time() - epoch_t0,
    #     #             "epoch/batches": num_batches,
    #     #         }
    #     #     )

    #     # pbar_epoch.close()
        
    #     ## original
        pbar = tqdm(total=self.epoch, desc=description, colour=color)
        for epoch in range(self.epoch):
            trainer.model.train()
            total_loss = 0.0

            for batch in trainer.train_loader:
                self.optimizer.zero_grad()
                train_loss = trainer.train_step(batch)
                train_loss.backward()
                # total_norm = 0.0
                # for p in self.model.parameters():
                #     if p.grad is not None:
                #         total_norm += p.grad.data.norm(2).item() ** 2
                # total_norm = total_norm ** 0.5
                # print(f"[SANITY] grad L2 ≈ {total_norm:.3e}")

                self.optimizer.step()

                total_loss += train_loss.detach()

                # (optional) step-wise logging (lightweight)
                global_step += 1
                if log_every and (global_step % log_every == 0):
                    _wb_log(
                        trainer,
                        {
                            "train/step_loss": float(train_loss.detach().cpu()),
                            "train/lr": _current_lr(self.optimizer, self.scheduler),
                        },
                        step=global_step,
                    )
                # if (global_step % ckpt_every_steps) == 0:
                #     trainer.save_ckpt_last(epoch=epoch, extra={"global_step": global_step})


            if self.scheduler is not None:
                self.scheduler.step()

            # epoch-end eval
            train_loss = total_loss.cpu().item() / len(trainer.train_loader)
            val_result = None
            val_loss = None
            val_rel_l1 = None
            val_rel_l2 = None
            if (epoch + 1) % self.eval_every_eps == 0:
                val_result = trainer.validate(trainer.val_loader)
                # Handle both dict return (new) and float return (old) for backward compatibility
                if isinstance(val_result, dict):
                    val_loss = val_result.get("loss", 0.0)
                    val_rel_l1 = val_result.get("rel_l1", 0.0)
                    val_rel_l2 = val_result.get("rel_l2", 0.0)
                else:
                    val_loss = val_result
                pbar.set_postfix({"loss": train_loss, "val_loss": val_loss})
                val_losses.append(val_loss)
                val_epochs.append(epoch)
            else:
                pbar.set_postfix({"loss": train_loss})

            losses.append(train_loss)
            epochs.append(epoch)
            pbar.update(1)

            # choose which metric to early-save on
            current_loss = val_loss if (self.early_save_metric == 'val' and val_loss is not None) else train_loss
            if current_loss < best_loss:
                best_loss, best_epoch = current_loss, epoch
                best_state = deepcopy(trainer.model.state_dict())
                trainer.save_ckpt_best(epoch=epoch, best_loss=best_loss)

            if ckpt_every_epochs > 0 and ((epoch + 1) % ckpt_every_epochs == 0):
                trainer.save_ckpt_epoch(epoch=epoch + 1)

            # W&B epoch logging
            wb_payload = {
                "train/epoch": epoch,
                "train/loss": float(train_loss),
                "train/lr": _current_lr(self.optimizer, self.scheduler),
            }
            if val_loss is not None:
                wb_payload["valid/loss"] = float(val_loss)
                if val_rel_l1 is not None:
                    wb_payload["valid/rel_l1"] = float(val_rel_l1)
                if val_rel_l2 is not None:
                    wb_payload["valid/rel_l2"] = float(val_rel_l2)
            _wb_log(trainer, wb_payload, step=global_step)


        if best_state is not None:
            trainer.model.load_state_dict(best_state)

        # Push summary so Sweeps/Dashboards can snapshot bests
        run = _wb_run(trainer)
        if run is not None:
            run.summary["best/epoch"] = int(best_epoch)
            run.summary["best/loss"] = float(best_loss)

        pbar.close()
        trainer.global_step = global_step  # persist across calls if needed

        return {
            "train": {"loss": losses, "epoch": epochs},
            "valid": {"loss": val_losses, "epoch": val_epochs},
            "best": {"epoch": best_epoch, "loss": best_loss},
            "time": time_total,
        }
    # def optimize(self, trainer: 'BaseTrainer',
    #              description: str = "AdamWOptimizer",
    #              color: str = "blue"):
    #     profile_enabled = bool(getattr(trainer.setup_config, "profile", True))
    #     print_every = int(getattr(trainer.setup_config, "profile_print_every", 50))
    #     prof = IterProfiler(enabled=profile_enabled)
    #     trainer.model._profiler = prof

    #     time_total = 0.0
    #     best_loss, best_epoch, best_state = np.inf, -1, None
    #     losses = []
    #     epochs = []
    #     val_epochs = []
    #     val_losses = []
    #     pbar = tqdm(total=self.epoch, desc=description, colour=color)
    #     for epoch in range(self.epoch):
    #         trainer.model.train()
    #         total_loss = 0.0
    #         # for batch in trainer.train_loader:
    #         fetch_t0 = prof.iter_reset_anchor()
    #         for i, batch in enumerate(trainer.train_loader):
    #             prof.record_data_wait(fetch_t0)

    #             start_time = time.time()
    #             self.optimizer.zero_grad()
    #             # train_loss = trainer.train_step(batch)

    #             with prof.section("forward"):
    #                 train_loss = trainer.train_step(batch)
    #             with prof.section("loss"):
    #                 total_loss = train_loss
    #             with prof.section("backward"):
    #                 train_loss.backward()
    #             with prof.section("step"):
    #                 self.optimizer.step()
    #             with prof.section("zero_grad"):
    #                 pass  # already zeroed above; kept for table consistency
    #             # if getattr(trainer, "scheduler", None) is not None:
    #             #     with prof.section("sched"):
    #             #         trainer.scheduler.step()

    #             if profile_enabled and (i + 1) % print_every == 0:
    #                 # Optional: memory snapshot
    #                 try:
    #                     import torch
    #                     mem_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    #                     torch.cuda.reset_peak_memory_stats()
    #                     mem_str = f" | max_alloc={mem_gb:.2f} GiB"
    #                 except Exception:
    #                     mem_str = ""
    #                 print(f"[PROFILE e{epoch} i{i+1}] {prof.format_table()}{mem_str}")

    #             # next data wait anchor
    #             fetch_t0 = prof.iter_reset_anchor()

    #             # train_loss.backward()
    #             # self.optimizer.step()

    #             total_loss += train_loss.detach()
    #             # if trainer.device.type == 'cuda':
    #             #     torch.cuda.synchronize()
    #             # time_total += time.time() - start_time

    #         if self.scheduler is not None:
    #             with prof.section("sched"):
    #                 self.scheduler.step()

    #         pbar.update(1)

    #         if (epoch + 1) % self.eval_every_eps == 0:
    #             train_loss = total_loss.cpu().item() / len(trainer.train_loader)
    #             losses.append(train_loss)
    #             epochs.append(epoch)
    #             val_loss = trainer.validate(trainer.val_loader)
    #             pbar.set_postfix({"loss": train_loss, "val_loss": val_loss})
    #             val_losses.append(val_loss)
    #             val_epochs.append(epoch)

    #             if self.early_save_metric == 'val':
    #                 current_loss = val_loss
    #             else:
    #                 current_loss = train_loss

    #             if current_loss < best_loss:
    #                 best_loss = current_loss
    #                 best_epoch = epoch
    #                 best_state = deepcopy(trainer.model.state_dict())

    #     if best_state is not None:
    #         trainer.model.load_state_dict(best_state)

    #     pbar.close()

    #     return {
    #         "train": {
    #             "loss": losses,
    #             "epoch": epochs,
    #         },
    #         "valid": {
    #             "loss": val_losses,
    #             "epoch": val_epochs,
    #         },
    #         "best": {
    #             "epoch": best_epoch,
    #             "loss": best_loss,
    #         },
    #         "time": time_total
    #     }  
