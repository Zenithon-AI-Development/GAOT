# MagLIF runs: checkpoint, log, and WandB locations

Reference for runs launched via `maglif_*_skypilot.yaml`. Paths are relative to the run working directory on the cluster (`/tmp/gaot` when using the YAML run section).

## Where to follow on WandB

- **Project:** https://wandb.ai/Zenithon-AI/gaot-maglif  
- **No-rollout run (1072):** Filter by group `maglif-long-rollout-no-rollouts` or run name `GAOT_maglif_long_rollout_no_rollouts`.  
- **Long-rollout run (1074):** Filter by group `maglif-long-rollout` or run name `GAOT_maglif_long_rollout`.  
- Smoke (1071) uses `wandb.enabled: false`, so it does not appear online.

## Sky managed job IDs (current)

| Job ID | Name                     | Status (as of last check) |
|--------|--------------------------|----------------------------|
| 1071   | maglif-smoke             | SUCCEEDED                  |
| 1561   | maglif-no-rollout-v100   | STARTING; V100 spot |
| 1560   | maglif-no-rollout-ondemand | STARTING; T4 on-demand |
| 1559   | maglif-no-rollout-east1  | STARTING; T4 us-east1 |
| 1557   | maglif-no-rollout-l4     | STARTING; L4 spot |
| 1556   | maglif-no-rollout        | STARTING; T4 (multi-region) |
| 1555   | maglif-no-rollout-west   | STARTING; T4 us-west1 |
| 1554   | maglif-no-rollout        | STARTING; T4 |
| 1553   | maglif-no-rollout        | PENDING |
| 1416   | maglif-long-rollout      | PENDING; had ~1d 1h run |
| (run when best.pt in GCS) | maglif-long-rollout-eval | `sky jobs launch maglif_long_rollout_eval_skypilot.yaml -n maglif-long-rollout-eval -y` |

**Sky job logs (stream or fetch):**
```bash
sky jobs queue
sky jobs logs <job_id>           # stream
sky jobs logs <job_id> --no-follow   # fetch tail
```

**WandB:** Both runs have WandB enabled in the YAML. Pass your API key when launching so logging works:
`sky jobs launch <yaml> -n <name> -y --env 'WANDB_API_KEY=<your_40+_char_key>'`. Do not store the key in the repo.

**Local animation (fully autoregressive, step=20, memory-efficient):**  
Use the no-rollout best checkpoint to generate an animation locally without Sky:
1. Get the checkpoint: download from GCS (`gsutil cp gs://zen_ml_checkpoints/gaot_maglif_long_rollout_no_rollouts/maglif_long_rollout_no_rollouts.best.pt .ckpt/examples/time_dep/`) or copy from a completed run.
2. Point `--data_dir` to the directory that contains `new_processed_maglif/` (same layout as on Sky: `.../new_processed_maglif/data/test/*.hdf5`).
3. Run:
   `python scripts/generate_maglif_animation_local.py --ckpt .ckpt/examples/time_dep/maglif_long_rollout_no_rollouts.best.pt --data_dir /path/to/parent_of_new_processed_maglif --output .results/examples/time_dep/maglif_no_rollout_ar20.gif --step 20 --max_frames 80`
   This uses the biggest training step size (20) and limits frames for low memory use.

**Monitoring no-rollout and long-rollout eval:**
```bash
./scripts/monitor_maglif_runs.sh   # queue, GCS checkpoints, eval hint
sky jobs queue                     # filter for maglif; note RUNNING IDs
sky jobs logs <job_id> --no-follow # e.g. 1555 (no-rollout), 1282 (long-rollout-eval), 1416 (long-rollout)
```
To confirm no-rollout checkpoints are available after the cluster is down:  
`gsutil stat gs://zen_ml_checkpoints/gaot_maglif_long_rollout_no_rollouts/maglif_long_rollout_no_rollouts.best.pt`

**Bugs/errors to watch:**
- **RECOVERING / STARTING long:** No-rollout may sit in STARTING if L4 spot is scarce in one region. The no-rollout YAML no longer pins `region`, so Sky can use multiple regions. Cancel and relaunch with `--env 'WANDB_API_KEY=...'` to get a new job (e.g. 1476).
- **WandB:** If logs show "WANDB_API_KEY not set; wandb will use mode=offline", the key was not passed to the worker. Relaunch with `--env 'WANDB_API_KEY=<your_key>'`. If the key is set, runs appear at https://wandb.ai/Zenithon-AI/gaot-maglif (filter by group).
- **Sequential trainer / plotting:** `rel_l1`/`rel_l2` are in normalized space; timing uses `est_first` so no NameError. Animations use `_create_animation` → `create_sequential_animation_1d` (MagLIF is 1D, `coord_mode=fx`); eval step is non-fatal if `best.pt` is missing. Animation generation for this dataset is supported: test run creates `*_standard.gif` and `*_autoregressive.gif`; any animation failure is caught and logged without aborting the run.

**WandB fallback:** If you launch without `--env WANDB_API_KEY=...`, the code now switches to `mode=offline` so the job does not crash; you can sync later with `wandb sync <run_dir>`.

---

## Checkpoint sync (so you can access them after run stops or cluster is down)

- **When:** Every time a new best loss is found, `save_ckpt_best()` writes `best.pt` locally, copies to `GAOT_CHECKPOINTS_DIR` (mount), and if `GAOT_CHECKPOINTS_GCS_URI` is set runs `gsutil cp` so the object appears in the bucket. At the end of training, `fit()` does a final sync and final `gsutil cp`. The no-rollout run script also runs `gsutil cp` after training.
- **Where:** Long-rollout: `gs://zen_ml_checkpoints/gaot_maglif_long_rollout/maglif_long_rollout.best.pt`. No-rollout: `gs://zen_ml_checkpoints/gaot_maglif_long_rollout_no_rollouts/maglif_long_rollout_no_rollouts.best.pt`.
- **Why checkpoint was not visible for job 1555:** Writes to the GCS FUSE mount (`/tmp/gcs_checkpoints/...`) on the Sky VM did not appear when listing `gs://zen_ml_checkpoints/` from this client (mount write-back may be to a different project or async). The trainer now also runs explicit `gsutil cp` when `GAOT_CHECKPOINTS_GCS_URI` is set so each new best is uploaded and visible. The no-rollout YAML sets this env.
- **For job 1555 (already running):** Checkpoint will appear in GCS when the job completes and the run script’s final `gsutil cp` runs. Until then it exists only on the VM at `/tmp/gaot/.ckpt/.../maglif_long_rollout_no_rollouts.best.pt` and on the mount path.
- **Verify no-rollout in GCS:** `gsutil stat gs://zen_ml_checkpoints/gaot_maglif_long_rollout_no_rollouts/maglif_long_rollout_no_rollouts.best.pt`
- **Access:** Use `gsutil cp ...` to download, or run the eval Sky jobs which read from these paths.

## Bugs/errors (current runs)

- **1416 (long-rollout):** Currently PENDING (waiting for a node). When RUNNING, checkpoints sync to GCS on each new best and at end of `fit()`. WandB works if the job was launched with `--env 'WANDB_API_KEY=...'`.
- **1553 (no-rollout):** Currently PENDING. To ensure WandB works when it runs, (re)launch with:  
  `sky jobs launch maglif_long_rollout_no_rollouts_skypilot.yaml -n maglif-no-rollout -y --env 'WANDB_API_KEY=<your_40+_char_key>'`  
  Checkpoints sync to GCS on each best and at end of training.
- **Controller disk full:** If `sky jobs launch` fails with "No space left on device" (rsync receiver), the Sky managed-jobs controller has run out of disk. Free space on the machine that runs `sky jobs` (or the controller node) and retry. To submit no-rollout on another region after that:  
  `sky jobs launch maglif_long_rollout_no_rollouts_skypilot.yaml -n maglif-no-rollout-west -y --infra gcp/us-west1 --env 'WANDB_API_KEY=<key>'`.

---

## Previous problems (and fixes)

- **No-rollout (1072, 1283):**  
  - **1072:** Failed on *recovery* with `SyntaxError: unmatched '}'` in `sequential_trainer.py` after `git pull` (code at that time had a syntax error).  
  - **1283:** Failed after ~24 min (1 recovery). Likely causes: **WandB** required an API key; job failed when `WANDB_API_KEY` was not set or invalid. Eval step could also fail and abort the whole run.  
  - **Fixes:** Eval step runs only if `best.pt` exists and is non-fatal. WandB re-enabled; pass `WANDB_API_KEY` via `--env` when launching.

- **Long-rollout (1074):**  
  - **FAILED_CONTROLLER** after ~4 days (controller/Spot issue, not training code).  
  - Best checkpoint was only on the node’s local disk; **no sync to GCS** in the original code, so no way to run eval after failure.  
  - **Fixes:** Training now syncs `best.pt` to `GAOT_CHECKPOINTS_DIR` (GCS) on each best save. WandB re-enabled; pass `WANDB_API_KEY` via `--env` when launching.

- **Long-rollout eval (1282):**  
  - Eval job stayed STARTING/PENDING because **no `maglif_long_rollout.best.pt` in GCS** (training had never synced it).  
  - **Fix:** Run the eval job only after long-rollout training has written at least one best checkpoint to GCS; monitor with `./scripts/monitor_maglif_runs.sh` and run `sky jobs launch maglif_long_rollout_eval_skypilot.yaml -n maglif-long-rollout-eval -y` when the script reports the checkpoint is present.

---

## Long-rollout run (maglif_long_rollout_skypilot.yaml)

**Checkpoint (config path):** `.ckpt/examples/time_dep/maglif_long_rollout.pt`  
**Loss plot (config path):** `.loss/examples/time_dep/maglif_long_rollout.png`  
**Result plot (config path):** `.results/examples/time_dep/maglif_long_rollout.png`  
**Database (config path):** `.database/examples/time_dep/maglif_long_rollout.csv`

**On cluster (under `/tmp/gaot`):**  
`/tmp/gaot/.ckpt/examples/time_dep/maglif_long_rollout.pt` (and same base for .loss, .results, .database).

**GCS mounts (YAML):**  
- Checkpoints dir (mount): `/tmp/gcs_checkpoints` → `gs://zen_ml_checkpoints`  
- Logs dir (mount): `/tmp/gcs_logs` → `gs://zen_ml_logs`  
- Data dir (mount): `/tmp/gcs_data` → `gs://zen_ml_data`  
- Env vars set in run: `GAOT_CHECKPOINTS_DIR=/tmp/gcs_checkpoints/gaot_maglif_long_rollout`, `GAOT_LOGS_DIR=/tmp/gcs_logs/gaot_maglif_long_rollout`, `GAOT_DATA_DIR=/tmp/gcs_data/gaot_maglif_long_rollout`.

**WandB:**  
- Project: `gaot-maglif`  
- Entity: `Zenithon-AI`  
- Group: `maglif-long-rollout`  
- Run name: `GAOT_maglif_long_rollout`  
- URL: https://wandb.ai/Zenithon-AI/gaot-maglif (filter by group or run name).

---

## No-rollout run (maglif_long_rollout_no_rollouts_skypilot.yaml)

**Checkpoint (config path):** `.ckpt/examples/time_dep/maglif_long_rollout_no_rollouts.pt`  
**Loss plot (config path):** `.loss/examples/time_dep/maglif_long_rollout_no_rollouts.png`  
**Result plot (config path):** `.results/examples/time_dep/maglif_long_rollout_no_rollouts.png`  
**Database (config path):** `.database/examples/time_dep/maglif_long_rollout_no_rollouts.csv`

**On cluster:**  
`/tmp/gaot/.ckpt/examples/time_dep/maglif_long_rollout_no_rollouts.pt` (and same base for .loss, .results, .database).

**GCS env vars:**  
`GAOT_CHECKPOINTS_DIR=/tmp/gcs_checkpoints/gaot_maglif_long_rollout_no_rollouts`, `GAOT_LOGS_DIR=/tmp/gcs_logs/gaot_maglif_long_rollout_no_rollouts`, `GAOT_DATA_DIR=/tmp/gcs_data/gaot_maglif_long_rollout_no_rollouts`.

**WandB:**  
- Project: `gaot-maglif`  
- Entity: `Zenithon-AI`  
- Group: `maglif-long-rollout-no-rollouts`  
- Run name: `GAOT_maglif_long_rollout_no_rollouts`  
- URL: https://wandb.ai/Zenithon-AI/gaot-maglif

---

## Smoke run (maglif_smoke_skypilot.yaml)

**Checkpoint (config path):** `.ckpt/examples/time_dep/maglif_smoke.pt`  
**Loss plot (config path):** `.loss/examples/time_dep/maglif_smoke.png`  
**Result plot (config path):** `.results/examples/time_dep/maglif_smoke.png`  
**On cluster:** `/tmp/gaot/.ckpt/examples/time_dep/maglif_smoke.pt` (and same base for .loss, .results).  
**GCS env vars:** `GAOT_CHECKPOINTS_DIR=/tmp/gcs_checkpoints/gaot_maglif_smoke`, etc.  
**WandB:** Same project/entity; group `maglif-smoke`, run name `GAOT_maglif_smoke`; smoke config uses `wandb.enabled: false` (offline/no sync).

---

## Generating AR trajectory and animation from best checkpoint

Eval-only configs and Sky YAMLs run **test** (no training): load the **best** checkpoint, run autoregressive trajectory evaluation, and write animations to GCS.

**Best checkpoint filenames:** `maglif_long_rollout_no_rollouts.best.pt`, `maglif_long_rollout.best.pt` (saved by the optimizer when validation improves). Training now **syncs best to GCS** when `GAOT_CHECKPOINTS_DIR` is set (see `src/core/base_trainer.py`).

**Eval configs (test-only, load `.best.pt`):**
- `config/examples/time_dep/maglif_long_rollout_no_rollouts_eval.json`
- `config/examples/time_dep/maglif_long_rollout_eval.json`

**Sky eval jobs (run test + copy animations to GCS):**
- No-rollout: `sky jobs launch maglif_no_rollout_eval_skypilot.yaml -n maglif-no-rollout-eval -y`
- Long-rollout: `sky jobs launch maglif_long_rollout_eval_skypilot.yaml -n maglif-long-rollout-eval -y`

**Where to find the animations (after a successful eval job):**
- No-rollout: `gs://zen_ml_logs/gaot_maglif_long_rollout_no_rollouts_eval/`
  - `maglif_long_rollout_no_rollouts_eval_standard.gif`, `maglif_long_rollout_no_rollouts_eval_autoregressive.gif`
- Long-rollout: `gs://zen_ml_logs/gaot_maglif_long_rollout_eval/`
  - `maglif_long_rollout_eval_standard.gif`, `maglif_long_rollout_eval_autoregressive.gif`

**Checkpoint availability:**
- **No-rollout (1072):** Run failed on recovery; the best checkpoint was on the first node and was **not** synced to GCS. To get an animation: re-run no-rollout training (with the updated code that syncs best to GCS), then run the no-rollout eval job above; or copy a checkpoint to `gs://zen_ml_checkpoints/gaot_maglif_long_rollout_no_rollouts/maglif_long_rollout_no_rollouts.best.pt` if you have it elsewhere.
- **Long-rollout (1074):** Checkpoint is on the running cluster’s local disk (`/tmp/gaot/.ckpt/...`). It is **not** in GCS until (1) the running job pulls the latest code (e.g. on next recovery) and saves another best (then it will sync), or (2) you copy it from the node to GCS (e.g. `gsutil cp` from the cluster). After the checkpoint exists at `gs://zen_ml_checkpoints/gaot_maglif_long_rollout/maglif_long_rollout.best.pt`, run the long-rollout eval job above.

**When you can see long-rollout animations:** Once `maglif_long_rollout.best.pt` is in GCS and the long-rollout eval job has run successfully, animations appear at **`gs://zen_ml_logs/gaot_maglif_long_rollout_eval/`** (`*_standard.gif`, `*_autoregressive.gif`).

**No-rollout (job 1413, L4):** After training completes, test runs and animations are copied to **`gs://zen_ml_logs/gaot_maglif_long_rollout_no_rollouts/animations/`** (`maglif_long_rollout_no_rollouts_eval_standard.gif`, `maglif_long_rollout_no_rollouts_eval_autoregressive.gif`). WandB disabled for robustness; eval step runs only if best.pt exists.
