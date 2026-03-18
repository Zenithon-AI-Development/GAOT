#!/usr/bin/env bash
# Monitor no-rollout run, long-rollout training, and long-rollout eval/animations.
# Usage: ./scripts/monitor_maglif_runs.sh   or   bash scripts/monitor_maglif_runs.sh

set -e
echo "===== Sky jobs (maglif) ====="
sky jobs queue 2>/dev/null | grep -E "maglif" || true

echo ""
echo "===== GCS: long-rollout checkpoint (for eval animations) ====="
gsutil ls gs://zen_ml_checkpoints/gaot_maglif_long_rollout/ 2>/dev/null || echo "(none or no access)"

echo ""
echo "===== GCS: long-rollout eval animations ====="
gsutil ls gs://zen_ml_logs/gaot_maglif_long_rollout_eval/ 2>/dev/null || echo "(none or no access)"

echo ""
echo "===== GCS: no-rollout animations (after no-rollout job completes) ====="
gsutil ls gs://zen_ml_logs/gaot_maglif_long_rollout_no_rollouts/animations/ 2>/dev/null || echo "(none or no access)"

echo ""
echo "===== GCS: no-rollout checkpoint (synced during training + explicit gsutil at end; available after cluster down) ====="
if gsutil -q stat gs://zen_ml_checkpoints/gaot_maglif_long_rollout_no_rollouts/maglif_long_rollout_no_rollouts.best.pt 2>/dev/null; then
  gsutil ls -l gs://zen_ml_checkpoints/gaot_maglif_long_rollout_no_rollouts/maglif_long_rollout_no_rollouts.best.pt 2>/dev/null || true
else
  echo "(none yet or no access)"
fi

echo ""
echo "===== Long-rollout: run eval when best.pt in GCS ====="
if gsutil -q stat gs://zen_ml_checkpoints/gaot_maglif_long_rollout/maglif_long_rollout.best.pt 2>/dev/null; then
  echo "Checkpoint present. Run: sky jobs launch maglif_long_rollout_eval_skypilot.yaml -n maglif-long-rollout-eval -y"
else
  echo "No best.pt in GCS yet; wait for long-rollout training to save one, then run eval job above."
fi

echo ""
echo "===== Recent logs (use job ID from queue above) ====="
echo "  sky jobs logs <job_id> --no-follow   # e.g. 1554/1555 (no-rollout), 1416 (long-rollout)"
