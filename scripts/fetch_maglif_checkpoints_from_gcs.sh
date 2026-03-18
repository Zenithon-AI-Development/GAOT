#!/usr/bin/env bash
# Download MagLIF checkpoints from GCS to local .ckpt (if they exist).
# Use after a no-rollout or long-rollout run has finished and uploaded to GCS.
# For *running* jobs: Sky managed jobs do not expose SSH to the VM, so you cannot
# pull checkpoints from the cluster until the job ends and the run script's
# gsutil cp (or EXIT trap) has run.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CKPT_DIR="$REPO_ROOT/.ckpt/examples/time_dep"
mkdir -p "$CKPT_DIR"

fetch() {
  local gcs_uri="$1"
  local local_name="$2"
  local dest="$CKPT_DIR/$local_name"
  if gsutil -q stat "$gcs_uri" 2>/dev/null; then
    gsutil -m cp "$gcs_uri" "$dest" && echo "Fetched: $dest"
  else
    echo "Not found in GCS (job may not have finished yet): $gcs_uri"
  fi
}

fetch "gs://zen_ml_checkpoints/gaot_maglif_long_rollout_no_rollouts/maglif_long_rollout_no_rollouts.best.pt" \
  "maglif_long_rollout_no_rollouts.best.pt"
fetch "gs://zen_ml_checkpoints/gaot_maglif_long_rollout/maglif_long_rollout.best.pt" \
  "maglif_long_rollout.best.pt"
