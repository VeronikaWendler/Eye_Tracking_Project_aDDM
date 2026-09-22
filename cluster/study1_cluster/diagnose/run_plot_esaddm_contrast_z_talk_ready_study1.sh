#!/bin/bash
#SBATCH --job-name=s1_esaddm_talkplots
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives/logs/esaddm_contrast_z_talkplots_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives/logs/esaddm_contrast_z_talkplots_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$repo_root"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache"
mkdir -p "$MPLCONFIGDIR"

image="${IMAGE:-$HOME/containers/hddm_latest.sif}"

base="/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives"
model_dir="$base/models/es_identity_S_upper_contrast_z_final"
out_dir="$base/figures/es_identity_S_upper_contrast_z_final/talk_ready"
log_dir="$base/logs"

mkdir -p "$out_dir" "$log_dir"

echo "=============================================================================="
echo "STUDY 1 TALK-READY ESaDDM PLOTS"
echo "Model: S-upper / E-lower, contrast + z, NO drift intercept"
echo "model_dir=$model_dir"
echo "out_dir=$out_dir"
echo "image=$image"
echo "=============================================================================="

test -f "$repo_root/py_diagnose/plot_esaddm_contrast_z_nointercept_talk_ready.py"
test -d "$model_dir"
test -f "$image"

apptainer exec --cleanenv \
  --bind "$repo_root:/workspace" \
  --bind "$model_dir:/target_models:ro" \
  --bind "$out_dir:/talk_ready" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/mplcache \
  "$image" \
  python /workspace/py_diagnose/plot_esaddm_contrast_z_nointercept_talk_ready.py \
    --study 1 \
    --model-dir /target_models \
    --out-dir /talk_ready

echo "=============================================================================="
echo "FINISHED STUDY 1 TALK-READY ESaDDM PLOTS"
echo "Saved to: $out_dir"
echo "=============================================================================="
