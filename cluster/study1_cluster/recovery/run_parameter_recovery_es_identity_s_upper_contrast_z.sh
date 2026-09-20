#!/bin/bash
#SBATCH --job-name=s1_rec_cz
#SBATCH --array=0-9%3
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=96G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives/logs/recovery/es_identity_s_upper_contrast_z_main_%A_%a.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives/logs/recovery/es_identity_s_upper_contrast_z_main_%A_%a.err
#SBATCH --mail-type=ALL
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
recovery_dir="$base/recovery/es_identity_s_upper_contrast_z_main"
log_dir="$base/logs/recovery"
mkdir -p "$recovery_dir" "$log_dir"
empirical_chains="${EMPIRICAL_CHAINS:-3}"
recovery_chains="${RECOVERY_CHAINS:-3}"
samples="${SAMPLES:-4000}"
burn="${BURN:-1000}"

echo "=============================================================================="
echo "study1 parameter recovery: es identity s upper contrast z"
echo "rep=${SLURM_ARRAY_TASK_ID}"
echo "model_dir=$model_dir"
echo "recovery_dir=$recovery_dir"
echo "empirical_chains=$empirical_chains"
echo "recovery_chains=$recovery_chains"
echo "samples=$samples"
echo "burn=$burn"
echo "=============================================================================="

test -f "$repo_root/py_recovery/addm_parameter_recovery_s_upper_contrast_z.py"
test -d "$model_dir"

apptainer exec --cleanenv \
  --bind "$repo_root:/workspace" \
  --bind "$model_dir:/target_models:ro" \
  --bind "$recovery_dir:/recovery" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/mplcache \
  "$image" \
  python /workspace/py_recovery/addm_parameter_recovery_s_upper_contrast_z.py \
    --study study1 \
    --model-dir /target_models \
    --recovery-dir /recovery \
    --rep "${SLURM_ARRAY_TASK_ID}" \
    --empirical-chains "$empirical_chains" \
    --recovery-chains "$recovery_chains" \
    --samples "$samples" \
    --burn "$burn"

echo "finished study1 recovery rep ${SLURM_ARRAY_TASK_ID}"
