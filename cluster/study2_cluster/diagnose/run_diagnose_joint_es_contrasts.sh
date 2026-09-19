#!/bin/bash
#SBATCH --job-name=s2_diag_es_contr
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/diagnostics/es_joint_es_contrasts_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/diagnostics/es_joint_es_contrasts_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache"
mkdir -p "$MPLCONFIGDIR"

IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"

OUT_ROOT="/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives"
MODEL_DIR_HOST="${OUT_ROOT}/models/es_joint_es_contrasts_final"
DIAG_DIR_HOST="${OUT_ROOT}/figures/es_joint_es_contrasts/diagnostics"
CHAINS="${CHAINS:-3}"

mkdir -p "${DIAG_DIR_HOST}" "${OUT_ROOT}/logs/diagnostics"

[[ -f "${REPO_ROOT}/py_diagnose/study2_diagnose/addm_diagnose_joint_es_contrasts.py" ]] || {
  echo "ERROR: missing diagnostic script." >&2
  exit 1
}

[[ -d "${MODEL_DIR_HOST}" ]] || {
  echo "ERROR: missing final model directory: ${MODEL_DIR_HOST}" >&2
  exit 1
}

apptainer exec --cleanenv \
  --bind "${REPO_ROOT}:/workspace" \
  --bind "${MODEL_DIR_HOST}:/target_models:ro" \
  --bind "${DIAG_DIR_HOST}:/diagnostics" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/mplcache \
  "${IMAGE}" \
  python /workspace/py_diagnose/study2_diagnose/addm_diagnose_joint_es_contrasts.py \
    --model-dir /target_models \
    --out-dir /diagnostics \
    --chains "${CHAINS}"

echo "Finished Study 2 accuracy-coded joint E/S contrast diagnostics."
