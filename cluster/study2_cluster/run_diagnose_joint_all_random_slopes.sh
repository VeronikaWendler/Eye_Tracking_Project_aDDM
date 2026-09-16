#!/bin/bash
#SBATCH --job-name=s2_diag_all_reg
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/diag_joint_all_random_slopes_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/diag_joint_all_random_slopes_%j.err
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

OUT_DIR_HOST="/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives"
MODEL_DIR_HOST="${OUT_DIR_HOST}/models/joint_all_random_slopes_final"
DIAG_DIR_HOST="${OUT_DIR_HOST}/figures/joint_all_random_slopes/diagnostics"
CHAINS="${CHAINS:-3}"

mkdir -p "${DIAG_DIR_HOST}" "${OUT_DIR_HOST}/logs"

echo "========================================================================"
echo "STUDY 2 ALL-RANDOM-SLOPES JOINT aDDM DIAGNOSTICS"
echo "========================================================================"
echo "REPO_ROOT=${REPO_ROOT}"
echo "DIAG_SCRIPT=${REPO_ROOT}/py_diagnose/study2_diagnose/addm_diagnose_joint_all_random_slopes.py"
echo "MODEL_DIR=${MODEL_DIR_HOST}"
echo "DIAG_DIR=${DIAG_DIR_HOST}"
echo "CHAINS=${CHAINS}"
echo ""
echo "IMPORTANT: the model directory is mounted at /target_models because"
echo "the saved HDDM objects remember their original database path there."
echo "========================================================================"

[[ -f "${REPO_ROOT}/py_diagnose/study2_diagnose/addm_diagnose_joint_all_random_slopes.py" ]] || {
  echo "ERROR: missing diagnostic script: ${REPO_ROOT}/py_diagnose/study2_diagnose/addm_diagnose_joint_all_random_slopes.py" >&2
  exit 1
}

[[ -d "${MODEL_DIR_HOST}" ]] || {
  echo "ERROR: missing model directory: ${MODEL_DIR_HOST}" >&2
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
  python /workspace/py_diagnose/study2_diagnose/addm_diagnose_joint_all_random_slopes.py \
    --model-dir /target_models \
    --out-dir /diagnostics \
    --chains "${CHAINS}"

echo "Finished Study 2 all-random-slopes joint aDDM diagnostics."
