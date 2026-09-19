#!/bin/bash
#SBATCH --job-name=s2_basic_ES
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/basic_ES_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/basic_ES_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail

# Slurm copies the submitted script into /var/spool before running it,
# so BASH_SOURCE points there. SLURM_SUBMIT_DIR is the actual repo directory.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache"
mkdir -p "$MPLCONFIGDIR"

IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"

DATA_DIR_HOST="/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/data"
OUT_DIR_HOST="/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives"

DATA_FILE_NAME="Study2_Behaviour_with_Gaze_AnalysisReady.csv"
PHASE="ES"

CHAINS="${CHAINS:-3}"
SAMPLES="${SAMPLES:-2000}"
BURN="${BURN:-500}"

mkdir -p \
  "${OUT_DIR_HOST}/models/basic_ES" \
  "${OUT_DIR_HOST}/figures/basic_ES" \
  "${OUT_DIR_HOST}/logs"

echo "========================================"
echo "STUDY 2 BASIC aDDM — ES"
echo "REPO_ROOT=${REPO_ROOT}"
echo "FIT_SCRIPT=${REPO_ROOT}/py_fit/study2_fit/addm_fit_basic.py"
echo "PREP_DIR=${REPO_ROOT}/py_prep/study2_prep"
echo "DATA=${DATA_DIR_HOST}/${DATA_FILE_NAME}"
echo "OUT=${OUT_DIR_HOST}/models/basic_ES"
echo "CHAINS=${CHAINS}"
echo "SAMPLES=${SAMPLES}"
echo "BURN=${BURN}"
echo "========================================"

# Fail with an informative message if any required input is missing.
[[ -f "${REPO_ROOT}/py_fit/study2_fit/addm_fit_basic.py" ]] || {
  echo "ERROR: missing fit script: ${REPO_ROOT}/py_fit/study2_fit/addm_fit_basic.py" >&2
  exit 1
}

[[ -f "${REPO_ROOT}/py_prep/study2_prep/addm_prepare_data.py" ]] || {
  echo "ERROR: missing prep module: ${REPO_ROOT}/py_prep/study2_prep/addm_prepare_data.py" >&2
  exit 1
}

[[ -f "${DATA_DIR_HOST}/${DATA_FILE_NAME}" ]] || {
  echo "ERROR: missing data file: ${DATA_DIR_HOST}/${DATA_FILE_NAME}" >&2
  exit 1
}

apptainer exec --cleanenv \
  --bind "${REPO_ROOT}:/workspace" \
  --bind "${DATA_DIR_HOST}:/data:ro" \
  --bind "${OUT_DIR_HOST}:/out" \
  --env PYTHONPATH=/workspace/py_prep/study2_prep \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/mplcache \
  "${IMAGE}" \
  python /workspace/py_fit/study2_fit/addm_fit_basic.py \
    --data "/data/${DATA_FILE_NAME}" \
    --phase "${PHASE}" \
    --model-dir "/out/models/basic_ES" \
    --samples "${SAMPLES}" \
    --burn "${BURN}" \
    --chains "${CHAINS}" \
    --jobs "${SLURM_CPUS_PER_TASK}"

echo "Finished Study 2 basic aDDM for ES."
