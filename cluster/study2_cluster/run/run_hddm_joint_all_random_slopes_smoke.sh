#!/bin/bash
#SBATCH --job-name=s2_joint_smoke
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/smoke/joint_all_random_slopes_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/smoke/joint_all_random_slopes_%j.err
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
MODEL_DIR_HOST="${OUT_DIR_HOST}/models/joint_all_random_slopes_smoke"

ES_INPUT_HOST="${OUT_DIR_HOST}/prepared_data/model_input_ES.csv"
EE_INPUT_HOST="${OUT_DIR_HOST}/prepared_data/model_input_EE.csv"

CHAINS="${CHAINS:-2}"
SAMPLES="${SAMPLES:-100}"
BURN="${BURN:-50}"

mkdir -p \
  "${MODEL_DIR_HOST}" \
  "${OUT_DIR_HOST}/logs/smoke"

echo "========================================================================"
echo "STUDY 2 JOINT aDDM — ALL RANDOM SLOPES — SMOKE TEST"
echo "========================================================================"
echo "REPO_ROOT=${REPO_ROOT}"
echo "FIT_SCRIPT=${REPO_ROOT}/py_fit/study2_fit/addm_fit_joint_all_random_slopes.py"
echo "ES_INPUT=${ES_INPUT_HOST}"
echo "EE_INPUT=${EE_INPUT_HOST}"
echo "MODEL_DIR=${MODEL_DIR_HOST}"
echo "CHAINS=${CHAINS}"
echo "SAMPLES=${SAMPLES}"
echo "BURN=${BURN}"
echo "========================================================================"

[[ -f "${REPO_ROOT}/py_fit/study2_fit/addm_fit_joint_all_random_slopes.py" ]] || {
  echo "ERROR: missing fit script: ${REPO_ROOT}/py_fit/study2_fit/addm_fit_joint_all_random_slopes.py" >&2
  exit 1
}

[[ -f "${ES_INPUT_HOST}" ]] || {
  echo "ERROR: missing ES input: ${ES_INPUT_HOST}" >&2
  exit 1
}

[[ -f "${EE_INPUT_HOST}" ]] || {
  echo "ERROR: missing EE input: ${EE_INPUT_HOST}" >&2
  exit 1
}

apptainer exec --cleanenv \
  --bind "${REPO_ROOT}:/workspace" \
  --bind "${OUT_DIR_HOST}:/out" \
  --bind "${MODEL_DIR_HOST}:/target_models" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/mplcache \
  "${IMAGE}" \
  python /workspace/py_fit/study2_fit/addm_fit_joint_all_random_slopes.py \
    --es-input /out/prepared_data/model_input_ES.csv \
    --ee-input /out/prepared_data/model_input_EE.csv \
    --model-dir /target_models \
    --samples "${SAMPLES}" \
    --burn "${BURN}" \
    --chains "${CHAINS}" \
    --jobs "${SLURM_CPUS_PER_TASK}"

echo "Finished Study 2 joint all-random-slopes SMOKE TEST."
