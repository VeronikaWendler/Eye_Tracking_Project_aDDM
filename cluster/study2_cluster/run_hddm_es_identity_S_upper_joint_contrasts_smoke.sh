#!/bin/bash
#SBATCH --job-name=s2_es_id_joint_smoke
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/smoke/es_identity_S_upper_joint_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/smoke/es_identity_S_upper_joint_%j.err
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
INPUT_HOST="${OUT_ROOT}/prepared_data/model_input_ES_identity_S_upper.csv"
MODEL_DIR_HOST="${OUT_ROOT}/models/es_identity_S_upper_joint_contrasts_smoke"

CHAINS="${CHAINS:-2}"
SAMPLES="${SAMPLES:-100}"
BURN="${BURN:-50}"

mkdir -p "${MODEL_DIR_HOST}" "${OUT_ROOT}/logs/smoke"

echo "========================================================================"
echo "STUDY 2 ES S-UPPER IDENTITY JOINT-CONTRAST aDDM — SMOKE"
echo "========================================================================"
echo "INPUT=${INPUT_HOST}"
echo "MODEL_DIR=${MODEL_DIR_HOST}"
echo "CHAINS=${CHAINS}"
echo "SAMPLES=${SAMPLES}"
echo "BURN=${BURN}"
echo "========================================================================"

[[ -f "${REPO_ROOT}/py_fit/study2_fit/addm_fit_es_identity_S_upper_joint_contrasts.py" ]] || {
  echo "ERROR: missing fit script." >&2
  exit 1
}

[[ -f "${INPUT_HOST}" ]] || {
  echo "ERROR: missing prepared S-upper input: ${INPUT_HOST}" >&2
  exit 1
}

apptainer exec --cleanenv \
  --bind "${REPO_ROOT}:/workspace" \
  --bind "${OUT_ROOT}:/out" \
  --bind "${MODEL_DIR_HOST}:/target_models" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/mplcache \
  "${IMAGE}" \
  python /workspace/py_fit/study2_fit/addm_fit_es_identity_S_upper_joint_contrasts.py \
    --data /out/prepared_data/model_input_ES_identity_S_upper.csv \
    --model-dir /target_models \
    --samples "${SAMPLES}" \
    --burn "${BURN}" \
    --chains "${CHAINS}" \
    --jobs "${SLURM_CPUS_PER_TASK}"

echo "Finished Study 2 S-upper identity joint-contrast SMOKE TEST."
