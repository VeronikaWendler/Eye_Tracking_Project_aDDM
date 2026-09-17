#!/bin/bash
#SBATCH --job-name=s2_basic_ES
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=128G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/final/basic_ES_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/final/basic_ES_%j.err
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

DATA_DIR_HOST="/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/data"
OUT_DIR_HOST="/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives"
DATA_FILE_NAME="Study2_Behaviour_with_Gaze_AnalysisReady.csv"

PHASE="ES"
MODEL_SUBDIR="basic_ES"
MODEL_NAME="basic_aDDM_ES"

CHAINS="${CHAINS:-3}"
SAMPLES="${SAMPLES:-1000}"
BURN="${BURN:-200}"

MODEL_DIR_HOST="${OUT_DIR_HOST}/models/${MODEL_SUBDIR}"
FIG_DIR_HOST="${OUT_DIR_HOST}/figures/${MODEL_SUBDIR}"

# ------------------------------------------------------------------
# Safety checks: the phase, folder, and model name MUST agree.
# This prevents an EE job from silently writing into basic_ES again.
# ------------------------------------------------------------------
[[ "${MODEL_SUBDIR}" == "basic_${PHASE}" ]] || {
  echo "ERROR: MODEL_SUBDIR/PHASE mismatch." >&2
  exit 1
}

[[ "${MODEL_NAME}" == "basic_aDDM_${PHASE}" ]] || {
  echo "ERROR: MODEL_NAME/PHASE mismatch." >&2
  exit 1
}

[[ -f "${REPO_ROOT}/py_fit/study2_fit/addm_fit_basic.py" ]] || {
  echo "ERROR: missing fit script." >&2
  exit 1
}

[[ -f "${REPO_ROOT}/py_prep/study2_prep/addm_prepare_data.py" ]] || {
  echo "ERROR: missing Study 2 prep module." >&2
  exit 1
}

[[ -f "${DATA_DIR_HOST}/${DATA_FILE_NAME}" ]] || {
  echo "ERROR: missing Study 2 AnalysisReady data." >&2
  exit 1
}

mkdir -p \
  "${MODEL_DIR_HOST}" \
  "${FIG_DIR_HOST}" \
  "${OUT_DIR_HOST}/logs/final"

# Refuse to mix a new run with old fitted chains.
if compgen -G "${MODEL_DIR_HOST}/${MODEL_NAME}_*.hddm" > /dev/null; then
  echo "ERROR: existing fitted chains found in ${MODEL_DIR_HOST}" >&2
  echo "Archive/remove that old folder before rerunning so results cannot be mixed." >&2
  exit 1
fi

echo "========================================================================"
echo "STUDY 2 BASIC aDDM — ${PHASE}"
echo "========================================================================"
echo "DATA=${DATA_DIR_HOST}/${DATA_FILE_NAME}"
echo "PHASE=${PHASE}"
echo "MODEL_DIR=${MODEL_DIR_HOST}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "CHAINS=${CHAINS}"
echo "SAMPLES=${SAMPLES}"
echo "BURN=${BURN}"
echo "========================================================================"

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
    --model-dir "/out/models/${MODEL_SUBDIR}" \
    --model-name "${MODEL_NAME}" \
    --samples "${SAMPLES}" \
    --burn "${BURN}" \
    --chains "${CHAINS}" \
    --jobs "${CHAINS}"

# ------------------------------------------------------------------
# Post-fit validation: fail if the expected outputs were not created.
# ------------------------------------------------------------------
for i in $(seq 0 $((CHAINS - 1))); do
  for ext in hddm pkl nc; do
    expected="${MODEL_DIR_HOST}/${MODEL_NAME}_${i}.${ext}"
    [[ -f "${expected}" ]] || {
      echo "ERROR: expected output missing: ${expected}" >&2
      exit 1
    }
  done
done

[[ -f "${MODEL_DIR_HOST}/${MODEL_NAME}_MODEL_INPUT.csv" ]] || {
  echo "ERROR: exact model input was not written to expected folder." >&2
  exit 1
}

echo ""
echo "SUCCESS: Study 2 basic ${PHASE} fit completed."
echo "Outputs verified in: ${MODEL_DIR_HOST}"
