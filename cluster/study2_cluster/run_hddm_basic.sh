#!/bin/bash
#SBATCH --job-name=s2_basic_ES
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=180G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/basic_ES_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/basic_ES_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail

# Submit from the repository root:
#   .../repo/Eye_Tracking_Project_aDDM

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache"
mkdir -p "$MPLCONFIGDIR"

IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

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
echo "DATA=${DATA_DIR_HOST}/${DATA_FILE_NAME}"
echo "OUT=${OUT_DIR_HOST}/models/basic_ES"
echo "CHAINS=${CHAINS}"
echo "SAMPLES=${SAMPLES}"
echo "BURN=${BURN}"
echo "========================================"

apptainer exec --cleanenv \
  --bind "${CODE_DIR}:/workspace" \
  --bind "${DATA_DIR_HOST}:/data:ro" \
  --bind "${OUT_DIR_HOST}:/out" \
  --env PYTHONPATH=/workspace/py_prep/study2_prep \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/mplcache \
  "${IMAGE}" \
  python /workspace/py_fit/addm_fit_basic.py \
    --data "/data/${DATA_FILE_NAME}" \
    --phase "${PHASE}" \
    --model-dir "/out/models/basic_ES" \
    --samples "${SAMPLES}" \
    --burn "${BURN}" \
    --chains "${CHAINS}" \
    --jobs "${SLURM_CPUS_PER_TASK}"

echo "Finished Study 2 basic aDDM for ES."
