#!/bin/bash
#SBATCH --job-name=s2_basic_ES
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/basic_ES_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/basic_ES_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail

# Resolve repository root from this script:
# cluster/study2_cluster/run_hddm_basic.sh -> repository root is ../..
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

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

# Defaults are full-run settings; override at sbatch for a smoke test.
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

test -f "${REPO_ROOT}/py_fit/study2_fit/addm_fit_basic.py"
test -f "${REPO_ROOT}/py_prep/study2_prep/addm_prepare_data.py"
test -f "${DATA_DIR_HOST}/${DATA_FILE_NAME}"

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
