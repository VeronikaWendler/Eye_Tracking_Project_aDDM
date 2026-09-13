#!/bin/bash
#SBATCH --job-name=basic_addm
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=96G
#SBATCH --output=logs/addm_%j.out
#SBATCH --error=logs/addm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail

# Submit from the repository root.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

module purge
module load bb-singularity-conf/live

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache"
mkdir -p "$MPLCONFIGDIR"

# ------------------------------------------------------------------
# EDIT THESE THREE LOCATIONS ONCE FOR THE NEW REPOSITORY / RDS SETUP
# ------------------------------------------------------------------
IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
DATA_DIR_HOST="${DATA_DIR_HOST:-/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/data}"
OUT_DIR_HOST="${OUT_DIR_HOST:-/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives}"

# Choose ES or EE when submitting:
#   sbatch --export=ALL,PHASE=ES cluster/run_hddm.sh
#   sbatch --export=ALL,PHASE=EE cluster/run_hddm.sh
PHASE="${PHASE:-ES}"

DATA_FILE_NAME="${DATA_FILE_NAME:-Study1_Behaviour_with_Gaze_AnalysisReady.csv}"
CHAINS="${CHAINS:-3}"
SAMPLES="${SAMPLES:-2000}"
BURN="${BURN:-500}"

mkdir -p "${OUT_DIR_HOST}"/{models,figures,logs}

echo "PHASE=${PHASE}"
echo "CODE_DIR=${CODE_DIR}"
echo "DATA=${DATA_DIR_HOST}/${DATA_FILE_NAME}"
echo "OUT=${OUT_DIR_HOST}"

apptainer exec --cleanenv \
  --bind "${CODE_DIR}:/workspace" \
  --bind "${DATA_DIR_HOST}:/data:ro" \
  --bind "${OUT_DIR_HOST}:/out" \
  --env PYTHONUNBUFFERED=1 \
  "${IMAGE}" \
  python /workspace/addm_fit_basic.py \
    --data "/data/${DATA_FILE_NAME}" \
    --phase "${PHASE}" \
    --model-dir "/out/models" \
    --samples "${SAMPLES}" \
    --burn "${BURN}" \
    --chains "${CHAINS}" \
    --jobs "${SLURM_CPUS_PER_TASK}"

apptainer exec --cleanenv \
  --bind "${CODE_DIR}:/workspace" \
  --bind "${OUT_DIR_HOST}:/out" \
  --env PYTHONUNBUFFERED=1 \
  "${IMAGE}" \
  python /workspace/addm_diagnose.py \
    --phase "${PHASE}" \
    --model-dir "/out/models" \
    --fig-dir "/out/figures" \
    --chains "${CHAINS}"

echo "Finished basic aDDM for ${PHASE}"
