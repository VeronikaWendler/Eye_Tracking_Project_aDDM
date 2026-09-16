#!/bin/bash
#SBATCH --job-name=s2_diag_ES
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/diagnose_basic_ES_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/diagnose_basic_ES_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache"
mkdir -p "$MPLCONFIGDIR"

IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUT_DIR_HOST="/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives"

CHAINS="${CHAINS:-3}"
PPC_SAMPLES="${PPC_SAMPLES:-50}"

mkdir -p \
  "${OUT_DIR_HOST}/models/basic_ES" \
  "${OUT_DIR_HOST}/figures/basic_ES" \
  "${OUT_DIR_HOST}/logs"

echo "========================================"
echo "STUDY 2 BASIC aDDM DIAGNOSTICS — ES"
echo "MODEL DIR=${OUT_DIR_HOST}/models/basic_ES"
echo "FIG DIR=${OUT_DIR_HOST}/figures/basic_ES"
echo "========================================"

apptainer exec --cleanenv \
  --bind "${CODE_DIR}:/workspace" \
  --bind "${OUT_DIR_HOST}:/out" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/mplcache \
  "${IMAGE}" \
  python /workspace/py_diagnose/addm_diagnose.py \
    --phase ES \
    --model-dir /out/models/basic_ES \
    --fig-dir /out/figures/basic_ES \
    --chains "${CHAINS}" \
    --ppc-samples "${PPC_SAMPLES}"

echo "Finished Study 2 basic aDDM diagnostics for ES."
