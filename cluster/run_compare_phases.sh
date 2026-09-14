#!/bin/bash
#SBATCH --job-name=addm_compare
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=logs/compare_%j.out
#SBATCH --error=logs/compare_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache"
mkdir -p "$MPLCONFIGDIR"

IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

OUT_DIR_HOST="${OUT_DIR_HOST:-/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives}"

mkdir -p "${OUT_DIR_HOST}/phase_comparison"

echo "========================================"
echo "POSTERIOR ES vs EE COMPARISON"
echo "CODE_DIR=${CODE_DIR}"
echo "OUT_DIR=${OUT_DIR_HOST}"
echo "========================================"

apptainer exec --cleanenv \
  --bind "${CODE_DIR}:/workspace" \
  --bind "${OUT_DIR_HOST}:/out" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/mplcache \
  "${IMAGE}" \
  python /workspace/addm_compare_phases.py \
    --model-dir /out/models \
    --out-dir /out/phase_comparison \
    --chains 3 \
    --contrast-draws 20000

echo "Finished ES vs EE posterior comparison."