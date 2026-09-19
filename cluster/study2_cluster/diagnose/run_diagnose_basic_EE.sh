#!/bin/bash
#SBATCH --job-name=s2_diag_basic_EE
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/diagnostics/basic_EE_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/diagnostics/basic_EE_%j.err
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

PHASE="EE"
MODEL_DIR_HOST="${OUT_DIR_HOST}/models/basic_EE"
FIG_DIR_HOST="${OUT_DIR_HOST}/figures/basic_EE"
CHAINS="${CHAINS:-3}"
PPC_SAMPLES="${PPC_SAMPLES:-50}"

mkdir -p "${FIG_DIR_HOST}" "${OUT_DIR_HOST}/logs/diagnostics"

echo "========================================================================"
echo "STUDY 2 BASIC aDDM DIAGNOSTICS — EE"
echo "========================================================================"
echo "MODEL_DIR=${MODEL_DIR_HOST}"
echo "FIG_DIR=${FIG_DIR_HOST}"
echo "CHAINS=${CHAINS}"
echo "PPC_SAMPLES=${PPC_SAMPLES}"
echo "========================================================================"

[[ -f "${REPO_ROOT}/py_diagnose/study2_diagnose/addm_diagnose.py" ]] || {
  echo "ERROR: missing diagnostic script." >&2
  exit 1
}

for i in $(seq 0 $((CHAINS - 1))); do
  [[ -f "${MODEL_DIR_HOST}/basic_aDDM_EE_${i}.hddm" ]] || {
    echo "ERROR: missing chain ${i}: ${MODEL_DIR_HOST}/basic_aDDM_EE_${i}.hddm" >&2
    exit 1
  }
done

apptainer exec --cleanenv \
  --bind "${REPO_ROOT}:/workspace" \
  --bind "${OUT_DIR_HOST}:/out" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/mplcache \
  "${IMAGE}" \
  python /workspace/py_diagnose/study2_diagnose/addm_diagnose.py \
    --phase "${PHASE}" \
    --model-dir /out/models/basic_EE \
    --fig-dir /out/figures/basic_EE \
    --chains "${CHAINS}" \
    --ppc-samples "${PPC_SAMPLES}"

echo "Finished Study 2 basic aDDM diagnostics for EE."
