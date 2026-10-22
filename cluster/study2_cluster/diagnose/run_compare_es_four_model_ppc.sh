#!/bin/bash
#SBATCH --job-name=s2_ppc_4cmp
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/es_four_model_ppc_compare_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/es_four_model_ppc_compare_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache"
mkdir -p "$MPLCONFIGDIR"

IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"
STUDY_ROOT="/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM"
PPC_ROOT="$STUDY_ROOT/derivatives/ppc"
OUT_DIR_HOST="$PPC_ROOT/es_four_model_comparison"
SCRIPT_HOST="$REPO_ROOT/py_diagnose/compare_es_four_model_ppc_study2.py"

mkdir -p "$OUT_DIR_HOST" "$STUDY_ROOT/derivatives/logs"

[[ -f "$SCRIPT_HOST" ]] || { echo "MISSING: $SCRIPT_HOST" >&2; exit 1; }

for tag in basic_noz basic_z contrast_noz contrast_z; do
  ROOT="$PPC_ROOT/es_identity_s_upper_${tag}_main"
  [[ -f "$ROOT/metadata.json" ]] || {
    echo "MISSING completed PPC metadata: $ROOT/metadata.json" >&2
    exit 1
  }
  [[ -f "$ROOT/tables/observed_replot_trials.csv.gz" ]] || {
    echo "MISSING replot trial data: $ROOT/tables/observed_replot_trials.csv.gz" >&2
    exit 1
  }
done

echo "========================================================================"
echo "STUDY 2 FOUR-MODEL PPC COMPARISON"
echo "========================================================================"
echo "PPC_ROOT=$PPC_ROOT"
echo "OUT_DIR_HOST=$OUT_DIR_HOST"
echo "========================================================================"

apptainer exec --cleanenv \
  --bind "$REPO_ROOT:/workspace" \
  --bind "$PPC_ROOT:/ppc" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/mplcache \
  "$IMAGE" \
  python /workspace/py_diagnose/compare_es_four_model_ppc_study2.py \
    --ppc-root /ppc \
    --out-dir /ppc/es_four_model_comparison

echo "Completed Study 2 four-model comparison: $OUT_DIR_HOST"
