#!/bin/bash
#SBATCH --job-name=s1_ppc_targeted
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives/logs/targeted_source_ppc_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives/logs/targeted_source_ppc_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"
STUDY_ROOT="/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM"
PPC_ROOT="$STUDY_ROOT/derivatives/ppc"
MODEL_INPUT="$STUDY_ROOT/derivatives/models/model_input_ES_identity_S_upper.csv"
OUT_DIR="$PPC_ROOT/es_targeted_source_sensitivity"
SCRIPT="$REPO_ROOT/py_diagnose/targeted_ppc_source_sensitivity.py"

mkdir -p "$OUT_DIR" "$STUDY_ROOT/derivatives/logs"

[[ -f "$IMAGE" ]] || { echo "MISSING: $IMAGE" >&2; exit 1; }
[[ -f "$SCRIPT" ]] || { echo "MISSING: $SCRIPT" >&2; exit 1; }
[[ -f "$MODEL_INPUT" ]] || { echo "MISSING: $MODEL_INPUT" >&2; exit 1; }

for tag in basic_noz basic_z contrast_noz contrast_z; do
  ROOT="$PPC_ROOT/es_identity_s_upper_${tag}_main"
  [[ -f "$ROOT/metadata.json" ]] || { echo "MISSING: $ROOT/metadata.json" >&2; exit 1; }
  [[ -f "$ROOT/tables/observed_replot_trials.csv.gz" ]] || { echo "MISSING observed replot data: $ROOT" >&2; exit 1; }
  [[ -f "$ROOT/tables/ppc_replot_trials.csv.gz" ]] || { echo "MISSING PPC replot data: $ROOT" >&2; exit 1; }
done

apptainer exec --cleanenv \
  --bind "$REPO_ROOT:/workspace" \
  --bind "$STUDY_ROOT:/study:ro" \
  --bind "$PPC_ROOT:/ppc" \
  --bind "$OUT_DIR:/targeted" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/mplcache \
  "$IMAGE" \
  python /workspace/py_diagnose/targeted_ppc_source_sensitivity.py \
    --study study1 \
    --ppc-root /ppc \
    --model-input /study/derivatives/models/model_input_ES_identity_S_upper.csv \
    --out-dir /targeted \
    --low-gaze-quantile 0.20

echo "Completed Study 1 targeted source-specific PPC: $OUT_DIR"
