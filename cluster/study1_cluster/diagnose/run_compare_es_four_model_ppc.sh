#!/bin/bash
#SBATCH --job-name=s1_ppc_4cmp
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives/logs/es_four_model_ppc_compare_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives/logs/es_four_model_ppc_compare_%j.err
#SBATCH --mail-type=FAIL,END
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
[[ -d "$REPO_ROOT/py_diagnose/study1_diagnose" && -d "$REPO_ROOT/cluster/study1_cluster/run" ]] || { echo "Submit from Eye_Tracking_Project_aDDM repository root; got $REPO_ROOT" >&2; exit 1; }
IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"
STUDY_ROOT=/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM
PPC_ROOT="$STUDY_ROOT/derivatives/ppc"
mkdir -p "$PPC_ROOT/es_four_model_comparison" "$STUDY_ROOT/derivatives/logs"
[[ -f "$REPO_ROOT/py_diagnose/study1_diagnose/compare_es_four_model_ppc.py" ]] || { echo 'Missing comparison script' >&2; exit 1; }
apptainer exec --cleanenv \
  --bind "$REPO_ROOT:/workspace" \
  --bind "$PPC_ROOT:/ppc" \
  --env PYTHONUNBUFFERED=1 --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg --env MPLCONFIGDIR=/tmp/mplcache \
  "$IMAGE" python /workspace/py_diagnose/study1_diagnose/compare_es_four_model_ppc.py \
     --ppc-root /ppc --out-dir /ppc/es_four_model_comparison
