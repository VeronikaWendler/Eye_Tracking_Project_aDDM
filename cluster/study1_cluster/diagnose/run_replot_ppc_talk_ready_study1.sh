#!/bin/bash
#SBATCH --job-name=s1_ppc_talk
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives/logs/ppc_talk_ready_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives/logs/ppc_talk_ready_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=VAW508@student.bham.ac.uk
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache"; mkdir -p "$MPLCONFIGDIR"
IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"
STUDY_ROOT="/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM"
PPC_DIR_HOST="$STUDY_ROOT/derivatives/ppc/es_identity_s_upper_contrast_z_main"
mkdir -p "$STUDY_ROOT/derivatives/logs" "$PPC_DIR_HOST/figures/talk_ready"
test -f "$REPO_ROOT/py_diagnose/replot_ppc_talk_ready.py"
test -d "$PPC_DIR_HOST/tables"
apptainer exec --cleanenv \
  --bind "$REPO_ROOT:/workspace" \
  --bind "$PPC_DIR_HOST:/ppc" \
  --env PYTHONUNBUFFERED=1 --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg --env MPLCONFIGDIR=/tmp/mplcache \
  "$IMAGE" \
  python /workspace/py_diagnose/replot_ppc_talk_ready.py \
    --study study1 --ppc-dir /ppc --out-dir /ppc/figures/talk_ready
echo "Finished Study 1 talk-ready PPC replot."
