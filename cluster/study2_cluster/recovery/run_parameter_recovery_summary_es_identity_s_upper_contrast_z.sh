#!/bin/bash
#SBATCH --job-name=s2_rec_sum
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/recovery/es_identity_s_upper_contrast_z_main_summary_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/recovery/es_identity_s_upper_contrast_z_main_summary_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache"
mkdir -p "$MPLCONFIGDIR"
image="${IMAGE:-$HOME/containers/hddm_latest.sif}"
base="/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives"
recovery_dir="$base/recovery/es_identity_s_upper_contrast_z_main"
mkdir -p "$recovery_dir" "$base/logs/recovery"

apptainer exec --cleanenv \
  --bind "$repo_root:/workspace" \
  --bind "$recovery_dir:/recovery" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/mplcache \
  "$image" \
  python /workspace/py_recovery/summarize_parameter_recovery_s_upper_contrast_z.py \
    --study study2 \
    --recovery-dir /recovery

echo "finished study2 parameter recovery summary"
