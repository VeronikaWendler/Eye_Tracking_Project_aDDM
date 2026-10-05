#!/bin/bash
#SBATCH --job-name=s2_markerplots
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/s2_markerplots_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/s2_markerplots_%j.err
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
compact_dir="/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/figures/es_identity_S_upper_contrast_z_final/individual_differences_compact"
merged_file="$compact_dir/Study2_behaviour_addm_merged.csv"
out_dir="$compact_dir/selected_marker_plots"

mkdir -p "$out_dir"

test -f "$repo_root/py_correlate/07_make_cross_task_marker_plots.py"
test -f "$merged_file"
test -f "$image"

apptainer exec --cleanenv \
  --bind "$repo_root:/workspace" \
  --bind "$compact_dir:/results" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/mplcache \
  "$image" \
  python /workspace/py_correlate/07_make_cross_task_marker_plots.py \
    --study 2 \
    --merged /results/Study2_behaviour_addm_merged.csv \
    --out-dir /results/selected_marker_plots

echo "Finished Study 2 marker plots."
