#!/bin/bash
#SBATCH --job-name=s2_es_indivcorr
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/s2_es_indivcorr_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/s2_es_indivcorr_%j.err
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

data_root="/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/data"
behaviour_file="$data_root/Study2_Behaviour_with_Gaze_AnalysisReady.csv"
slope_file="$repo_root/data_sets/study2_data_sets/Study2_phase_slopes_full.csv"

base="/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives"
model_dir="$base/models/es_identity_S_upper_contrast_z_final"
out_dir="$base/figures/es_identity_S_upper_contrast_z_final/individual_differences_compact"

mkdir -p "$out_dir"

echo "=============================================================================="
echo "STUDY 2: compact behaviour x ESaDDM+z individual differences"
echo "repo_root=$repo_root"
echo "slope_file=$slope_file"
echo "behaviour_file=$behaviour_file"
echo "model_dir=$model_dir"
echo "out_dir=$out_dir"
echo "=============================================================================="

test -f "$repo_root/py_correlate/05_ES_selective_neglect_addm_individual_differences_ALLPARAMS.py"
test -f "$slope_file"
test -f "$behaviour_file"
test -d "$model_dir"

apptainer exec --cleanenv \
  --bind "$repo_root:/workspace" \
  --bind "$data_root:/source:ro" \
  --bind "$model_dir:/target_models:ro" \
  --bind "$out_dir:/results" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/mplcache \
  "$image" \
  python /workspace/py_correlate/05_ES_selective_neglect_addm_individual_differences_ALLPARAMS.py \
    --study 2 \
    --slopes /workspace/data_sets/study2_data_sets/Study2_phase_slopes_full.csv \
    --behaviour-data /source/Study2_Behaviour_with_Gaze_AnalysisReady.csv \
    --model-dir /target_models \
    --out-dir /results

echo "Finished Study 2 compact individual-difference correlations."
