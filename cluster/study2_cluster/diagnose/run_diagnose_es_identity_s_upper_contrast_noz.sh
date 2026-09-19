#!/bin/bash
#SBATCH --job-name=s2_diag_contrast_noz
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/diagnostics/es_identity_s_upper_contrast_noz_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/diagnostics/es_identity_s_upper_contrast_noz_%j.err
#SBATCH --mail-type=ALL
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

out_root="/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives"
model_dir_host="$out_root/models/es_identity_S_upper_contrast_noz_final"
diag_dir_host="$out_root/figures/es_identity_s_upper_contrast_noz/diagnostics"

chains="${CHAINS:-3}"
ppc_samples="${PPC_SAMPLES:-50}"

mkdir -p "$diag_dir_host" "$out_root/logs/diagnostics"

echo "=============================================================================="
echo "study 2 diagnostics: es identity s upper contrast_noz"
echo "model dir=$model_dir_host"
echo "diagnostic dir=$diag_dir_host"
echo "chains=$chains"
echo "ppc samples=$ppc_samples"
echo "=============================================================================="

test -f "$repo_root/py_diagnose/study2_diagnose/addm_diagnose_es_identity_s_upper_family.py"
test -d "$model_dir_host"

apptainer exec --cleanenv   --bind "$repo_root:/workspace"   --bind "$model_dir_host:/target_models:ro"   --bind "$diag_dir_host:/diagnostics"   --env PYTHONUNBUFFERED=1   --env PYTHONNOUSERSITE=1   --env MPLBACKEND=Agg   --env MPLCONFIGDIR=/tmp/mplcache   "$image"   python /workspace/py_diagnose/study2_diagnose/addm_diagnose_es_identity_s_upper_family.py     --model-dir /target_models     --out-dir /diagnostics     --model contrast     --chains "$chains"     --ppc-samples "$ppc_samples"

echo "finished study 2 diagnostics: es identity s upper contrast_noz"
