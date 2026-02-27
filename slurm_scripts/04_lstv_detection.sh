#!/bin/bash
# =============================================================================
# 04_lstv_detection.sh  —  LSTV Detection + Morphometrics
# =============================================================================
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=lstv_detect
#SBATCH -o logs/lstv_detect_%j.out
#SBATCH -e logs/lstv_detect_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
ALL=true                   # true  → classify every study with SPINEPS segmentation
STUDY_ID=""                # single study override (leave empty for batch)
NO_MORPHO=false            # true  → skip extended lstv_engine morphometrics (faster)
# ─────────────────────────────────────────────────────────────────────────────

echo "=============================================================="
echo "LSTV DETECTION + MORPHOMETRICS"
echo "ALL=$ALL  STUDY_ID=${STUDY_ID:-<batch>}  NO_MORPHO=$NO_MORPHO"
echo "Job: $SLURM_JOB_ID  |  Start: $(date)"
echo "=============================================================="

export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME
export XDG_RUNTIME_DIR="${HOME}/xdr"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "$XDG_RUNTIME_DIR" "$NXF_SINGULARITY_CACHEDIR" logs results/lstv_detection

PROJECT_DIR="$(pwd)"
CONTAINER="docker://go2432/spineps-preprocessing:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spineps-preprocessing.sif"
[[ ! -f "$IMG_PATH" ]] && singularity pull "$IMG_PATH" "$CONTAINER"

ARGS=()
if [[ -n "$STUDY_ID" ]]; then
    ARGS+=("--study_id" "$STUDY_ID")
elif [[ "$ALL" == "true" ]]; then
    ARGS+=("--all")
fi
[[ "$NO_MORPHO" == "true" ]] && ARGS+=("--no_morpho")

singularity exec \
    --bind "${PROJECT_DIR}:/work" \
    --env PYTHONUNBUFFERED=1 \
    --pwd /work \
    "$IMG_PATH" \
    python3 -u /work/scripts/04_detect_lstv.py \
        --spineps_dir    /work/results/spineps \
        --totalspine_dir /work/results/totalspineseg \
        --registered_dir /work/results/registered \
        --nifti_dir      /work/results/nifti \
        --output_dir     /work/results/lstv_detection \
        "${ARGS[@]}"

echo "=============================================================="
echo "Done | End: $(date)"
echo "  results/lstv_detection/lstv_results.json"
echo "  results/lstv_detection/lstv_summary.json"
echo "=============================================================="
