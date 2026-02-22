#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=lstv_detect
#SBATCH -o logs/lstv_detect_%j.out
#SBATCH -e logs/lstv_detect_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ── Configuration — edit these to change behaviour ────────────────────────────
TOP_N=1                    # studies from each end — must match 02b + 03b + register settings
RANK_BY=l5_s1_confidence   # column to rank by — must match 02b + 03b + register settings
ALL=false                  # set to true to classify every study with SPINEPS segmentation
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "LSTV DETECTION (Hybrid Two-Phase Castellvi Classifier)"
echo "TOP_N=$TOP_N  RANK_BY=$RANK_BY  ALL=$ALL"
echo "Job: $SLURM_JOB_ID  |  Start: $(date)"
echo "================================================================"

# --- Environment ---
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME
which singularity || echo "WARNING: singularity not found"
export XDG_RUNTIME_DIR="${HOME}/xdr"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "$XDG_RUNTIME_DIR" "$NXF_SINGULARITY_CACHEDIR"
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# --- Paths ---
PROJECT_DIR="$(pwd)"
UNCERTAINTY_CSV="${PROJECT_DIR}/results/epistemic_uncertainty/lstv_uncertainty_metrics.csv"
MODELS_DIR="${PROJECT_DIR}/models"

mkdir -p logs results/lstv_detection

# --- Preflight ---
if [[ ! -d "${PROJECT_DIR}/results/spineps/segmentations" ]]; then
    echo "ERROR: SPINEPS segmentations not found. Run 02b_spineps_selective.sh first"
    exit 1
fi

if [[ "$ALL" != "true" ]]; then
    if [[ ! -f "$UNCERTAINTY_CSV" ]]; then
        echo "ERROR: Uncertainty CSV not found: $UNCERTAINTY_CSV"
        echo "Run 00_ian_pan_inference.sh first, or set ALL=true"
        exit 1
    fi
    if [[ ! -f "${MODELS_DIR}/valid_id.npy" ]]; then
        echo "ERROR: valid_id.npy not found at ${MODELS_DIR}/valid_id.npy"
        exit 1
    fi
fi

# --- Container ---
CONTAINER="docker://go2432/spineps-preprocessing:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spineps-preprocessing.sif"
if [[ ! -f "$IMG_PATH" ]]; then
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# --- Build selection args (bash array avoids multiline string splitting bugs) ---
SELECTION_ARGS=()
if [[ "$ALL" == "true" ]]; then
    SELECTION_ARGS+=( "--all" )
    echo "ALL mode: classifying every study with SPINEPS segmentations"
    echo "Note: Phase 2 will be skipped for studies missing TSS or registration."
else
    SELECTION_ARGS+=(
        "--uncertainty_csv" "/work/results/epistemic_uncertainty/lstv_uncertainty_metrics.csv"
        "--valid_ids"       "/app/models/valid_id.npy"
        "--top_n"           "$TOP_N"
        "--rank_by"         "$RANK_BY"
    )
    echo "Selective mode: top/bottom $TOP_N by $RANK_BY"
fi

# --- Run ---
singularity exec \
    --bind "${PROJECT_DIR}":/work \
    --bind "${MODELS_DIR}":/app/models \
    --env PYTHONUNBUFFERED=1 \
    --pwd /work \
    "$IMG_PATH" \
    python3 -u /work/scripts/04_detect_lstv.py \
        --spineps_dir    /work/results/spineps \
        --totalspine_dir /work/results/totalspineseg \
        --registered_dir /work/results/registered \
        --nifti_dir      /work/results/nifti \
        --output_dir     /work/results/lstv_detection \
        "${SELECTION_ARGS[@]}"

echo "================================================================"
echo "LSTV detection complete | End: $(date)"
echo "Results: results/lstv_detection/lstv_results.json"
echo "================================================================"
