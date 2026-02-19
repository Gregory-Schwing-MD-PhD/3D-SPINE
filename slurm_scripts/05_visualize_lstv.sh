#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --job-name=lstv_viz
#SBATCH -o logs/lstv_viz_%j.out
#SBATCH -e logs/lstv_viz_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

echo "================================================================"
echo "LSTV VISUALIZATION"
echo "Job ID: $SLURM_JOB_ID"
echo "Start:  $(date)"
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
SPINEPS_DIR="${PROJECT_DIR}/results/spineps"
TOTALSPINE_DIR="${PROJECT_DIR}/results/totalspineseg"
LSTV_DIR="${PROJECT_DIR}/results/lstv_detection"
OUTPUT_DIR="${PROJECT_DIR}/results/lstv_viz"

mkdir -p logs "$OUTPUT_DIR"

# --- Optional: visualize a single study ID instead of the full batch ---
# Set STUDY_ID to a specific ID, or leave empty for batch mode.
STUDY_ID="${STUDY_ID:-}"

# --- Container ---
CONTAINER="docker://go2432/spineps-preprocessing:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spineps-preprocessing.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling container image..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# --- Build run command ---
# --lstv_json is optional but strongly recommended: annotates the summary
# panel with per-side metrics so you can audit the classifier visually.
# Remove or comment it out if 04_detect_lstv.py has not been run yet.

LSTV_JSON="${LSTV_DIR}/lstv_results.json"
LSTV_JSON_ARG=""
if [[ -f "$LSTV_JSON" ]]; then
    LSTV_JSON_ARG="--lstv_json /data/lstv/lstv_results.json"
    echo "Detection results found — summary panels will be annotated."
else
    echo "WARNING: lstv_results.json not found — summary panels will be unannotated."
    echo "         Run 04_lstv_detection.sh first for fully annotated output."
fi

STUDY_ID_ARG=""
if [[ -n "$STUDY_ID" ]]; then
    STUDY_ID_ARG="--study_id ${STUDY_ID}"
    echo "Single-study mode: ${STUDY_ID}"
else
    echo "Batch mode: all studies in ${SPINEPS_DIR}/segmentations/"
fi

# --- Run ---
singularity exec \
    --bind "${PROJECT_DIR}":/work \
    --bind "${SPINEPS_DIR}":/data/spineps \
    --bind "${TOTALSPINE_DIR}":/data/totalspine \
    --bind "${LSTV_DIR}":/data/lstv \
    --bind "${OUTPUT_DIR}":/data/output \
    --pwd /work \
    "$IMG_PATH" \
    python /work/scripts/05_visualize_overlay.py \
        --spineps_dir    /data/spineps \
        --totalspine_dir /data/totalspine \
        --output_dir     /data/output \
        ${STUDY_ID_ARG} \
        ${LSTV_JSON_ARG}

echo "================================================================"
echo "Visualization complete"
echo "PNGs written to: ${OUTPUT_DIR}"
echo "End: $(date)"
echo "================================================================"
