#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --job-name=lstv_report
#SBATCH -o logs/lstv_report_%j.out
#SBATCH -e logs/lstv_report_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
N_REPS=3   # number of representative examples per Castellvi type in the report
# ─────────────────────────────────────────────────────────────────────────────
# No TOP_N / RANK_BY here — the report operates on whatever studies were
# classified by 04_lstv_detection.sh and visualized by 05_visualize_lstv.sh.
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "LSTV HTML REPORT GENERATION"
echo "N_REPS=$N_REPS"
echo "Job: $SLURM_JOB_ID  |  Start: $(date)"
echo "================================================================"

# --- Environment ---
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME
which singularity || { echo "ERROR: singularity not found"; exit 1; }
export XDG_RUNTIME_DIR="${HOME}/xdr"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "$XDG_RUNTIME_DIR" "$NXF_SINGULARITY_CACHEDIR"
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# --- Paths ---
PROJECT_DIR="$(pwd)"
LSTV_JSON="${PROJECT_DIR}/results/lstv_detection/lstv_results.json"
IMAGE_DIR="${PROJECT_DIR}/results/lstv_viz"
OUTPUT_HTML="${PROJECT_DIR}/results/lstv_report.html"

mkdir -p logs

# --- Preflight ---
if [[ ! -f "$LSTV_JSON" ]]; then
    echo "ERROR: lstv_results.json not found at $LSTV_JSON"
    echo "Run 04_lstv_detection.sh first"
    exit 1
fi

N_IMAGES=$(ls "$IMAGE_DIR"/*.png 2>/dev/null | wc -l)
if [[ "$N_IMAGES" -eq 0 ]]; then
    echo "WARNING: No PNGs found in $IMAGE_DIR"
    echo "Run 05_visualize_lstv.sh first for images in the report."
    echo "Continuing — report will be generated without images."
else
    echo "Found $N_IMAGES overlay PNGs"
fi

echo "Classified studies in JSON: $(python3 -c "import json; d=json.load(open('$LSTV_JSON')); print(len(d))" 2>/dev/null || echo '?')"

# --- Container ---
CONTAINER="docker://go2432/spineps-preprocessing:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spineps-preprocessing.sif"
if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling container image..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# --- Run ---
singularity exec \
    --bind "${PROJECT_DIR}":/work \
    --env PYTHONUNBUFFERED=1 \
    --pwd /work \
    "$IMG_PATH" \
    python3 -u /work/scripts/06_html_report.py \
        --lstv_json   /work/results/lstv_detection/lstv_results.json \
        --image_dir   /work/results/lstv_viz \
        --output_html /work/results/lstv_report.html \
        --n_reps      "$N_REPS"

echo "================================================================"
echo "Report complete -> results/lstv_report.html"
echo "End: $(date)"
echo "================================================================"
