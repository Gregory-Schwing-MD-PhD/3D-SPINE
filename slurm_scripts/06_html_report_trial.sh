#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --job-name=lstv_report_trial
#SBATCH -o logs/lstv_report_%j.out
#SBATCH -e logs/lstv_report_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

echo "================================================================"
echo "LSTV HTML REPORT (trial)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "================================================================"

export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

PROJECT_DIR="$(pwd)"
LSTV_JSON="${PROJECT_DIR}/results/lstv_detection/lstv_results.json"
IMAGE_DIR="${PROJECT_DIR}/results/lstv_viz"
OUTPUT_HTML="${PROJECT_DIR}/results/lstv_report_trial.html"

if [[ ! -f "$LSTV_JSON" ]]; then
    echo "ERROR: lstv_results.json not found at $LSTV_JSON"
    echo "Run 04_lstv_detection.sh first."
    exit 1
fi

if [[ ! -d "$IMAGE_DIR" ]]; then
    echo "WARNING: image_dir not found at $IMAGE_DIR — report will have no images"
    mkdir -p "$IMAGE_DIR"
fi

python "${PROJECT_DIR}/scripts/06_html_report.py" \
    --lstv_json   "$LSTV_JSON" \
    --image_dir   "$IMAGE_DIR" \
    --output_html "$OUTPUT_HTML" \
    --n_reps      3

echo ""
echo "Report: $OUTPUT_HTML"
echo "================================================================"
echo "Done | End: $(date)"
echo "================================================================"
