#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=96:00:00
#SBATCH --job-name=lstv_pipeline_prod
#SBATCH -o logs/lstv_pipeline_prod_%j.out
#SBATCH -e logs/lstv_pipeline_prod_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

echo "================================================================"
echo "LSTV DETECTION PIPELINE — PROD (full dataset)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "================================================================"

export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME
export XDG_RUNTIME_DIR="${HOME}/xdr"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p $XDG_RUNTIME_DIR $NXF_SINGULARITY_CACHEDIR
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

PROJECT_DIR="$(pwd)"
cd "$PROJECT_DIR"
mkdir -p logs

# ── Step 0: Download (if needed) ──────────────────────────────────────────────
PREV_JOB=""
if [[ ! -d "data/raw/train_images" ]] || [[ $(ls -1 data/raw/train_images 2>/dev/null | wc -l) -eq 0 ]]; then
    echo "Data not found — submitting download job"
    JOB0=$(sbatch --parsable slurm_scripts/00_download_data.sh)
    echo "  Job ID: $JOB0  (download)"
    PREV_JOB="--dependency=afterok:$JOB0"
else
    echo "Data found — skipping download"
fi

# ── Step 1: DICOM → NIfTI ────────────────────────────────────────────────────
echo "Submitting: 01_dicom_to_nifti.sh"
JOB1=$(MODE=prod sbatch --parsable $PREV_JOB slurm_scripts/01_dicom_to_nifti.sh)
echo "  Job ID: $JOB1"

# ── Step 2: SPINEPS (GPU) ─────────────────────────────────────────────────────
echo "Submitting: 02_spineps.sh"
JOB2=$(MODE=prod sbatch --parsable --dependency=afterok:$JOB1 slurm_scripts/02_spineps.sh)
echo "  Job ID: $JOB2"

# ── Step 4: LSTV Detection ────────────────────────────────────────────────────
echo "Submitting: 04_lstv_detection.sh"
JOB4=$(sbatch --parsable --dependency=afterok:$JOB2 slurm_scripts/04_lstv_detection.sh)
echo "  Job ID: $JOB4"

# ── Step 5: Visualization ─────────────────────────────────────────────────────
echo "Submitting: 05_visualize_lstv.sh"
JOB5=$(sbatch --parsable --dependency=afterok:$JOB4 slurm_scripts/05_visualize_lstv.sh)
echo "  Job ID: $JOB5"

# ── Step 6: HTML Report ───────────────────────────────────────────────────────
echo "Submitting: 06_html_report_prod.sh"
JOB6=$(sbatch --parsable --dependency=afterok:$JOB5 slurm_scripts/06_html_report_prod.sh)
echo "  Job ID: $JOB6"

echo ""
echo "================================================================"
echo "Dependency chain:"
[[ -n "${JOB0:-}" ]] && echo "  $JOB0  Download" && echo "    ↓"
echo "  $JOB1  DICOM → NIfTI"
echo "    ↓"
echo "  $JOB2  SPINEPS (GPU)"
echo "    ↓"
echo "  $JOB4  LSTV Detection"
echo "    ↓"
echo "  $JOB5  Visualization"
echo "    ↓"
echo "  $JOB6  HTML Report"
echo ""
echo "Monitor:  squeue -u $USER"
echo "Logs:     ls -ltr logs/"
echo "Report:   results/lstv_report.html"
echo "================================================================"

# ── Wait and check ────────────────────────────────────────────────────────────
ALL_JOBS="${JOB1} ${JOB2} ${JOB4} ${JOB5} ${JOB6}"
[[ -n "${JOB0:-}" ]] && ALL_JOBS="${JOB0} ${ALL_JOBS}"

for job_id in $ALL_JOBS; do
    while squeue -j $job_id 2>/dev/null | grep -q $job_id; do
        sleep 60
    done
    STATE=$(sacct -j $job_id --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
    if [[ "$STATE" == "COMPLETED" ]]; then
        echo "  ✓ Job $job_id: $STATE"
    else
        echo "  ✗ Job $job_id: $STATE — check logs"
        exit 1
    fi
done

echo ""
echo "================================================================"
echo "PROD PIPELINE COMPLETE"
echo "Results:  results/lstv_detection/"
echo "Images:   results/lstv_viz/"
echo "Report:   results/lstv_report.html"
echo "End: $(date)"
echo "================================================================"
