#!/bin/bash
#SBATCH -q standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=96:00:00
#SBATCH --job-name=lstv_pipeline
#SBATCH -o logs/lstv_pipeline_%j.out
#SBATCH -e logs/lstv_pipeline_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

MODE=${MODE:-prod}

echo "================================================================"
echo "LSTV DETECTION PIPELINE - MASTER ORCHESTRATOR"
echo "Mode: $MODE"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "================================================================"

PROJECT_DIR="$(pwd)"
cd "$PROJECT_DIR"

mkdir -p logs

echo ""
echo "This script will submit jobs in dependency chain:"
echo "  Step 0: Download data (if needed)"
echo "  Step 1: DICOM → NIfTI conversion"
echo "  Step 2: SPINEPS segmentation (GPU)"
echo "  Step 3: TotalSpineSeg (GPU, parallel with Step 2)"
echo "  Step 4: LSTV detection"
echo ""

# Check if data exists
DATA_EXISTS=false
if [[ -d "data/raw/train_images" ]] && [[ $(ls -1 data/raw/train_images | wc -l) -gt 0 ]]; then
    DATA_EXISTS=true
    echo "✓ Data found in data/raw/train_images"
else
    echo "⚠ Data not found, will download first"
fi

# Job submission
JOBS=()

# Step 0: Download (if needed)
if [[ "$DATA_EXISTS" == "false" ]]; then
    echo ""
    echo "Submitting: 00_download_data.sh"
    JOB0=$(sbatch --parsable slurm_scripts/00_download_data.sh)
    echo "  Job ID: $JOB0"
    JOBS+=("$JOB0")
    PREV_JOB=$JOB0
else
    echo ""
    echo "Skipping download (data exists)"
    PREV_JOB=""
fi

# Step 1: DICOM → NIfTI
echo ""
echo "Submitting: 01_dicom_to_nifti.sh"
if [[ -n "$PREV_JOB" ]]; then
    JOB1=$(sbatch --parsable --dependency=afterok:$PREV_JOB slurm_scripts/01_dicom_to_nifti.sh)
else
    JOB1=$(sbatch --parsable slurm_scripts/01_dicom_to_nifti.sh)
fi
echo "  Job ID: $JOB1"
JOBS+=("$JOB1")

# Step 2: SPINEPS (depends on Step 1)
echo ""
echo "Submitting: 02_spineps.sh"
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm_scripts/02_spineps.sh)
echo "  Job ID: $JOB2 (GPU, after DICOM conversion)"
JOBS+=("$JOB2")

# Step 3: TotalSpineSeg (depends on Step 1, runs in parallel with SPINEPS)
echo ""
echo "Submitting: 03_totalspineseg.sh"
JOB3=$(sbatch --parsable --dependency=afterok:$JOB1 slurm_scripts/03_totalspineseg.sh)
echo "  Job ID: $JOB3 (GPU, parallel with SPINEPS)"
JOBS+=("$JOB3")

# Step 4: LSTV Detection (depends on Steps 2 & 3)
echo ""
echo "Submitting: 04_lstv_detection.sh"
JOB4=$(sbatch --parsable --dependency=afterok:$JOB2:$JOB3 slurm_scripts/04_lstv_detection.sh)
echo "  Job ID: $JOB4 (after SPINEPS + TotalSpineSeg)"
JOBS+=("$JOB4")

echo ""
echo "================================================================"
echo "PIPELINE SUBMITTED"
echo "================================================================"
echo ""
echo "Job dependency chain:"
if [[ "$DATA_EXISTS" == "false" ]]; then
    echo "  $JOB0 (download)"
    echo "    ↓"
fi
echo "  $JOB1 (DICOM → NIfTI)"
echo "    ├─→ $JOB2 (SPINEPS)"
echo "    └─→ $JOB3 (TotalSpineSeg)"
echo "         ↓"
echo "       $JOB4 (LSTV detection)"
echo ""
echo "Monitor jobs:"
echo "  squeue -u $USER"
echo "  squeue -j ${JOBS[*]}"
echo ""
echo "Check logs:"
echo "  tail -f logs/lstv_pipeline_${SLURM_JOB_ID}.out"
echo "  ls -ltr logs/"
echo ""
echo "Expected completion: ~24-48 hours (depending on GPU availability)"
echo "================================================================"

# Wait for all jobs to complete
echo ""
echo "Waiting for pipeline to complete..."
echo "(This master job will stay alive to track progress)"
echo ""

for job_id in "${JOBS[@]}"; do
    echo "Waiting for job $job_id..."
    while squeue -j $job_id 2>/dev/null | grep -q $job_id; do
        sleep 60
    done
    
    # Check if job succeeded
    if sacct -j $job_id --format=State --noheader | grep -q "COMPLETED"; then
        echo "  ✓ Job $job_id completed successfully"
    else
        echo "  ✗ Job $job_id failed"
        STATE=$(sacct -j $job_id --format=State --noheader | head -1 | tr -d ' ')
        echo "    State: $STATE"
        echo ""
        echo "Check logs for details:"
        echo "  ls -ltr logs/ | grep $job_id"
        exit 1
    fi
done

echo ""
echo "================================================================"
echo "PIPELINE COMPLETE!"
echo "================================================================"
echo ""
echo "Results:"
echo "  DICOM → NIfTI:  results/nifti/"
echo "  SPINEPS:        results/spineps/segmentations/"
echo "  TotalSpineSeg:  results/totalspineseg/"
echo "  LSTV Detection: results/lstv_detection/"
echo ""
echo "View LSTV results:"
echo "  cat results/lstv_detection/lstv_summary.json"
echo "  cat results/lstv_detection/lstv_results.json"
echo ""
echo "End: $(date)"
echo "================================================================"
