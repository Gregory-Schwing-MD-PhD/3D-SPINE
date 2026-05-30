#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --job-name=totalspineseg_selective
#SBATCH -o logs/totalspineseg_selective_%j.out
#SBATCH -e logs/totalspineseg_selective_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
# Set MODE to "all" to process every valid study, or "selective" for top/bottom N
MODE=all                   # "all" or "selective"
TOP_N=30                   # used only when MODE=selective
RANK_BY=l5_s1_confidence   # used only when MODE=selective
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "TOTALSPINESEG SEGMENTATION"
echo "Mode:         $MODE"
[[ "$MODE" == "selective" ]] && echo "Top/Bottom N: $TOP_N  |  Rank by: $RANK_BY"
echo "Job ID:       $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES"
echo "Start:        $(date)"
echo "================================================================"

# --- Environment ---
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME
which singularity || echo "WARNING: singularity not found"
export XDG_RUNTIME_DIR="${HOME}/xdr"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"

mkdir -p "$XDG_RUNTIME_DIR" "$NXF_SINGULARITY_CACHEDIR"

unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# --- Resource limits (prevent OOM -> BrokenProcessPool -> Intel MKL crash) ---
# TotalSpineSeg's worker pool defaults to os.cpu_count() (the whole node), which
# oversubscribes the RAM SLURM actually granted us. Pin every thread/worker
# count to the allocated CPU count instead.
N_CPUS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$N_CPUS"
export MKL_NUM_THREADS="$N_CPUS"
export OPENBLAS_NUM_THREADS="$N_CPUS"
export TSS_MAX_WORKERS="$N_CPUS"     # read by 03b_totalspineseg_selective.py
echo "Resource caps: CPUS=$N_CPUS  MEM=96G  TSS_MAX_WORKERS=$N_CPUS"

# --- Paths ---
PROJECT_DIR="$(pwd)"
UNCERTAINTY_CSV="${PROJECT_DIR}/results/epistemic_uncertainty/lstv_uncertainty_metrics.csv"
NIFTI_DIR="${PROJECT_DIR}/results/nifti"
OUTPUT_DIR="${PROJECT_DIR}/results/totalspineseg"
SERIES_CSV="${PROJECT_DIR}/data/raw/train_series_descriptions.csv"
MODELS_DIR="${PROJECT_DIR}/models"
TOTALSPINESEG_MODELS="${PROJECT_DIR}/models/totalspineseg_models"
NNUNET_TRAINER_DIR="${PROJECT_DIR}/models/nnunetv2_trainer"

# Create ALL bind-mount targets before singularity exec (critical!)
mkdir -p logs \
         "$OUTPUT_DIR" \
         "$TOTALSPINESEG_MODELS" \
         "$NNUNET_TRAINER_DIR"

# --- Preflight ---
if [[ "$MODE" == "selective" && ! -f "$UNCERTAINTY_CSV" ]]; then
    echo "ERROR: Uncertainty CSV not found: $UNCERTAINTY_CSV"
    echo "Run sbatch slurm_scripts/00_ian_pan_inference.sh first"
    exit 1
fi

if [[ ! -f "${MODELS_DIR}/valid_id.npy" ]]; then
    echo "ERROR: valid_id.npy not found at ${MODELS_DIR}/valid_id.npy"
    exit 1
fi

# --- Container ---
CONTAINER="docker://go2432/totalspineseg:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/totalspineseg.sif"
if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling TotalSpineSeg container (first time only)..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# Extract nnUNetTrainer directory from container if not already done (one-time setup)
if [[ -z "$(ls -A $NNUNET_TRAINER_DIR 2>/dev/null)" ]]; then
    echo "Extracting nnUNetTrainer directory from container (one-time setup)..."
    singularity exec "$IMG_PATH" \
        cp -r /opt/conda/lib/python3.10/site-packages/nnunetv2/training/nnUNetTrainer/. \
        "$NNUNET_TRAINER_DIR/"
    echo "Extracted $(ls $NNUNET_TRAINER_DIR | wc -l) files"
fi

# --- Build python args ---
if [[ "$MODE" == "all" ]]; then
    PYTHON_ARGS=(
        --nifti_dir   /work/results/nifti
        --output_dir  /work/results/totalspineseg
        --series_csv  /work/data/raw/train_series_descriptions.csv
        --valid_ids   /app/models/valid_id.npy
        --all
    )
else
    PYTHON_ARGS=(
        --uncertainty_csv /work/results/epistemic_uncertainty/lstv_uncertainty_metrics.csv
        --nifti_dir       /work/results/nifti
        --output_dir      /work/results/totalspineseg
        --series_csv      /work/data/raw/train_series_descriptions.csv
        --valid_ids       /app/models/valid_id.npy
        --top_n           "$TOP_N"
        --rank_by         "$RANK_BY"
    )
fi

# --- Run ---
singularity exec --nv \
    --bind "${PROJECT_DIR}":/work \
    --bind "${NIFTI_DIR}":/work/results/nifti \
    --bind "${OUTPUT_DIR}":/work/results/totalspineseg \
    --bind "${MODELS_DIR}":/app/models \
    --bind "${TOTALSPINESEG_MODELS}":/app/totalspineseg_models \
    --bind "${NNUNET_TRAINER_DIR}":/opt/conda/lib/python3.10/site-packages/nnunetv2/training/nnUNetTrainer \
    --env TOTALSPINESEG_DATA=/app/totalspineseg_models \
    --env PYTHONUNBUFFERED=1 \
    --env OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    --env MKL_NUM_THREADS="$MKL_NUM_THREADS" \
    --env OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" \
    --env TSS_MAX_WORKERS="$TSS_MAX_WORKERS" \
    --pwd /work \
    "$IMG_PATH" \
    python3 -u /work/scripts/03b_totalspineseg_selective.py "${PYTHON_ARGS[@]}"

echo "================================================================"
echo "TotalSpineSeg complete | End: $(date)"
echo "Already-done studies are skipped automatically."
echo "================================================================"
