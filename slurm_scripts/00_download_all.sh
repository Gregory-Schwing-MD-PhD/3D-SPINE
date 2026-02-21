#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=dl_rsna
#SBATCH -o logs/dl_rsna_%j.out
#SBATCH -e logs/dl_rsna_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
set -x

echo "================================================================"
echo "DOWNLOAD RSNA DATA + MODEL CHECKPOINT + VALIDATION IDs"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "================================================================"

# --- Environment ---
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME
which singularity || echo "WARNING: singularity not found"
export XDG_RUNTIME_DIR="${HOME}/xdr"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p $XDG_RUNTIME_DIR $NXF_SINGULARITY_CACHEDIR
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# --- Paths ---
PROJECT_DIR="$(pwd)"
DATA_DIR="${PROJECT_DIR}/data/raw"
MODELS_DIR="${PROJECT_DIR}/models"
TMP_DIR="${PROJECT_DIR}/.tmp_dl"
mkdir -p "$DATA_DIR" "$MODELS_DIR" "$TMP_DIR" logs

# --- Cleanup on exit ---
cleanup() {
    rm -rf "${PROJECT_DIR}/.kaggle_tmp"
}
trap cleanup EXIT

# --- Kaggle credentials check ---
KAGGLE_JSON="${HOME}/.kaggle/kaggle.json"
if [[ ! -f "$KAGGLE_JSON" ]]; then
    echo "ERROR: Kaggle credentials not found at $KAGGLE_JSON"
    echo ""
    echo "Setup instructions:"
    echo "  1. Go to: https://www.kaggle.com/settings/account"
    echo "  2. Click 'Create New Token' under API section"
    echo "  3. Save kaggle.json to ~/.kaggle/"
    echo "  4. Run: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi
echo "✓ Kaggle credentials found"

# --- Container (spineps-preprocessing used for all downloads) ---
SPINEPS_CONTAINER="docker://go2432/spineps-preprocessing:latest"
SPINEPS_IMG="${NXF_SINGULARITY_CACHEDIR}/spineps-preprocessing.sif"

if [[ ! -f "$SPINEPS_IMG" ]]; then
    echo "Pulling spineps-preprocessing container..."
    singularity pull "$SPINEPS_IMG" "$SPINEPS_CONTAINER"
fi
echo "✓ spineps-preprocessing container ready: $SPINEPS_IMG"

# --- Prepare Kaggle credentials for container bind mount ---
mkdir -p "${PROJECT_DIR}/.kaggle_tmp"
cp "${HOME}/.kaggle/kaggle.json" "${PROJECT_DIR}/.kaggle_tmp/"
chmod 600 "${PROJECT_DIR}/.kaggle_tmp/kaggle.json"

# ================================================================
# PART 1: MODEL CHECKPOINT + VALIDATION IDs
# Must run before DICOM extraction — valid_id.npy gates what we extract
# ================================================================
echo ""
echo "================================================================"
echo "PART 1: POINT NET MODEL CHECKPOINT + VALIDATION IDs"
echo "Dataset: rsna2024-demo-workflow (by hengck23)"
echo "================================================================"

MODEL_PTH="${MODELS_DIR}/point_net_checkpoint.pth"
VALID_NPY="${MODELS_DIR}/valid_id.npy"

if [[ -f "$MODEL_PTH" && -f "$VALID_NPY" ]]; then
    echo "✓ Model checkpoint and validation IDs already exist, skipping download."
else
    singularity exec \
        --bind "${PROJECT_DIR}:/work" \
        --bind "${PROJECT_DIR}/.kaggle_tmp:/root/.kaggle" \
        --pwd /work \
        "$SPINEPS_IMG" \
        bash -c '
            set -euo pipefail
            cd /work/.tmp_dl

            echo "Downloading rsna2024-demo-workflow dataset..."
            kaggle datasets download -d hengck23/rsna2024-demo-workflow

            echo "Extracting model checkpoint and validation IDs..."
            unzip -j rsna2024-demo-workflow.zip 00002484.pth valid_id.npy

            mv 00002484.pth /work/models/point_net_checkpoint.pth
            mv valid_id.npy /work/models/valid_id.npy
            echo "✓ Model artifacts extracted"
        '

    echo "✓ Model checkpoint and validation IDs downloaded"
fi

# ================================================================
# PART 2: RSNA COMPETITION DATA (train CSV + DICOM images)
# Runs after Part 1 so valid_id.npy is available to gate extraction
# ================================================================
echo ""
echo "================================================================"
echo "PART 2: RSNA COMPETITION DATA"
echo "================================================================"

# Use single-quoted heredoc so the outer shell does NO variable expansion.
# All $ references inside belong to the inner bash/python processes.
singularity exec \
    --bind "${PROJECT_DIR}:/work" \
    --bind "${PROJECT_DIR}/.kaggle_tmp:/root/.kaggle" \
    --pwd /work \
    "$SPINEPS_IMG" \
    bash -c '
        set -euo pipefail

        cd /work/.tmp_dl

        # --- train_series_descriptions.csv ---
        if [ -f "/work/data/raw/train_series_descriptions.csv" ]; then
            echo "✓ train_series_descriptions.csv already exists, skipping."
        else
            echo "Downloading train_series_descriptions.csv..."
            kaggle competitions download \
                -c rsna-2024-lumbar-spine-degenerative-classification \
                -f train_series_descriptions.csv

            if [ -f "train_series_descriptions.csv.zip" ]; then
                unzip -o train_series_descriptions.csv.zip
            fi
            mv train_series_descriptions.csv /work/data/raw/
            echo "✓ train_series_descriptions.csv downloaded"
        fi

        # --- DICOM images ---
        # Zip is stored in data/raw so it persists across runs and survives
        # .tmp_dl cleanup. We only re-download if it is genuinely absent.
        ZIP_FILE="/work/data/raw/rsna-2024-lumbar-spine-degenerative-classification.zip"

        if [ ! -f "$ZIP_FILE" ]; then
            echo "Downloading full competition zip (ALL DICOM images)..."
            kaggle competitions download \
                -c rsna-2024-lumbar-spine-degenerative-classification \
                -p /work/data/raw
        else
            echo "✓ Competition zip already present at $ZIP_FILE, skipping download."
        fi

        echo "Extracting missing validation studies..."
        # Only extract studies in valid_id.npy — no point pulling training data.
        # valid_id.npy is guaranteed to exist at this point (downloaded in Part 1).
        python3 - "$ZIP_FILE" <<'"'"'PYEOF'"'"'
import sys, zipfile, numpy as np
from pathlib import Path

zip_path   = sys.argv[1]
output_dir = Path("/work/data/raw/train_images")
valid_ids  = set(str(v) for v in np.load("/work/models/valid_id.npy"))

existing_studies = set(d.name for d in output_dir.iterdir() if d.is_dir()) if output_dir.exists() else set()
missing_valid    = valid_ids - existing_studies

print(f"Validation IDs total:  {len(valid_ids)}")
print(f"Already on disk:       {len(existing_studies)} studies")
print(f"Need to extract:       {len(missing_valid)} studies")

if not missing_valid:
    print("✓ All validation studies already extracted, nothing to do")
    sys.exit(0)

with zipfile.ZipFile(zip_path, "r") as z:
    to_extract = [
        f for f in z.namelist()
        if f.startswith("train_images/") and (
            len(f.split("/")) <= 1 or f.split("/")[1] in missing_valid
        )
    ]
    print(f"Files to extract: {len(to_extract)}")
    z.extractall("/work/data/raw", members=to_extract)
    print("✓ Extraction complete")
PYEOF
    '

# ================================================================
# VERIFICATION
# ================================================================
echo ""
echo "================================================================"
echo "VERIFICATION"
echo "================================================================"

echo ""
echo "RSNA data:"
ls -lh "${DATA_DIR}/train_series_descriptions.csv"
N_STUDIES=$(ls "${DATA_DIR}/train_images/" | wc -l)
echo "DICOM studies extracted: $N_STUDIES"

echo ""
echo "Model artifacts:"
ls -lh "$MODEL_PTH"
ls -lh "$VALID_NPY"

echo ""
echo "================================================================"
echo "ALL DOWNLOADS COMPLETE"
echo "End time: $(date)"
echo "================================================================"
echo ""
echo "Next steps:"
echo "  1. Convert DICOM:  sbatch slurm_scripts/01_dicom_to_nifti.sh"
echo "  2. Run pipeline:   sbatch slurm_scripts/00_run_all.sh"
echo ""
echo "IMPORTANT: Inference will ONLY use validation set studies to avoid data leakage!"
echo "================================================================"
