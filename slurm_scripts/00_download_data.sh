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

echo "================================================================"
echo "DOWNLOAD RSNA DATA"
echo "Job ID: $SLURM_JOB_ID"
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
mkdir -p "$DATA_DIR" logs "${PROJECT_DIR}/.tmp_dl"

# --- Kaggle check ---
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

# --- Container ---
CONTAINER="docker://go2432/spineps-preprocessing:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spineps-preprocessing.sif"
if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling container..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# --- Prepare Kaggle mount ---
mkdir -p ${PROJECT_DIR}/.kaggle_tmp
cp ${HOME}/.kaggle/kaggle.json ${PROJECT_DIR}/.kaggle_tmp/
chmod 600 ${PROJECT_DIR}/.kaggle_tmp/kaggle.json

echo ""
echo "================================================================"
echo "DOWNLOADING RSNA COMPETITION DATA"
echo "================================================================"

singularity exec \
    --bind $PROJECT_DIR:/work \
    --bind ${PROJECT_DIR}/.kaggle_tmp:/root/.kaggle \
    --pwd /work \
    "$IMG_PATH" \
    bash -c "
        set -e
        cd .tmp_dl

        echo 'Downloading train_series_descriptions.csv...'
        kaggle competitions download -c rsna-2024-lumbar-spine-degenerative-classification -f train_series_descriptions.csv

        if [ -f 'train_series_descriptions.csv.zip' ]; then
            echo 'Unzipping CSV...'
            unzip -o train_series_descriptions.csv.zip
        elif [ -f 'train_series_descriptions.csv' ]; then
            echo 'CSV downloaded uncompressed (skipping unzip).'
        else
            echo 'ERROR: CSV file not found after download'
            ls -lh
            exit 1
        fi

        mv train_series_descriptions.csv /work/data/raw/
        echo '✓ Series CSV extracted'

        echo ''
        echo 'Downloading full competition zip (ALL DICOM images)...'
        ZIP_FILE='rsna-2024-lumbar-spine-degenerative-classification.zip'

        if [ ! -f \"\$ZIP_FILE\" ]; then
            kaggle competitions download -c rsna-2024-lumbar-spine-degenerative-classification
        else
            echo '  Zip already downloaded, skipping.'
        fi

        echo ''
        echo 'Extracting ALL studies (no validation filter)...'
        python3 -c \"
import zipfile, os
from pathlib import Path

output_dir = Path('/work/data/raw/train_images')
existing_studies = set(d.name for d in output_dir.iterdir() if d.is_dir()) if output_dir.exists() else set()

print(f'Already extracted: {len(existing_studies)} studies')
print('Extracting all train_images/ files...')

with zipfile.ZipFile('\$ZIP_FILE', 'r') as z:
    # Get all train_images files
    to_extract = [f for f in z.namelist() if f.startswith('train_images/')]
    
    # Skip already extracted studies
    if existing_studies:
        to_extract = [
            f for f in to_extract 
            if len(f.split('/')) <= 1 or f.split('/')[1] not in existing_studies
        ]
    
    print(f'Total files to extract: {len(to_extract)}')
    
    if to_extract:
        z.extractall('/work/data/raw', members=to_extract)
        print('✓ Extraction complete')
    else:
        print('✓ All files already extracted')
\"
    "

exit_code=$?

# Cleanup
rm -rf ${PROJECT_DIR}/.kaggle_tmp

if [ $exit_code -ne 0 ]; then
    echo "ERROR: Download failed"
    exit $exit_code
fi

# Verify everything
echo ""
echo "================================================================"
echo "VERIFICATION"
echo "================================================================"

echo ""
echo "Data files:"
ls -lh ${DATA_DIR}/train_series_descriptions.csv
N_STUDIES=$(ls ${DATA_DIR}/train_images/ | wc -l)
echo "DICOM studies extracted: $N_STUDIES"

echo ""
echo "================================================================"
echo "DOWNLOAD COMPLETE"
echo "================================================================"
echo ""
echo "Next steps:"
echo "  1. Convert DICOM:  sbatch slurm_scripts/01_dicom_to_nifti.sh"
echo "  2. Run pipeline:   sbatch slurm_scripts/00_run_all.sh"
echo "================================================================"
