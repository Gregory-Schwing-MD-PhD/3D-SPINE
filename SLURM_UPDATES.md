# SLURM Scripts - Updated Summary

All SLURM scripts now follow consistent format with:
- Proper environment setup (CONDA_PREFIX, Singularity paths)
- Singularity container execution
- Email notifications to go2432@wayne.edu
- Consistent bind mount patterns

## Container Requirements

### 1. spineps-preprocessing.sif
- **Used by**: 00_download_data.sh, 01_dicom_to_nifti.sh, 04_lstv_detection.sh
- **Image**: docker://go2432/spineps-preprocessing:latest
- **Contains**: Python, dcm2niix, numpy, pandas, nibabel, kaggle CLI
- **Purpose**: Data download, DICOM conversion, basic Python analysis

### 2. spineps-segmentation.sif
- **Used by**: 02_spineps.sh
- **Image**: docker://go2432/spineps-segmentation:latest
- **Contains**: SPINEPS models, CUDA support, deep learning stack
- **Purpose**: Vertebra and subregion segmentation
- **GPU**: Required (--nv flag)

### 3. totalsegmentator.sif
- **Used by**: 03_totalspineseg.sh
- **Image**: docker://wasserth/totalsegmentator:latest
- **Contains**: TotalSegmentator models, nnU-Net
- **Purpose**: Vertebra segmentation (sagittal + axial)
- **GPU**: Required (--nv flag)

## File Paths

All scripts use consistent path mapping:

```bash
Host                              Container
====================================================================================================
$PROJECT_DIR                   → /work
data/raw/train_images          → /data/input
results/nifti                  → /data/nifti
results/spineps                → /data/output (SPINEPS)
results/totalspineseg          → /data/output (TotalSpineSeg)
results/lstv_detection         → /data/output (LSTV)
models/spineps_cache           → /app/models
models/spineps_pkg_models      → /opt/conda/lib/.../spineps/models
```

## Key Changes from Original

### 00_download_data.sh
- ✅ Kept original format (already correct)
- ❌ Removed Ian Pan model download
- ❌ Removed validation filter - downloads ALL studies

### 01_dicom_to_nifti.sh
- ✅ Added Singularity execution
- ✅ Uses spineps-preprocessing container
- ✅ Maps paths: /data/input, /data/output, /data/raw
- ⏰ Extended time: 4h → 8h

### 02_spineps.sh
- ✅ Already using Singularity (kept existing format)
- ✅ Updated job name: spineps → spineps_run
- ⏰ Extended time: 24h → 36h
- ✅ Matches user's existing format exactly

### 03_totalspineseg.sh
- ✅ Complete rewrite with Singularity
- ✅ Uses TotalSegmentator container
- ✅ Maps paths: /data/nifti, /data/output
- ⏰ Extended time: 24h → 36h
- 🔧 SERIES variable support (sagittal, axial, both)

### 04_lstv_detection.sh
- ✅ Added Singularity execution
- ✅ Uses spineps-preprocessing container (has numpy/nibabel)
- ✅ Maps paths: /data/spineps, /data/totalspine, /data/output
- ⏰ Extended time: 1h → 2h

### 00_run_all.sh
- ✅ Added environment setup (Singularity paths)
- ✅ Passes MODE variable to child jobs
- ✅ Monitors all jobs and reports status
- ⏰ 96h limit (enough for full pipeline)

## Usage

### Quick Start
```bash
cd ~/lstv-detector

# Copy all updated scripts
cp /path/to/outputs/*.sh slurm_scripts/
chmod +x slurm_scripts/*.sh

# Run trial mode (3 studies)
MODE=trial sbatch slurm_scripts/00_run_all.sh

# Run production (all studies)
MODE=prod sbatch slurm_scripts/00_run_all.sh
```

### Manual Execution
```bash
# If data already exists
MODE=prod sbatch slurm_scripts/01_dicom_to_nifti.sh
# Wait, then:
MODE=prod sbatch slurm_scripts/02_spineps.sh
MODE=prod sbatch slurm_scripts/03_totalspineseg.sh
# Wait, then:
sbatch slurm_scripts/04_lstv_detection.sh
```

### Monitor Jobs
```bash
squeue -u $USER
tail -f logs/lstv_pipeline_*.out
ls -ltr logs/
```

## Expected Timeline

| Step | Duration | Resource |
|------|----------|----------|
| 0. Download | ~2-4h | CPU |
| 1. DICOM→NIfTI | ~4-8h | CPU |
| 2. SPINEPS | ~24-36h | GPU (parallel) |
| 3. TotalSpineSeg | ~24-36h | GPU (parallel) |
| 4. LSTV Detection | ~0.5-2h | CPU |
| **Total** | **~24-48h** | (with GPU availability) |

## Outputs

```
results/
├── nifti/
│   ├── {study_id}_sag_t2.nii.gz
│   └── {study_id}_axial_t2.nii.gz
├── spineps/
│   └── segmentations/
│       ├── {study_id}_seg-vert_msk.nii.gz
│       ├── {study_id}_seg-spine_msk.nii.gz
│       ├── {study_id}_ctd.json (ALL structures!)
│       └── {study_id}_unc.nii.gz
├── totalspineseg/
│   ├── {study_id}_sagittal_vertebrae.nii.gz
│   └── {study_id}_axial_vertebrae.nii.gz
└── lstv_detection/
    ├── lstv_results.json
    └── lstv_summary.json
```

## Troubleshooting

### Container not found
```bash
ls ~/singularity_cache/
# If missing, scripts will auto-pull on first run
```

### GPU not available
```bash
sinfo -o "%20N %10c %10m %25f %10G"
# Check v100 GPU availability
```

### Bind mount errors
```bash
# Ensure directories exist before job runs
mkdir -p results/nifti results/spineps results/totalspineseg
mkdir -p models/spineps_cache models/spineps_pkg_models
```

### Python script not found
```bash
# Ensure scripts are in PROJECT_DIR/scripts/
ls scripts/*.py
```

## Email Notifications

All scripts send email to: **go2432@wayne.edu**
- BEGIN: When job starts
- END: When job completes successfully
- FAIL: When job fails

Update email in each script header if needed:
```bash
#SBATCH --mail-user=YOUR_EMAIL@wayne.edu
```
