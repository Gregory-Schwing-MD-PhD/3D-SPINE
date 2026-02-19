# LSTV Detection Pipeline

Automated detection and classification of lumbosacral transitional vertebrae (LSTV) using SPINEPS and TotalSpineSeg.

## Overview

This pipeline detects LSTV using Castellvi classification:
- **Type I**: Enlarged transverse process (>19mm)
- **Type II**: Extra lumbar vertebra (L6)
- **Type III**: L5-S1 fusion
- **Type IV**: Type II + Type III

## Architecture

```
Input: DICOM studies
├─ Step 1: DICOM → NIfTI conversion
│  └─ Outputs: *_sag_t2.nii.gz, *_axial_t2.nii.gz
│
├─ Step 2: SPINEPS (Sagittal T2)
│  └─ Outputs: Costal processes, centroids, uncertainty
│
├─ Step 3: TotalSpineSeg (Sagittal + Axial T2)
│  └─ Outputs: Vertebra segmentations
│
└─ Step 4: LSTV Detection
   └─ Outputs: Castellvi classification results
```

## File Structure

```
spineps-segmentation/
├── scripts/
│   ├── 01_dicom_to_nifti.py       # DICOM conversion
│   ├── 02_run_spineps.py          # SPINEPS segmentation
│   ├── 03_run_totalspineseg.py    # TotalSpineSeg
│   ├── 04_detect_lstv.py          # LSTV detection
│   └── visualize_overlay.py       # Visualization
├── slurm_scripts/
│   ├── 01_dicom_to_nifti.sh
│   ├── 02_spineps.sh
│   ├── 03_totalspineseg.sh
│   └── 04_lstv_detection.sh
├── results/
│   ├── nifti/                     # Converted NIfTI files (shared)
│   ├── spineps/                   # SPINEPS outputs
│   ├── totalspineseg/             # TotalSpineSeg outputs
│   └── lstv_detection/            # Final LSTV results
└── data/raw/train_images/         # Input DICOMs
```

## Quick Start

### 1. Convert DICOM to NIfTI

```bash
# Trial mode (3 studies)
MODE=trial sbatch slurm_scripts/01_dicom_to_nifti.sh

# Production (all studies)
MODE=prod sbatch slurm_scripts/01_dicom_to_nifti.sh
```

### 2. Run SPINEPS (can run in parallel with TotalSpineSeg)

```bash
MODE=prod sbatch slurm_scripts/02_spineps.sh
```

### 3. Run TotalSpineSeg (can run in parallel with SPINEPS)

```bash
# Process both sagittal and axial
MODE=prod SERIES=both sbatch slurm_scripts/03_totalspineseg.sh

# Or just sagittal
MODE=prod SERIES=sagittal sbatch slurm_scripts/03_totalspineseg.sh
```

### 4. Detect LSTV

```bash
# After both SPINEPS and TotalSpineSeg finish
sbatch slurm_scripts/04_lstv_detection.sh
```

### 5. Parallel Execution

```bash
# Step 1: Convert
JOB1=$(sbatch --parsable slurm_scripts/01_dicom_to_nifti.sh)

# Step 2 & 3: Segment in parallel
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm_scripts/02_spineps.sh)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB1 slurm_scripts/03_totalspineseg.sh)

# Step 4: Detect (waits for both)
sbatch --dependency=afterok:$JOB2:$JOB3 slurm_scripts/04_lstv_detection.sh
```

## Output Files

### Per Study

**SPINEPS:**
```
results/spineps/segmentations/
├── {study_id}_seg-vert_msk.nii.gz    # Instance (vertebrae, discs, endplates)
├── {study_id}_seg-spine_msk.nii.gz   # Semantic (costal processes!)
├── {study_id}_ctd.json               # Centroids (ALL structures)
└── {study_id}_unc.nii.gz             # Uncertainty map
```

**TotalSpineSeg:**
```
results/totalspineseg/
├── {study_id}_sagittal_vertebrae.nii.gz
└── {study_id}_axial_vertebrae.nii.gz
```

**LSTV Detection:**
```
results/lstv_detection/
├── lstv_results.json     # Per-study classifications
└── lstv_summary.json     # Aggregate statistics
```

## Visualization

```bash
python scripts/visualize_overlay.py \
  --nifti results/nifti/1020394063_sag_t2.nii.gz \
  --instance results/spineps/segmentations/1020394063_seg-vert_msk.nii.gz \
  --semantic results/spineps/segmentations/1020394063_seg-spine_msk.nii.gz \
  --output overlay_1020394063.png
```

Highlights:
- **Red**: Costal processes (critical for Type I detection)
- **Blue**: Vertebrae
- **Green**: Discs
- **Yellow**: Endplates

## Troubleshooting

### DICOM Conversion Fails
- Check series descriptions CSV is correct
- Verify dcm2niix is installed: `which dcm2niix`

### SPINEPS Fails
- Check GPU availability: `nvidia-smi`
- Verify Singularity container: `ls ~/singularity_cache/spineps-segmentation.sif`
- Check model cache: `ls models/spineps_cache`

### TotalSpineSeg Fails
- Verify environment: `conda activate totalsegmentator && which TotalSegmentator`
- Check GPU memory: SLURM may need more than 64GB

### No LSTV Detected
- Check SPINEPS costal process segmentation quality
- Verify TotalSpineSeg vertebra counts
- Review uncertainty maps for low-confidence regions

## Key Features

1. **No validation filter** - Processes all studies
2. **Parallel execution** - SPINEPS + TotalSpineSeg run simultaneously
3. **Shared NIfTI** - Convert once, segment twice
4. **Complete centroids** - All structures (vertebrae, discs, endplates, subregions)
5. **Uncertainty maps** - Confidence assessment for LSTV calls
6. **Resume support** - Progress tracking in all scripts

## Next Steps

1. **Validate LSTV detector** - Compare against radiologist annotations
2. **Tune thresholds** - TP height, fusion detection criteria
3. **Add T1 support** - Improve Type III fusion detection
4. **Generate reports** - PDF summaries with overlays

## Citation

If you use this pipeline, cite:
- SPINEPS: [paper/repo]
- TotalSegmentator: [paper/repo]
- Castellvi Classification: Castellvi et al. 1984

## Contact

For issues or questions, open a GitHub issue or contact [your email].
