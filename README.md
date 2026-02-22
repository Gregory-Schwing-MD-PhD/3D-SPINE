# 3D Spine

**Automated quantitative spine MRI analysis — morphometrics, pathology screening, and deep learning classification at scale.**

3D Spine is a research pipeline for comprehensive, fully automated analysis of lumbar spine MRI studies. It combines state-of-the-art open-source segmentation models (SPINEPS, TotalSpineSeg) with a custom morphometrics engine to produce structured, clinically grounded measurements across entire datasets — without manual intervention.

Built to run on HPC clusters via SLURM. All heavy computation runs inside Singularity containers. Results ship as structured JSON, CSV, and self-contained HTML reports.

---

## What It Does

### Pathology Screening

Each study is automatically screened for the following conditions:

| Domain | Measurements | Classification |
|---|---|---|
| **LSTV** | Transverse process height, TP–sacrum distance, axial fusion confirmation | Castellvi Type I–IV (bilateral) |
| **Spinal Canal Stenosis** | AP diameter, dural sac cross-sectional area (DSCA) | Normal / Relative / Absolute (Schizas-equivalent) |
| **Cord Compression** | MSCC along full cord length, AP cord diameter, CSA | Normal / Mild / Moderate / Severe |
| **Disc Degeneration** | Disc Height Index (Farfan method), endplate-to-endplate distance | Pfirrmann-equivalent grading |
| **Spondylolisthesis** | Sagittal translation per level | Positive ≥ 3 mm |
| **Baastrup Disease** | Inter-spinous process gap | Contact / Risk / Normal |
| **Ligamentum Flavum** | LFT proxy (arcus → canal), ligamentum flavum area | Normal / Hypertrophy / Severe |
| **Facet Tropism** | Left/right facet angles, asymmetry | Ko Grade 0–2 |
| **Foraminal Stenosis** | Neural foraminal volume (bilateral, per level) | Lee Grade equivalent |
| **Vertebral Fracture** | Anterior/mid/posterior body heights, wedge ratio | Genant Grade 0–3 |

All measurements are computed at **L1–L2 through L5–S1** and reported both per-study and as dataset-level summary statistics (mean ± SD, min/median/max, prevalence frequencies).

### Uncertainty Quantification

An epistemic uncertainty model (Ian-Pan PointNet) assigns per-study L5-S1 junction confidence scores, enabling quality stratification and flagging studies that warrant closer review.

---

## Architecture

```
Input: DICOM studies (Sagittal T2, Axial T2)
│
├── Step 0: Ian-Pan Epistemic Uncertainty Inference    [GPU]
│   └── Outputs: l5_s1_confidence scores, uncertainty CSV
│
├── Step 1: DICOM → NIfTI Conversion                  [CPU, parallel with Step 0]
│   └── Outputs: sub-{id}_acq-sag_T2w.nii.gz
│                sub-{id}_acq-ax_T2w.nii.gz
│
├── Step 2: SPINEPS Segmentation                       [GPU, after Step 1]
│   └── Outputs: Instance mask, semantic mask, centroids,
│                sub-region mask, uncertainty map
│
├── Step 3: TotalSpineSeg Segmentation                 [GPU, after Step 1, parallel with Step 2]
│   └── Outputs: Labeled vertebrae, spinal cord, canal,
│                disc level markers, uncertainty
│
├── Step 4: Morphometrics Engine                       [CPU, after Steps 2 & 3]
│   └── Outputs: morphometrics_all.json / .csv
│                Per-study HTML clinical reports
│
└── Step 5: Dataset Report                             [CPU, after Step 4]
    └── Outputs: dataset_morphometrics_report.html
                 lstv_report.html
```

Steps 0 and 1 run simultaneously on submission. Steps 2 and 3 run in parallel after NIfTI conversion. The pipeline is fully resumable — each step tracks progress and skips completed studies.

---

## Quick Start

### Full Dataset (recommended)

```bash
# From project root — submits all steps as a SLURM dependency chain
sbatch slurm_scripts/00_run_full_dataset.sh
```

This single command submits six jobs in the correct dependency order and monitors them to completion.

### Individual Steps

```bash
# Step 0: Uncertainty inference
sbatch slurm_scripts/00_ian_pan_inference.sh

# Step 1: DICOM → NIfTI
sbatch slurm_scripts/01_dicom_to_nifti.sh

# Step 2: SPINEPS (all valid studies)
sbatch slurm_scripts/02b_spineps_selective.sh   # edit TOP_N or use ALL=true

# Step 3: TotalSpineSeg (all valid studies)
sbatch slurm_scripts/03b_totalspineseg_selective.sh

# Step 4: Morphometrics
ALL=true sbatch slurm_scripts/05_morphometrics.sh

# Step 5: Reports
sbatch slurm_scripts/06_html_report.sh
```

### Single Study (development / debugging)

```bash
# Morphometrics for one study
STUDY_ID=1020394063 sbatch slurm_scripts/05_morphometrics.sh

# Generate report from existing morphometrics
python scripts/06_html_report.py \
    --morphometrics_json results/morphometrics/morphometrics_all.json \
    --output_html        results/dataset_morphometrics_report.html \
    --morpho_only
```

---

## Output Files

```
results/
├── epistemic_uncertainty/
│   └── lstv_uncertainty_metrics.csv        # Per-study L5-S1 confidence scores
│
├── nifti/
│   └── {study_id}/{series_id}/             # Converted NIfTI files
│
├── spineps/
│   └── segmentations/{study_id}/
│       ├── {id}_seg-vert_msk.nii.gz        # Instance mask (vertebrae, discs, endplates)
│       ├── {id}_seg-spine_msk.nii.gz       # Semantic mask (incl. costal processes)
│       ├── {id}_seg-subreg_msk.nii.gz      # Sub-region mask
│       ├── {id}_ctd.json                   # Centroids (all structures)
│       └── {id}_unc.nii.gz                 # Segmentation uncertainty map
│
├── totalspineseg/{study_id}/
│   ├── sagittal/
│   │   ├── {id}_sagittal_labeled.nii.gz    # Labeled vertebrae + discs
│   │   ├── {id}_sagittal_cord.nii.gz       # Spinal cord
│   │   ├── {id}_sagittal_canal.nii.gz      # Spinal canal
│   │   └── {id}_sagittal_unc.nii.gz        # Uncertainty
│   └── axial/
│       └── {id}_axial_labeled.nii.gz
│
├── morphometrics/
│   ├── morphometrics_all.json              # Full nested results (feed into 3D viewer)
│   ├── morphometrics_all.csv               # Flat table for R / pandas / Excel
│   ├── morphometrics_summary.json          # Dataset-level statistics
│   └── reports/{study_id}_report.html      # Per-study clinical HTML report
│
├── lstv_detection/
│   ├── lstv_results.json                   # Per-study Castellvi classifications
│   └── lstv_summary.json                   # Aggregate LSTV statistics
│
├── dataset_morphometrics_report.html        # ← Dataset summary report
└── lstv_report.html                         # ← LSTV classification report
```

---

## Reports

### Dataset Summary Report (`dataset_morphometrics_report.html`)

A self-contained interactive HTML report covering the entire dataset:

- **Pathology prevalence cards** — counts and percentages for each condition
- **Per-level bar charts** — spondylolisthesis and disc degeneration by spinal level
- **Descriptive statistics tables** — mean ± SD, min, median, max for every measurement
- **Canal shape distribution** — trefoil / round / triangular / oval frequencies
- **LSTV tab** — Castellvi class distribution, morphometric comparison, representative overlays

### Per-Study Clinical Report (`reports/{id}_report.html`)

Individual study reports include:
- Canal stenosis classification with AP diameter and DSCA
- Full-length spinal cord compression profile (MSCC per slice, mini bar chart)
- Disc Height Index table with Pfirrmann-equivalent grading
- Vertebral height ratios and Genant fracture grading
- Spondylolisthesis, Baastrup, LFT, facet tropism, foraminal volume
- Clinical threshold reference card
- LSTV / Castellvi annotation (if detection results are available)

### LSTV Classification Report (`lstv_report.html`)

- Castellvi type distribution with count and prevalence
- Morphometric comparison table across classes
- Representative overlay images per class
- Full LSTV results table with Phase 1 and Phase 2 features

---

## Morphometrics Engine

`scripts/morphometrics_engine.py` implements a fully documented, threshold-referenced measurement library. All thresholds are sourced from peer-reviewed literature:

| Measurement | Reference |
|---|---|
| Canal AP stenosis grading | Schizas et al. (2010) |
| DSCA stenosis grading | Haig et al. (2006) |
| MSCC cord compression | Fehlings et al. (2010) |
| DHI (Farfan method) | Farfan et al. (1972) |
| Vertebral fracture (Genant) | Genant et al. (1993) |
| Spondylolisthesis | Meyerding classification |
| Baastrup disease | Baastrup (1933), Sonne-Holm et al. (2007) |
| Facet tropism (Ko grade) | Ko et al. (2014) |
| Foraminal stenosis (Lee grade) | Lee et al. (2010) |
| Ligamentum flavum hypertrophy | Yoshida et al. (2013) |
| Castellvi LSTV classification | Castellvi et al. (1984) |

---

## Infrastructure

### Containers

| Step | Image | Registry |
|---|---|---|
| DICOM → NIfTI, SPINEPS | `go2432/spineps-segmentation` | Docker Hub |
| Morphometrics, Reports | `go2432/spineps-preprocessing` | Docker Hub |
| TotalSpineSeg | `go2432/totalspineseg` | Docker Hub |
| Ian-Pan Inference | `go2432/lstv-uncertainty` | Docker Hub |

All containers are pulled automatically on first run and cached in `~/singularity_cache/`.

### Requirements

- SLURM cluster with GPU nodes (V100 or better recommended for TotalSpineSeg)
- Singularity ≥ 3.8
- `models/valid_id.npy` — study ID allowlist
- `models/point_net_checkpoint.pth` — Ian-Pan uncertainty model weights
- `data/raw/train_images/` — DICOM input
- `data/raw/train_series_descriptions.csv` — series metadata

---

## Roadmap

The following integrations are planned or under active development:

### Near-term

- **Ian-Pan axial stenosis classifier** — extend the existing epistemic uncertainty model to produce full central canal stenosis severity grades from axial T2 series, directly comparable to radiologist grading (Schizas A–D)
- **3D interactive viewer** — Plotly-based whole-spine 3D visualization with morphometric annotations overlaid on the segmentation meshes
- **Pfirrmann grading (direct)** — CNN-based disc degeneration grading from sagittal T2 intensity, replacing the DHI proxy

### Medium-term

- **VertXNet / SpineMetNet integration** — automated vertebral body compression fracture detection and OVCF grading from sagittal T1/T2
- **Disc herniation classification** — protrusion / extrusion / sequestration detection using axial T2, following recent YOLOv8-based approaches (Sustersic et al. 2022, extended)
- **Scoliosis / Cobb angle measurement** — automated coronal Cobb angle measurement from whole-spine sagittal MRI, following van der Graaf et al. (2024)
- **Bone marrow signal analysis** — fatty infiltration and edema scoring using Dixon sequences, leveraging nnU-Net bone marrow segmentation

### Longer-term

- **Cervical spine extension** — extend morphometrics and stenosis grading to C2–C7 using TotalSpineSeg cervical outputs and the CSpineSeg benchmark (Zhou et al. 2025)
- **Multi-modal fusion** — combine T1, T2, STIR, and Dixon series for more robust Modic change detection and marrow pathology characterization
- **Report generation → structured radiology output** — map morphometric findings to structured reporting templates (RADS-style) for direct integration with radiology workflows
- **Longitudinal tracking** — delta morphometrics between serial studies for disease progression monitoring

---

## Troubleshooting

**DICOM conversion fails**
Check that `train_series_descriptions.csv` contains the expected `series_description` values. Sagittal T2 series are matched against patterns: `Sagittal T2/STIR`, `SAG T2`, `Sag T2`. Add variants to `SAGITTAL_T2_PATTERNS` if your scanner uses different descriptions.

**SPINEPS produces no instance mask**
The `derivatives_seg/` directory is expected adjacent to the input NIfTI. If it is missing, SPINEPS likely failed silently — check `logs/spineps_all_*.err` and verify GPU availability with `nvidia-smi`.

**TotalSpineSeg OOM**
Requires a V100 (32 GB) or better. The SLURM script requests `--constraint=v100`. If your cluster uses different node labels, update the constraint in `slurm_scripts/03b_totalspineseg_selective.sh`.

**Morphometrics returns all N/A**
This usually means masks were not found. Check that `results/spineps/segmentations/{study_id}/` exists and contains `_seg-vert_msk.nii.gz`. TotalSpineSeg outputs are optional — morphometrics degrades gracefully when only SPINEPS masks are available.

**Resume a failed run**
All steps write a `progress_selective.json` file tracking success/failure per study. Simply resubmit the same SLURM script — completed studies are skipped automatically.

---

## Citation

If you use this pipeline in your research, please cite the underlying tools:

```
SPINEPS: Möller et al. (2025). SPINEPS — automatic whole spine segmentation of
  T2-weighted MR images. European Radiology. doi:10.1007/s00330-024-11155-y

TotalSpineSeg: Warszawer et al. (2025). TotalSpineSeg: Robust Spine Segmentation
  with Landmark-Based Labeling in MRI.

Castellvi Classification: Castellvi AE et al. (1984). Intransverse process
  impingement of the superior gluteal nerve. Spine 9(1):31–35.
```

---

## License

Research use. See `LICENSE` for details.

## Contact

Open an issue on GitHub or email go2432@wayne.edu.
