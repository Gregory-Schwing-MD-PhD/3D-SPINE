# lstv-detector

**Automated MRI-based detection and classification of Lumbosacral Transitional Vertebrae (LSTV) using deep learning segmentation, radiologically-grounded morphometrics, vertebral angle analysis, and interactive 3D visualisation.**

> Target audience: spine neurosurgeons, musculoskeletal radiologists, and deep learning researchers working with spinal MRI.

---

## Clinical Background

### What is an LSTV?

A lumbosacral transitional vertebra (LSTV) is a congenital anomaly in which the last mobile lumbar segment displays morphology intermediate between a lumbar and a sacral vertebra. The reported prevalence varies from 4% to 35.9% depending on modality, counting methodology, and population (Nardo et al., *Radiology* 2012; Hughes & Saifuddin, *Skeletal Radiol* 2006).

LSTVs have significant clinical importance in spine surgery:

- **Wrong-level surgery risk**: The transitional vertebra is the most common cause of numbering errors on MRI. Up to 30% of studies fail to identify the correct lumbosacral level without dedicated whole-spine imaging (Carrino et al., *Radiology* 2011).
- **Disc herniation pattern alteration**: LSTVs shift the biomechanical fulcrum, causing accelerated degeneration at the mobile level above the TV and relative protection below (Bertolotti's syndrome).
- **Bertolotti's syndrome**: Low back pain attributed to a Castellvi TP articulating with the ilium or sacrum, occurring in 4–8% of patients presenting with low back pain (Alonzo et al., *J Neurosurg Spine* 2020).
- **Altered surgical anatomy**: Pedicle screw trajectories, neuromonitoring electrode placement, and intraoperative fluoroscopic counting must all account for LSTV.

### The Two LSTV Phenotypes

These phenotypes are radiologically distinct and clinically independent. A patient may have **both** a Castellvi classification (TP morphology) and a phenotype (overall transition pattern) simultaneously.

#### Sacralization
L5 (or occasionally L4) progressively incorporates into the sacrum. The L5 segment loses lumbar characteristics — the disc below becomes reduced or absent (the most reliable radiologic sign; Seyfert, *Neuroradiology* 1997), the vertebral body becomes squarer (H/AP ratio decreases toward sacral range), and the transverse processes may articulate or fuse with the sacrum ala (= Castellvi classification).

#### Lumbarization
S1 acquires lumbar characteristics. The S1 segment separates from the remainder of the sacrum, developing a mobile intervertebral disc below it (L6-S1), and adopts lumbar vertebral body proportions (H/AP ratio ≥ 0.68). This creates a 6-lumbar-segment spine. Castellvi TP enlargement may co-occur on L6 (i.e., L6 TP may be enlarged and contact the remaining sacrum), but this is classified separately.

---

## Castellvi Classification System

The Castellvi system (Castellvi et al., *Spine* 1984;9(1):31–35) classifies TP morphology at the lumbosacral junction. It is the most widely adopted radiologic classification for LSTV and remains the clinical standard.

| Type | Definition | Unilateral | Bilateral |
|------|-----------|-----------|----------|
| **I** | Dysplastic TP ≥ 19 mm craniocaudal height, no sacral contact | Ia | Ib |
| **II** | Pseudo-articulation (diarthrodial joint) between enlarged TP and sacrum | IIa | IIb |
| **III** | Complete bony fusion of TP with sacrum ala | IIIa | IIIb |
| **IV** | Mixed: Type II one side, Type III the other | — | — |

**Key threshold**: TP height ≥ 19 mm (craniocaudal extent), originally defined on plain film. On MRI, this maps to the full SI extent of the SPINEPS costal process mask.

**MRI adaptation** (Konin & Walz, *Semin Musculoskelet Radiol* 2010; Nidecker et al., *Eur Radiol* 2018):

- **Type II on MRI**: Heterogeneous or dark T2w signal at the TP–sacrum junction — fibrocartilaginous pseudo-joint or synovial cleft. Intermediate signal = fibrocartilage; dark = synovial fluid.
- **Type III on MRI**: Homogeneous high T2w signal continuous with sacral bone marrow — osseous bridge with marrow fat.
- **CT remains the gold standard** for Type III confirmation (bony cortical continuity). MRI Phase 2 classification in this pipeline should be treated as provisional.

---

## Vertebral Angle Analysis (Seilanian Toosi 2025) — NEW in v5.x

This pipeline implements the full five-angle measurement system introduced by Seilanian Toosi et al. (*Arch Bone Jt Surg.* 2025;13(5):271–280), computed automatically from 3D segmentation masks on the reconstructed midsagittal slice.

### Angle Definitions

All angles are measured on the midsagittal view, exactly as illustrated in the paper figures. The pipeline reconstructs an optimal midsagittal 2D slice from the 3D segmentation via maximum-intensity projection of a 10mm midline slab.

| Angle | Definition | Normal Median | LSTV Direction |
|-------|-----------|--------------|----------------|
| **A** | Line parallel to the superior sacral surface vs. vertical (scan axis) | ~37° | ↑ in LSTV (median 41.5° vs 37°, P=0.038) |
| **B** | Line parallel to L3 superior endplate vs. sacral superior surface | — | Non-significant |
| **C** | Largest angle formed by posterior body lines of TV±1 and S1±1 | ~37° | ↓ in LSTV (P<0.001) |
| **D** | Sacral superior surface vs. TV (most caudal lumbar) superior surface | ~26° | ↓ in LSTV (P=0.028) |
| **D1** | TV superior surface vs. TV-1 (supra-adjacent) superior surface | ~14° | ↓ in Type 2 (P<0.001) |
| **δ (delta)** | D − D1 | ~15° | ↓ in LSTV (P=0.003); **≤8.5° → Type 2 LSTV** |

### Diagnostic Thresholds (from ROC analysis, n=220)

| Criterion | Threshold | Sensitivity | Specificity | NPV | Target |
|-----------|-----------|-------------|-------------|-----|--------|
| **δ ≤ 8.5°** | 8.5° | **92.3%** | **87.9%** | **99.5%** | Type 2 LSTV |
| C ≤ 35.5° | 35.5° | 72.2% | 57.6% | 91.4% | Any LSTV |
| δ ≤ 14.5° | 14.5° | 66.7% | 52.2% | 88.9% | Any LSTV |

The **δ-angle (delta)** is the most powerful single predictor. A value ≤ 8.5° indicates that at least two levels contribute to the lumbosacral curvature rather than a single, isolated acute angle at L5-S1 — the geometric signature of Type 2 LSTV (pseudo-articular fusion distributing the angulation across multiple levels).

### Clinical Interpretation of the Delta Angle

- **High delta (>15°)**: Concentrated acute angulation at a single L5-S1 junction — normal anatomy, or Type 3 sacralization (where L5 is so incorporated that the spine *appears* normal)
- **Low delta (≤8.5°)**: Distributed angulation across two levels, implying that the TV participates in both the L4-TV and TV-S1 transitions — Type 2 LSTV with high diagnostic certainty
- **Important caveat**: Type 3B sacralization with complete fusion may produce normal-appearing angles (the fused L5 is miscounted as S1), which is why the pipeline combines angle analysis with Castellvi TP detection and disc metrics

### Multivariate Independent Predictors (Seilanian Toosi 2025, Table 4)

From logistic regression on 220 subjects:

| Variable | Odds Ratio | 95% CI | P |
|----------|-----------|--------|---|
| Increased A-angle | 1.141 | 1.019–1.279 | 0.023 |
| Decreased D-angle | 0.719 | 0.530–0.976 | 0.034 |
| L4-L5 disc dehydration | 0.157 | 0.057–0.430 | <0.001 |
| Non-dehydrated L5-S1 disc | 19.869 | 5.743–68.741 | <0.001 |

The disc dehydration pattern (L4-L5 dehydrated + L5-S1 preserved) has an OR of 19.87 for LSTV and is incorporated as a dedicated criterion in the Bayesian model.

### Implementation

Angles are computed in `lstv_angles.py` via midsagittal 2D OLS line fitting with one-pass outlier rejection. Sagittal slices are extracted as max-projections of a midline slab (±10mm ML). The Bayesian model in `lstv_engine.py` applies log-odds updates from angle findings:

```python
# LR values from Seilanian Toosi 2025
delta ≤ 8.5°    →  LR+ ≈ 7.7  (log update: +2.04 nats)
C ≤ 35.5°       →  LR+ ≈ 4.5  (log update: +1.50 nats)
A > 41°         →  LR+ ≈ 2.0  (log update: +0.69 nats)
```

---

## Radiologic Criteria Implemented

### Primary Criteria (each independently sufficient to flag LSTV)

| Criterion | Threshold | Reference |
|-----------|-----------|-----------|
| **Castellvi TP height** | ≥ 19 mm craniocaudal | Castellvi et al. 1984 |
| **TP–sacrum contact** | ≤ 2 mm 3D distance | Castellvi et al. 1984 |
| **Disc below TV absent / severely reduced** | DHI < 50% | Seyfert 1997; Farfan et al. 1972 |
| **6-lumbar count** (L6 present) | VERIDAH label 25 detected | Hughes & Saifuddin 2006 |
| **4-lumbar count** (confirmed both sources) | TSS + VERIDAH = 4 | Konin & Walz 2010 |
| **δ-angle ≤ 8.5°** | Seilanian Toosi threshold | Seilanian Toosi et al. 2025 |

### Supporting Criteria (increase phenotype confidence)

| Criterion | Threshold | Reference |
|-----------|-----------|-----------|
| **TV body H/AP ratio — sacral-like** | < 0.52 | Nardo et al. 2012; Panjabi et al. 1992 |
| **TV body H/AP ratio — transitional** | 0.52–0.68 | Nardo et al. 2012 |
| **TV body H/AP ratio — lumbar-like** | > 0.68 | Nardo et al. 2012 |
| **TV/L4 normalised H:AP ratio** | < 0.80 → squarer than L4 | Nardo et al. 2012 |
| **Disc below TV moderately reduced** | DHI 50–70% | Farfan et al. 1972 |
| **Disc below TV preserved** | DHI ≥ 80% | Konin & Walz 2010 |
| **Disc above TV preserved** | DHI ≥ 80% | Localises pathology to L5-S1 |
| **C-angle ≤ 35.5°** | Seilanian Toosi threshold | Seilanian Toosi et al. 2025 |
| **A-angle > 41°** | Seilanian Toosi threshold | Seilanian Toosi et al. 2025 |
| **Disc pattern: L4-L5 dehydrated + L5-S1 preserved** | OR 19.87 | Seilanian Toosi et al. 2025 |

### Disc Height Index (DHI) — Farfan Method

DHI = (disc height / mean of adjacent vertebral body heights) × 100

Normal lumbar DHI: 80–100 %. The L5-S1 disc (disc below the TV in a standard 5-lumbar spine) is the **most reliable single radiologic marker** of sacralization; its reduction or absence indicates L5 is transitioning into the sacrum (Seyfert 1997; Quinlan et al. 1984).

### TV Body Morphology — Nardo Classification

Nardo et al. (*Radiology* 2012) established H/AP ratio thresholds for the TV body on sagittal MRI, validated against CT:

- Normal lumbar reference (Panjabi et al. 1992): L3=0.82±0.09, L4=0.78±0.08, L5=0.72±0.10
- TV 0.55–0.70 → intermediate / transitional morphology
- TV < 0.52 → sacral-like morphology (high specificity for sacralization)

The pipeline normalises TV H/AP against the ipsilateral L4 body (TV/L4 ratio) to account for inter-individual variation.

---

## Architecture

```
Input: DICOM studies (Sagittal T2w ± Axial T2w)
│
├── Step 01: DICOM → NIfTI conversion
│
├── Step 02b: SPINEPS Segmentation                        [GPU]
│   ├── seg-spine_msk.nii.gz  (subregion semantic labels)
│   │     43=Costal_Process_Left  ← TP source
│   │     44=Costal_Process_Right ← TP source
│   │     26=Sacrum  41=Arcus  42=Spinous  49=Corpus
│   │     60=Cord  61=Canal
│   └── seg-vert_msk.nii.gz   (VERIDAH per-vertebra instance labels)
│         20=L1  21=L2  22=L3  23=L4  24=L5  25=L6  26=Sacrum
│         100+X=IVD below X   200+X=Endplate of X
│
├── Step 03b: TotalSpineSeg Segmentation                  [GPU]
│   └── sagittal_labeled.nii.gz
│         1=Cord  2=Canal
│         41=L1  42=L2  43=L3  44=L4  45=L5  50=Sacrum
│         91=T12-L1  92=L1-L2  93=L2-L3  94=L3-L4  95=L4-L5  100=L5-S1
│         ⚠ TSS labels 43/44 = L3/L4 vertebral bodies ≠ SPINEPS TPs
│
├── Step 03c: Registration (SPINEPS → axial T2w space)
│
├── Step 04: LSTV Detection + Morphometrics + Angle Analysis  [CPU]
│   ├── Phase 1: Sagittal geometric Castellvi (TP height, TP–sacrum distance)
│   ├── Phase 2: Axial T2w signal classification (Type II vs III)
│   ├── Step 8.5: Vertebral angle analysis (lstv_angles.py)
│   │   └── A, B, C, D, D1, δ angles — Seilanian Toosi 2025
│   ├── Phenotype: Radiologically-grounded multi-criteria classifier
│   │   (independent of Castellvi — both co-reported)
│   └── Outputs: lstv_results.json  lstv_summary.json
│
└── Step 06: 3D Visualisation                             [CPU]
    └── results/lstv_3d/{study_id}_lstv_3d.html
        └── Paper-accurate dorsal angle overlays (v5.2+)
```

### Critical label disambiguation

| Label value | In SPINEPS seg-spine_msk | In TotalSpineSeg sagittal |
|------------|--------------------------|--------------------------|
| 43 | **Costal_Process_Left** ← TP source | **L3 vertebral body** — NOT a TP |
| 44 | **Costal_Process_Right** ← TP source | **L4 vertebral body** — NOT a TP |
| 45 | Sup_Articular_Left | **L5 vertebral body** |
| 50 | Not used | **Sacrum** ← preferred sacrum source |

The pipeline always sources TPs from SPINEPS `seg-spine_msk` (labels 43/44) and sacrum from TSS label 50. Mixing these would produce grossly incorrect Castellvi classifications.

---

## Version History

### v5.3 (current)
- **BUG FIX**: All vertebral angle key names corrected throughout `04_detect_lstv.py`. v5.2 read wrong keys from the `vertebral_angles` dict (e.g. `delta_angle`, `delta_le8p5`, `c_le35p5`) — these do not exist in `VertebralAngles`. Correct names: `delta_angle_deg`, `delta_positive`, `c_positive`, `a_angle_elevated`, `disc_pattern_l4dehy_l5preserved`. This caused all angle-based LSTV detection and logging to silently produce N/A values.
- **BUG FIX**: TP concordance result now correctly passed to `lstv_engine` via `tp_concordance_precomputed=`, preventing the engine from re-running its broken cross-NIfTI-space comparison.

### v5.2
- Paper-accurate dorsal angle overlays in `06_visualize_3d.py` matching Seilanian Toosi 2025 Figures 1–4
- Fixed arc slerp calculation (correct spherical linear interpolation)
- Color conflict fix: TP-Left = `#00ccff` (cyan), TP-Right = `#ff6600` (orange)

### v5.1
- TP disc-boundary concordance validation (`tp_in_correct_zone`)
- Disc dehydration asymmetry pattern in Bayesian model (OR 19.87)

### v5.0
- Initial vertebral angle analysis integration (Seilanian Toosi 2025)
- All five angles (A, B, C, D, δ) computed from segmentation masks

---

## Classification Logic

### Step 1: Vertebral counting (lumbar)

TSS labels 41–45 provide L1–L5 counts. VERIDAH label 25 provides L6 detection (only VERIDAH can identify L6; TSS has no L6 label). Reconciliation rule:

- If VERIDAH label 25 (L6) present → consensus = TSS_count + 1 (lumbarization indicator)
- If TSS < 5 and VERIDAH corroborates → 4-lumbar count (sacralization indicator)
- If VERIDAH > TSS without L6 label → trust TSS (over-segmentation artifact)

### Step 2: Castellvi Phase 1 (sagittal geometric)

For each side (left/right), isolate the SPINEPS costal process (TP) label at the TV z-extent ± 3 voxels. Measure:
1. **TP craniocaudal height** = (z_max − z_min + 1) × voxel_size_z mm (global mask extent)
2. **TP–sacrum 3D minimum distance** using EDT (scipy.ndimage.distance_transform_edt)
3. **TP zone validation** via `tp_in_correct_zone()` — confirms TP centroid lies between the inferior face of the L4-L5 disc (TSS 95) and the superior face of the L5-S1 disc (TSS 100) / sacrum

Classification:
- dist > 2 mm AND height ≥ 19 mm → **Type I**
- dist ≤ 2 mm → **CONTACT_PENDING_P2** → proceed to Phase 2

### Step 3: Castellvi Phase 2 (axial T2w signal)

Extract a 32×32 voxel patch centred at the midpoint between the closest TP and sacrum voxels in the registered axial space. Classify:
- patch_mean < 55% × global_p95 → dark/intermediate → **Type II** (fibrocartilage)
- CV < 0.12 (uniform bright) → **Type III** (osseous marrow bridge)
- Ambiguous → **Type II** (conservative fallback; CT recommended for definitive Type III)

### Step 4: Phenotype classification (independent of Castellvi)

A tiered multi-criteria classifier grounded in the literature above.

**Tier 1 — Count anomaly (highest specificity, immediate classification)**
- count = 6 → lumbarization (high confidence)
- count = 4 → sacralization (high confidence)

**Tier 2 — count = 5 (morphometric criteria required)**

Sacralization pathway requires ≥ 1 primary criterion:
- S1: Castellvi detected (any type)
- S2: Disc below TV DHI < 50% or absent (most reliable sign; Seyfert 1997)
- S3: 4-lumbar count (covered in Tier 1)
- S4: TV body sacral-like (H/AP < 0.52) + corroborating finding

Lumbarization pathway requires ≥ 1 primary criterion:
- L1: 6-lumbar count (covered in Tier 1)
- L2: disc below TV preserved (DHI ≥ 80%) indicating mobile L6-S1
- L3: TV body lumbar-like (H/AP ≥ 0.68) with count = 6

### Step 8.5: Vertebral angle analysis (Seilanian Toosi 2025)

Computed by `lstv_angles.py` via midsagittal OLS line fitting. Bayesian log-odds updates applied when angle criteria are met. Detection triggers:
- **δ ≤ 8.5°** → `lstv_detected = True` + reason logged with sens/spec/NPV
- **C ≤ 35.5° + corroborating finding** (disc pattern, elevated A, or decreased D) → `lstv_detected = True`
- **Disc pattern alone** (L4-L5 dehydrated + L5-S1 preserved, without Castellvi) → `lstv_detected = True`

### What triggers `lstv_detected = True`

```
lstv_detected = True  iff ANY of:
  • Castellvi Type I-IV on either side
  • lumbar_count ≠ 5  (confirmed by reconciled TSS + VERIDAH)
  • phenotype ∈ {sacralization, lumbarization}  (primary criterion confirmed)
  • δ-angle ≤ 8.5°
  • C-angle ≤ 35.5° + corroborating angle/disc finding
  • Disc asymmetry pattern (L4-L5 dehydrated + L5-S1 preserved)
```

---

## File Structure

```
lstv-detector/
├── scripts/
│   ├── lstv_angles.py         ← vertebral angle computation (Seilanian Toosi 2025)
│   ├── lstv_engine.py         ← all morphometric calculations (importable)
│   ├── 04_detect_lstv.py      ← Castellvi classifier + phenotype + angle engine
│   └── 06_visualize_3d.py     ← interactive 3D HTML renderer with angle overlays
│
├── slurm_scripts/
│   ├── 01_dicom_to_nifti.sh
│   ├── 02b_spineps_selective.sh
│   ├── 03b_totalspineseg_selective.sh
│   ├── 03c_register.sh
│   ├── 04_lstv_detection.sh
│   └── 06_visualize_3d.sh
│
└── results/
    ├── spineps/segmentations/{study_id}/
    │   ├── {id}_seg-spine_msk.nii.gz
    │   └── {id}_seg-vert_msk.nii.gz
    ├── totalspineseg/{study_id}/sagittal/
    │   └── {id}_sagittal_labeled.nii.gz
    ├── registered/{study_id}/
    │   └── {id}_spineps_reg.nii.gz
    ├── lstv_detection/
    │   ├── lstv_results.json      ← per-study full results
    │   └── lstv_summary.json      ← aggregate statistics + angle stats
    └── lstv_3d/
        └── {study_id}_lstv_3d.html
```

---

## Output Schema

### `lstv_results.json` — per-study

```json
{
  "study_id": "1307819508",
  "lstv_detected": true,
  "lstv_reason": [
    "Angle: delta=6.2 <= 8.5 -- predicts Castellvi Type 2 LSTV (sens 92.3%, spec 87.9%, NPV 99.5%; Seilanian Toosi 2025)",
    "Phenotype: SACRALIZATION (high confidence) -- criteria: S1:Castellvi-Type-IIb; S2:disc-below-DHI-38pct"
  ],
  "castellvi_type": "Type IIb",
  "lstv_morphometrics": {
    "lumbar_count_consensus": 5,
    "tv_name": "L5",
    "disc_below": { "level": "L5-S1", "dhi_pct": 38.2, "grade": "Severely reduced" },
    "lstv_phenotype": "sacralization",
    "phenotype_confidence": "high",
    "vertebral_angles": {
      "angles_available": true,
      "a_angle_deg": 43.1,
      "b_angle_deg": 47.3,
      "c_angle_deg": 28.4,
      "d_angle_deg": 20.1,
      "d1_angle_deg": 13.9,
      "delta_angle_deg": 6.2,
      "delta_positive": true,
      "c_positive": true,
      "a_angle_elevated": true,
      "angle_flags": { "A": "OK", "C": "OK", "D": "OK", "delta": "OK" }
    },
    "probabilities": {
      "p_sacralization": 0.9312,
      "p_lumbarization": 0.0041,
      "p_normal": 0.0647
    }
  },
  "pathology_score": 14.5
}
```

### `lstv_summary.json` — batch statistics

```json
{
  "total": 283,
  "lstv_detected": 61,
  "lstv_rate": 0.2156,
  "angle_stats": {
    "delta_le8p5_count": 22,
    "c_le35p5_count": 31,
    "disc_pattern_count": 18,
    "angle_only_lstv": 5,
    "reference": "Seilanian Toosi F et al. Arch Bone Jt Surg. 2025;13(5):271-280"
  }
}
```

---

## Quick Start

### Full pipeline (SLURM dependency chain)

```bash
J1=$(sbatch --parsable slurm_scripts/01_dicom_to_nifti.sh)
J2=$(sbatch --parsable --dependency=afterok:$J1 slurm_scripts/02b_spineps_selective.sh)
J3=$(sbatch --parsable --dependency=afterok:$J1 slurm_scripts/03b_totalspineseg_selective.sh)
J4=$(sbatch --parsable --dependency=afterok:$J2:$J3 slurm_scripts/03c_register.sh)
J5=$(sbatch --parsable --dependency=afterok:$J4 slurm_scripts/04_lstv_detection.sh)
sbatch --dependency=afterok:$J5 slurm_scripts/06_visualize_3d.sh
```

### Single study (development)

```bash
python scripts/04_detect_lstv.py \
    --study_id 1307819508 \
    --spineps_dir    results/spineps \
    --totalspine_dir results/totalspineseg \
    --registered_dir results/registered \
    --nifti_dir      results/nifti \
    --output_dir     results/lstv_detection

python scripts/06_visualize_3d.py \
    --study-id 1307819508 \
    --spineps-dir    results/spineps \
    --totalspine-dir results/totalspineseg \
    --output-dir     results/lstv_3d \
    --lstv-json      results/lstv_detection/lstv_results.json
```

---

## Python API

```python
from lstv_engine import (
    load_lstv_masks, analyze_lstv, compute_lstv_pathology_score,
)
from lstv_angles import compute_angles_from_spineps

# Load and resample masks to 1mm isotropic
masks = load_lstv_masks("1307819508", spineps_dir, totalspine_dir)

# Run full morphometric + angle analysis
morpho = analyze_lstv(masks, castellvi_result=detect_result)

print(f"TV:         {morpho.tv_name}")
print(f"Phenotype:  {morpho.lstv_phenotype} ({morpho.phenotype_confidence})")
print(f"P(sac):     {morpho.probabilities.p_sacralization:.1%}")

va = morpho.vertebral_angles
if va and va.angles_available:
    print(f"δ-angle:    {va.delta_angle_deg:.1f}°  ({'⚠ TYPE 2' if va.delta_positive else 'normal'})")
    print(f"C-angle:    {va.c_angle_deg:.1f}°  ({'⚠ LSTV' if va.c_positive else 'normal'})")
    print(f"A-angle:    {va.a_angle_deg:.1f}°  ({'↑ elevated' if va.a_angle_elevated else 'normal'})")

# Pathology burden score for ranking
score = compute_lstv_pathology_score(detect_result, morpho.to_dict())
```

---

## 3D Visualiser

Each HTML output (`{study_id}_lstv_3d.html`) contains:

- **Colour-coded phenotype banner**: SACRALIZATION (red) / LUMBARIZATION (orange) / TRANSITIONAL (yellow) / NORMAL (green)
- **Castellvi badge**: displayed alongside phenotype — both shown simultaneously if applicable
- **TP concordance badge**: L/R status showing whether TPs are validated within the L4-L5 / L5-S1 disc bounds
- **TP height rulers**: craniocaudal extent overlaid on 3D TP mesh (≥19mm flag)
- **TP–sacrum gap rulers**: dashed line to nearest sacrum point; contact (≤2mm) shown in red
- **Paper-accurate dorsal angle overlays** (v5.2+): all six angles rendered at distinct dorsal depths matching Seilanian Toosi 2025 Figures 1–4
  - δ (white/red, furthest dorsal, thickest lines) — turns red and appends "⚠ Type2 LSTV" if ≤8.5°
  - D (orange) / D1 (cyan) — intermediate dorsal depths
  - C (magenta) — posterior body lines with yellow vertical reference
  - A (yellow) / B (red) — near-spine placement
  - Each overlay: tilted endplate line, dashed vertical connector, slerp arc, bold label
- **Angle panel sidebar**: all six angle values with colour-coded threshold badges (red = threshold exceeded, green = normal)
- **Probability panel**: P(sacralization) / P(lumbarization) / P(normal) with bar visualizations
- **Surgical risk panel**: wrong-level risk category, Bertolotti probability, surgical flags

---

## Pathology Score

Used for study ranking only — not a diagnosis. Higher = more interesting LSTV case.

| Feature | Points |
|---------|--------|
| Castellvi Type IV | 5 |
| Castellvi Type III | 4 |
| Castellvi Type II | 3 |
| Castellvi Type I | 1 |
| Phenotype (sacralization/lumbarization), high confidence | +3 |
| Phenotype (sacralization/lumbarization), moderate | +2 |
| Phenotype transitional_indeterminate | +1 |
| Lumbar count anomaly (≠5) | +2 |
| Disc below TV DHI < 50% or absent | +2 |
| Disc below TV DHI 50–70% | +1 |
| TV body sacral-like (H/AP < 0.52) | +2 |
| TV body transitional (H/AP 0.52–0.68) | +1 |
| Rib anomaly (lumbar rib / thoracic count mismatch) | +1 |
| **δ-angle ≤ 8.5° (Type 2 predictor)** | **+3** |
| **δ-angle ≤ 15° (borderline)** | **+1.5** |
| **C-angle ≤ 35.5°** | **+1.5** |
| **A-angle > 41° (elevated sacral tilt)** | **+0.5** |

---

## Known Limitations

**Lumbarization count edge case**: TSS labels stop at L5 (label 45). When L6 is present, TSS will label L1–L5 of the 6-lumbar spine correctly, but the TSS count will read as 5 (normal). The L6 signal comes exclusively from VERIDAH label 25. If SPINEPS mis-labels L6 (e.g., as a second L5), the consensus count will be 5 and the L6 will be missed. The cross-validation warning (`L5 centroid dist > 20mm`) will flag such cases.

**Type III over-reporting**: Phase 2 MRI Type III classification is provisional. Homogeneous T2 signal at the TP junction may occur with periosteal bone marrow without true cortical bridging. CT confirmation is recommended before operative planning.

**Castellvi on 4-lumbar spines**: When count=4, the lowest mobile segment is typically L4. The VERIDAH TV search will identify L4 (label 23) as the TV, and Castellvi will be assessed on L4 TP. This is radiologically correct — the sacralizing segment's TP should be evaluated — but the label printed will be "L4."

**DHI at L6-S1**: TotalSpineSeg has no disc label for the L6-S1 level. DHI at L6-S1 uses VERIDAH IVD label 125 (100 + VD_L6=25) if present. If SPINEPS does not label the L6-S1 disc, DHI will be reported as undetected (not absent).

**Type 3B angle limitation**: Seilanian Toosi 2025 specifically notes that Type 3B sacralization with complete fusion can produce apparently normal angles because L5 is so incorporated that the MRI appears to have normal lumbar anatomy — the "L5" is simply miscounted as S1. The delta angle was documented to be ineffective in these cases. The pipeline's Castellvi + disc + count criteria remain the primary detection pathway for Type 3.

**Angle coordinate space**: All five angles are computed on a reconstructed midsagittal 2D slice from the 3D segmentation via maximum-intensity projection. Rotation of the patient relative to the scanner (e.g., scoliosis >10°) can bias the midline projection and should be interpreted with caution. The `_identify_axes()` function in `lstv_angles.py` automatically detects the CC/AP/ML axes, which partially mitigates this.

---

## Future Directions

### 1. Head-to-Head Comparison with Competing Angle Metrics

The Seilanian Toosi paper situates the δ- and C-angles against a field of competing quantitative approaches that the pipeline does not yet implement. A natural extension is a direct comparative validation study on the pipeline's output:

- **Chalian et al. 2012 A/B angles** (*World J Radiol*): The original sagittal angle proposal (sensitivity/specificity 80% as reported by Chalian). Seilanian Toosi found the A-angle to remain a significant independent predictor (OR 1.141) but insufficient alone. Direct pipeline comparison would test whether the A/B angles add marginal information beyond δ in a segmentation-derived framework.
- **Farshad Diff-VMVA** (*Bone Joint J* 2013): Draws four vertical midvertebral lines from the last fully-developed disc cranially, computes the difference between the two most caudal inter-line angles. Diff-VMVA ≤ 10° reportedly identifies Type 3/4 LSTV with 100% sensitivity and 89% specificity on symptomatic patients — the complementary regime to δ (which is strong for Type 2 but weak for Type 3). Implementing Diff-VMVA would allow the pipeline to cover both ends of the fusion spectrum.
- **AVA (Anterior-edge Vertebral Angle) — Zhou et al. 2022** (*Eur Radiol*): Quantitative PET/CT-derived metric with sensitivity 77.5% and specificity 88.3% at a cutoff of 73°. While CT-based, the anterior endplate angle is measurable on sagittal MRI and may be extractable from TotalSpineSeg endplate labels. Comparison against δ and C on the same cohort would establish which angle provides greatest diagnostic yield on MRI specifically.
- **RISE (Ratio of Inferior-to-Superior Endplate Length)**: Another PET/CT parameter introduced for LSTV enumeration; not yet validated on MRI segmentation data.

### 2. Axial MRI Nerve Root Morphology for Level Counting

A highly promising but unimplemented approach uses axial T2-weighted MRI at the sacral level to determine the presacral vertebral count directly from nerve root caliber, without requiring whole-spine imaging (described on Radiopaedia, sourced from clinical practice guidelines):

The method exploits the fact that the L5 nerve characteristically does not split proximally and has approximately twice the caliber of the L4 peroneal branch at this level. The pattern at the lateral sacrum determines the count:

- **4 lumbar segments** (completely sacralized L5, 23 presacral vertebrae): a bundle of several splitting nerves at the lateral sacrum represents the L4 nerve
- **5 lumbar segments** (normal anatomy, or partial sacralization/lumbarization, 24 presacral vertebrae): a thin nerve joining a thicker nerve at the lateral sacrum represents the peroneal branch of L4 and the L5 nerve
- **6 lumbar segments** (completely lumbarized S1, 25 presacral vertebrae): two nerves of similar caliber at the lateral sacrum represent the L5 and S1 nerves

**Implementation pathway**: The pipeline already registers SPINEPS into the axial T2w space (Step 03c). SPINEPS labels the spinal cord (label 60) and canal (label 61) but does not currently segment individual nerve roots. Integration options include:
  - Fine-tuning SPINEPS with a nerve-root segmentation head on axial slices at the L5-S1 level
  - Using a separate dedicated nerve root segmentation model (e.g., based on nnU-Net trained on sacral axial slices) and registering output into the pipeline's shared coordinate space
  - Semi-automated measurement: extracting the L5-S1 axial slice from the registered data and computing cross-sectional area of the two lateral nerve bundles; a caliber ratio > 1.8 would indicate the 5-lumbar pattern

This modality is orthogonal to all sagittal-slice methods and could serve as a strong independent confirmatory criterion — particularly for cases where the sagittal angle analysis is confounded by scoliosis or Type 3 fusion, or where the whole-spine localizer is unavailable.

### 3. Iliolumbar Ligament Landmark Integration

The iliolumbar ligament (ILL) typically arises from the transverse process of L5 and has been proposed as a landmark for vertebral numbering. Its utility is mixed: Farshad-Amacker et al. found it unreliable in LSTV cases (multiple origins), and the Seilanian Toosi cohort did not use it. However:
- In cases where SPINEPS labels the iliolumbar ligament or the ILL is visible on axial MRI, cross-referencing its origin with the VERIDAH TV identification could provide a soft Bayesian prior
- The ILL appears to reliably originate from the **last mobile** lumbar vertebra even in LSTV — making it a marker of TV identity rather than absolute count

### 4. Multi-Center Angle Validation

Seilanian Toosi 2025 explicitly calls for multi-center validation in larger populations. The pipeline's automated angle extraction provides a direct vehicle for this: running the pipeline on a held-out multi-center cohort (e.g., from the DRYAD spine imaging repositories or the SPIDER dataset) and comparing automatically computed angles against radiologist ground truth would establish inter-site reproducibility of the δ-angle threshold.

### 5. Clinical Outcome Correlation

The paper recommends correlating vertebral angle measurements with clinical findings (symptoms, pain scores, functional outcomes). The pipeline's surgical relevance module already outputs Bertolotti syndrome probability and wrong-level risk scores. Linking these to postoperative outcomes data from an institutional spine surgery registry would enable prospective validation of the surgical risk model.

### 6. Longitudinal Angle Tracking

The δ-angle has a mechanistic interpretation — it reflects the degree to which the TV participates in the lumbosacral curvature. Serial imaging in patients with Bertolotti syndrome could test whether progressive sacralization correlates with δ-angle decrease over time, providing a quantitative imaging biomarker of transition progression.

---

## Infrastructure

### Segmentation models

| Model | Container | Source |
|-------|-----------|--------|
| SPINEPS | `go2432/spineps-segmentation` | Möller et al. *Eur Radiol* 2025 |
| TotalSpineSeg | `go2432/totalspineseg` | Warszawer et al. 2025 |

### SLURM resource requirements

| Step | CPUs | Memory | GPU | Time |
|------|------|--------|-----|------|
| 02b SPINEPS | 4 | 32 GB | V100 32 GB | 8h |
| 03b TotalSpineSeg | 4 | 32 GB | V100 32 GB | 8h |
| 04 LSTV detection | 8 | 48 GB | None | 12h |
| 06 3D visualisation | 4 | 32 GB | None | 6h |

### Resumability

Each step tracks progress in `progress_selective.json`. Resubmitting any SLURM script automatically skips completed studies.

---

## References

All thresholds are sourced directly from peer-reviewed literature. No arbitrary values.

1. **Castellvi AE**, Goldstein LA, Chan DPK. *Intertransverse process impingement of the superior gluteal nerve*. Spine. 1984;9(1):31–35. — Original Castellvi classification; ≥19mm TP threshold.

2. **Konin GP**, Walz DM. *Lumbosacral transitional vertebrae: classification, imaging findings, and clinical relevance*. Semin Musculoskelet Radiol. 2010;14(1):67–76. — Comprehensive MRI classification review; disc reduction as sacralization criterion.

3. **Nardo L**, Alizai H, Virayavanich W, et al. *Lumbosacral transitional vertebrae: association with low back pain*. Radiology. 2012;265(2):497–503. — H/AP ratio thresholds (0.52, 0.68); transitional morphology on MRI; large population study.

4. **Hughes RJ**, Saifuddin A. *Imaging of lumbosacral transitional vertebrae*. Clin Radiol. 2004;59(11):984–991. — Lumbarization definition; L6 disc criteria; MRI counting methodology.

5. **Hughes RJ**, Saifuddin A. *Numbering of lumbo-sacral transitional vertebrae on MRI: role of the iliolumbar ligament*. AJR Am J Roentgenol. 2006;187(1):W59–65. — Iliolumbar ligament as level-identification anchor; lumbarization vs sacralization distinction.

6. **Seyfert S**. *Dermatome changes after lumbosacral transitional vertebra treatment*. Neuroradiology. 1997;39(8):584–587. — L5-S1 disc loss as most reliable sacralization sign.

7. **Farfan HF**, Cossette JW, Robertson GH, Wells RV, Kraus H. *The effects of torsion on the lumbar intervertebral joints*. J Bone Joint Surg Am. 1972;54(3):492–510. — Disc Height Index methodology; Farfan method.

8. **Panjabi MM**, Goel V, Oxland T, et al. *Human lumbar vertebrae: quantitative three-dimensional anatomy*. Spine. 1992;17(3):299–306. — Normal H/AP ratios for L1–L5 (L3=0.82, L4=0.78, L5=0.72).

9. **Quinlan JF**, Duke D, Eustace S. *Bertolotti's syndrome: a cause of back pain in young people*. J Bone Joint Surg Br. 2006;88(9):1183–1186. — Castellvi Type I clinical significance; unilateral vs bilateral morbidity.

10. **Farshad-Amacker NA**, Farshad M, Winklehner A, Andreisek G. *MR imaging of the intervertebral disc*. Eur Spine J. 2014;23(Suppl 3):S386–395. — Disc signal and DHI at transitional levels.

11. **Carrino JA**, Campbell PD, Lin DC, et al. *Effect of spinal segment variants on numbering of lumbar vertebrae by use of CT and MR imaging*. Radiology. 2011;259(1):196–202. — 30% error rate in level identification without whole-spine imaging.

12. **Nidecker AE**, Woernle CM, Sprott H. *Sacral transitional vertebra and L5 sacralization: considerations for lumbar spine surgery*. Eur Radiol. 2018;28(4):1376–1383. — MRI Phase 2 T2w signal classification criteria; surgical implications.

13. **Seilanian Toosi F**, Mahdianfar B, Zarifian A, et al. *Lumbosacral vertebral angles can predict lumbosacral transitional vertebrae on routine sagittal MRI*. Arch Bone Jt Surg. 2025;13(5):271–280. doi:10.22038/ABJS.2025.83244.3790. — A/B/C/D/δ angles; δ ≤ 8.5° for Type 2 LSTV (sens 92.3%, spec 87.9%, NPV 99.5%); C ≤ 35.5° for any LSTV; disc asymmetry OR 19.87.

14. **Chalian M**, Soldatos T, Carrino JA, Belzberg AJ, Khanna J, Chhabra A. *Prediction of transitional lumbosacral anatomy on magnetic resonance imaging of the lumbar spine*. World J Radiol. 2012;4(3):97–101. — Original A/B angle proposal; 80% sensitivity/specificity claim.

15. **Farshad M**, Aichmair A, Hughes AP, Herzog RJ, Farshad-Amacker NA. *A reliable measurement for identifying a lumbosacral transitional vertebra with a solid bony bridge on a single-slice midsagittal MRI or plain lateral radiograph*. Bone Joint J. 2013;95-B(11):1533–7. — Diff-VMVA ≤10° for Type 3/4 LSTV; 100% sensitivity, 89% specificity.

16. **Zhou S**, Du L, Liu X, et al. *Quantitative measurements at the lumbosacral junction are more reliable parameters for identifying and numbering lumbosacral transitional vertebrae*. Eur Radiol. 2022;32(8):5650–5658. — AVA (anterior-edge vertebral angle) at cutoff 73°; sensitivity 77.5%, specificity 88.3%.

17. **Farshad-Amacker NA**, Lurie B, Herzog RJ, Farshad M. *Is the iliolumbar ligament a reliable identifier of the L5 vertebra in lumbosacral transitional anomalies?* Eur Radiol. 2014;24(10):2623–30. — ILL unreliable in LSTV; multiple origins.

18. **Möller H** et al. *SPINEPS — automatic whole spine segmentation of T2-weighted MR images using a two-step approach for iterative segmentation of individual spine structures*. Eur Radiol. 2025. doi:10.1007/s00330-024-11155-y

19. **Warszawer Y** et al. *TotalSpineSeg: Robust spine segmentation and landmark labeling in MRI*. 2025. arXiv:2411.09344.

---

## Contact

Pipeline questions: go2432@wayne.edu  
Wayne State University School of Medicine — Spine Imaging & AI Lab
