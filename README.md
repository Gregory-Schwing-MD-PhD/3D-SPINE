# lstv-detector

**Automated MRI-based detection and classification of Lumbosacral Transitional Vertebrae (LSTV) using deep learning segmentation, radiologically-grounded morphometrics, and interactive 3D visualisation.**

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

## Radiologic Criteria Implemented

### Primary Criteria (each independently sufficient to flag LSTV)

| Criterion | Threshold | Reference |
|-----------|-----------|-----------|
| **Castellvi TP height** | ≥ 19 mm craniocaudal | Castellvi et al. 1984 |
| **TP–sacrum contact** | ≤ 2 mm 3D distance | Castellvi et al. 1984 |
| **Disc below TV absent / severely reduced** | DHI < 50% | Seyfert 1997; Farfan et al. 1972 |
| **6-lumbar count** (L6 present) | VERIDAH label 25 detected | Hughes & Saifuddin 2006 |
| **4-lumbar count** (confirmed both sources) | TSS + VERIDAH = 4 | Konin & Walz 2010 |

### Supporting Criteria (increase phenotype confidence)

| Criterion | Threshold | Reference |
|-----------|-----------|-----------|
| **TV body H/AP ratio — sacral-like** | < 0.52 | Nardo et al. 2012; Panjabi et al. 1992 |
| **TV body H/AP ratio — transitional** | 0.52–0.68 | Nardo et al. 2012 |
| **TV body H/AP ratio — lumbar-like** | > 0.68 | Nardo et al. 2012 |
| **TV/L4 normalised H:AP ratio** | < 0.80 → squarer than L4 | Nardo et al. 2012 |
| **Disc below TV moderately reduced** | DHI 50–70% | Farfan et al. 1972 |
| **Disc below TV preserved** | DHI ≥ 80% | Konin & Walz 2010 |
| **Disc above TV preserved** | DHI ≥ 80% | localises pathology to L5-S1 |

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
├── Step 04: LSTV Detection + Morphometrics               [CPU]
│   ├── Phase 1: Sagittal geometric Castellvi (TP height, TP–sacrum distance)
│   ├── Phase 2: Axial T2w signal classification (Type II vs III)
│   ├── Phenotype: Radiologically-grounded multi-criteria classifier
│   │   (independent of Castellvi — both co-reported)
│   └── Outputs: lstv_results.json  lstv_summary.json
│
└── Step 06: 3D Visualisation                             [CPU]
    └── results/lstv_3d/{study_id}_lstv_3d.html
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

Classification:
- dist > 2 mm AND height ≥ 19 mm → **Type I**
- dist ≤ 2 mm → **CONTACT_PENDING_P2** → proceed to Phase 2

### Step 3: Castellvi Phase 2 (axial T2w signal)

Extract a 32×32 voxel patch centred at the midpoint between the closest TP and sacrum voxels in the registered axial space. Classify:
- patch_mean < 55% × global_p95 → dark/intermediate → **Type II** (fibrocartilage)
- CV < 0.12 (uniform bright) → **Type III** (osseous marrow bridge)
- Ambiguous → **Type II** (conservative fallback; CT recommended for definitive Type III)

### Step 4: Phenotype classification (independent of Castellvi)

A tiered multi-criteria classifier grounded in the literature above:

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

Castellvi alone without disc or morphometric corroboration → **transitional_indeterminate** (Castellvi I has lowest clinical impact; Quinlan et al. 1984).

### What triggers lstv_detected = True

```
lstv_detected = True  iff ANY of:
  • Castellvi Type I-IV on either side
  • lumbar_count ≠ 5  (confirmed by reconciled TSS + VERIDAH)
  • phenotype ∈ {sacralization, lumbarization}  (primary criterion confirmed)
```

---

## File Structure

```
lstv-detector/
├── scripts/
│   ├── lstv_engine.py         ← all morphometric calculations (importable)
│   ├── 04_detect_lstv.py      ← Castellvi classifier + phenotype engine
│   └── 06_visualize_3d.py     ← interactive 3D HTML renderer
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
    │   └── lstv_summary.json      ← aggregate statistics
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
    "Lumbar count = 6 (expected 5) — LUMBARIZATION by vertebral counting",
    "Phenotype: LUMBARIZATION (high confidence) — criteria: L1:6-lumbar-count; L3:TV-is-L6"
  ],
  "castellvi_type": "Type IIb",
  "confidence": "high",
  "left":  { "classification": "Type II", "tp_height_mm": 22.4, "dist_mm": 0.8, ... },
  "right": { "classification": "Type II", "tp_height_mm": 21.9, "dist_mm": 1.1, ... },
  "lstv_morphometrics": {
    "lumbar_count_tss": 5,
    "lumbar_count_veridah": 6,
    "lumbar_count_consensus": 6,
    "lumbar_count_anomaly": true,
    "tv_name": "L6",
    "has_l6": true,
    "tv_shape": {
      "h_ap_ratio": 0.74,
      "shape_class": "lumbar-like",
      "norm_ratio": 0.95
    },
    "disc_below": {
      "level": "L6-S1",
      "dhi_pct": 88.2,
      "grade": "Normal"
    },
    "lstv_phenotype": "lumbarization",
    "phenotype_confidence": "high",
    "primary_criteria_met": ["L1:6-lumbar-count", "L2:disc-below-preserved-DHI-88pct", "L3:TV-is-L6"]
  },
  "pathology_score": 11.0
}
```

Note that in this example, `castellvi_type = "Type IIb"` and `lstv_phenotype = "lumbarization"` are **both set**. The TP of L6 forms pseudo-articulations with the sacrum bilaterally (Castellvi IIb) while the overall pattern is lumbarization. These are orthogonal findings and co-reported.

### `lstv_summary.json`

```json
{
  "total": 283,
  "lstv_detected": 61,
  "lstv_rate": 0.2156,
  "castellvi_breakdown": { "Type Ia": 8, "Type Ib": 4, "Type IIa": 12, ... },
  "phenotype_breakdown": { "lumbarization": 14, "sacralization": 28, "transitional_indeterminate": 7, "normal": 234 }
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
    --study_id 1307819508 \
    --spineps_dir    results/spineps \
    --totalspine_dir results/totalspineseg \
    --output_dir     results/lstv_3d \
    --lstv_json      results/lstv_detection/lstv_results.json
```

### Batch — all studies, render top 5 pathologic + 2 normal

```bash
sbatch slurm_scripts/04_lstv_detection.sh   # ALL=true

# After completion:
python scripts/06_visualize_3d.py \
    --rank_by lstv --top_n 5 --top_normal 2 \
    --lstv_json results/lstv_detection/lstv_results.json \
    --spineps_dir results/spineps \
    --totalspine_dir results/totalspineseg \
    --output_dir results/lstv_3d
```

---

## Python API

```python
from lstv_engine import (
    load_lstv_masks, analyze_lstv, compute_lstv_pathology_score,
    TP_HEIGHT_MM, TV_SHAPE_LUMBAR, DHI_REDUCED_PCT,
)

# Load and resample masks to 1mm isotropic
masks = load_lstv_masks("1307819508", spineps_dir, totalspine_dir)

# Run full morphometric analysis; pass Castellvi result if already computed
morpho = analyze_lstv(masks, castellvi_result=detect_result)

print(f"TV:         {morpho.tv_name}")
print(f"Count:      {morpho.lumbar_count_consensus}")
print(f"Phenotype:  {morpho.lstv_phenotype} ({morpho.phenotype_confidence})")
print(f"Primary:    {morpho.primary_criteria_met}")
print(f"DHI below:  {morpho.disc_below.dhi_pct:.1f}%")
print(f"H/AP ratio: {morpho.tv_shape.h_ap_ratio:.2f} ({morpho.tv_shape.shape_class})")

# Pathology burden score for ranking
score = compute_lstv_pathology_score(detect_result, morpho.to_dict())
```

---

## 3D Visualiser

Each HTML output (`{study_id}_lstv_3d.html`) contains:

- **Colour-coded phenotype banner**: SACRALIZATION (red) / LUMBARIZATION (orange) / TRANSITIONAL (yellow) / NORMAL (green)
- **Castellvi badge**: displayed alongside phenotype — both shown simultaneously if applicable
- **TP height rulers**: craniocaudal extent overlaid on 3D TP mesh (red/cyan = left/right; diamond marker with ≥19mm flag)
- **TP–sacrum gap rulers**: dashed line to nearest sacrum point; contact (≤2mm) shown in red
- **TV body shape annotation**: SI-height and AP-depth rulers; H/AP ratio labelled with literature classification
- **Lumbar count badge**: `4`, `5`, or `6` with anomaly flag
- **LSTV detection reasons panel**: full list of which criteria triggered lstv_detected
- **Phenotype rationale panel**: multi-sentence radiologic justification with primary criteria
- **Disc DHI strip**: above/below TV DHI percentages colour-coded by grade
- **Camera presets**: Oblique / Lateral / Posterior / Anterior / Axial

The visualiser ranks studies by pathology score (configurable via `--top_n`, `--top_normal`). "Normal" controls are strictly defined: lstv_detected=False AND count=5 AND score=0.

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

## Known Limitations

**Lumbarization count edge case**: TSS labels stop at L5 (label 45). When L6 is present, TSS will label L1–L5 of the 6-lumbar spine correctly, but the TSS count will read as 5 (normal). The L6 signal comes exclusively from VERIDAH label 25. If SPINEPS mis-labels L6 (e.g., as a second L5), the consensus count will be 5 and the L6 will be missed. The cross-validation warning (`L5 centroid dist > 20mm`) will flag such cases.

**Type III over-reporting**: Phase 2 MRI Type III classification is provisional. Homogeneous T2 signal at the TP junction may occur with periosteal bone marrow without true cortical bridging. CT confirmation is recommended before operative planning.

**Castellvi on 4-lumbar spines**: When count=4, the lowest mobile segment is typically L4. The VERIDAH TV search will identify L4 (label 23) as the TV, and Castellvi will be assessed on L4 TP. This is radiologically correct — the sacralizing segment's TP should be evaluated — but the label printed will be "L4."

**DHI at L6-S1**: TotalSpineSeg has no disc label for the L6-S1 level. DHI at L6-S1 uses VERIDAH IVD label 125 (100 + VD_L6=25) if present. If SPINEPS does not label the L6-S1 disc, DHI will be reported as undetected (not absent).

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

13. **Möller H** et al. *SPINEPS — automatic whole spine segmentation of T2-weighted MR images using a two-step approach for iterative segmentation of individual spine structures*. Eur Radiol. 2025. doi:10.1007/s00330-024-11155-y

14. **Warszawer Y** et al. *TotalSpineSeg: Robust spine segmentation and landmark labeling in MRI*. 2025. arXiv:2411.09344.

---

## Contact

Pipeline questions: go2432@wayne.edu  
Wayne State University School of Medicine — Spine Imaging & AI Lab
