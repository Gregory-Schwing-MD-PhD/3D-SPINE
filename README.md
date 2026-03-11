# lstv-detector

**Automated MRI-based detection and Castellvi classification of Lumbosacral Transitional Vertebrae (LSTV) using a three-model deep learning ensemble, radiologically-grounded morphometrics, and interactive 3D visualisation.**

> Target audience: spine neurosurgeons, musculoskeletal radiologists, and deep learning researchers working with spinal MRI.

---

## Table of Contents

1. [Clinical Background](#1-clinical-background)
2. [Castellvi Classification](#2-castellvi-classification)
3. [Radiologic Criteria Implemented](#3-radiologic-criteria-implemented)
4. [System Architecture](#4-system-architecture)
5. [Three-Model Ensemble — Label Dictionaries](#5-three-model-ensemble--label-dictionaries)
6. [Vertebral Alignment: TSS vs VERIDAH Offset Model](#6-vertebral-alignment-tss-vs-veridah-offset-model)
7. [Ian Pan Vertebral Arbitration (v4.4)](#7-ian-pan-vertebral-arbitration-v44)
8. [Classification Logic](#8-classification-logic)
9. [Output Schema](#9-output-schema)
10. [Quick Start](#10-quick-start)
11. [Python API](#11-python-api)
12. [3D Visualiser](#12-3d-visualiser)
13. [Pathology Score](#13-pathology-score)
14. [Infrastructure](#14-infrastructure)
15. [Known Limitations](#15-known-limitations)
16. [Potential Improvements](#16-potential-improvements)
17. [Likelihood of Working: Honest Assessment](#17-likelihood-of-working-honest-assessment)
18. [References](#18-references)

---

## 1. Clinical Background

### What is an LSTV?

A lumbosacral transitional vertebra (LSTV) is a congenital anomaly in which the last mobile lumbar segment displays morphology intermediate between a lumbar and sacral vertebra. Reported prevalence ranges from **4–35.9%** depending on modality, counting methodology, and population (Nardo et al. 2012; Hughes & Saifuddin 2006).

### Why It Matters Surgically

| Risk | Mechanism | Magnitude |
|------|-----------|-----------|
| **Wrong-level surgery** | MRI level counting fails without whole-spine imaging | ~30% of studies (Carrino et al. 2011) |
| **Bertolotti's syndrome** | Low back pain from TP–sacrum pseudo-articulation | 4–8% of LBP patients |
| **Altered disc biomechanics** | Fulcrum shift accelerates degeneration at mobile level above TV | Well-established |
| **Surgical anatomy distortion** | Pedicle screw trajectories, fluoroscopic counting, neuromonitoring | Per-patient |

### The Two LSTV Phenotypes

These are **radiologically distinct and clinically independent**. A patient may carry both a Castellvi classification (TP morphology) and a phenotype (transition pattern) simultaneously.

**Sacralization**: L5 incorporates into the sacrum. The disc below becomes reduced or absent — the single most reliable radiologic marker (Seyfert 1997). The vertebral body becomes squarer (H/AP ratio decreases toward sacral range). The TPs may articulate or fuse with the sacral ala (Castellvi classification).

**Lumbarization**: S1 acquires lumbar characteristics. A mobile L6-S1 disc develops below S1, and the vertebral body adopts lumbar proportions (H/AP ≥ 0.68). Creates a 6-lumbar-segment spine. Castellvi TP enlargement may co-occur on L6 but is classified separately.

---

## 2. Castellvi Classification

Source: Castellvi et al., *Spine* 1984;9(1):31–35.

| Type | Definition | Unilateral | Bilateral |
|------|-----------|-----------|----------|
| **I** | Dysplastic TP ≥ 19 mm craniocaudal height, no sacral contact | Ia | Ib |
| **II** | Pseudo-articulation (diarthrodial joint) between enlarged TP and sacrum | IIa | IIb |
| **III** | Complete bony fusion of TP with sacrum ala | IIIa | IIIb |
| **IV** | Mixed: Type II one side, Type III the other | — | — |

**Key threshold**: TP height ≥ 19 mm craniocaudal, originally defined on plain film. On MRI this maps to the full SI extent of the SPINEPS costal process mask.

**MRI Phase 2 T2w signal classification** (Konin & Walz 2010; Nidecker et al. 2018):

| Signal at TP–sacrum junction | Interpretation | Classification |
|------------------------------|----------------|----------------|
| Dark / heterogeneous | Fibrocartilaginous pseudo-joint or synovial cleft | **Type II** |
| Uniform high T2w (bright, homogeneous) | Osseous bridge with marrow fat continuity | **Type III** |
| Ambiguous | Conservative fallback | **Type II** |

> ⚠ CT remains the gold standard for Type III confirmation. MRI Phase 2 results are provisional.

---

## 3. Radiologic Criteria Implemented

### Primary Criteria (each independently sufficient to flag LSTV)

| Criterion | Threshold | Reference |
|-----------|-----------|-----------|
| Castellvi TP height | ≥ 19 mm craniocaudal | Castellvi et al. 1984 |
| TP–sacrum contact | ≤ 2 mm 3D distance | Castellvi et al. 1984 |
| Disc below TV absent / severely reduced | DHI < 50% | Seyfert 1997; Farfan et al. 1972 |
| 6-lumbar count (L6 present) | VERIDAH label 25 detected | Hughes & Saifuddin 2006 |
| 4-lumbar count (confirmed both sources) | TSS + VERIDAH = 4 | Konin & Walz 2010 |

### Supporting Criteria (increase phenotype confidence)

| Criterion | Threshold | Reference |
|-----------|-----------|-----------|
| TV body sacral-like | H/AP < 0.52 | Nardo et al. 2012; Panjabi et al. 1992 |
| TV body transitional | H/AP 0.52–0.68 | Nardo et al. 2012 |
| TV body lumbar-like | H/AP ≥ 0.68 | Nardo et al. 2012 |
| TV/L4 normalised H:AP | < 0.80 → squarer than L4 | Nardo et al. 2012 |
| Disc below TV moderately reduced | DHI 50–70% | Farfan et al. 1972 |
| Disc below TV preserved | DHI ≥ 80% | Konin & Walz 2010 |
| Disc above TV preserved | DHI ≥ 80% | localises pathology to L5-S1 |

### Disc Height Index — Farfan Method

```
DHI = (disc height / mean of adjacent vertebral body heights) × 100

Normal lumbar:   80–100%
Moderate loss:   50–70%
Severe loss:     < 50%  ← primary sacralization criterion (Seyfert 1997)
Absent:          detected as label present but < 3% of expected volume
```

Normal lumbar reference from Panjabi et al. 1992: L3 H/AP = 0.82 ± 0.09; L4 = 0.78 ± 0.08; L5 = 0.72 ± 0.10.

---

## 4. System Architecture

### Full Pipeline

```
Input: DICOM studies (Sagittal T2w  ±  Axial T2w)
│
├── Step 01 ─ DICOM → NIfTI  (dcm2niix)
│
├── Step 02b ─ SPINEPS Segmentation                              [GPU]
│   ├── seg-spine_msk.nii.gz    subregion semantic labels
│   │       43 = Costal_Process_LEFT   ← TP source (left side)
│   │       44 = Costal_Process_RIGHT  ← TP source (right side)
│   │       26 = Sacrum  41 = Arcus  42 = Spinous  49 = Corpus
│   │       60 = Cord    61 = Canal
│   └── seg-vert_msk.nii.gz     VERIDAH per-vertebra instance labels
│           20=L1  21=L2  22=L3  23=L4  24=L5  25=L6  26=Sacrum
│           120=L1-L2 IVD  121=L2-L3  122=L3-L4  123=L4-L5  124=L5-S1
│           126 = S1-S2 IVD  ← sacralization marker
│
├── Step 03b ─ TotalSpineSeg (TSS) Segmentation                  [GPU]
│   └── sagittal_labeled.nii.gz
│           41=L1  42=L2  43=L3  44=L4  45=L5  50=Sacrum
│           91=T12-L1  92=L1-L2  93=L2-L3  94=L3-L4  95=L4-L5  100=L5-S1
│           ⚠  TSS labels 43/44 = L3/L4 vertebral bodies ≠ SPINEPS TPs
│
├── Step 00b ─ Ian Pan Disc Localiser                            [GPU]
│   └── ian_pan_disc_coords.json
│           per-disc: world_ras_mm, peak_prob, entropy
│           disc keys: l1_l2  l2_l3  l3_l4  l4_l5  l5_s1
│
├── Step 03c ─ SPINEPS → Axial T2w Registration                  [CPU]
│
└── Step 04 ─ LSTV Detection + Classification                    [CPU]
    │
    ├── [A] Vertebral Alignment Analysis
    │         TSS vs VERIDAH Dice offset model
    │         Finds integer offset between lumbar level labellings
    │
    ├── [B] Ian Pan Vertebral Arbitration (when offset ≠ 0)
    │         Infers vertebral body positions from disc midpoints
    │         Scores both models in RAS mm space; overrides if margin ≥ 5 mm
    │
    ├── [C] TV Identification
    │         TSS-first, VERIDAH fallback, L6 cross-verification
    │
    ├── [D] Castellvi Phase 1  (sagittal: PCA TP height + EDT distance)
    │
    ├── [E] Castellvi Phase 2  (axial: T2w patch signal classification)
    │
    ├── [F] Ensemble offset re-classification
    │         Castellvi re-run at best_offset TV when offset ≠ 0
    │
    ├── [G] Phenotype Classification
    │         Count anomaly → morphometric tiering → confidence
    │
    └── Outputs: lstv_results.json  lstv_summary.json
                 lstv_per_study.csv  lstv_alignment.csv
                 lstv_cohort_summary.csv
```

### Critical Label Disambiguation

> The single most dangerous confusion in the entire codebase. Getting this wrong causes silently incorrect Castellvi classifications.

| Label value | In SPINEPS `seg-spine_msk` | In TotalSpineSeg `sagittal_labeled` |
|-------------|---------------------------|-------------------------------------|
| **43** | **Costal_Process_LEFT ← TP source** | **L3 vertebral body — NOT a TP** |
| **44** | **Costal_Process_RIGHT ← TP source** | **L4 vertebral body — NOT a TP** |
| 45 | Superior articular process LEFT | **L5 vertebral body** |
| 50 | Not used | **Sacrum ← preferred sacrum source** |

**Rule**: Always source TPs from SPINEPS `seg-spine_msk` labels 43/44. Always source sacrum from TSS label 50 (with SPINEPS label 26 as fallback). Never mix these across models.

---

## 5. Three-Model Ensemble — Label Dictionaries

### Model 1: TotalSpineSeg (TSS)

Primary role: **vertebral body level ground truth and sacrum source.**

| Label | Structure | Role |
|-------|-----------|------|
| 41 | L1 | Level counting |
| 42 | L2 | Level counting |
| 43 | L3 | Level counting |
| 44 | L4 | H/AP normalisation reference |
| 45 | L5 | TV candidate in sacralization |
| 50 | Sacrum | EDT distance source for Castellvi |
| 91 | T12–L1 disc | Disc height calculation |
| 92 | L1–L2 disc | Disc height calculation |
| 93 | L2–L3 disc | Disc height calculation |
| 94 | L3–L4 disc | Disc height calculation |
| 95 | L4–L5 disc | Disc height calculation |
| 100 | L5–S1 disc | **DHI below TV; primary sacralization marker** |

### Model 2: SPINEPS `seg-vert_msk` (VERIDAH)

Primary role: **L6 detection and per-vertebra instance segmentation.**

| Label | Structure | Role |
|-------|-----------|------|
| 20 | L1 | Level counting |
| 21 | L2 | Level counting |
| 22 | L3 | Level counting |
| 23 | L4 | Level counting |
| 24 | L5 | TV candidate |
| **25** | **L6** | **Lumbarization indicator — only model with this label** |
| 26 | Sacrum | Fallback sacrum source |
| 120 | L1–L2 IVD | Disc height calculation |
| 121 | L2–L3 IVD | Disc height calculation |
| 122 | L3–L4 IVD | Disc height calculation |
| 123 | L4–L5 IVD | Disc height calculation |
| 124 | L5–S1 IVD | DHI below TV |
| **126** | **S1–S2 IVD** | **Sacralization marker — only present when L5 fuses to sacrum** |

### Model 2: SPINEPS `seg-spine_msk` (subregion semantic)

Primary role: **transverse process geometry source for Castellvi classification.**

| Label | Structure | Role |
|-------|-----------|------|
| 26 | Sacrum | Fallback sacrum |
| 41 | Neural arch (arcus) | — |
| 42 | Spinous process | — |
| **43** | **Costal process LEFT** | **TP height (PCA SI extent); EDT distance to sacrum** |
| **44** | **Costal process RIGHT** | **TP height (PCA SI extent); EDT distance to sacrum** |
| 45 | Sup. articular process LEFT | — |
| 46 | Sup. articular process RIGHT | — |
| 49 | Corpus (vertebral body) | Supplementary body mask |
| 60 | Spinal cord | — |
| 61 | Spinal canal | — |

### Model 3: Ian Pan Disc Localiser

Primary role: **independent 3D disc position reference for vertebral arbitration.**

| JSON disc key | Level |
|---------------|-------|
| `l1_l2` | L1–L2 |
| `l2_l3` | L2–L3 |
| `l3_l4` | L3–L4 |
| `l4_l5` | L4–L5 |
| `l5_s1` | L5–S1 |

Per-disc fields:

| Field | Type | Description |
|-------|------|-------------|
| `world_ras_mm` | float[3] | 3D RAS coordinate of heatmap peak |
| `peak_prob` | float 0–1 | Detection confidence (threshold: 0.30) |
| `entropy` | float | Heatmap entropy (lower = sharper localisation) |

---

## 6. Vertebral Alignment: TSS vs VERIDAH Offset Model

### The Problem

TSS and VERIDAH independently assign level identities to lumbar vertebrae. In a normal spine they agree (offset = 0). In an LSTV spine, the transitional vertebra may be assigned a different level by each model — TSS may call it L5 while VERIDAH calls it L6, or TSS may incorporate it into the sacrum while VERIDAH counts it as L5. If this disagreement is not detected, Castellvi morphometrics will be computed on the **wrong vertebra**.

### Offset Definition

```
TSS anatomical index:     i = TSS_label − 41      (L1→0, L2→1, … L5→4)
VERIDAH anatomical index: j = VD_label  − 20      (L1→0, L2→1, … L6→5)

Offset k:  TSS index i  ←→  VERIDAH index (i + k)
           i.e. TSS label (41+i)  ←→  VD label (20+i+k)
```

| Offset | Meaning | LSTV implication |
|--------|---------|------------------|
| **0** | Models agree | Normal, or both correctly identify LSTV |
| **+1** | VERIDAH one level cranial to TSS | Lumbarization — VERIDAH found L6, TSS did not |
| **−1** | TSS one level cranial to VERIDAH | Sacralization — TSS subsumed L5 into sacrum |
| **+2** | Extreme lumbarization | Two extra caudal levels in VERIDAH |
| **−2** | Extreme sacralization | Two extra caudal levels in TSS |

### Scoring Algorithm

For each candidate offset k ∈ {−2, −1, 0, +1, +2}:

```
For anatomical indices i = 0..4 (L1..L5):
    pair: TSS label (41+i)  ←→  VD label (20+i+k)
    compute Dice overlap of their 3D voxel masks

score(k) = mean Dice over valid pairs
           where: valid means Dice ≥ MIN_PAIR_DICE (0.10)
           and:   n_valid ≥ MIN_PAIRS_REQUIRED (3)
```

**Decision rule — accept non-zero offset k over baseline (k=0) when:**

```
(a) score(k) > score(0) + SCORE_MARGIN (0.08)
    OR offset=0 has no valid pairs at all
(b) ≥ CONSISTENCY_FRAC (60%) of individual pairs
    individually favour k over 0
```

### Example: Lumbarization (5-TSS × 6-VERIDAH)

```
offset = 0:   TSS L1↔VD L1   TSS L2↔VD L2   …   TSS L5↔VD L5
              Dice: 0.83       0.80                0.34  ← VD L5 only partially overlaps TSS L5

offset = +1:  TSS L1↔VD L2   TSS L2↔VD L3   …   TSS L5↔VD L6
              Dice: 0.89       0.91                0.88  ← anatomically correct pairings

margin = 0.89 − 0.57 = 0.32  >>  SCORE_MARGIN (0.08)
consistency = 5/5 pairs favour offset +1

→ ACCEPTED: offset = +1
→ preferred_hypothesis = 'shifted_plus_1'  (lumbarization)
```

### `preferred_hypothesis` Values

| Value | Offset | Interpretation |
|-------|--------|----------------|
| `aligned` | 0 | TSS and VERIDAH agree |
| `shifted_plus_1` | +1 | Lumbarization candidate |
| `shifted_minus_1` | −1 | Sacralization candidate |
| `shifted_plus_2` | +2 | Extreme lumbarization |
| `shifted_minus_2` | −2 | Extreme sacralization |
| `insufficient_data` | None | <3 valid pairs at all offsets |

---

## 7. Ian Pan Vertebral Arbitration (v4.4)

### Motivation

When TSS and VERIDAH disagree (offset ≠ 0), the Dice model has made a geometric determination based on voxel overlap. However it may be overconfident in borderline LSTV cases where label boundaries are inherently ambiguous. Ian Pan provides an **independent anatomical constraint** that is completely agnostic to voxel-label overlap: the 3D positions of disc centres in world (RAS mm) space. By inferring where vertebral bodies _must_ be from the midpoints of their bounding discs, we can score both segmentation models against a physics-based reference.

### Vertebral Body Position Inference

A vertebral body centroid lies approximately at the midpoint between the two discs that bound it. Both adjacent disc detections must have `peak_prob ≥ 0.30` to contribute.

```
Disc peaks (world_ras_mm) from Ian Pan:

   L1/L2 peak ──┐
                ├── midpoint ──► inferred L2 centroid (RAS mm)
   L2/L3 peak ──┘

   L2/L3 peak ──┐
                ├── midpoint ──► inferred L3 centroid (RAS mm)
   L3/L4 peak ──┘

   L3/L4 peak ──┐
                ├── midpoint ──► inferred L4 centroid (RAS mm)
   L4/L5 peak ──┘

   L4/L5 peak ──┐
                ├── midpoint ──► inferred L5 centroid (RAS mm)
   L5/S1 peak ──┘
```

### Disc Pair → Inferred Vertebra Lookup Table

| Disc above | Disc below | Inferred body | TSS label | VERIDAH label |
|------------|------------|--------------|-----------|---------------|
| l1_l2 | l2_l3 | **L2** | 42 | 21 |
| l2_l3 | l3_l4 | **L3** | 43 | 22 |
| l3_l4 | l4_l5 | **L4** | 44 | 23 |
| l4_l5 | l5_s1 | **L5** | 45 | 24 |

### Scoring

```
For each inferred vertebral body k ∈ {L2, L3, L4, L5}:

    dist_TSS(k)     = ||inferred_pos_k − TSS_vert_k_centroid_RAS||₂
    dist_VERIDAH(k) = ||inferred_pos_k − VD_vert_k_centroid_RAS||₂

score_TSS     = mean(dist_TSS)     over qualifying pairs
score_VERIDAH = mean(dist_VERIDAH) over qualifying pairs

margin_mm = score_VERIDAH − score_TSS

    margin_mm > 0  →  TSS centroids closer  →  TSS labelling correct
    margin_mm < 0  →  VERIDAH centroids closer  →  VERIDAH labelling correct
```

### Override Decision

```
IF offset == 0:
    ─ informational only; preferred_hypothesis unchanged

IF offset ≠ 0  AND  n_pairs ≥ 2  AND  |margin_mm| ≥ 5.0 mm:
    TSS wins  (margin > 0)  →  preferred_hypothesis = 'aligned'
    VERIDAH wins, offset > 0  →  preferred_hypothesis = 'shifted_plus_1'
    VERIDAH wins, offset < 0  →  preferred_hypothesis = 'shifted_minus_1'
    ip_tiebreak_applied = True

ELSE:
    ─ arbitration skipped; Dice offset result stands
```

### Full Decision Flow

```
                 ┌──────────────────────────────────┐
                 │  TSS + VERIDAH Dice offset model  │
                 │  → best_offset, preferred_hyp     │
                 └─────────────────┬────────────────┘
                                   │
                  ┌────────────────▼────────────────┐
                  │          offset == 0?            │
                  └──────┬───────────────┬──────────┘
                         │ YES           │ NO
                         ▼               ▼
                   preferred =     Ian Pan: sufficient data?
                   'aligned'       (n_pairs ≥ 2)
                   [done]          ┌──────┴──────┐
                                   │ NO          │ YES
                                   ▼              ▼
                             Dice result     |margin| ≥ 5mm?
                             stands          ┌──────┴──────┐
                                             │ NO          │ YES
                                             ▼              ▼
                                        Dice result    IP overrides
                                        stands         preferred_hyp
                                             │              │
                                             └──────┬───────┘
                                                    ▼
                                          preferred_hypothesis (final)
                                                    │
                         ┌──────────────────────────▼────────────────────────┐
                         │    Castellvi at preferred TV                       │
                         │    Phase 1 (sagittal: PCA TP height + EDT dist)    │
                         │    Phase 2 (axial: T2w patch signal)               │
                         └───────────────────────────────────────────────────┘
```

### `AlignmentResult` Ian Pan Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ip_score_tss_mm` | float | None | Mean dist: inferred bodies → TSS centroids |
| `ip_score_vd_mm` | float | None | Mean dist: inferred bodies → VERIDAH centroids |
| `ip_vert_margin_mm` | float | None | score_vd − score_tss (+ve = TSS wins) |
| `ip_n_vert_pairs` | int | 0 | Number of inferred body pairs that voted |
| `ip_winner` | str | None | `'TSS'` \| `'VERIDAH'` \| `'tie'` \| `'insufficient'` |
| `ip_inferred_verts` | dict | None | Per-vertebra detail (L2, L3, L4, L5) |
| `ip_tiebreak_applied` | bool | False | True if preferred_hypothesis was changed |
| `ip_sequence_vote` | str | None | Legacy disc-vs-disc vote (audit trail) |
| `ip_vote_confidence` | str | None | Legacy confidence (audit trail) |
| `ip_per_level` | dict | None | Legacy per-disc detail (audit trail) |

---

## 8. Classification Logic

### Step 1: Vertebral Counting

```
TSS lumbar count   = |{41, 42, 43, 44, 45} ∩ labels_present|
VERIDAH count      = |{20, 21, 22, 23, 24, 25} ∩ labels_present|

Reconciliation:
  VERIDAH label 25 present  →  consensus = TSS_count + 1   (lumbarization)
  TSS < 5 AND VERIDAH corroborates  →  4-lumbar count       (sacralization)
  VERIDAH > TSS without label 25    →  trust TSS             (over-seg artifact)
  SPINEPS label 126 present  →  sacralization flag           (S1-S2 IVD marker)
```

### Step 2: Castellvi Phase 1 (Sagittal Geometric)

```
For each side {left=43, right=44}:

  1. Isolate SPINEPS costal process at TV z-extent ± 3 voxels
  2. PCA decomposition of TP mask point cloud
     → principal axis projected onto segmental SI axis
     → SI axis = unit vector: TSS disc-below-TV centroid → TSS disc-above-TV centroid
  3. TP craniocaudal height = SI projection extent × voxel_size  [mm]
  4. TP–sacrum 3D minimum distance via EDT on TSS sacrum mask  [mm]

Classification:
  dist > 2.0 mm  AND  height ≥ 19.0 mm  →  Type I
  dist ≤ 2.0 mm                          →  CONTACT → Phase 2
  else                                   →  Normal TP
```

### Step 3: Castellvi Phase 2 (Axial T2w Signal)

```
  Extract 32×32 voxel patch centred at midpoint(nearest_TP_vox, nearest_sacrum_vox)
  in registered axial space.

  patch_mean < 0.55 × global_p95   →  Type II  (fibrocartilage / dark)
  CV < 0.12 (uniform bright)        →  Type III (osseous marrow bridge)
  else                              →  Type II  (conservative fallback)
```

### Step 4: Phenotype Classification

**Tier 1 — Count anomaly (highest specificity; no morphometrics required)**

| Consensus count | Phenotype | Confidence |
|----------------|-----------|-----------|
| 6 | Lumbarization | High |
| 4 (both models) | Sacralization | High |

**Tier 2 — count = 5 (requires at least one primary criterion)**

Sacralization primary criteria:

| Code | Criterion |
|------|-----------|
| S1 | Castellvi detected (any type) |
| S2 | Disc below TV: DHI < 50% or absent |
| S4 | TV body sacral-like (H/AP < 0.52) + corroborating finding |

Lumbarization primary criteria:

| Code | Criterion |
|------|-----------|
| L2 | Disc below TV preserved (DHI ≥ 80%) → mobile L6-S1 |
| L3 | TV body lumbar-like (H/AP ≥ 0.68) with count = 6 |

Castellvi alone without disc or morphometric corroboration → `transitional_indeterminate`.

### `lstv_detected = True` when ANY of:

```
• Castellvi Type I–IV on either side
• lumbar_count ≠ 5  (confirmed by reconciled TSS + VERIDAH)
• phenotype ∈ {sacralization, lumbarization}  (primary criterion met)
```

---

## 9. Output Schema

### `lstv_results.json` — per-study

```json
{
  "study_id": "1307819508",
  "lstv_detected": true,
  "castellvi_type": "Type IIb",
  "confidence": "high",
  "lstv_reason": [
    "Lumbar count = 6 — LUMBARIZATION by vertebral counting",
    "Phenotype: LUMBARIZATION (high) — L1:6-lumbar-count; L2:disc-below-preserved-DHI-88pct"
  ],
  "left":  { "classification": "Type II", "tp_height_mm": 22.4, "dist_mm": 0.8 },
  "right": { "classification": "Type II", "tp_height_mm": 21.9, "dist_mm": 1.1 },
  "alignment": {
    "best_offset": 1,
    "preferred_hypothesis": "shifted_plus_1",
    "confidence": "moderate",
    "ip_winner": "VERIDAH",
    "ip_vert_margin_mm": -7.3,
    "ip_score_tss_mm": 14.2,
    "ip_score_vd_mm": 6.9,
    "ip_n_vert_pairs": 3,
    "ip_tiebreak_applied": true
  },
  "lstv_morphometrics": {
    "lumbar_count_tss": 5,
    "lumbar_count_veridah": 6,
    "lumbar_count_consensus": 6,
    "tv_name": "L6",
    "tv_shape": { "h_ap_ratio": 0.74, "shape_class": "lumbar-like", "norm_ratio": 0.95 },
    "disc_below": { "level": "L6-S1", "dhi_pct": 88.2, "grade": "Normal" },
    "lstv_phenotype": "lumbarization",
    "phenotype_confidence": "high",
    "primary_criteria_met": ["L1:6-lumbar-count", "L2:disc-below-preserved-DHI-88pct", "L3:TV-is-L6"]
  },
  "pathology_score": 11.0
}
```

### CSV Outputs

| File | Contents |
|------|----------|
| `lstv_per_study.csv` | All classification + alignment fields (1 row/study) |
| `lstv_alignment.csv` | Alignment-focused table including all `ip_*` fields |
| `lstv_l6_subgroup.csv` | Studies where VERIDAH found L6 |
| `lstv_cohort_summary.csv` | Single-row aggregate cohort statistics |

---

## 10. Quick Start

### Full SLURM Pipeline

```bash
J1=$(sbatch --parsable slurm_scripts/01_dicom_to_nifti.sh)
J2=$(sbatch --parsable --dependency=afterok:$J1 slurm_scripts/02b_spineps_selective.sh)
J3=$(sbatch --parsable --dependency=afterok:$J1 slurm_scripts/03b_totalspineseg_selective.sh)
J4=$(sbatch --parsable --dependency=afterok:$J2:$J3 slurm_scripts/03c_register.sh)
J5=$(sbatch --parsable --dependency=afterok:$J4 slurm_scripts/00b_ian_pan_disc_coords.sh)
J6=$(sbatch --parsable --dependency=afterok:$J4:$J5 slurm_scripts/04_lstv_detection.sh)
sbatch --dependency=afterok:$J6 slurm_scripts/06_visualize_3d.sh
```

### Single Study (Development)

```bash
python scripts/04_detect_lstv.py \
    --study_id 1307819508 \
    --spineps_dir    results/spineps \
    --totalspine_dir results/totalspineseg \
    --registered_dir results/registered \
    --nifti_dir      results/nifti \
    --output_dir     results/lstv_detection \
    --ian_pan_coords results/ian_pan_disc_coords/ian_pan_disc_coords.json
```

### Without Ian Pan (arbitration gracefully disabled)

```bash
# Omit --ian_pan_coords; all ip_* fields will be null but pipeline runs normally
python scripts/04_detect_lstv.py \
    --study_id 1307819508 \
    --spineps_dir    results/spineps \
    --totalspine_dir results/totalspineseg \
    --registered_dir results/registered \
    --nifti_dir      results/nifti \
    --output_dir     results/lstv_detection
```

---

## 11. Python API

```python
from lstv_engine import load_lstv_masks, analyze_lstv, compute_lstv_pathology_score
from vertebral_alignment import analyse_vertebral_alignment

# Load and resample masks to 1 mm isotropic
masks = load_lstv_masks("1307819508", spineps_dir, totalspine_dir)

# Vertebral alignment analysis
ar = analyse_vertebral_alignment("1307819508", sag_tss, sag_vert)
print(f"Offset:     {ar.best_offset}")
print(f"Hypothesis: {ar.preferred_hypothesis}  [{ar.confidence}]")
print(f"IP winner:  {ar.ip_winner}  margin = {ar.ip_vert_margin_mm} mm")

# Full morphometrics
morpho = analyze_lstv(masks, castellvi_result=detect_result)
print(f"TV:         {morpho.tv_name}")
print(f"Phenotype:  {morpho.lstv_phenotype} ({morpho.phenotype_confidence})")
print(f"DHI below:  {morpho.disc_below.dhi_pct:.1f}%")
print(f"H/AP:       {morpho.tv_shape.h_ap_ratio:.2f} ({morpho.tv_shape.shape_class})")

# Study ranking score
score = compute_lstv_pathology_score(detect_result, morpho.to_dict())
```

---

## 12. 3D Visualiser

Each HTML output (`results/lstv_3d/{study_id}_lstv_3d.html`) contains:

- **Colour-coded phenotype banner**: SACRALIZATION (red) / LUMBARIZATION (orange) / TRANSITIONAL (yellow) / NORMAL (green)
- **Castellvi badge**: displayed alongside phenotype (both shown simultaneously when applicable — they are orthogonal findings)
- **TP height rulers**: PCA-derived craniocaudal extent on 3D TP mesh; diamond marker with ≥19mm flag
- **TP–sacrum gap rulers**: dashed line to nearest sacrum point; contact (≤2mm) shown in red
- **TV body shape annotation**: SI-height and AP-depth rulers; H/AP ratio labelled with literature classification
- **Lumbar count badge**: `4`, `5`, or `6` with anomaly flag
- **LSTV detection reasons panel**: full list of which criteria triggered `lstv_detected`
- **Phenotype rationale panel**: multi-sentence radiologic justification with primary criteria codes
- **Disc DHI strip**: above/below TV DHI percentages colour-coded by grade
- **Camera presets**: Oblique / Lateral / Posterior / Anterior / Axial

Studies are ranked by pathology score. "Normal" controls: `lstv_detected = False` AND count = 5 AND score = 0.

---

## 13. Pathology Score

Used for study ranking only — not a diagnosis. Higher = more severe or interesting LSTV case.

| Feature | Points |
|---------|--------|
| Castellvi Type IV | 5 |
| Castellvi Type III (either side) | 4 |
| Castellvi Type II (either side) | 3 |
| Castellvi Type I (either side) | 1 |
| Phenotype confirmed, high confidence | +3 |
| Phenotype confirmed, moderate confidence | +2 |
| Phenotype transitional_indeterminate | +1 |
| Lumbar count anomaly (≠5) | +2 |
| Disc below TV DHI < 50% or absent | +2 |
| Disc below TV DHI 50–70% | +1 |
| TV body sacral-like (H/AP < 0.52) | +2 |
| TV body transitional (H/AP 0.52–0.68) | +1 |
| Rib anomaly / thoracic count mismatch | +1 |

---

## 14. Infrastructure

### Segmentation Models

| Model | Container | Reference |
|-------|-----------|-----------|
| SPINEPS | `go2432/spineps-preprocessing:latest` | Möller et al. *Eur Radiol* 2025 |
| TotalSpineSeg | `go2432/totalspineseg` | Warszawer et al. 2025 |
| Ian Pan disc localiser | (integrated into container) | Pan et al. |

### SLURM Resource Requirements

| Step | CPUs | Memory | GPU | Walltime |
|------|------|--------|-----|---------|
| 02b SPINEPS | 4 | 32 GB | V100 32 GB | 8h |
| 03b TotalSpineSeg | 4 | 32 GB | V100 32 GB | 8h |
| 00b Ian Pan | 4 | 16 GB | V100 32 GB | 4h |
| 04 LSTV detection | 8 | 48 GB | None | 12h |
| 06 3D visualisation | 4 | 32 GB | None | 6h |

### File Structure

```
lstv-detector/
├── scripts/
│   ├── lstv_engine.py            ← morphometric calculations (importable)
│   ├── vertebral_alignment.py    ← TSS/VERIDAH Dice offset analyser (v2.1)
│   ├── ian_pan_disc_coords.py    ← Ian Pan 2D heatmap → 3D RAS extractor
│   ├── lstv_csv_reporter.py      ← CSV output writer
│   ├── 04_detect_lstv.py         ← main classifier + Ian Pan arbitration (v4.4)
│   └── 06_visualize_3d.py        ← interactive 3D HTML renderer (v3.2)
├── slurm_scripts/
│   ├── 00b_ian_pan_disc_coords.sh
│   ├── 01_dicom_to_nifti.sh
│   ├── 02b_spineps_selective.sh
│   ├── 03b_totalspineseg_selective.sh
│   ├── 03c_register.sh
│   ├── 04_lstv_detection.sh
│   └── 06_visualize_3d.sh
└── results/
    ├── spineps/segmentations/{study_id}/
    │   ├── {id}_seg-spine_msk.nii.gz
    │   └── {id}_seg-vert_msk.nii.gz
    ├── totalspineseg/{study_id}/sagittal/
    │   └── {id}_sagittal_labeled.nii.gz
    ├── registered/{study_id}/
    ├── ian_pan_disc_coords/
    │   └── ian_pan_disc_coords.json
    ├── lstv_detection/
    │   ├── lstv_results.json
    │   ├── lstv_summary.json
    │   ├── lstv_per_study.csv
    │   └── lstv_alignment.csv
    └── lstv_3d/
        └── {study_id}_lstv_3d.html
```

### Resumability

Each step tracks progress in `progress_selective.json`. Resubmitting any SLURM script automatically skips completed studies. To force reprocessing: `RETRY_FAILED=true MODE=prod`.

---

## 15. Known Limitations

**Lumbarization count edge case**: TSS labels stop at L5 (label 45). When L6 is present, TSS will label L1–L5 of the 6-lumbar spine correctly but return count = 5. The L6 signal comes exclusively from VERIDAH label 25. If SPINEPS mis-labels L6 (e.g. as a second L5), the consensus count will be 5 and lumbarization will be missed. The cross-validation warning (`L5 centroid distance > 20mm`) will flag these cases for manual review.

**Type III over-reporting**: Phase 2 MRI Type III classification is provisional. Homogeneous T2w signal at the TP junction can occur with periosteal marrow without true cortical bridging. CT confirmation is recommended before operative planning.

**Castellvi on 4-lumbar spines**: When count = 4, the sacralizing segment is typically L4. Castellvi will be assessed on L4 TP, which is radiologically correct, but the printed label will be "L4" rather than "L5."

**DHI at L6-S1**: TSS has no disc label for the L6-S1 level. DHI at L6-S1 uses VERIDAH IVD label 125 (100 + VD_L6=25) if present. If SPINEPS does not detect the L6-S1 disc, DHI is reported as undetected (not absent).

**Ian Pan 2D→3D projection**: The `world_ras_mm` coordinate is derived from a 2D sagittal heatmap peak projected into 3D space. If this projection naively assumes image centreline for the x-coordinate, lateral disc displacement introduces systematic error in the arbitration distance calculations.

---

## 16. Potential Improvements

### Tier 1 — High Impact, Relatively Achievable

**Iliolumbar Ligament (ILL) Detection**
The ILL originates from the L5 TP in ~97% of individuals and is the single most reliable MRI anchor for level identification (Hughes & Saifuddin 2006). Detecting it on axial sequences would provide a direct anatomic ground truth independent of all three segmentation models. It would effectively replace the entire offset-model arbitration with a definitive anchor. Difficulty: **hard** — requires reliable axial segmentation and ILL-specific detection or atlas matching. Payoff: very high.

**Ian Pan L6 Extension**
Currently Ian Pan provides 5 disc levels (L1/L2–L5/S1). In lumbarization the L6/S1 disc exists. Extending the localiser to detect `l6_s1` when present would resolve the most common offset = +1 ambiguity directly — the inferred L6 body from `l5_s1` and `l6_s1` midpoints would give a concrete vote independent of voxel overlap. Difficulty: **moderate** — model fine-tuning or heuristic threshold extension on existing heatmaps.

**Nerve Root Morphology (Farshad Framework)**
Farshad et al. describe axial MRI-based classification of nerve root exit patterns at the lumbosacral junction. L5 nerve roots exit above the TP; S1 roots exit below. This is model-agnostic and could be implemented as Phase 2b following Castellvi Phase 2. Difficulty: **moderate-hard** — requires nerve root segmentation or atlas-based landmark detection on axial T2w.

**Whole-Spine Scout Counting from TSS Thoracic Labels**
TSS already labels thoracic vertebrae (labels 8–19 = T1–T12). Counting verified thoracic levels from T1 downward provides an independent route to the lumbosacral junction without relying on lumbar-level label assignment. Difficulty: **moderate** — implement caudal counting from confirmed T12 (label 19) through lumbar chain; requires rib co-verification.

**Full Integration of SPINEPS Label 126**
SPINEPS labels the S1-S2 IVD as label 126, which can only exist when L5 has partially or fully sacralized and S1 has a mobile caudal interface. This is a SPINEPS-native sacralization signal that currently fires a warning but is not fully wired into the phenotype classifier. Difficulty: **easy** — ~20 lines of integration code in `04_detect_lstv.py`.

**Confidence-Weighted Ensemble Vote**
The current decision flow is sequential (Dice offset first, Ian Pan if disagreement). A probabilistic ensemble weighting each model by empirical accuracy on a validation set would be more principled. Ian Pan weights could be modulated by per-disc `peak_prob` and `entropy`. Difficulty: **moderate** — requires a labelled validation set (even 30–40 CT-confirmed cases would be sufficient to calibrate weights).

### Tier 2 — Moderate Impact

**T1w Integration for Vertebral Body Morphometry**
H/AP ratio measurements currently use TSS masks on T2w. T1w sequences show the vertebral body–endplate interface with higher contrast (marrow T1 vs endplate cortex) and would give more accurate body morphometry, particularly at the lumbosacral junction where T2w partial-volume effects are worst. Difficulty: **moderate** — requires T1w → T2w registration.

**Pfirrmann Grading of Disc Below TV**
The pipeline uses binary disc presence and DHI. Adding Pfirrmann T2w signal grading (I–V) for the disc below the TV provides a more sensitive sacralization marker — Pfirrmann IV–V discs are functionally absent even when DHI is preserved. Difficulty: **easy** — patch-based T2w intensity classification analogous to Phase 2 already implemented.

**TP Width as Second Castellvi Geometric Criterion**
The current PCA decomposition already produces all three principal axes of the TP mask. Adding the second-axis (mediolateral) measurement costs one line. This would enable a two-criterion Type I threshold (height AND width), potentially reducing false positives from tall but narrow TPs. Difficulty: **trivial**.

**Structured Radiology Report Generation**
The pipeline produces JSON and CSV but no human-readable output. A structured report generator (LSTV phenotype, Castellvi type, disc findings, surgical risk flags) formatted as PDF or DICOM SR would enable direct clinical workflow integration. Difficulty: **moderate**.

---

## 17. Likelihood of Working: Honest Assessment

### What Almost Certainly Works

| Component | Confidence | Reasoning |
|-----------|-----------|-----------|
| TSS vertebral level labelling (L1–L5) | Very high | Published, validated on large datasets; labels 41–45 are highly reliable in standard anatomy |
| SPINEPS TP detection | High | Costal process labels 43/44 are SPINEPS's strongest output; well-validated in Möller et al. |
| Castellvi Phase 1 geometry | High | Deterministic voxel geometry on reliable masks; PCA TP height and EDT distance are robust |
| Sacrum identification | High | TSS label 50 is the most reliably segmented structure across both models |
| L6 detection via VERIDAH label 25 | Moderate-high | SPINEPS was specifically trained to detect L6; false positive rate is dataset-dependent |

### What Probably Works But Needs Empirical Validation

| Component | Confidence | Main concern |
|-----------|-----------|-------------|
| Dice offset model (offset ≠ 0 detection) | Moderate | The 8% Dice margin and 60% consistency thresholds are reasonable but unvalidated against CT ground truth. Borderline LSTV with partial fusion may produce near-identical Dice scores at adjacent offsets |
| Castellvi Phase 2 (axial T2w signal) | Moderate | The 55% × p95 dark threshold and 0.12 CV for Type II vs III are literature-grounded but not calibrated to this dataset. MRI Type III is notoriously hard to confirm without CT |
| H/AP ratio morphometry | Moderate | Accurate body shape requires clean label boundaries at the lumbosacral junction — exactly where sacrum label bleed-over is most common |
| Phenotype classifier | Moderate | Tiered logic is literature-grounded, but the interaction between count anomaly, disc findings, and morphometry has not been prospectively validated in this specific combination |

### The Known Weak Points

**The core problem**: In LSTV spines — the cases you most need to classify correctly — both TSS and VERIDAH are least reliable. They were trained predominantly on normal spines. A sacralized L5 with partial fusion may be inconsistently labelled as L5, sacrum, or have genuinely uncertain boundaries. This is exactly the geometry where the Dice offset detection is most critical and most likely to fail quietly.

**Ian Pan midpoint assumption**: The geometric inference that a vertebral body centroid lies at the midpoint of its bounding discs assumes approximately equal disc heights above and below. In LSTV cases with severe disc reduction at the lumbosacral junction, the inferred L5 body position will be pulled inferiorly, potentially biasing the score against the anatomically correct hypothesis. The 5 mm margin threshold provides some protection, but this is a genuine error source, particularly in sacralization cases where the L5-S1 disc is the one that disappears.

**2D→3D Ian Pan projection**: The quality of the `world_ras_mm` coordinate depends on how the heatmap peak is projected from 2D sagittal space into 3D RAS space. A naive projection that assumes the image centreline for the x-coordinate will systematically misplace disc centres in patients with scoliosis or lateral disc displacement.

**Registration quality**: Phase 2 requires SPINEPS to be registered into axial T2w space. Rigid registration between sagittal and axial acquisitions is imperfect at the lumbosacral junction due to patient repositioning, lordosis changes between sequences, and geometric distortion from gradient nonlinearity. A 3–5 mm registration error at the TP–sacrum interface is sufficient to flip a contact/no-contact classification.

**No ground truth**: No thresholds have been empirically optimised for this dataset and pipeline. All values (SCORE_MARGIN = 0.08, IP_VERT_MIN_MARGIN_MM = 5.0, Phase 2 signal thresholds) are literature-derived starting points that require prospective calibration.

### Estimated Accuracy (Literature-Derived, No Empirical Validation Available)

| Task | Estimated accuracy | Limiting factor |
|------|-------------------|----------------|
| Any LSTV detected | 75–85% sensitivity | Phase 1 TP height measurement quality |
| Castellvi type (given correct TV) | 70–80% | Phase 2 Type II/III signal ambiguity |
| Correct TV identification (offset model alone) | 80–90% | Dice margin in borderline LSTV |
| Ian Pan arbitration (when it fires) | ~70–80% correct decisions | Midpoint geometry + 2D→3D projection |
| Sacralization phenotype | 65–80% | H/AP morphometry at fused lumbosacral junction |
| Lumbarization phenotype | 75–85% | VERIDAH L6 detection is relatively reliable |

**Bottom line**: The architecture is sound and represents a genuinely novel approach — a three-model deep learning ensemble with geometry-grounded vertebral arbitration goes well beyond any published automated LSTV classifier. The main risks are (1) segmentation quality degradation in precisely the anomalous anatomy you are trying to characterise, (2) unvalidated decision thresholds, and (3) registration error in Phase 2. Validation against even 20–30 CT-confirmed LSTV cases would provide calibration data sufficient to tune the key thresholds and give you a meaningful sensitivity/specificity estimate for the AACA abstract and any follow-on manuscript.

---

## 18. References

1. **Castellvi AE**, Goldstein LA, Chan DPK. *Intertransverse process impingement of the superior gluteal nerve*. Spine. 1984;9(1):31–35. — Original classification; ≥19 mm TP threshold.

2. **Konin GP**, Walz DM. *Lumbosacral transitional vertebrae: classification, imaging findings, and clinical relevance*. Semin Musculoskelet Radiol. 2010;14(1):67–76. — MRI classification review; disc reduction as sacralization criterion.

3. **Nardo L** et al. *Lumbosacral transitional vertebrae: association with low back pain*. Radiology. 2012;265(2):497–503. — H/AP ratio thresholds (0.52, 0.68); transitional morphology on MRI.

4. **Hughes RJ**, Saifuddin A. *Imaging of lumbosacral transitional vertebrae*. Clin Radiol. 2004;59(11):984–991. — Lumbarization definition; L6 disc criteria.

5. **Hughes RJ**, Saifuddin A. *Numbering of lumbo-sacral transitional vertebrae on MRI: role of the iliolumbar ligament*. AJR. 2006;187(1):W59–65. — Iliolumbar ligament as level-identification anchor.

6. **Seyfert S**. *Dermatome changes after lumbosacral transitional vertebra treatment*. Neuroradiology. 1997;39(8):584–587. — L5-S1 disc loss as most reliable sacralization marker.

7. **Farfan HF** et al. *The effects of torsion on the lumbar intervertebral joints*. J Bone Joint Surg Am. 1972;54(3):492–510. — Disc Height Index methodology.

8. **Panjabi MM** et al. *Human lumbar vertebrae: quantitative three-dimensional anatomy*. Spine. 1992;17(3):299–306. — Normal H/AP ratios (L3=0.82, L4=0.78, L5=0.72).

9. **Quinlan JF**, Duke D, Eustace S. *Bertolotti's syndrome: a cause of back pain in young people*. J Bone Joint Surg Br. 2006;88(9):1183–1186. — Castellvi Type I clinical significance.

10. **Carrino JA** et al. *Effect of spinal segment variants on numbering of lumbar vertebrae*. Radiology. 2011;259(1):196–202. — 30% error rate without whole-spine imaging.

11. **Nidecker AE** et al. *Sacral transitional vertebra and L5 sacralization: considerations for lumbar spine surgery*. Eur Radiol. 2018;28(4):1376–1383. — MRI Phase 2 T2w signal classification.

12. **Farshad-Amacker NA** et al. *MR imaging of the intervertebral disc*. Eur Spine J. 2014;23(Suppl 3):S386–395. — Disc signal and DHI at transitional levels.

13. **Möller H** et al. *SPINEPS — automatic whole spine segmentation of T2-weighted MR images*. Eur Radiol. 2025. doi:10.1007/s00330-024-11155-y

14. **Warszawer Y** et al. *TotalSpineSeg: Robust spine segmentation and landmark labeling in MRI*. arXiv:2411.09344. 2025.

15. **Seilanian Toosi F** et al. *Angle-based MRI classification of lumbosacral transitional vertebrae*. 2025. — Vertebral angle metrics (A, B, C, D, D1, δ) for future integration.

---

*Contact: go2432@wayne.edu — Wayne State University School of Medicine, Neurosurgery / Spine Imaging & AI*
