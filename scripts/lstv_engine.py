#!/usr/bin/env python3
"""
lstv_engine.py — LSTV Morphometrics Engine (Radiologically Grounded)
======================================================================
Single-responsibility module for all measurements needed to classify
Lumbosacral Transitional Vertebrae (LSTV).

RADIOLOGIC DEFINITION OF LSTV
------------------------------
An LSTV is a congenital spinal anomaly in which the last mobile lumbar
vertebra (the "transitional vertebra," TV) displays morphologic features
intermediate between a lumbar and a sacral segment, resulting in either
a lumbar-numbered vertebra acquiring sacral characteristics (sacralization)
or a sacral segment acquiring lumbar mobility (lumbarization).

Prevalence: 4–36% of the population depending on imaging modality and
counting methodology (Konin & Walz, Radiology 2010; Nardo et al., Radiology
2012; Hughes & Saifuddin, Skeletal Radiol 2006).

SACRALIZATION — Radiologic Criteria
-------------------------------------
Sacralization = L5 (or occasionally L4) fuses or partially fuses with the
sacrum. Diagnostic criteria derived from:
  • Castellvi et al. (1984): TP ≥ 19 mm CC height; contact with sacrum
  • Nardo et al. (2012): TV body H/AP ratio < 0.68 → transitional/sacral
  • Konin & Walz (2010): disc space L5-S1 reduced/absent (DHI < 50%)
  • Farshad-Amacker et al. (2014): TV disc signal loss / height reduction
  • Seyfert (1997): loss of normal L5-S1 disc space is the most reliable sign
  • Quinlan et al. (1984): unilateral > bilateral (functional significance)

  PRIMARY criteria (ANY ONE sufficient for sacralization flag):
    ① Castellvi Type I-IV (transverse process enlargement / fusion)
    ② TV disc below severely reduced or absent (DHI < 50%)
    ③ TV body H/AP ratio in transitional or sacral-like range AND 4-lumbar count
    ④ 4-lumbar count confirmed by BOTH TSS and VERIDAH (rare, high specificity)

  SUPPORTING criteria (increase confidence):
    ⑤ TV/L4 H:AP normalised ratio < 0.80 (TV squarer than L4)
    ⑥ TV disc above normal (L4-L5 preserved → pathology localised to L5-S1)

LUMBARIZATION — Radiologic Criteria
--------------------------------------
Lumbarization = S1 acquires lumbar characteristics — it separates from the
sacrum and develops a mobile disc, normal lumbar morphology, and resembles L6.
Criteria:
  • Hughes & Saifuddin (2006): L6 label; disc present below TV
  • Nardo et al. (2012): TV H/AP ≥ 0.68 (lumbar-like)
  • Konin & Walz (2010): disc preserved below TV (DHI ≥ 80%)
  • Castellvi classification may also be present (TP enlargement of L6)

  PRIMARY criteria:
    ① 6-lumbar count (L6 detected by VERIDAH label 25)
    ② TV is the lowest segment above a non-fused disc (L6-S1 disc present)
    ③ TV body H/AP ≥ 0.68 (lumbar-like body morphology)

  SUPPORTING criteria:
    ④ Castellvi positive on L6 TP (co-occurring pathology)
    ⑤ TV/L4 H:AP ratio ≥ 0.90 (L6 body resembles L4/L5)

IMPORTANT: Sacralization and lumbarization are NOT exclusive of Castellvi
classification. A lumbarized S1 (L6) may have TP enlargement (Castellvi I-IV).
A sacralizing L5 ALWAYS has Castellvi classification if TP is involved.
The phenotype and Castellvi classifications are orthogonal axes.

CASTELLVI CLASSIFICATION (Castellvi et al. 1984, Spine 9(1):31–35)
----------------------------------------------------------------------
Type I   : Dysplastic TP ≥ 19 mm CC height, no sacral contact
           Ia = unilateral; Ib = bilateral
Type II  : Diarthrodial (pseudo-articular) joint between TP and sacrum
           IIa = unilateral; IIb = bilateral
Type III : Complete bony fusion of TP and sacrum
           IIIa = unilateral; IIIb = bilateral
Type IV  : Mixed — Type II one side, Type III the other

MRI adaptation (Konin & Walz 2010; Nidecker et al. 2018):
  Type II on MRI: Heterogeneous / dark T2w signal at TP–sacrum junction
                  (fibrocartilaginous pseudo-joint; synovial fluid)
  Type III on MRI: Homogeneous bright T2w signal (marrow continuity)
  Note: CT remains gold standard for Type III confirmation.

DISC HEIGHT INDEX (DHI)
------------------------
Farfan et al. (1972), J Bone Joint Surg: DHI = disc height / mean adjacent
vertebral body height × 100.
Normal lumbar: 80–100 %
Mild reduction: 70–80 %
Moderate reduction: 50–70 %
Severely reduced / absent: < 50 % — most reliable sacralization marker
                                     (Seyfert 1997; Quinlan et al. 1984)

TV BODY MORPHOLOGY
-------------------
H/AP ratio (SI-height / AP-depth), from Nardo et al. (2012) and
Panjabi et al. (1992, Spine):
  > 0.68 → lumbar-like (normal lumbar: L3=0.82, L4=0.78, L5=0.72)
  0.52–0.68 → transitional morphology
  < 0.52 → sacral-like

LABEL REFERENCE
---------------
TotalSpineSeg sagittal step2_output:
  Vertebrae : 11-17=C1-C7  21-32=T1-T12  41-45=L1-L5  50=Sacrum
  Discs     : 91=T12-L1  92=L1-L2  93=L2-L3  94=L3-L4  95=L4-L5  100=L5-S1

VERIDAH (SPINEPS seg-vert_msk):
  20=L1  21=L2  22=L3  23=L4  24=L5  25=L6  26=Sacrum
  100+X = IVD below vertebra X    (per-level disc)
  200+X = Endplate of vertebra X  (per-level endplate)

SPINEPS seg-spine_msk (subregion):
  43=Costal_Process_Left  44=Costal_Process_Right  26=Sacrum
  41=Arcus  42=Spinous  45=Sup_Art_L  46=Sup_Art_R  49=Corpus_Border
  60=Spinal_Cord  61=Spinal_Canal

REFERENCES
----------
Castellvi AE et al. Spine. 1984;9(1):31–35.
Konin GP & Walz DM. Semin Musculoskelet Radiol. 2010;14(1):67–76.
Nardo L et al. Radiology. 2012;265(2):497–503.
Hughes RJ & Saifuddin A. Skeletal Radiol. 2006;35(5):299–316.
Farshad-Amacker NA et al. Eur Spine J. 2014;23(2):396–402.
Seyfert S. Neuroradiology. 1997;39(8):584–587.
Quinlan JF et al. J Bone Joint Surg Br. 1984;66(4):556–558.
Farfan HF et al. J Bone Joint Surg Am. 1972;54(3):492–510.
Panjabi MM et al. Spine. 1992;17(3):299–306.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import label as cc_label, zoom as ndizoom

logger = logging.getLogger(__name__)

# ── Isotropic resampling target ───────────────────────────────────────────────
ISO_MM = 1.0

# ── TotalSpineSeg label tables ────────────────────────────────────────────────
TSS_CORD    = 1
TSS_CANAL   = 2
TSS_CERVICAL  : Dict[int, str] = {11:'C1',12:'C2',13:'C3',14:'C4',15:'C5',16:'C6',17:'C7'}
TSS_THORACIC  : Dict[int, str] = {
    21:'T1',22:'T2',23:'T3',24:'T4',25:'T5',26:'T6',
    27:'T7',28:'T8',29:'T9',30:'T10',31:'T11',32:'T12',
}
TSS_LUMBAR    : Dict[int, str] = {41:'L1',42:'L2',43:'L3',44:'L4',45:'L5'}
TSS_SACRUM    = 50
TSS_DISCS     : Dict[int, str] = {
    91:'T12-L1', 92:'L1-L2', 93:'L2-L3', 94:'L3-L4',
    95:'L4-L5',  100:'L5-S1',
}

# ── VERIDAH label tables ──────────────────────────────────────────────────────
VD_L1=20; VD_L2=21; VD_L3=22; VD_L4=23; VD_L5=24; VD_L6=25; VD_SAC=26
VD_IVD_BASE = 100
VD_EP_BASE  = 200
VERIDAH_LUMBAR  : Dict[int, str] = {20:'L1',21:'L2',22:'L3',23:'L4',24:'L5',25:'L6'}
VERIDAH_NAMES   : Dict[int, str] = {20:'L1',21:'L2',22:'L3',23:'L4',24:'L5',25:'L6',26:'Sacrum'}
# VERIDAH preferred TV search order — L6 first so it is identified if present
VERIDAH_TV_SEARCH = [25, 24, 23, 22, 21, 20]

# TSS vert label matching VERIDAH TV label
VD_TO_TSS_VERT  : Dict[int, int] = {20:41, 21:42, 22:43, 23:44, 24:45}  # no L6

# ── SPINEPS subregion labels ──────────────────────────────────────────────────
SP_TP_L    = 43;  SP_TP_R   = 44
SP_SACRUM  = 26
SP_ARCUS   = 41;  SP_SPINOUS = 42
SP_SAL     = 45;  SP_SAR    = 46
SP_CORPUS  = 49
SP_CORD    = 60;  SP_CANAL  = 61

# ── Morphology thresholds — all literature-derived ───────────────────────────
TP_HEIGHT_MM      = 19.0    # Castellvi 1984: Type I TP craniocaudal height ≥ 19 mm
CONTACT_DIST_MM   = 2.0     # Phase 1 contact criterion (mm)

# TV body shape thresholds (Nardo et al. 2012; Panjabi et al. 1992)
TV_SHAPE_LUMBAR   = 0.68    # H/AP > 0.68 → lumbar-like
TV_SHAPE_SACRAL   = 0.52    # H/AP < 0.52 → sacral-like

# Disc Height Index thresholds (Farfan 1972; Seyfert 1997)
DHI_NORMAL_PCT    = 80.0    # ≥ 80 % → normal
DHI_MILD_PCT      = 80.0    # < 80 % → mild reduction
DHI_MODERATE_PCT  = 70.0    # < 70 % → moderate reduction
DHI_REDUCED_PCT   = 50.0    # < 50 % → severely reduced / absent — PRIMARY sacralization criterion

EXPECTED_LUMBAR   = 5
EXPECTED_THORACIC = 12


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class LSTVMaskSet:
    """Isotropically resampled masks for one study."""
    study_id:    str
    sp_iso:      np.ndarray            # SPINEPS seg-spine_msk  @ 1 mm
    vert_iso:    np.ndarray            # VERIDAH seg-vert_msk   @ 1 mm
    tss_iso:     Optional[np.ndarray]  # TotalSpineSeg sagittal @ 1 mm (may be None)
    sp_labels:   frozenset
    vert_labels: frozenset
    tss_labels:  frozenset


@dataclass
class DiscMetrics:
    level:       str
    height_mm:   Optional[float] = None
    vert_sup_h:  Optional[float] = None   # superior adjacent vertebra SI height
    vert_inf_h:  Optional[float] = None   # inferior adjacent vertebra SI height
    dhi_pct:     Optional[float] = None   # disc height index (%)
    grade:       Optional[str]   = None   # Normal / Mild / Moderate / Severely Reduced
    source:      Optional[str]   = None   # TSS / VERIDAH / none
    is_absent:   bool            = False  # disc absent (fused) — primary sacralization criterion


@dataclass
class TVBodyShape:
    """
    Morphometric shape analysis of the transitional vertebra body.
    Reference: Nardo et al. Radiology 2012; Panjabi et al. Spine 1992.
    """
    h_mm:          Optional[float] = None  # craniocaudal height (SI axis)
    ap_mm:         Optional[float] = None  # anteroposterior depth
    ml_mm:         Optional[float] = None  # mediolateral width
    h_ap_ratio:    Optional[float] = None  # SI/AP — main discriminator (Nardo 2012)
    h_ml_ratio:    Optional[float] = None
    shape_class:   Optional[str]   = None  # lumbar-like / transitional / sacral-like
    ref_l4_h_ap:   Optional[float] = None  # L4 H/AP for normalisation
    norm_ratio:    Optional[float] = None  # tv_h_ap / l4_h_ap
    source:        Optional[str]   = None  # TSS / VERIDAH


@dataclass
class RibAnomalyResult:
    thoracic_count:      Optional[int]  = None
    expected_thoracic:   int            = EXPECTED_THORACIC
    count_anomaly:       bool           = False
    count_description:   Optional[str]  = None
    lumbar_rib_l1:       bool           = False
    lumbar_rib_l1_h_mm:  Optional[float]= None
    any_anomaly:         bool           = False
    description:         Optional[str]  = None


@dataclass
class LSTVMorphometrics:
    """
    Complete LSTV morphometric result for one study.

    lstv_phenotype is independent of castellvi_type — both may be set
    simultaneously, because Castellvi classifies TP morphology while
    phenotype classifies the overall lumbosacral transition pattern.
    A lumbarized S1 (L6) may carry Castellvi I-IV on its TP;
    a sacralizing L5 always has Castellvi classification if TP is involved.
    """
    study_id: str
    error:    Optional[str] = None

    # ── Lumbar count ──────────────────────────────────────────────────────────
    lumbar_count_tss:       Optional[int] = None
    lumbar_count_veridah:   Optional[int] = None
    lumbar_count_consensus: Optional[int] = None
    lumbar_count_anomaly:   bool          = False
    lumbar_count_note:      Optional[str] = None

    # ── TV identification ─────────────────────────────────────────────────────
    tv_label_veridah:  Optional[int] = None
    tv_name:           Optional[str] = None
    tv_tss_label:      Optional[int] = None
    has_l6:            bool          = False

    # ── TV body shape ─────────────────────────────────────────────────────────
    tv_shape: Optional[TVBodyShape] = None

    # ── Adjacent disc metrics ─────────────────────────────────────────────────
    disc_above: Optional[DiscMetrics] = None
    disc_below: Optional[DiscMetrics] = None

    # ── Rib anomaly ───────────────────────────────────────────────────────────
    rib_anomaly: Optional[RibAnomalyResult] = None

    # ── LSTV phenotype — INDEPENDENT of Castellvi ─────────────────────────────
    # phenotype classifies the overall lumbosacral transition pattern
    # castellvi classifies TP morphology; both may be positive simultaneously
    lstv_phenotype:       Optional[str] = None
    # sacralization / lumbarization / transitional_indeterminate / normal
    phenotype_confidence:  Optional[str] = None   # high / moderate / low
    phenotype_criteria:    List[str]     = field(default_factory=list)
    phenotype_rationale:   Optional[str] = None

    # ── Evidence summary (for downstream reporting) ────────────────────────────
    primary_criteria_met:  List[str]     = field(default_factory=list)
    # Which primary radiologic criteria triggered the phenotype

    def to_dict(self) -> dict:
        return asdict(self)


# ── NIfTI helpers ──────────────────────────────────────────────────────────────

def _load_canonical(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    nii  = nib.load(str(path))
    nii  = nib.as_closest_canonical(nii)
    data = nii.get_fdata()
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{path.name}: cannot reduce to 3D (shape={data.shape})")
    return data, nii


def _voxel_mm(nii: nib.Nifti1Image) -> np.ndarray:
    return np.abs(np.array(nii.header.get_zooms()[:3], dtype=float))


def _resample(vol: np.ndarray, vox_mm: np.ndarray, target: float = ISO_MM) -> np.ndarray:
    factors = (vox_mm / target).tolist()
    return ndizoom(vol.astype(np.int32), factors,
                   order=0, mode='nearest', prefilter=False).astype(np.int32)


# ── Mask loading ───────────────────────────────────────────────────────────────

def load_lstv_masks(study_id: str,
                    spineps_dir: Path,
                    totalspine_dir: Path) -> LSTVMaskSet:
    """Load and resample all masks needed for LSTV analysis."""
    seg_dir   = spineps_dir / 'segmentations' / study_id
    sp_path   = seg_dir / f"{study_id}_seg-spine_msk.nii.gz"
    vert_path = seg_dir / f"{study_id}_seg-vert_msk.nii.gz"
    tss_path  = (totalspine_dir / study_id / 'sagittal'
                 / f"{study_id}_sagittal_labeled.nii.gz")

    sp_raw, sp_nii  = _load_canonical(sp_path)
    vert_raw, _     = _load_canonical(vert_path)
    vox_mm          = _voxel_mm(sp_nii)

    sp_iso   = _resample(sp_raw.astype(np.int32),   vox_mm)
    vert_iso = _resample(vert_raw.astype(np.int32), vox_mm)

    tss_iso = None
    if tss_path.exists():
        try:
            tss_raw, tss_nii = _load_canonical(tss_path)
            tss_vox          = _voxel_mm(tss_nii)
            tss_iso          = _resample(tss_raw.astype(np.int32), tss_vox)
        except Exception as exc:
            logger.warning(f"[{study_id}] TSS load failed: {exc}")

    return LSTVMaskSet(
        study_id    = study_id,
        sp_iso      = sp_iso,
        vert_iso    = vert_iso,
        tss_iso     = tss_iso,
        sp_labels   = frozenset(np.unique(sp_iso).tolist())   - {0},
        vert_labels = frozenset(np.unique(vert_iso).tolist()) - {0},
        tss_labels  = (frozenset(np.unique(tss_iso).tolist()) - {0}
                       if tss_iso is not None else frozenset()),
    )


# ── Geometry primitives ────────────────────────────────────────────────────────

def _si_height(mask: np.ndarray) -> Optional[float]:
    """Craniocaudal (Z-axis) extent of a binary mask in mm."""
    if not mask.any(): return None
    zc = np.where(mask)[2]
    return float((int(zc.max()) - int(zc.min()) + 1) * ISO_MM)


def _ap_depth(mask: np.ndarray) -> Optional[float]:
    """Anteroposterior (Y-axis) extent in mm."""
    if not mask.any(): return None
    yc = np.where(mask)[1]
    return float((int(yc.max()) - int(yc.min()) + 1) * ISO_MM)


def _ml_width(mask: np.ndarray) -> Optional[float]:
    """Mediolateral (X-axis) extent in mm."""
    if not mask.any(): return None
    xc = np.where(mask)[0]
    return float((int(xc.max()) - int(xc.min()) + 1) * ISO_MM)


# ── Lumbar count ───────────────────────────────────────────────────────────────

def count_lumbar_tss(tss_iso: np.ndarray, tss_labels: frozenset) -> Tuple[int, List[str]]:
    """
    Count lumbar vertebrae from TotalSpineSeg labels (41=L1 … 45=L5).
    TSS is authoritative for L1-L5 — good boundary quality, well-validated.
    Returns (count, list_of_detected_names).
    """
    detected = [name for lbl, name in TSS_LUMBAR.items() if lbl in tss_labels]
    return len(detected), detected


def detect_l6_veridah(vert_labels: frozenset) -> bool:
    """
    Return True if VERIDAH (SPINEPS seg-vert_msk) label 25 is present.
    Label 25 = L6 — only VERIDAH can detect this since TSS has no L6 label.
    This is the primary segmentation indicator of lumbarization.
    """
    return VD_L6 in vert_labels


def count_lumbar_veridah(vert_labels: frozenset) -> Tuple[int, List[str]]:
    """Count lumbar vertebrae from VERIDAH labels (20=L1 … 25=L6)."""
    detected = [name for lbl, name in VERIDAH_LUMBAR.items() if lbl in vert_labels]
    return len(detected), detected


def reconcile_lumbar_count(tss_count: int,
                            veridah_count: int,
                            tss_names: List[str],
                            veridah_names: List[str],
                            vert_labels: frozenset) -> Tuple[int, str]:
    """
    Reconcile lumbar count with TSS as primary source for L1-L5.

    Strategy:
    - TSS (41-45) is authoritative for L1-L5 count
    - VERIDAH label 25 (L6) is the ONLY basis for count > 5
    - If VERIDAH detects L6 (label 25 present), consensus = tss_count + 1
    - If tss_count < 5 and VERIDAH corroborates, likely 4-lumbar spine
    - VERIDAH > TSS without L6 label → likely over-segmentation, trust TSS
    """
    has_l6_veridah = detect_l6_veridah(vert_labels)

    if has_l6_veridah:
        consensus = tss_count + 1
        note = (f"TSS={tss_count} (L1-L5) + VERIDAH L6 label 25 confirmed "
                f"→ consensus={consensus} lumbar vertebrae — LUMBARIZATION indicator")
        return consensus, note

    if tss_count == veridah_count:
        return tss_count, f"TSS={tss_count}, VERIDAH={veridah_count} — consistent"

    if tss_count < veridah_count:
        note = (f"TSS={tss_count} < VERIDAH={veridah_count} but no L6 label; "
                f"TSS trusted (VERIDAH over-segmentation likely)")
        return tss_count, note

    note = (f"TSS={tss_count} > VERIDAH={veridah_count}; "
            f"TSS trusted (detected: {', '.join(tss_names)})")
    return tss_count, note


# ── TV body shape analysis ─────────────────────────────────────────────────────

def _vert_shape(iso: np.ndarray, vert_label: int,
                source: str) -> Optional[TVBodyShape]:
    mask = (iso == vert_label)
    if not mask.any(): return None
    h = _si_height(mask); ap = _ap_depth(mask); ml = _ml_width(mask)
    if h is None or ap is None or ap == 0: return None
    h_ap  = h / ap
    h_ml  = (h / ml) if (ml and ml > 0) else None
    shape = ('lumbar-like' if h_ap > TV_SHAPE_LUMBAR
             else 'sacral-like' if h_ap < TV_SHAPE_SACRAL
             else 'transitional')
    return TVBodyShape(h_mm=h, ap_mm=ap, ml_mm=ml,
                       h_ap_ratio=round(h_ap, 3),
                       h_ml_ratio=round(h_ml, 3) if h_ml else None,
                       shape_class=shape, source=source)


def analyze_tv_body_shape(masks: LSTVMaskSet,
                           tv_veridah_label: int,
                           tv_tss_label: Optional[int]) -> TVBodyShape:
    """
    Analyze TV body morphology (Nardo et al. 2012; Panjabi et al. 1992).
    TSS preferred; falls back to VERIDAH. Normalises against L4 H/AP.
    """
    shape: Optional[TVBodyShape] = None
    if tv_tss_label is not None and masks.tss_iso is not None:
        shape = _vert_shape(masks.tss_iso, tv_tss_label, 'TSS')
    if shape is None:
        shape = _vert_shape(masks.vert_iso, tv_veridah_label, 'VERIDAH')
    if shape is None:
        return TVBodyShape()

    # Normalise against L4 (stable reference)
    l4_shape: Optional[TVBodyShape] = None
    if masks.tss_iso is not None and 44 in masks.tss_labels:
        l4_shape = _vert_shape(masks.tss_iso, 44, 'TSS')
    if l4_shape is None and VD_L4 in masks.vert_labels:
        l4_shape = _vert_shape(masks.vert_iso, VD_L4, 'VERIDAH')
    if l4_shape and l4_shape.h_ap_ratio and shape.h_ap_ratio:
        shape.ref_l4_h_ap = l4_shape.h_ap_ratio
        shape.norm_ratio  = round(shape.h_ap_ratio / l4_shape.h_ap_ratio, 3)

    return shape


# ── Disc height metrics ────────────────────────────────────────────────────────

def _disc_height_mm(iso: np.ndarray, label: int) -> Optional[float]:
    return _si_height(iso == label)


def _vert_si_height(iso: np.ndarray, label: int) -> Optional[float]:
    return _si_height(iso == label)


def _disc_grade(dhi: Optional[float]) -> Optional[str]:
    if dhi is None: return None
    if dhi >= DHI_MILD_PCT:    return 'Normal'
    if dhi >= DHI_MODERATE_PCT: return 'Mild reduction'
    if dhi >= DHI_REDUCED_PCT:  return 'Moderate reduction'
    return 'Severely reduced / absent'


def measure_disc_metrics(masks: LSTVMaskSet,
                          disc_label_tss: Optional[int],
                          sup_tss_label: Optional[int],
                          inf_tss_label: Optional[int],
                          sup_vd_label:  Optional[int],
                          inf_vd_label:  Optional[int],
                          level_name: str) -> DiscMetrics:
    """
    Measure disc height and DHI (Farfan 1972 method).
    DHI = disc_height / mean(sup_vert_height, inf_vert_height) × 100
    """
    dm     = DiscMetrics(level=level_name)
    disc_h = None; sup_h = None; inf_h = None

    # TSS disc + vertebrae
    if (disc_label_tss is not None and masks.tss_iso is not None
            and disc_label_tss in masks.tss_labels):
        disc_h    = _disc_height_mm(masks.tss_iso, disc_label_tss)
        dm.source = 'TSS'
        if sup_tss_label and sup_tss_label in masks.tss_labels:
            sup_h = _vert_si_height(masks.tss_iso, sup_tss_label)
        if inf_tss_label and inf_tss_label in masks.tss_labels:
            inf_h = _vert_si_height(masks.tss_iso, inf_tss_label)

    # Fall back to VERIDAH IVD label
    if disc_h is None and sup_vd_label is not None:
        vd_disc_lbl = VD_IVD_BASE + sup_vd_label
        if vd_disc_lbl in masks.vert_labels:
            disc_h    = _disc_height_mm(masks.vert_iso, vd_disc_lbl)
            dm.source = 'VERIDAH'
        if sup_vd_label in masks.vert_labels:
            sup_h = _vert_si_height(masks.vert_iso, sup_vd_label)
        if inf_vd_label and inf_vd_label in masks.vert_labels:
            inf_h = _vert_si_height(masks.vert_iso, inf_vd_label)

    dm.height_mm  = round(disc_h, 2) if disc_h else None
    dm.vert_sup_h = round(sup_h, 2)  if sup_h  else None
    dm.vert_inf_h = round(inf_h, 2)  if inf_h  else None
    dm.is_absent  = (disc_h is None or disc_h == 0.0)

    denom_heights = [h for h in (sup_h, inf_h) if h and h > 0]
    if disc_h and denom_heights:
        ref_h      = float(np.mean(denom_heights))
        dhi        = (disc_h / ref_h) * 100.0
        dm.dhi_pct = round(dhi, 1)
        dm.grade   = _disc_grade(dhi)

    return dm


def get_tv_adjacent_discs(masks: LSTVMaskSet,
                           tv_veridah_label: int,
                           tv_tss_label: Optional[int]) -> Tuple[DiscMetrics, DiscMetrics]:
    """Return (disc_above_TV, disc_below_TV) DiscMetrics objects."""
    if tv_veridah_label == VD_L5:
        disc_above = measure_disc_metrics(masks, 95, 44, 45, VD_L4, VD_L5, 'L4-L5')
        disc_below = measure_disc_metrics(masks, 100, 45, TSS_SACRUM, VD_L5, VD_SAC, 'L5-S1')
    elif tv_veridah_label == VD_L4:
        disc_above = measure_disc_metrics(masks, 94, 43, 44, VD_L3, VD_L4, 'L3-L4')
        disc_below = measure_disc_metrics(masks, 95, 44, 45, VD_L4, VD_L5, 'L4-L5')
    elif tv_veridah_label == VD_L6:
        # L6 discs are not labeled by TSS — rely on VERIDAH IVDs
        disc_above = measure_disc_metrics(masks, None, None, None, VD_L5, VD_L6, 'L5-L6')
        disc_below = measure_disc_metrics(masks, None, None, None, VD_L6, VD_SAC, 'L6-S1')
    else:
        tv_name    = VERIDAH_NAMES.get(tv_veridah_label, str(tv_veridah_label))
        disc_above = DiscMetrics(level=f'above-{tv_name}')
        disc_below = DiscMetrics(level=f'below-{tv_name}')

    return disc_above, disc_below


# ── Rib anomaly detection ──────────────────────────────────────────────────────

def detect_rib_anomaly(masks: LSTVMaskSet) -> RibAnomalyResult:
    """Detect thoracic count anomalies and lumbar rib (elongated L1 costal process ≥19mm)."""
    result = RibAnomalyResult()

    if masks.tss_iso is not None:
        thr_present = [lbl for lbl in TSS_THORACIC if lbl in masks.tss_labels]
        result.thoracic_count = len(thr_present)
        if result.thoracic_count != EXPECTED_THORACIC:
            result.count_anomaly = True
            delta = result.thoracic_count - EXPECTED_THORACIC
            if delta < 0:
                result.count_description = (
                    f"Only {result.thoracic_count} thoracic vertebrae detected "
                    f"(expected {EXPECTED_THORACIC}); possible missing rib or T12 fusion")
            else:
                result.count_description = (
                    f"{result.thoracic_count} thoracic vertebrae detected "
                    f"(expected {EXPECTED_THORACIC}); possible supernumerary rib")

    if SP_TP_L in masks.sp_labels or SP_TP_R in masks.sp_labels:
        l1_mask = (masks.vert_iso == VD_L1) if VD_L1 in masks.vert_labels else None
        if l1_mask is not None and l1_mask.any():
            zc   = np.where(l1_mask)[2]
            z_lo = max(0, int(zc.min()) - 5)
            z_hi = min(masks.sp_iso.shape[2] - 1, int(zc.max()) + 5)
            max_tp_h = 0.0
            for tp_lbl in (SP_TP_L, SP_TP_R):
                if tp_lbl not in masks.sp_labels: continue
                tp_full  = (masks.sp_iso == tp_lbl)
                tp_at_l1 = np.zeros_like(tp_full)
                tp_at_l1[:, :, z_lo:z_hi + 1] = tp_full[:, :, z_lo:z_hi + 1]
                if tp_at_l1.any():
                    h = _si_height(tp_at_l1)
                    if h and h > max_tp_h: max_tp_h = h
            if max_tp_h > 0:
                result.lumbar_rib_l1_h_mm = round(max_tp_h, 1)
                if max_tp_h >= TP_HEIGHT_MM:
                    result.lumbar_rib_l1 = True

    result.any_anomaly = result.count_anomaly or result.lumbar_rib_l1
    parts = []
    if result.count_description: parts.append(result.count_description)
    if result.lumbar_rib_l1:
        parts.append(
            f"Suspected lumbar rib at L1 (L1 costal process height="
            f"{result.lumbar_rib_l1_h_mm:.1f}mm ≥ {TP_HEIGHT_MM}mm)")
    result.description = '; '.join(parts) if parts else None
    return result


# ── LSTV phenotype classification (multi-criteria, literature-grounded) ─────────

def classify_lstv_phenotype(
        lumbar_count:   int,
        tv_name:        Optional[str],
        castellvi_type: Optional[str],
        tv_shape:       Optional[TVBodyShape],
        disc_above:     Optional[DiscMetrics],
        disc_below:     Optional[DiscMetrics],
) -> Tuple[str, str, List[str], str, List[str]]:
    """
    Classify LSTV phenotype using radiologically-grounded multi-criteria approach.

    IMPORTANT: Castellvi classification and phenotype are ORTHOGONAL.
    A study may have BOTH a Castellvi type AND a lumbarization/sacralization
    phenotype. This function classifies the overall lumbosacral transition
    pattern, not the TP morphology.

    Classification is based on PRIMARY criteria (any one sufficient to trigger
    phenotype classification) and SUPPORTING criteria (increase confidence).

    PRIMARY criteria for SACRALIZATION (Castellvi 1984; Seyfert 1997; Konin 2010):
      S1: Castellvi Type I-IV detected
      S2: TV disc below absent or severely reduced (DHI < 50%) — most reliable
      S3: 4-lumbar count confirmed by both TSS and VERIDAH (rare, high specificity)
      S4: TV body sacral-like H/AP < 0.52 with ANY corroborating finding

    PRIMARY criteria for LUMBARIZATION (Hughes & Saifuddin 2006; Konin 2010):
      L1: 6-lumbar count (VERIDAH L6 label 25 present)
      L2: TV is L6 AND disc preserved below TV (L6-S1, DHI ≥ 50%)
      L3: TV body lumbar-like H/AP ≥ 0.68 with 6-lumbar count

    Castellvi type may co-occur with EITHER phenotype — it is always reported
    separately in the output dict and the phenotype does not suppress it.

    Returns (phenotype, confidence, criteria, rationale, primary_criteria_met)
    """
    criteria:      List[str] = []
    primary:       List[str] = []  # which primary criteria were met
    sac_score  = 0
    lumb_score = 0

    has_castellvi = bool(castellvi_type and castellvi_type not in ('None', 'N/A', None))

    # ── PRIMARY S1: Castellvi (TP morphology — sacralization signal) ───────────
    # Castellvi indicates TP interaction with sacrum, which is definitionally
    # a sacralization-spectrum finding (L5 TP approaching/contacting sacrum)
    if has_castellvi:
        criteria.append(f"S1 ✓ Castellvi {castellvi_type} — TP enlargement / sacral contact")
        primary.append(f"S1:Castellvi-{castellvi_type}")
        sac_score += 3
        if any(x in castellvi_type for x in ('III', 'IV')): sac_score += 2
        elif 'II' in castellvi_type: sac_score += 1

    # ── PRIMARY L1 / S3: Lumbar count anomaly ─────────────────────────────────
    if lumbar_count == 6:
        criteria.append(
            f"L1 ✓ 6-lumbar count (L6 present, VERIDAH label 25) — "
            f"PRIMARY lumbarization indicator (Hughes & Saifuddin 2006)")
        primary.append("L1:6-lumbar-count")
        lumb_score += 5  # highest weight — most specific lumbarization signal
    elif lumbar_count == 4:
        criteria.append(
            f"S3 ✓ 4-lumbar count — "
            f"PRIMARY sacralization indicator (L5 incorporated into sacrum)")
        primary.append("S3:4-lumbar-count")
        sac_score += 5
    else:
        criteria.append(f"Lumbar count = {lumbar_count} (normal)")

    # ── TV identification ──────────────────────────────────────────────────────
    if tv_name == 'L6':
        criteria.append(f"L3 ✓ TV = L6 (extra lumbar segment, lumbarization morphology)")
        primary.append("L3:TV-is-L6")
        lumb_score += 3
    elif tv_name == 'L5':
        criteria.append(f"TV = L5 (standard lowest lumbar — common LSTV site)")

    # ── TV body shape (Nardo 2012; Panjabi 1992) ───────────────────────────────
    if tv_shape and tv_shape.h_ap_ratio:
        ratio_str = f"H/AP={tv_shape.h_ap_ratio:.2f}"
        ref_str   = f"(ref Nardo 2012: lumbar >0.68, transitional 0.52–0.68, sacral <0.52)"
        if tv_shape.shape_class == 'sacral-like':
            criteria.append(
                f"TV body sacral-like morphology — {ratio_str} < {TV_SHAPE_SACRAL} {ref_str}")
            sac_score += 2
            if not has_castellvi and lumbar_count == 5:
                primary.append(f"S4:sacral-like-body")
        elif tv_shape.shape_class == 'transitional':
            criteria.append(
                f"TV body transitional morphology — {ratio_str}, range "
                f"{TV_SHAPE_SACRAL}–{TV_SHAPE_LUMBAR} {ref_str}")
            sac_score += 1
        else:
            criteria.append(
                f"TV body lumbar-like morphology — {ratio_str} ≥ {TV_SHAPE_LUMBAR} {ref_str}")
            lumb_score += 2

        if tv_shape.norm_ratio:
            if tv_shape.norm_ratio < 0.80:
                criteria.append(
                    f"TV/L4 H:AP ratio={tv_shape.norm_ratio:.2f} — TV notably squarer "
                    f"than L4 (supporting sacralization)")
                sac_score += 1

    # ── PRIMARY S2 / L2: Disc below TV (Seyfert 1997; Farfan 1972) ────────────
    if disc_below and disc_below.dhi_pct is not None:
        dhi   = disc_below.dhi_pct
        level = disc_below.level
        if dhi < DHI_REDUCED_PCT:
            criteria.append(
                f"S2 ✓ Disc BELOW TV ({level}) severely reduced — DHI={dhi:.0f}% "
                f"< {DHI_REDUCED_PCT}% — PRIMARY sacralization criterion (Seyfert 1997; "
                f"Farfan 1972). Disc height loss at L5-S1 is the most reliable "
                f"radiologic sign of sacralization.")
            primary.append(f"S2:disc-below-DHI-{dhi:.0f}pct")
            sac_score += 4
        elif dhi < DHI_MODERATE_PCT:
            criteria.append(
                f"Disc BELOW TV ({level}) moderately reduced — DHI={dhi:.0f}% "
                f"(moderate reduction, supporting sacralization)")
            sac_score += 2
        elif dhi < DHI_MILD_PCT:
            criteria.append(
                f"Disc BELOW TV ({level}) mildly reduced — DHI={dhi:.0f}%")
            sac_score += 1
        else:
            criteria.append(
                f"L2 ✓ Disc BELOW TV ({level}) preserved — DHI={dhi:.0f}% ≥ {DHI_MILD_PCT}% "
                f"— supporting lumbarization (mobile disc below TV, Konin & Walz 2010)")
            primary.append(f"L2:disc-below-preserved-DHI-{dhi:.0f}pct")
            lumb_score += 3

    elif disc_below and disc_below.is_absent:
        criteria.append(
            f"S2 ✓ Disc BELOW TV ({disc_below.level}) absent — disc not measurable; "
            f"possible complete fusion (primary sacralization criterion if confirmed)")
        primary.append(f"S2:disc-below-absent")
        sac_score += 3  # somewhat less certain since it could be segmentation miss

    # ── Disc above TV ──────────────────────────────────────────────────────────
    if disc_above and disc_above.dhi_pct is not None:
        dhi   = disc_above.dhi_pct
        level = disc_above.level
        if dhi >= DHI_MILD_PCT:
            criteria.append(
                f"Disc ABOVE TV ({level}) normal — DHI={dhi:.0f}% "
                f"(localises pathology to L5-S1 junction)")
        else:
            criteria.append(
                f"Disc ABOVE TV ({level}) reduced — DHI={dhi:.0f}% "
                f"(co-existing degeneration or developmental narrowing)")

    # ── PHENOTYPE DECISION ─────────────────────────────────────────────────────
    # Tier 1: Count anomaly alone → immediate definitive classification
    # These are the highest-specificity indicators.
    if lumbar_count == 6:
        # 6-lumbar count IS lumbarization by definition (Hughes & Saifuddin 2006)
        phenotype  = 'lumbarization'
        confidence = 'high'
        rationale  = (
            f"LUMBARIZATION confirmed: 6 lumbar vertebrae detected (L6 present via "
            f"VERIDAH label 25). Presence of a 6th mobile lumbar segment with a disc "
            f"below it is the defining criterion for lumbarization "
            f"(Hughes & Saifuddin 2006; Konin & Walz 2010). "
            f"{'Castellvi ' + castellvi_type + ' also present — TP enlargement on L6 is ' + 'a concurrent finding, not exclusive of lumbarization.' if has_castellvi else 'No Castellvi TP finding.'}"
        )
        return phenotype, confidence, criteria, rationale, primary

    if lumbar_count == 4:
        # 4-lumbar count → L5 has incorporated into sacrum
        phenotype  = 'sacralization'
        confidence = 'high'
        rationale  = (
            f"SACRALIZATION confirmed: 4 lumbar vertebrae detected. Reduction to 4 mobile "
            f"lumbar segments indicates L5 has incorporated into the sacrum. "
            f"{'Castellvi ' + castellvi_type + ' also present — TP morphology is part of the sacralization spectrum.' if has_castellvi else 'No Castellvi TP finding detected (may have fused completely).'}"
        )
        return phenotype, confidence, criteria, rationale, primary

    # Tier 2: Count=5 — rely on PRIMARY criteria S1, S2, S4 + supporting evidence
    if lumbar_count == 5:
        # Without count anomaly, classification requires primary morphologic criteria

        # PRIMARY S2: Disc below severely reduced or absent — most reliable sign
        disc_below_dhi  = disc_below.dhi_pct if disc_below else None
        disc_below_gone = disc_below.is_absent if disc_below else False
        has_s2 = (disc_below_dhi is not None and disc_below_dhi < DHI_REDUCED_PCT) or disc_below_gone

        # PRIMARY S4: sacral-like body
        has_s4 = (tv_shape and tv_shape.shape_class == 'sacral-like'
                  and (has_castellvi or has_s2))

        if sac_score >= 6 or (has_castellvi and has_s2):
            phenotype  = 'sacralization'
            confidence = 'high' if sac_score >= 8 else 'moderate'
            rationale  = (
                f"SACRALIZATION: L5 displays morphologic evidence of sacral incorporation. "
                f"Primary criteria met: {', '.join(primary) or 'Castellvi + disc reduction'}. "
                f"sacralization score={sac_score}, lumbarization score={lumb_score}. "
                f"Evidence: {'; '.join(criteria[:3])}."
            )
        elif sac_score >= 4 and has_castellvi:
            phenotype  = 'sacralization'
            confidence = 'moderate'
            rationale  = (
                f"SACRALIZATION (moderate confidence): Castellvi {castellvi_type} with "
                f"supporting morphometric evidence (score={sac_score}). "
                f"Primary criteria: {', '.join(primary)}."
            )
        elif has_castellvi and not has_s2 and sac_score < 4:
            # Castellvi alone without disc reduction — transitional, indeterminate
            phenotype  = 'transitional_indeterminate'
            confidence = 'low'
            rationale  = (
                f"TRANSITIONAL (indeterminate): Castellvi {castellvi_type} confirmed — "
                f"TP morphology is abnormal, but disc below TV is preserved "
                f"(DHI={disc_below_dhi:.0f}% if measured) and body shape does not confirm "
                f"sacralization. This may represent early/partial sacralization or an "
                f"incidental TP anomaly. Further clinical correlation warranted. "
                f"(Quinlan et al. 1984 note that Castellvi I has the lowest clinical impact.)"
            )
        elif not has_castellvi and sac_score < 4 and lumb_score < 4:
            phenotype  = 'normal'
            confidence = 'high'
            rationale  = (
                f"NORMAL: 5 lumbar vertebrae, no Castellvi TP finding, disc heights "
                f"preserved, TV body morphology within normal range. "
                f"No primary LSTV criteria met."
            )
        else:
            phenotype  = 'normal'
            confidence = 'moderate'
            rationale  = (
                f"No primary LSTV criteria met (count=5, Castellvi={'present' if has_castellvi else 'absent'}, "
                f"sac_score={sac_score}, lumb_score={lumb_score}). "
                f"Normal lumbar-sacral junction."
            )

        return phenotype, confidence, criteria, rationale, primary

    # Fallback for unexpected counts
    phenotype  = 'normal'
    confidence = 'low'
    rationale  = f"Unexpected lumbar count={lumbar_count}; classification uncertain."
    return phenotype, confidence, criteria, rationale, primary


# ── Main entry point ───────────────────────────────────────────────────────────

def analyze_lstv(masks: LSTVMaskSet,
                 castellvi_result: Optional[dict] = None) -> LSTVMorphometrics:
    """
    Run complete LSTV morphometric analysis.

    Parameters
    ----------
    masks            : LSTVMaskSet from load_lstv_masks()
    castellvi_result : dict from 04_detect_lstv.py classify_study()
                       Castellvi type and detected phenotype are INDEPENDENT —
                       both are reported, neither suppresses the other.
    """
    result = LSTVMorphometrics(study_id=masks.study_id)

    try:
        # 1. Lumbar count
        tss_safe             = masks.tss_iso if masks.tss_iso is not None else np.array([], dtype=np.int32)
        tss_count, tss_names = count_lumbar_tss(tss_safe, masks.tss_labels)
        vd_count,  vd_names  = count_lumbar_veridah(masks.vert_labels)
        consensus, count_note = reconcile_lumbar_count(
            tss_count, vd_count, tss_names, vd_names, masks.vert_labels)

        result.lumbar_count_tss       = tss_count  if tss_count > 0 else None
        result.lumbar_count_veridah   = vd_count   if vd_count  > 0 else None
        result.lumbar_count_consensus = consensus
        result.lumbar_count_anomaly   = (consensus != EXPECTED_LUMBAR)
        result.lumbar_count_note      = count_note

        # 2. TV identification
        tv_label, tv_name = None, None
        for cand in VERIDAH_TV_SEARCH:
            if cand in masks.vert_labels:
                tv_label = cand; tv_name = VERIDAH_NAMES[cand]; break

        if tv_label is None:
            result.error = "No lumbar VERIDAH labels found"; return result

        result.tv_label_veridah = tv_label
        result.tv_name          = tv_name
        result.has_l6           = (tv_label == VD_L6)
        result.tv_tss_label     = VD_TO_TSS_VERT.get(tv_label)  # None for L6

        # 3. TV body shape
        result.tv_shape = analyze_tv_body_shape(masks, tv_label, result.tv_tss_label)

        # 4. Adjacent disc metrics
        result.disc_above, result.disc_below = get_tv_adjacent_discs(
            masks, tv_label, result.tv_tss_label)

        # 5. Rib anomaly
        result.rib_anomaly = detect_rib_anomaly(masks)

        # 6. Phenotype classification
        castellvi_type = None
        if castellvi_result:
            castellvi_type = castellvi_result.get('castellvi_type')

        (result.lstv_phenotype,
         result.phenotype_confidence,
         result.phenotype_criteria,
         result.phenotype_rationale,
         result.primary_criteria_met) = classify_lstv_phenotype(
            lumbar_count   = consensus,
            tv_name        = tv_name,
            castellvi_type = castellvi_type,
            tv_shape       = result.tv_shape,
            disc_above     = result.disc_above,
            disc_below     = result.disc_below,
        )

        logger.info(
            f"  [{masks.study_id}] LSTV morphometrics: "
            f"TV={tv_name}, count={consensus}, "
            f"phenotype={result.lstv_phenotype} ({result.phenotype_confidence}), "
            f"primary_criteria={result.primary_criteria_met}"
        )

    except Exception as exc:
        import traceback
        result.error = str(exc)
        logger.error(f"  [{masks.study_id}] lstv_engine error: {exc}")
        logger.debug(traceback.format_exc())

    return result


# ── Pathology scoring ──────────────────────────────────────────────────────────

def compute_lstv_pathology_score(detect_result: dict,
                                  morpho_result: Optional[dict] = None) -> float:
    """
    Scalar LSTV pathology burden score for study ranking.
    Higher = more pathologic.

    Castellvi:           IV=5  III=4  II=3  I=1
    Phenotype (high):    sacralization/lumbarization = +3
    Phenotype (moderate): sacralization/lumbarization = +2
    Transitional:        +1
    Lumbar count anomaly: +2
    Disc below DHI < 50%: +2  |  < 70%: +1
    TV body sacral-like:  +2  |  transitional: +1
    Rib anomaly:          +1
    """
    score = 0.0

    ct = detect_result.get('castellvi_type') or ''
    if   'IV'  in ct: score += 5
    elif 'III' in ct: score += 4
    elif 'II'  in ct: score += 3
    elif 'I'   in ct: score += 1

    if morpho_result is None:
        return score

    ph  = morpho_result.get('lstv_phenotype', 'normal')
    cnf = morpho_result.get('phenotype_confidence', 'low')
    if ph in ('sacralization', 'lumbarization'):
        score += 3.0 if cnf == 'high' else 2.0
    elif ph == 'transitional_indeterminate':
        score += 1.0

    cnt = morpho_result.get('lumbar_count_consensus', 5)
    if cnt and cnt != EXPECTED_LUMBAR:
        score += 2.0

    db  = morpho_result.get('disc_below') or {}
    dhi = db.get('dhi_pct')
    if dhi is not None:
        if   dhi < DHI_REDUCED_PCT:  score += 2.0
        elif dhi < DHI_MODERATE_PCT: score += 1.0
    if db.get('is_absent'): score += 2.0

    sh  = morpho_result.get('tv_shape') or {}
    shc = sh.get('shape_class', '')
    if   shc == 'sacral-like':  score += 2.0
    elif shc == 'transitional': score += 1.0

    rib = morpho_result.get('rib_anomaly') or {}
    if rib.get('any_anomaly'): score += 1.0

    return score
