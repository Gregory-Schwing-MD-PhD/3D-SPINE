#!/usr/bin/env python3
"""
04_detect_lstv.py — Hybrid Two-Phase LSTV Castellvi Classifier
===============================================================
Classifies Lumbosacral Transitional Vertebrae (LSTV) using the Castellvi
system, then calls lstv_engine.py for radiologically-grounded phenotype
classification (lumbarization / sacralization / transitional_indeterminate).

CRITICAL DESIGN PRINCIPLE
--------------------------
Castellvi classification and phenotype classification are ORTHOGONAL.
A study may simultaneously have:
  • Castellvi IIb (bilateral pseudo-articulation of L5/L6 TP with sacrum)
  • Phenotype: lumbarization (6-lumbar count, L6 segment identified)
Neither classification suppresses the other.

LSTV DETECTION FLAGS (lstv_detected = True when ANY of the following):
  • Castellvi Type I-IV detected (TP morphology)
  • Lumbar count ≠ 5 (4 → sacralization, 6 → lumbarization)
  • Phenotype = sacralization or lumbarization (confirmed by morphometrics)

CASTELLVI CLASSIFICATION (Castellvi et al. 1984, Spine 9:31-35)
-----------------------------------------------------------------
Type I   : Dysplastic TP ≥ 19 mm craniocaudal height, no sacral contact
           Ia = unilateral; Ib = bilateral
Type II  : Pseudo-articulation (diarthrodial joint) between TP and sacrum
           IIa = unilateral; IIb = bilateral
Type III : Complete osseous fusion of TP with sacrum
           IIIa = unilateral; IIIb = bilateral
Type IV  : Mixed — one side Type II, other side Type III

LABEL REFERENCE — SOURCE DISAMBIGUATION
----------------------------------------
SPINEPS seg-spine_msk.nii.gz  (subregion / semantic labels):
  43 = Costal_Process_Left   ← TP SOURCE
  44 = Costal_Process_Right  ← TP SOURCE
  26 = Sacrum
  ⚠ TSS labels 43/44 = L3/L4 vertebral bodies — NEVER used as TP source

SPINEPS seg-vert_msk.nii.gz  (VERIDAH per-vertebra instance labels):
  20=L1  21=L2  22=L3  23=L4  24=L5  25=L6  26=Sacrum

TotalSpineSeg sagittal_labeled.nii.gz:
  41=L1  42=L2  43=L3  44=L4  45=L5  50=Sacrum
  91-100 = disc labels
  Preferred vertebra body source for TV Z-range and segmental axis.

TV Z-RANGE DETERMINATION
--------------------------
The Z range used to isolate SPINEPS TP instances is derived from the
TotalSpineSeg (TSS) vertebra body label whenever possible, with VERIDAH
as fallback.  See get_tv_z_range() for full preference order.

TP HEIGHT MEASUREMENT — PRINCIPAL AXIS METHOD
---------------------------------------------
Castellvi's craniocaudal TP height criterion (≥ 19 mm) is measured along
the anatomic SI axis of the spinal segment, NOT the scanner Z axis.

The segmental SI axis is defined as the unit vector from the centroid of
the disc below the transitional vertebra (TV) to the centroid of the disc
above the TV.  This axis is independent of scanner orientation, corrects
for spinal curvature, and remains valid for scoliotic and post-surgical
cases where the anatomic SI axis may deviate substantially from Z.

The three principal axes of the isolated TP voxel cloud are obtained via
SVD/PCA on the physical-space (mm) coordinates.  The principal axis with
the highest absolute cosine similarity to the segmental SI axis is
selected, and the TP height is the range of voxel projections onto that
axis.

SEGMENTAL AXIS COMPUTATION
---------------------------
Preference order for superior / inferior reference points:

  Preferred (disc centroids):
    sup: TSS disc label above TV (e.g. label 95 = L4-L5 for TV=L5)
    inf: TSS disc label below TV (e.g. label 100 = L5-S1 for TV=L5)

  Fallback 1 (vertebra body centroids):
    sup: TSS vertebra body above TV  (e.g. label 44 = L4 for TV=L5)
    inf: TSS sacrum (label 50) or TSS vert below

  Fallback 2 (VERIDAH vertebra body centroids):
    Used when TSS label is absent or TV = L6 (no TSS L6 label exists).

  Last resort: scanner Z axis [0, 0, 1].

TSS disc label mapping for each TV:
  TV=L1 (VD 20):  disc_above=None   disc_below=92 (L1-L2)
  TV=L2 (VD 21):  disc_above=92     disc_below=93
  TV=L3 (VD 22):  disc_above=93     disc_below=94
  TV=L4 (VD 23):  disc_above=94     disc_below=95
  TV=L5 (VD 24):  disc_above=95     disc_below=100
  TV=L6 (VD 25):  no TSS disc labels → VERIDAH body fallback

OUTPUT FIELDS (per-side Phase 1 dict)
--------------------------------------
  tp_height_mm              : TP extent along chosen axis (Castellvi criterion)
  tp_axis_chosen            : unit vector of chosen principal axis [x,y,z]
  tp_axis_cos_segmental     : |cos θ| between chosen axis and segmental SI axis
  tp_axis_deg_from_segmental: angle in degrees between chosen axis and segmental SI
  tp_segmental_axis         : segmental SI unit vector [x,y,z] used
  tp_segmental_axis_source  : description of how the segmental axis was computed
  tp_z_min_vox / tp_z_max_vox: raw Z indices of TP mask (for visualisation)
  tp_centroid_z_mm          : Z centroid of TP mask in mm (for visualisation)
"""

from __future__ import annotations

import argparse
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, label as cc_label

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from lstv_engine import (
    load_lstv_masks, analyze_lstv, compute_lstv_pathology_score,
    TP_HEIGHT_MM, CONTACT_DIST_MM,
    TSS_SACRUM, TSS_LUMBAR, SP_TP_L, SP_TP_R, SP_SACRUM,
    VD_L1, VD_L2, VD_L3, VD_L4, VD_L5, VD_L6, VD_SAC,
    VERIDAH_NAMES, VERIDAH_TV_SEARCH, EXPECTED_LUMBAR,
    VD_TO_TSS_VERT,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-7s  %(message)s')
logger = logging.getLogger(__name__)

# ── Phase 2 signal thresholds ─────────────────────────────────────────────────
BBOX_HALF          = 16
P2_DARK_CLEFT_FRAC = 0.55
P2_MIN_STD_RATIO   = 0.12

# ── Cross-validation thresholds ───────────────────────────────────────────────
XVAL_MIN_DICE      = 0.30
XVAL_MAX_CENTROID  = 20.0   # mm

# ── TSS disc labels adjacent to each possible TV ──────────────────────────────
# (disc_above_tss_label, disc_below_tss_label)
# None = that disc level has no TSS label (e.g. T12-L1 rarely segmented; no L6 disc).
TV_TO_DISC_LABELS: Dict[int, Tuple[Optional[int], Optional[int]]] = {
    VD_L1: (None,  92),   # inf = L1-L2; no TSS label above L1
    VD_L2: (92,    93),
    VD_L3: (93,    94),
    VD_L4: (94,    95),
    VD_L5: (95,   100),   # 100 = L5-S1
    VD_L6: (None, None),  # no TSS disc labels at L6 level
}

# TSS vertebra body fallback: (vert_above_tss_label, vert_below_tss_label_or_sacrum)
# Used when the preferred disc label is absent.
TV_TO_VERT_FALLBACK: Dict[int, Tuple[Optional[int], Optional[int]]] = {
    VD_L1: (None,       42),
    VD_L2: (41,         43),
    VD_L3: (42,         44),
    VD_L4: (43,         45),
    VD_L5: (44,         TSS_SACRUM),
    VD_L6: (45,         TSS_SACRUM),
}

# VERIDAH vertebra body fallback (used when TSS is absent or TV = L6)
TV_TO_VD_FALLBACK: Dict[int, Tuple[Optional[int], Optional[int]]] = {
    VD_L1: (None,    VD_L2),
    VD_L2: (VD_L1,  VD_L3),
    VD_L3: (VD_L2,  VD_L4),
    VD_L4: (VD_L3,  VD_L5),
    VD_L5: (VD_L4,  VD_SAC),
    VD_L6: (VD_L5,  VD_SAC),
}


# ══════════════════════════════════════════════════════════════════════════════
# NIfTI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_canonical(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    nii  = nib.load(str(path))
    nii  = nib.as_closest_canonical(nii)
    data = nii.get_fdata()
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Cannot reduce {path.name} to 3D: shape={data.shape}")
    return data, nii


def voxel_size_mm(nii: nib.Nifti1Image) -> np.ndarray:
    return np.abs(np.array(nii.header.get_zooms()[:3], dtype=float))


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def dice_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    a, b  = a.astype(bool), b.astype(bool)
    inter = (a & b).sum()
    denom = a.sum() + b.sum()
    return float(2 * inter / denom) if denom else float('nan')


def centroid_mm(mask: np.ndarray, vox_mm: np.ndarray) -> Optional[np.ndarray]:
    coords = np.array(np.where(mask))
    return coords.mean(axis=1) * vox_mm if coords.size else None


def run_cross_validation(sag_spineps: np.ndarray,
                          sag_vert:    np.ndarray,
                          sag_tss:     np.ndarray,
                          vox_mm:      np.ndarray,
                          study_id:    str) -> dict:
    xval: dict = {'warnings': []}

    sp_sac  = (sag_spineps == SP_SACRUM)
    tss_sac = (sag_tss     == TSS_SACRUM)
    if sp_sac.any() and tss_sac.any():
        d = dice_coefficient(sp_sac, tss_sac)
        xval['sacrum_dice'] = round(d, 4)
        if d < XVAL_MIN_DICE:
            msg = f"Sacrum Dice={d:.3f} < {XVAL_MIN_DICE} — SPINEPS/TSS sacrum mismatch"
            logger.warning(f"  [{study_id}] {msg}")
            xval['warnings'].append(msg)
        else:
            logger.info(f"  [{study_id}] Sacrum Dice={d:.3f} ✓")

    vd_l5  = (sag_vert == VD_L5)
    tss_l5 = (sag_tss  == 45)
    if vd_l5.any() and tss_l5.any():
        c_v = centroid_mm(vd_l5,  vox_mm)
        c_t = centroid_mm(tss_l5, vox_mm)
        if c_v is not None and c_t is not None:
            dist = float(np.linalg.norm(c_v - c_t))
            xval['l5_centroid_dist_mm'] = round(dist, 2)
            if dist > XVAL_MAX_CENTROID:
                msg = f"L5 centroid dist={dist:.1f}mm > {XVAL_MAX_CENTROID}mm"
                logger.warning(f"  [{study_id}] {msg}")
                xval['warnings'].append(msg)
            else:
                logger.info(f"  [{study_id}] L5 centroid dist={dist:.1f}mm ✓")

    return xval


# ══════════════════════════════════════════════════════════════════════════════
# MASK OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_z_range_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    if not mask.any():
        return None
    zc = np.where(mask)[2]
    return int(zc.min()), int(zc.max())


def isolate_tp_at_tv(sp_data: np.ndarray, tp_label: int,
                     z_min: int, z_max: int) -> np.ndarray:
    tp    = (sp_data == tp_label)
    out   = np.zeros_like(tp)
    z_lo  = max(0, z_min - 3)
    z_hi  = min(sp_data.shape[2] - 1, z_max + 3)
    out[:, :, z_lo:z_hi + 1] = tp[:, :, z_lo:z_hi + 1]
    return out


def inferiormost_tp_cc(tp_mask: np.ndarray,
                        sacrum_mask: Optional[np.ndarray] = None) -> np.ndarray:
    if not tp_mask.any(): return np.zeros_like(tp_mask, bool)
    labeled, n = cc_label(tp_mask)
    if n == 1: return tp_mask.astype(bool)

    sac_z_min = None
    if sacrum_mask is not None and sacrum_mask.any():
        sac_z_min = int(np.where(sacrum_mask)[2].min())

    comps = []
    for i in range(1, n + 1):
        comp = (labeled == i)
        zc   = np.where(comp)[2]
        comps.append((float(zc.mean()), int(zc.max()), comp))
    comps.sort(key=lambda t: t[0])

    if sac_z_min is not None:
        cands = [c for _, zm, c in comps if zm < sac_z_min]
        if cands: return cands[0].astype(bool)

    return comps[0][2].astype(bool)


def min_dist_3d(mask_a: np.ndarray, mask_b: np.ndarray,
                vox_mm: np.ndarray) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    if not mask_a.any() or not mask_b.any():
        return float('inf'), None, None
    dt       = distance_transform_edt(~mask_b, sampling=vox_mm)
    dist_at  = np.where(mask_a, dt, np.inf)
    flat_idx = int(np.argmin(dist_at))
    vox_a    = np.array(np.unravel_index(flat_idx, mask_a.shape))
    dist_mm  = float(dt[tuple(vox_a)])

    z_lo  = max(0, int(vox_a[2]) - 20)
    z_hi  = min(mask_b.shape[2], int(vox_a[2]) + 20)
    sub   = mask_b[:, :, z_lo:z_hi]
    if sub.any():
        coords       = np.array(np.where(sub))
        coords[2, :] += z_lo
    else:
        coords = np.array(np.where(mask_b))
    d2    = ((coords.T * vox_mm - vox_a * vox_mm) ** 2).sum(axis=1)
    vox_b = coords[:, int(np.argmin(d2))]
    return dist_mm, vox_a, vox_b


# ══════════════════════════════════════════════════════════════════════════════
# TV Z-RANGE DETERMINATION
# ══════════════════════════════════════════════════════════════════════════════

def get_tv_z_range(sag_vert:  np.ndarray,
                   sag_tss:   Optional[np.ndarray],
                   tv_label:  int,
                   study_id:  str) -> Tuple[Optional[Tuple[int, int]], str]:
    """
    Determine the Z voxel range of the transitional vertebra body.

    Preference:
    1. TSS vertebra body label (independent of SPINEPS TP label ambiguity)
    2. VERIDAH vertebra body label (also the only option for L6)
    """
    tss_lbl = VD_TO_TSS_VERT.get(tv_label)

    if tss_lbl is not None and sag_tss is not None:
        tss_mask = (sag_tss == tss_lbl)
        if tss_mask.any():
            zr  = get_z_range_from_mask(tss_mask)
            src = f'TSS label {tss_lbl} ({VERIDAH_NAMES.get(tv_label, str(tv_label))})'
            logger.info(f"  [{study_id}] TV Z range from {src}: {zr}")
            return zr, src
        logger.warning(
            f"  [{study_id}] TSS label {tss_lbl} yields empty mask — "
            f"falling back to VERIDAH")

    vd_mask = (sag_vert == tv_label)
    if vd_mask.any():
        zr  = get_z_range_from_mask(vd_mask)
        src = (f'VERIDAH label {tv_label} '
               f'({VERIDAH_NAMES.get(tv_label, str(tv_label))}) [fallback]')
        logger.info(f"  [{study_id}] TV Z range from {src}: {zr}")
        return zr, src

    logger.warning(f"  [{study_id}] TV label {tv_label} not found in TSS or VERIDAH")
    return None, 'not found'


# ══════════════════════════════════════════════════════════════════════════════
# SEGMENTAL SI AXIS
# ══════════════════════════════════════════════════════════════════════════════

def _mask_centroid_mm(mask: np.ndarray, vox_mm: np.ndarray) -> Optional[np.ndarray]:
    """Physical-space centroid of a boolean mask in mm."""
    if not mask.any():
        return None
    idx = np.array(np.where(mask), dtype=np.float64)   # (3, N)
    return idx.mean(axis=1) * vox_mm                    # (3,)


def compute_segmental_axis(
        sag_tss:  Optional[np.ndarray],
        sag_vert: np.ndarray,
        tv_label: int,
        vox_mm:   np.ndarray,
        study_id: str,
) -> Tuple[np.ndarray, str]:
    """
    Compute the anatomic superior–inferior (SI) unit axis of the spinal
    segment at the transitional vertebra (TV).

    The axis is the unit vector pointing from the centroid of the structure
    immediately INFERIOR to the TV (disc below, or sacrum) to the centroid
    of the structure immediately SUPERIOR to the TV (disc above, or vertebra
    body above).  This is the anatomic craniocaudal axis of the segment,
    independent of scanner orientation.

    Reference structures are obtained in preference order:
      1. TSS disc centroids  (most stable — disc centroids hug the endplates)
      2. TSS vertebra body centroids
      3. VERIDAH vertebra body centroids  (for L6 or when TSS is absent)
      4. Scanner Z axis [0, 0, 1]  (last resort; triggers a warning)

    Parameters
    ----------
    sag_tss   : TotalSpineSeg sagittal volume (may be None)
    sag_vert  : SPINEPS seg-vert_msk (VERIDAH) volume
    tv_label  : VERIDAH label of the transitional vertebra (20–25)
    vox_mm    : physical voxel size [x, y, z] in mm
    study_id  : for log messages

    Returns
    -------
    (axis_unit_vector, source_description)
      axis_unit_vector : np.ndarray shape (3,), cranially pointing
      source_description : human-readable string naming the reference landmarks
    """
    disc_above_lbl, disc_below_lbl = TV_TO_DISC_LABELS.get(tv_label, (None, None))
    vert_above_tss, vert_below_tss = TV_TO_VERT_FALLBACK.get(tv_label, (None, None))
    vd_above,       vd_below       = TV_TO_VD_FALLBACK.get(tv_label, (None, None))

    sup_pt:  Optional[np.ndarray] = None
    inf_pt:  Optional[np.ndarray] = None
    sup_src: str = ''
    inf_src: str = ''

    # ── Superior reference (disc above → TSS vert above → VERIDAH vert above) ─
    if disc_above_lbl is not None and sag_tss is not None:
        m = (sag_tss == disc_above_lbl)
        if m.any():
            sup_pt  = _mask_centroid_mm(m, vox_mm)
            sup_src = f'TSS disc {disc_above_lbl}'

    if sup_pt is None and vert_above_tss is not None and sag_tss is not None:
        m = (sag_tss == vert_above_tss)
        if m.any():
            sup_pt  = _mask_centroid_mm(m, vox_mm)
            sup_src = f'TSS vert {vert_above_tss}'

    if sup_pt is None and vd_above is not None:
        m = (sag_vert == vd_above)
        if m.any():
            sup_pt  = _mask_centroid_mm(m, vox_mm)
            sup_src = f'VERIDAH vert {vd_above}'

    # ── Inferior reference (disc below → TSS sacrum/vert → VERIDAH sacrum/vert) ─
    if disc_below_lbl is not None and sag_tss is not None:
        m = (sag_tss == disc_below_lbl)
        if m.any():
            inf_pt  = _mask_centroid_mm(m, vox_mm)
            inf_src = f'TSS disc {disc_below_lbl}'

    if inf_pt is None and vert_below_tss is not None and sag_tss is not None:
        m = (sag_tss == vert_below_tss)
        if m.any():
            inf_pt  = _mask_centroid_mm(m, vox_mm)
            inf_src = (f'TSS sacrum'
                       if vert_below_tss == TSS_SACRUM
                       else f'TSS vert {vert_below_tss}')

    if inf_pt is None and vd_below is not None:
        m = (sag_vert == vd_below)
        if m.any():
            inf_pt  = _mask_centroid_mm(m, vox_mm)
            inf_src = (f'VERIDAH sacrum'
                       if vd_below == VD_SAC
                       else f'VERIDAH vert {vd_below}')

    # ── Last resort: scanner Z axis ────────────────────────────────────────────
    if sup_pt is None or inf_pt is None:
        missing = ([' superior'] if sup_pt is None else []) + \
                  ([' inferior'] if inf_pt is None else [])
        logger.warning(
            f"  [{study_id}] Segmental axis: could not locate"
            f"{''.join(missing)} reference — falling back to scanner Z axis"
        )
        return np.array([0.0, 0.0, 1.0]), 'scanner Z axis (fallback)'

    # ── Build unit vector: inf → sup (caudal to cranial) ─────────────────────
    raw  = sup_pt - inf_pt
    norm = float(np.linalg.norm(raw))
    if norm < 1e-6:
        logger.warning(
            f"  [{study_id}] Segmental axis: {sup_src} and {inf_src} have "
            f"identical centroids — falling back to scanner Z axis"
        )
        return np.array([0.0, 0.0, 1.0]), 'scanner Z axis (degenerate centroids)'

    unit = raw / norm
    src  = f'{inf_src} → {sup_src}'
    logger.info(
        f"  [{study_id}] Segmental SI axis: {src}  "
        f"v=[{unit[0]:.3f},{unit[1]:.3f},{unit[2]:.3f}]  "
        f"Z-component={unit[2]:.3f}"
    )
    return unit, src


# ══════════════════════════════════════════════════════════════════════════════
# TP HEIGHT — PRINCIPAL AXIS METHOD
# ══════════════════════════════════════════════════════════════════════════════

def measure_tp_height_principal_axis(
        tp_mask:        np.ndarray,
        vox_mm:         np.ndarray,
        segmental_axis: np.ndarray,
) -> Tuple[float, np.ndarray, float, np.ndarray]:
    """
    Measure the craniocaudal extent of a transverse process along the
    principal axis that best aligns with the segmental SI axis.

    RATIONALE
    ---------
    Castellvi (1984) specifies "craniocaudal height" of the TP.  In a spine
    with normal alignment this equals the Z-extent, but in scoliosis, kyphosis,
    or any oblique acquisition the Z axis may deviate substantially from the
    anatomic SI axis of the segment.  Measuring along the dominant structural
    axis of the TP itself — the one that matches the segmental SI direction —
    is anatomically and geometrically correct.

    ALGORITHM
    ---------
    1. Convert TP mask voxel indices to physical mm coordinates.
    2. Mean-centre the point cloud.
    3. Decompose via SVD → three right singular vectors = principal axes
       sorted by descending explained variance (first = longest axis).
    4. Select the principal axis whose |dot product| with segmental_axis is
       maximised.  This is the axis that best represents the craniocaudal
       direction within the TP geometry.
    5. Orient the selected axis cranially (positive dot with segmental_axis).
    6. Project ALL TP voxels onto the selected axis.
    7. Height = max(projection) − min(projection).

    Parameters
    ----------
    tp_mask        : 3-D boolean array of isolated TP voxels
    vox_mm         : physical voxel size [x, y, z] mm
    segmental_axis : unit vector (cranially pointing) of the segmental SI axis

    Returns
    -------
    height_mm      : craniocaudal extent in mm along the chosen axis
    chosen_axis    : the principal axis vector used (unit, cranially oriented)
    cos_align      : |cos θ| between chosen_axis and segmental_axis
    principal_axes : (3, 3) array — all three principal axes as rows, by
                     descending explained variance
    """
    if not tp_mask.any():
        return 0.0, segmental_axis.copy(), 0.0, np.eye(3)

    # Physical coordinates of every TP voxel (N × 3, in mm)
    idx    = np.array(np.where(tp_mask), dtype=np.float64).T   # (N, 3)
    pts_mm = idx * vox_mm                                       # (N, 3)
    n_pts  = pts_mm.shape[0]

    # Degenerate case: too few points for PCA
    if n_pts < 3:
        proj = pts_mm @ segmental_axis
        height = float(proj.max() - proj.min()) if n_pts > 1 else 0.0
        return height, segmental_axis.copy(), 1.0, np.eye(3)

    centred = pts_mm - pts_mm.mean(axis=0)

    # Thin SVD — Vt rows are right singular vectors (principal axes),
    # ordered by descending singular value (= descending variance).
    try:
        _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        logger.warning("SVD failed — using segmental axis directly for TP height")
        proj = pts_mm @ segmental_axis
        return float(proj.max() - proj.min()), segmental_axis.copy(), 1.0, np.eye(3)

    principal_axes = Vt   # shape (3, 3); row i = i-th principal axis

    # ── Select the axis most aligned with the segmental SI direction ──────────
    dots    = np.abs(principal_axes @ segmental_axis)   # (3,)
    best    = int(np.argmax(dots))
    chosen  = principal_axes[best].copy()

    # Ensure consistent cranial orientation
    if np.dot(chosen, segmental_axis) < 0:
        chosen = -chosen

    cos_align = float(dots[best])
    deg       = float(np.degrees(np.arccos(np.clip(cos_align, 0.0, 1.0))))

    # ── Measure extent along chosen axis ─────────────────────────────────────
    proj      = pts_mm @ chosen
    height_mm = float(proj.max() - proj.min())

    logger.debug(
        f"    TP PCA: N={n_pts}  "
        f"axis=[{chosen[0]:.3f},{chosen[1]:.3f},{chosen[2]:.3f}]  "
        f"|cosθ|={cos_align:.3f} ({deg:.1f}°)  h={height_mm:.1f}mm"
    )

    return height_mm, chosen, cos_align, principal_axes


# ══════════════════════════════════════════════════════════════════════════════
# L6 VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

TSS_DISC_LABELS = set(range(91, 101))


def _verify_l6(sag_vert:  np.ndarray,
               sag_tss:   Optional[np.ndarray],
               vox_mm:    np.ndarray,
               tv_z:      Tuple[int, int],
               study_id:  str) -> Tuple[bool, str]:
    """
    Confirm that VERIDAH's L6 label is a genuine extra lumbar vertebra.

    Three checks (all must pass):
      1. Positional sanity: L6 centroid between TSS L5 inferior and TSS
         sacrum superior.
      2. Disc ABOVE L6: TSS disc centroid clearly above L6 superior edge
         (L5-L6 disc space present).
      3. Disc BELOW L6: TSS disc centroid between L6 inferior edge and
         sacrum superior edge (L6-S1 disc space present).
    """
    if sag_tss is None:
        return False, "no TSS available — cannot verify L6"

    l6_z_min, l6_z_max = tv_z
    l6_centroid_z = (l6_z_min + l6_z_max) / 2.0

    tss_l5   = (sag_tss == 45)
    tss_sac  = (sag_tss == TSS_SACRUM)

    if not tss_sac.any():
        return False, "TSS sacrum (label 50) absent — cannot verify L6 position"

    sac_z_sup = float(np.where(tss_sac)[2].max())

    if l6_centroid_z <= sac_z_sup:
        return False, (
            f"L6 centroid z={l6_centroid_z:.0f} ≤ TSS sacrum superior z={sac_z_sup:.0f} "
            f"— VERIDAH L6 inside sacrum (mislabeled sacrum)")

    if tss_l5.any():
        l5_z_min = float(np.where(tss_l5)[2].min())
        if l6_centroid_z >= l5_z_min:
            return False, (
                f"L6 centroid z={l6_centroid_z:.0f} ≥ TSS L5 inferior z={l5_z_min:.0f} "
                f"— VERIDAH L6 overlaps TSS L5")

    disc_above_z: Optional[float] = None
    disc_below_z: Optional[float] = None

    for disc_lbl in TSS_DISC_LABELS:
        disc_mask = (sag_tss == disc_lbl)
        if not disc_mask.any():
            continue
        disc_zc = float(np.mean(np.where(disc_mask)[2]))
        if disc_zc > l6_z_max + 2:
            if disc_above_z is None or disc_zc < disc_above_z:
                disc_above_z = disc_zc
        if disc_zc < l6_z_min - 2 and disc_zc > sac_z_sup:
            if disc_below_z is None or disc_zc > disc_below_z:
                disc_below_z = disc_zc

    if disc_above_z is None:
        return False, (
            f"no TSS disc above VERIDAH L6 (z=[{l6_z_min},{l6_z_max}]) — "
            f"no L5-L6 disc space")
    if disc_below_z is None:
        return False, (
            f"no TSS disc between L6 inferior (z={l6_z_min}) and "
            f"sacrum superior (z={sac_z_sup:.0f}) — no L6-S1 disc space")

    return True, (
        f"positional OK (centroid z={l6_centroid_z:.0f}, sac_sup={sac_z_sup:.0f}), "
        f"disc above z={disc_above_z:.0f}, disc below z={disc_below_z:.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — AXIAL T2w SIGNAL CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def _extract_bbox(axial_t2w: np.ndarray, midpoint: np.ndarray,
                  half: int = BBOX_HALF) -> Optional[np.ndarray]:
    x0, y0, z0 = int(midpoint[0]), int(midpoint[1]), int(midpoint[2])
    nx, ny, nz  = axial_t2w.shape
    if not (0 <= z0 < nz): return None
    patch = axial_t2w[max(0, x0 - half):min(nx, x0 + half),
                      max(0, y0 - half):min(ny, y0 + half), z0].copy()
    return patch if patch.size > 0 else None


def _classify_signal(patch: np.ndarray, axial_t2w: np.ndarray) -> Tuple[str, dict]:
    """
    Classify axial T2w signal at TP–sacrum junction.
    Dark/intermediate → fibrocartilaginous pseudo-joint → Type II
    Bright homogeneous → osseous marrow bridge           → Type III
    (Konin & Walz 2010; Nidecker et al. 2018)
    """
    vals = patch.astype(float).ravel()
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 'Type II', {'reason': 'empty patch', 'valid': False}

    p_mean    = float(np.mean(vals))
    p_std     = float(np.std(vals))
    cv        = p_std / (p_mean + 1e-6)
    global_fg = axial_t2w[axial_t2w > 0]
    p95       = float(np.percentile(global_fg, 95)) if global_fg.size else 1.0
    dark_thr  = P2_DARK_CLEFT_FRAC * p95

    feats = {
        'patch_mean': round(p_mean, 2), 'patch_std': round(p_std, 2),
        'coeff_var':  round(cv, 4),     'global_p95': round(p95, 2),
        'dark_thresh':round(dark_thr, 2), 'valid': True,
    }
    if p_mean < dark_thr:
        feats['reason'] = (f"mean={p_mean:.1f} < dark_thr={dark_thr:.1f} "
                           f"— dark/intermediate → fibrocartilage → Type II")
        return 'Type II', feats
    elif cv < P2_MIN_STD_RATIO:
        feats['reason'] = (f"CV={cv:.3f} < {P2_MIN_STD_RATIO} "
                           f"— uniform bright → marrow bridge → Type III")
        return 'Type III', feats
    else:
        feats['reason'] = "Bright heterogeneous — ambiguous; Type II (conservative)"
        return 'Type II', feats


def phase2_axial(side: str, tp_label: int,
                 ax_spineps: np.ndarray, ax_tss: np.ndarray,
                 ax_t2w: np.ndarray, ax_vox_mm: np.ndarray) -> dict:
    out: dict = {'phase2_attempted': True, 'classification': 'Type II',
                 'midpoint_vox': None, 'p2_features': None, 'p2_valid': False}

    tp_ax  = (ax_spineps == tp_label)
    sac_ax = (ax_tss     == TSS_SACRUM)

    if not tp_ax.any():
        out['p2_note'] = f"TP label {tp_label} absent in registered SPINEPS mask"
        return out
    if not sac_ax.any():
        out['p2_note'] = f"Sacrum label {TSS_SACRUM} absent in axial TSS"
        return out

    dist_mm, vox_a, vox_b = min_dist_3d(tp_ax, sac_ax, ax_vox_mm)
    if vox_a is None or vox_b is None:
        out['p2_note'] = 'min_dist_3d returned None'
        return out

    midpoint             = ((vox_a + vox_b) / 2.0).astype(int)
    out['midpoint_vox']  = midpoint.tolist()
    out['axial_dist_mm'] = round(float(dist_mm), 3)

    patch = _extract_bbox(ax_t2w, midpoint, BBOX_HALF)
    if patch is None:
        out['p2_note'] = 'Bounding box outside axial volume'
        return out

    cls, feats           = _classify_signal(patch, ax_t2w)
    out['classification'] = cls
    out['p2_features']    = feats
    out['p2_valid']       = feats.get('valid', False)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — SAGITTAL GEOMETRIC CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def phase1_sagittal(side:           str,
                    tp_label:       int,
                    sag_sp:         np.ndarray,
                    sag_tss:        Optional[np.ndarray],
                    sag_vox_mm:     np.ndarray,
                    tv_z_range:     Tuple[int, int],
                    segmental_axis: np.ndarray,
                    segmental_src:  str) -> dict:
    """
    Phase 1: sagittal geometric Castellvi analysis.

    TP craniocaudal height is measured along the principal axis of the TP
    voxel cloud that best aligns with the segmental SI axis (see
    measure_tp_height_principal_axis).  This replaces the naive Z-extent
    measurement and correctly handles oblique TP orientations.

    Contact threshold : ≤ 2 mm 3D Euclidean distance to sacrum.
    Type I threshold  : TP height ≥ 19 mm (Castellvi 1984) without contact.

    Parameters
    ----------
    segmental_axis : cranially-pointing SI unit vector for this TV level,
                     computed by compute_segmental_axis()
    segmental_src  : human-readable description of how the axis was derived
    """
    out = {
        'tp_present': False, 'tp_height_mm': 0.0, 'contact': False,
        'dist_mm': float('inf'), 'tp_vox': None, 'sacrum_vox': None,
        'classification': 'Normal', 'phase1_done': False, 'sacrum_source': None,
        # ── Principal axis fields ─────────────────────────────────────────────
        'tp_axis_chosen':             None,
        'tp_axis_cos_segmental':      None,
        'tp_axis_deg_from_segmental': None,
        'tp_segmental_axis':          segmental_axis.tolist(),
        'tp_segmental_axis_source':   segmental_src,
    }

    # ── Sacrum mask ───────────────────────────────────────────────────────────
    tss_sac = (sag_tss == TSS_SACRUM) if sag_tss is not None else None
    if tss_sac is not None and tss_sac.any():
        sac_mask             = tss_sac
        out['sacrum_source'] = f'TSS label {TSS_SACRUM}'
    else:
        sac_mask             = (sag_sp == SP_SACRUM)
        out['sacrum_source'] = 'SPINEPS label 26 (fallback)'

    # ── Isolate TP at TV level ────────────────────────────────────────────────
    tp_at_tv = isolate_tp_at_tv(sag_sp, tp_label, *tv_z_range)
    tp_mask  = inferiormost_tp_cc(tp_at_tv, sac_mask if sac_mask.any() else None)

    if not tp_mask.any():
        out['note'] = f"TP label {tp_label} absent at TV level"
        return out

    out['tp_present'] = True

    # Z indices retained for visualisation
    tp_z_coords         = np.where(tp_mask)[2]
    out['tp_z_min_vox'] = int(tp_z_coords.min())
    out['tp_z_max_vox'] = int(tp_z_coords.max())
    out['tp_centroid_z_mm'] = round(float(tp_z_coords.mean()) * sag_vox_mm[2], 2)

    # ── TP height via principal axis aligned to segmental SI axis ─────────────
    h_mm, chosen_axis, cos_align, _principal_axes = measure_tp_height_principal_axis(
        tp_mask, sag_vox_mm, segmental_axis)

    out['tp_height_mm']               = round(h_mm, 3)
    out['tp_axis_chosen']             = [round(float(v), 5) for v in chosen_axis]
    out['tp_axis_cos_segmental']      = round(cos_align, 4)
    out['tp_axis_deg_from_segmental'] = round(
        float(np.degrees(np.arccos(np.clip(cos_align, 0.0, 1.0)))), 2)

    if not sac_mask.any():
        return out

    # ── Distance to sacrum ────────────────────────────────────────────────────
    dist_mm, tp_vox, sac_vox = min_dist_3d(tp_mask, sac_mask, sag_vox_mm)
    out['dist_mm']     = round(float(dist_mm), 3)
    out['phase1_done'] = True
    if tp_vox  is not None: out['tp_vox']     = tp_vox.tolist()
    if sac_vox is not None: out['sacrum_vox'] = sac_vox.tolist()

    # ── Castellvi classification ──────────────────────────────────────────────
    if dist_mm > CONTACT_DIST_MM:
        out['contact'] = False
        if h_mm >= TP_HEIGHT_MM:
            out['classification'] = 'Type I'
    else:
        out['contact']        = True
        out['classification'] = 'CONTACT_PENDING_P2'

    return out


# ══════════════════════════════════════════════════════════════════════════════
# PER-STUDY CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def classify_study(study_id:       str,
                   spineps_dir:    Path,
                   totalspine_dir: Path,
                   registered_dir: Path,
                   nifti_dir:      Path,
                   run_morpho:     bool = True) -> dict:
    """
    Full LSTV classification for one study.

    Castellvi classification (TP morphology) and LSTV phenotype
    (lumbarization / sacralization) are computed independently and both
    reported — they may co-occur.
    """
    out: dict = {
        'study_id':           study_id,
        'lstv_detected':      False,
        'lstv_reason':        [],
        'castellvi_type':     None,
        'confidence':         'high',
        'left':               {},
        'right':              {},
        'details':            {},
        'cross_validation':   {},
        'lstv_morphometrics': None,
        'pathology_score':    None,
        'errors':             [],
    }

    # ── Paths ──────────────────────────────────────────────────────────────────
    seg_dir    = spineps_dir / 'segmentations' / study_id
    spine_path = seg_dir / f"{study_id}_seg-spine_msk.nii.gz"
    vert_path  = seg_dir / f"{study_id}_seg-vert_msk.nii.gz"
    tss_sag    = totalspine_dir / study_id / 'sagittal' / f"{study_id}_sagittal_labeled.nii.gz"
    tss_ax     = totalspine_dir / study_id / 'axial'    / f"{study_id}_axial_labeled.nii.gz"
    sp_ax      = registered_dir / study_id / f"{study_id}_spineps_reg.nii.gz"

    def _load(path: Path, tag: str):
        if not path.exists():
            logger.warning(f"  Missing: {path.name}")
            return None, None
        try:
            return load_canonical(path)
        except Exception as exc:
            logger.warning(f"  {tag}: {exc}")
            return None, None

    def _find_t2w(acq: str) -> Optional[Path]:
        sd = nifti_dir / study_id
        if not sd.exists(): return None
        for sub in sorted(sd.iterdir()):
            p = sub / f"sub-{study_id}_acq-{acq}_T2w.nii.gz"
            if p.exists(): return p
        return None

    sag_sp,  sp_nii  = _load(spine_path, 'seg-spine_msk')
    sag_vert, _      = _load(vert_path,  'seg-vert_msk')
    sag_tss,  _      = _load(tss_sag,    'TSS sagittal')

    if sag_sp   is None: out['errors'].append('Missing SPINEPS seg-spine_msk'); return out
    if sag_vert is None: out['errors'].append('Missing SPINEPS seg-vert_msk');  return out
    if sag_tss  is None: out['errors'].append('Missing TotalSpineSeg sagittal'); return out

    sag_sp   = sag_sp.astype(int)
    sag_vert = sag_vert.astype(int)
    sag_tss  = sag_tss.astype(int)
    vox_mm   = voxel_size_mm(sp_nii)

    # ── Log TSS lumbar labels ──────────────────────────────────────────────────
    tss_unique         = sorted(int(v) for v in np.unique(sag_tss) if v > 0)
    tss_lumbar_present = {lbl: name for lbl, name in TSS_LUMBAR.items()
                          if lbl in tss_unique}
    tss_lumbar_missing = {lbl: name for lbl, name in TSS_LUMBAR.items()
                          if lbl not in tss_unique}
    logger.info(
        f"  [{study_id}] TSS lumbar labels present: "
        f"{list(tss_lumbar_present.values()) or 'none'}"
        + (f"  MISSING: {list(tss_lumbar_missing.values())}" if tss_lumbar_missing else "")
    )

    # ── Cross-validation ───────────────────────────────────────────────────────
    xval = run_cross_validation(sag_sp, sag_vert, sag_tss, vox_mm, study_id)
    out['cross_validation'] = xval
    for w in xval.get('warnings', []): out['errors'].append(f'XVAL: {w}')

    # ── Axial data (Phase 2) ───────────────────────────────────────────────────
    ax_tss, _ = _load(tss_ax, 'TSS axial')
    ax_sp,  _ = _load(sp_ax,  'registered SPINEPS axial')
    ax_t2w, ax_vox_mm = None, None
    t2w_path = _find_t2w('ax')
    if t2w_path:
        arr, t2w_nii = _load(t2w_path, 'axial T2w')
        if arr is not None:
            ax_t2w    = arr
            ax_vox_mm = voxel_size_mm(t2w_nii)

    p2_available = (ax_tss is not None and ax_sp is not None and ax_t2w is not None)
    if not p2_available:
        logger.warning(f"  Phase 2 unavailable — contact cases → Type II (conservative)")

    if ax_tss is not None: ax_tss = ax_tss.astype(int)
    if ax_sp  is not None: ax_sp  = ax_sp.astype(int)

    # ── TV identification from VERIDAH ─────────────────────────────────────────
    vert_unique = sorted(int(v) for v in np.unique(sag_vert) if v > 0)
    named       = [VERIDAH_NAMES[l] for l in vert_unique if l in VERIDAH_NAMES]
    logger.info(f"  [{study_id}] VERIDAH labels: {named}")

    tv_label, tv_name = None, None
    for cand in VERIDAH_TV_SEARCH:
        if cand in vert_unique:
            tv_label = cand; tv_name = VERIDAH_NAMES[cand]; break

    if tv_label is None:
        out['errors'].append('No lumbar VERIDAH labels found'); return out

    # ── L6 verification ────────────────────────────────────────────────────────
    l6_verified = False
    if tv_label == VD_L6:
        vd_l6_z = get_z_range_from_mask(sag_vert == VD_L6)
        if vd_l6_z is None:
            out['errors'].append('VERIDAH L6 label present but empty mask')
            return out
        l6_ok, l6_reason = _verify_l6(sag_vert, sag_tss, vox_mm, vd_l6_z, study_id)
        l6_verified = l6_ok
        if not l6_ok:
            logger.warning(
                f"  [{study_id}] VERIDAH L6 FAILED verification: {l6_reason} "
                f"— demoting TV to L5")
            if VD_L5 in vert_unique:
                tv_label = VD_L5
                tv_name  = VERIDAH_NAMES[VD_L5]
            else:
                out['errors'].append('L6 verification failed and no VERIDAH L5 fallback')
                return out
        else:
            logger.info(f"  [{study_id}] VERIDAH L6 verified ✓ — {l6_reason}")

    # ── TV Z range from TSS body (preferred) or VERIDAH ───────────────────────
    tv_z, tv_z_source = get_tv_z_range(sag_vert, sag_tss, tv_label, study_id)
    if tv_z is None:
        out['errors'].append(f'TV label {tv_name} not found in TSS or VERIDAH mask')
        return out

    # ── Segmental SI axis ──────────────────────────────────────────────────────
    # Computed once per study; both TPs share the same transitional vertebra
    # so the disc references — and therefore the segmental axis — are identical
    # for left and right sides.
    seg_axis, seg_src = compute_segmental_axis(
        sag_tss, sag_vert, tv_label, vox_mm, study_id)

    out['details'] = {
        'tv_label':              tv_label,
        'tv_name':               tv_name,
        'has_l6':                tv_label == VD_L6,
        'l6_verified':           l6_verified if tv_label == VD_L6 else None,
        'tv_z_range':            list(tv_z),
        'tv_z_source':           tv_z_source,
        'segmental_axis':        [round(float(v), 5) for v in seg_axis],
        'segmental_axis_source': seg_src,
        'sag_vox_mm':            vox_mm.tolist(),
        'phase2_available':      p2_available,
        'tp_source':             'seg-spine_msk labels 43 (L) / 44 (R)',
        'sacrum_source':         'TSS label 50 (preferred) / SPINEPS 26 (fallback)',
        'label_note':            'TSS 43/44 = L3/L4 vertebrae — TP always from seg-spine_msk',
        'tss_lumbar_labels':     tss_lumbar_present,
    }
    logger.info(
        f"  [{study_id}] TV={tv_name}  z={tv_z}  z_src={tv_z_source}\n"
        f"             seg_axis=[{seg_axis[0]:.3f},{seg_axis[1]:.3f},{seg_axis[2]:.3f}]"
        f"  src={seg_src}"
    )

    # ── Phase 1 + 2 per side ───────────────────────────────────────────────────
    for side, tp_lbl in (('left', SP_TP_L), ('right', SP_TP_R)):
        try:
            p1 = phase1_sagittal(
                side, tp_lbl, sag_sp, sag_tss, vox_mm, tv_z,
                seg_axis, seg_src)

            logger.info(
                f"  {side:5s} P1: {p1['classification']:22s} "
                f"h={p1['tp_height_mm']:.1f}mm  "
                f"d={p1['dist_mm']:.1f}mm  "
                f"|cosθ|={p1.get('tp_axis_cos_segmental','?')}  "
                f"Δ={p1.get('tp_axis_deg_from_segmental','?')}°  "
                f"z=[{p1.get('tp_z_min_vox','?')},{p1.get('tp_z_max_vox','?')}]  "
                f"sac={p1.get('sacrum_source','?')}"
            )

            if p1['contact'] and p2_available:
                p2 = phase2_axial(side, tp_lbl, ax_sp, ax_tss, ax_t2w, ax_vox_mm)
                p1['phase2']         = p2
                p1['classification'] = p2['classification']
                logger.info(f"  {side:5s} P2: {p2['classification']}  "
                            f"valid={p2.get('p2_valid')}  "
                            f"reason={p2.get('p2_features', {}).get('reason','?')}")
            elif p1['contact'] and not p2_available:
                p1['classification'] = 'Type II'
                p1['phase2']         = {'phase2_attempted': False,
                                        'p2_note': 'Axial data unavailable — Type II (conservative)'}
                out['confidence']    = 'low'

            out[side] = p1

        except Exception as exc:
            out['errors'].append(f'{side}: {exc}')
            logger.error(f"  {side} failed: {exc}")
            logger.debug(traceback.format_exc())

    # ── Final Castellvi type ────────────────────────────────────────────────────
    l_cls = out['left'].get('classification',  'Normal')
    r_cls = out['right'].get('classification', 'Normal')
    valid = {l_cls, r_cls} - {'Normal', 'CONTACT_PENDING_P2'}

    if valid:
        RANK = {'Type I': 1, 'Type II': 2, 'Type III': 3, 'Type IV': 4}
        if (l_cls not in ('Normal', 'CONTACT_PENDING_P2') and
                r_cls not in ('Normal', 'CONTACT_PENDING_P2')):
            out['castellvi_type'] = (l_cls + 'b' if l_cls == r_cls else 'Type IV')
        else:
            dominant = max(valid, key=lambda t: RANK.get(t, 0))
            out['castellvi_type'] = dominant + 'a'

        out['lstv_detected'] = True
        out['lstv_reason'].append(f"Castellvi {out['castellvi_type']} — TP morphology")
        logger.info(f"  ✓ [{study_id}] Castellvi: {out['castellvi_type']}")
    else:
        logger.info(f"  ✗ [{study_id}] No Castellvi finding")

    # ── Extended LSTV morphometrics ─────────────────────────────────────────────
    if run_morpho:
        try:
            masks  = load_lstv_masks(study_id, spineps_dir, totalspine_dir)
            morpho = analyze_lstv(masks, castellvi_result=out)
            out['lstv_morphometrics'] = morpho.to_dict()

            consensus = morpho.lumbar_count_consensus
            phenotype = morpho.lstv_phenotype or 'normal'

            if consensus is not None and consensus != EXPECTED_LUMBAR:
                out['lstv_detected'] = True
                direction = 'LUMBARIZATION' if consensus > EXPECTED_LUMBAR else 'SACRALIZATION'
                reason = (f"Lumbar count = {consensus} (expected {EXPECTED_LUMBAR}) — "
                          f"{direction} by vertebral counting "
                          f"(TSS={morpho.lumbar_count_tss}, "
                          f"VERIDAH={morpho.lumbar_count_veridah})")
                out['lstv_reason'].append(reason)
                logger.info(f"  ✓ [{study_id}] LSTV: {reason}")

            if phenotype in ('sacralization', 'lumbarization'):
                out['lstv_detected'] = True
                primary = morpho.primary_criteria_met or []
                reason  = (f"Phenotype: {phenotype.upper()} "
                           f"({morpho.phenotype_confidence} confidence) — "
                           f"criteria: {'; '.join(primary)}")
                if not any('Phenotype' in r for r in out['lstv_reason']):
                    out['lstv_reason'].append(reason)
                logger.info(f"  ✓ [{study_id}] LSTV: {reason}")

            logger.info(
                f"  [{study_id}] Morphometrics: "
                f"TV={morpho.tv_name}, count={consensus}, "
                f"phenotype={phenotype} ({morpho.phenotype_confidence}), "
                f"primary={morpho.primary_criteria_met}"
            )

        except Exception as exc:
            logger.error(f"  [{study_id}] lstv_engine error: {exc}")
            out['errors'].append(f'lstv_engine: {exc}')

    # ── Final summary ───────────────────────────────────────────────────────────
    morpho_dict = out.get('lstv_morphometrics') or {}
    probs_dict  = morpho_dict.get('probabilities') or {}
    sr_dict     = morpho_dict.get('surgical_relevance') or {}
    p_sac       = probs_dict.get('p_sacralization', 0)
    p_lumb      = probs_dict.get('p_lumbarization', 0)
    p_norm      = probs_dict.get('p_normal', 0)
    wl_risk     = sr_dict.get('wrong_level_risk', '?')
    bert_prob   = sr_dict.get('bertolotti_probability', 0)
    rdr         = morpho_dict.get('relative_disc_ratio')
    deg_l       = out['left'].get('tp_axis_deg_from_segmental', '?')
    deg_r       = out['right'].get('tp_axis_deg_from_segmental', '?')

    if out['lstv_detected']:
        ph_str = ''
        if morpho_dict.get('lstv_phenotype'):
            ph_str = (f"phenotype={morpho_dict['lstv_phenotype']} "
                      f"({morpho_dict.get('phenotype_confidence','')})")
        logger.info(
            f"  ✓✓ [{study_id}] LSTV DETECTED  "
            f"Castellvi={out.get('castellvi_type','None')}  {ph_str}  "
            f"P(sac)={p_sac:.0%}  P(lumb)={p_lumb:.0%}  P(norm)={p_norm:.0%}  "
            f"surgical_risk={wl_risk}  bertolotti={bert_prob:.0%}"
            + (f"  disc_ratio={rdr:.2f}" if rdr is not None else "")
            + f"  axis_dev={deg_l}°/{deg_r}° (L/R)"
        )
    else:
        logger.info(
            f"  ✗✗ [{study_id}] No LSTV  "
            f"P(sac)={p_sac:.0%}  P(lumb)={p_lumb:.0%}  P(norm)={p_norm:.0%}"
        )

    out['pathology_score'] = compute_lstv_pathology_score(
        out, out.get('lstv_morphometrics'))

    return out


# ══════════════════════════════════════════════════════════════════════════════
# STUDY SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def select_studies_csv(csv_path: Path, top_n: int, rank_by: str,
                        valid_ids: Optional[set]) -> List[str]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    df['study_id'] = df['study_id'].astype(str)
    if valid_ids: df = df[df['study_id'].isin(valid_ids)]
    df = df.sort_values(rank_by, ascending=False).reset_index(drop=True)
    ids = df.head(top_n)['study_id'].tolist() + df.tail(top_n)['study_id'].tolist()
    seen, result = set(), []
    for sid in ids:
        if sid not in seen: result.append(sid); seen.add(sid)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Hybrid Two-Phase LSTV Castellvi Classifier + Morphometrics',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--spineps_dir',      required=True)
    parser.add_argument('--totalspine_dir',   required=True)
    parser.add_argument('--registered_dir',   required=True)
    parser.add_argument('--nifti_dir',        required=True)
    parser.add_argument('--output_dir',       required=True)
    parser.add_argument('--study_id',         default=None)
    parser.add_argument('--all',              action='store_true')
    parser.add_argument('--uncertainty_csv',  default=None)
    parser.add_argument('--valid_ids',        default=None)
    parser.add_argument('--top_n',            type=int, default=None)
    parser.add_argument('--rank_by',          default='l5_s1_confidence')
    parser.add_argument('--no_morpho',        action='store_true')
    args = parser.parse_args()

    spineps_dir    = Path(args.spineps_dir)
    totalspine_dir = Path(args.totalspine_dir)
    registered_dir = Path(args.registered_dir)
    nifti_dir      = Path(args.nifti_dir)
    output_dir     = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seg_root = spineps_dir / 'segmentations'

    if args.study_id:
        study_ids = [args.study_id]
    elif args.all:
        study_ids = sorted(d.name for d in seg_root.iterdir() if d.is_dir())
        logger.info(f"ALL mode: {len(study_ids)} studies")
    else:
        if not args.uncertainty_csv or not args.top_n:
            parser.error("--uncertainty_csv and --top_n required unless --all or --study_id")
        valid = (set(str(x) for x in np.load(args.valid_ids)) if args.valid_ids else None)
        study_ids = select_studies_csv(Path(args.uncertainty_csv), args.top_n,
                                        args.rank_by, valid)
        study_ids = [s for s in study_ids if (seg_root / s).is_dir()]

    logger.info(f"Processing {len(study_ids)} studies")

    results: List[dict]  = []
    errors               = 0
    castellvi_counts     = {k: 0 for k in
                            ['Type Ia','Type Ib','Type IIa','Type IIb',
                             'Type IIIa','Type IIIb','Type IV']}
    phenotype_counts: Dict[str, int] = {}
    axis_deviations: List[float]     = []   # per-TP axis-vs-segmental angle (deg)

    for sid in study_ids:
        logger.info(f"\n{'='*60}\n[{sid}]")
        try:
            r = classify_study(
                sid, spineps_dir, totalspine_dir, registered_dir, nifti_dir,
                run_morpho=not args.no_morpho,
            )
            results.append(r)
            if r.get('errors'): errors += 1
            ct = r.get('castellvi_type') or ''
            for k in castellvi_counts:
                if ct.replace(' ', '') == k.replace(' ', ''):
                    castellvi_counts[k] += 1
            morpho = r.get('lstv_morphometrics') or {}
            ph = morpho.get('lstv_phenotype', 'normal')
            phenotype_counts[ph] = phenotype_counts.get(ph, 0) + 1
            for side in ('left', 'right'):
                deg = r.get(side, {}).get('tp_axis_deg_from_segmental')
                if deg is not None:
                    axis_deviations.append(float(deg))
        except Exception as exc:
            logger.error(f"  Unhandled: {exc}")
            logger.debug(traceback.format_exc())
            errors += 1

    lstv_n = sum(1 for r in results if r.get('lstv_detected'))
    n      = max(len(results), 1)
    scores = sorted(
        ((r['study_id'], r.get('pathology_score') or 0) for r in results),
        key=lambda t: t[1], reverse=True,
    )

    p_sac_vals  = []
    p_lumb_vals = []
    wl_risk_counts: Dict[str, int] = {}
    bertolotti_ge50 = 0
    high_cert_sac   = 0
    high_cert_lumb  = 0
    nerve_ambig     = 0
    rel_disc_low    = 0

    for r in results:
        morpho = r.get('lstv_morphometrics') or {}
        probs  = morpho.get('probabilities') or {}
        ps = probs.get('p_sacralization', 0)
        pl = probs.get('p_lumbarization', 0)
        p_sac_vals.append(ps); p_lumb_vals.append(pl)
        if ps > 0.80: high_cert_sac  += 1
        if pl > 0.80: high_cert_lumb += 1
        sr  = morpho.get('surgical_relevance') or {}
        wlr = sr.get('wrong_level_risk', 'low')
        wl_risk_counts[wlr] = wl_risk_counts.get(wlr, 0) + 1
        if sr.get('nerve_root_ambiguity'): nerve_ambig += 1
        if sr.get('bertolotti_probability', 0) >= 0.50: bertolotti_ge50 += 1
        rdr = morpho.get('relative_disc_ratio')
        if rdr is not None and rdr < 0.65: rel_disc_low += 1

    sep = '=' * 60
    logger.info(f"\n{sep}")
    logger.info(f"{'LSTV DETECTION SUMMARY':^60}")
    logger.info(f"{sep}")
    logger.info(f"Studies processed:        {len(results)}")
    logger.info(f"LSTV detected:            {lstv_n}  ({100*lstv_n/n:.1f}%)")
    logger.info(f"  Sacralization:          {phenotype_counts.get('sacralization',0)}")
    logger.info(f"  Lumbarization:          {phenotype_counts.get('lumbarization',0)}")
    logger.info(f"  Transitional:           {phenotype_counts.get('transitional_indeterminate',0)}")
    logger.info(f"  Normal:                 {phenotype_counts.get('normal',0)}")
    logger.info(f"Errors:                   {errors}")

    if axis_deviations:
        logger.info(f"\n── Segmental Axis QA (TP principal axis vs segmental SI) ─")
        logger.info(f"  mean={np.mean(axis_deviations):.1f}°  "
                    f"median={np.median(axis_deviations):.1f}°  "
                    f"max={np.max(axis_deviations):.1f}°  "
                    f"n_TPs={len(axis_deviations)}")
        n_dev = sum(1 for d in axis_deviations if d > 20.0)
        logger.info(f"  TPs deviating >20° from segmental axis: "
                    f"{n_dev} ({100*n_dev/len(axis_deviations):.1f}%)")

    logger.info(f"\n── Castellvi Type Breakdown ──────────────────────────────")
    for t, cnt in castellvi_counts.items():
        if cnt: logger.info(f"  {t:12s}: {cnt}")
    total_ct = sum(castellvi_counts.values())
    logger.info(f"  {'TOTAL':12s}: {total_ct}  ({100*total_ct/n:.1f}%)")

    logger.info(f"\n── Probability Model Statistics ──────────────────────────")
    if p_sac_vals:
        logger.info(f"  P(sacralization):  mean={np.mean(p_sac_vals):.2%}  "
                    f"median={np.median(p_sac_vals):.2%}  >80%: {high_cert_sac}")
        logger.info(f"  P(lumbarization):  mean={np.mean(p_lumb_vals):.2%}  "
                    f"median={np.median(p_lumb_vals):.2%}  >80%: {high_cert_lumb}")
        logger.info(f"  Relative disc ratio <0.65: {rel_disc_low} studies")

    logger.info(f"\n── Surgical Risk Distribution ────────────────────────────")
    for risk_lvl in ('critical', 'high', 'moderate', 'low-moderate', 'low'):
        cnt = wl_risk_counts.get(risk_lvl, 0)
        if cnt: logger.info(f"  {risk_lvl:14s}: {cnt} studies  ({100*cnt/n:.1f}%)")
    logger.info(f"  Nerve root ambiguity:   {nerve_ambig}")
    logger.info(f"  Bertolotti P≥50%:       {bertolotti_ge50}")

    logger.info(f"\n── Top-10 Pathology Scores ───────────────────────────────")
    for sid, sc in scores[:10]:
        r_m  = next((r for r in results if r['study_id'] == sid), {})
        mo   = r_m.get('lstv_morphometrics') or {}
        pr   = mo.get('probabilities') or {}
        sr   = mo.get('surgical_relevance') or {}
        rdr  = mo.get('relative_disc_ratio')
        srfb = ' [SR-fallback]' if sr.get('calibration_note','').startswith('fallback') else ''
        logger.info(
            f"  {sid}: score={sc:.1f}  {mo.get('lstv_phenotype','?')}  "
            f"castellvi={r_m.get('castellvi_type','None')}  "
            f"P(sac)={pr.get('p_sacralization',0):.0%}  "
            f"P(lumb)={pr.get('p_lumbarization',0):.0%}  "
            f"surgical_risk={sr.get('wrong_level_risk','?')}  "
            f"bertolotti={sr.get('bertolotti_probability',0):.0%}"
            + (f"  disc_ratio={rdr:.2f}" if rdr is not None else "")
            + srfb
        )
    logger.info(f"\n{sep}")

    out_json = output_dir / 'lstv_results.json'
    with open(out_json, 'w') as fh:
        json.dump(results, fh, indent=2, default=str)

    summary = {
        'total':                        len(results),
        'lstv_detected':                lstv_n,
        'lstv_rate':                    round(lstv_n / n, 4),
        'errors':                       errors,
        'castellvi_breakdown':          castellvi_counts,
        'phenotype_breakdown':          phenotype_counts,
        'probability_stats': {
            'mean_p_sacralization':     round(float(np.mean(p_sac_vals)), 4) if p_sac_vals else None,
            'mean_p_lumbarization':     round(float(np.mean(p_lumb_vals)), 4) if p_lumb_vals else None,
            'high_confidence_sac':      high_cert_sac,
            'high_confidence_lumb':     high_cert_lumb,
            'relative_disc_ratio_low':  rel_disc_low,
        },
        'segmental_axis_qa': {
            'n_tps_measured':   len(axis_deviations),
            'mean_dev_deg':     round(float(np.mean(axis_deviations)),   2) if axis_deviations else None,
            'median_dev_deg':   round(float(np.median(axis_deviations)), 2) if axis_deviations else None,
            'max_dev_deg':      round(float(np.max(axis_deviations)),    2) if axis_deviations else None,
            'n_dev_gt20deg':    sum(1 for d in axis_deviations if d > 20.0),
        },
        'surgical_risk_breakdown':      wl_risk_counts,
        'nerve_root_ambiguity_count':   nerve_ambig,
        'bertolotti_probability_ge50':  bertolotti_ge50,
        'top_scores':                   scores[:20],
    }
    with open(output_dir / 'lstv_summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.info(f"Results → {out_json}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
