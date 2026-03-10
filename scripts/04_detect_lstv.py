#!/usr/bin/env python3
"""
04_detect_lstv.py — Hybrid Two-Phase LSTV Castellvi Classifier
===============================================================
v4.4  (IAN PAN VERTEBRAL ARBITRATION EDITION)
----------------------------------------------

NEW IN v4.4
-----------
IAN PAN VERTEBRAL BODY ARBITRATION
  Ian Pan's disc-level confidence predictions now act as a first-class
  arbitrator whenever TotalSpineSeg (TSS) and VERIDAH (SPINEPS seg-vert_msk)
  disagree on vertebral level identity (Dice offset != 0).

  When offset == 0 (both models agree): Ian Pan is informational only —
  its scores are logged and stored but preferred_hypothesis is not changed.

  When offset != 0 (models disagree): Ian Pan arbitrates by inferring
  vertebral body positions from adjacent disc midpoints and scoring each
  segmentation model by proximity in RAS mm space.

  VERTEBRAL BODY INFERENCE
  For each adjacent disc pair where both peaks meet SEQ_VOTE_MIN_PROB (0.30):
    inferred_body_RAS = midpoint(world_ras_mm[disc_above],
                                  world_ras_mm[disc_below])

  This gives four inferred vertebral body positions:
    L2 = midpoint(L1/L2 peak, L2/L3 peak)
    L3 = midpoint(L2/L3 peak, L3/L4 peak)
    L4 = midpoint(L3/L4 peak, L4/L5 peak)
    L5 = midpoint(L4/L5 peak, L5/S1 peak)

  SCORING
    score_TSS    = mean dist(inferred_body_k, TSS_vert_k_centroid_RAS)
    score_VERI   = mean dist(inferred_body_k, VERIDAH_vert_k_centroid_RAS)
    margin_mm    = score_VERI - score_TSS
      > 0  =>  TSS centroids are closer  =>  TSS labeling is anatomically correct
      < 0  =>  VERIDAH centroids are closer  =>  VERIDAH labeling is correct

  OVERRIDE POLICY
    offset != 0 and n_pairs >= IP_MIN_VERT_PAIRS (2) and
    |margin_mm| >= IP_VERT_MIN_MARGIN_MM (5.0 mm):
      TSS wins     => preferred_hypothesis = 'aligned'
      VERIDAH wins => 'shifted_plus_1' (offset > 0) or 'shifted_minus_1' (offset < 0)

  All Ian Pan fields stored on AlignmentResult and written to lstv_alignment.csv
  regardless of whether an override fires.

  The legacy disc-vs-disc sequence vote fields are preserved for auditability.

USAGE
  Pass --ian_pan_coords path/to/ian_pan_disc_coords.json to enable.
  No JSON regeneration needed — uses the existing world_ras_mm field.
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
from vertebral_alignment import (
    analyse_vertebral_alignment, compute_cohort_stats,
    AlignmentResult, CohortAlignmentStats,
)
from lstv_csv_reporter import write_csv_reports

try:
    from ian_pan_disc_coords import load_ian_pan_disc_coords
    _IAN_PAN_AVAILABLE = True
except ImportError:
    _IAN_PAN_AVAILABLE = False

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
TV_TO_DISC_LABELS: Dict[int, Tuple[Optional[int], Optional[int]]] = {
    VD_L1: (None,  92),
    VD_L2: (92,    93),
    VD_L3: (93,    94),
    VD_L4: (94,    95),
    VD_L5: (95,   100),
    VD_L6: (None, None),
}

TV_TO_VERT_FALLBACK: Dict[int, Tuple[Optional[int], Optional[int]]] = {
    VD_L1: (None,       42),
    VD_L2: (41,         43),
    VD_L3: (42,         44),
    VD_L4: (43,         45),
    VD_L5: (44,         TSS_SACRUM),
    VD_L6: (45,         TSS_SACRUM),
}

TV_TO_VD_FALLBACK: Dict[int, Tuple[Optional[int], Optional[int]]] = {
    VD_L1: (None,    VD_L2),
    VD_L2: (VD_L1,  VD_L3),
    VD_L3: (VD_L2,  VD_L4),
    VD_L4: (VD_L3,  VD_L5),
    VD_L5: (VD_L4,  VD_SAC),
    VD_L6: (VD_L5,  VD_SAC),
}

TSS_VERT_TO_VD: Dict[int, int] = {v: k for k, v in VD_TO_TSS_VERT.items()}

# ── Ian Pan arbitration constants ─────────────────────────────────────────────
DISC_NAMES            = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
SEQ_VOTE_MIN_PROB     = 0.30   # minimum peak_prob for a disc to contribute
IP_VERT_MIN_MARGIN_MM = 5.0    # Ian Pan must beat other model by this to arbitrate
IP_MIN_VERT_PAIRS     = 2      # minimum inferred vertebral body pairs required

# (disc_above, disc_below) -> (vert_name, TSS_label, VERIDAH_label)
# The body between two adjacent discs is approximated by their midpoint.
#   TSS labels:     41=L1  42=L2  43=L3  44=L4  45=L5
#   VERIDAH labels: 20=L1  21=L2  22=L3  23=L4  24=L5  25=L6
IP_DISC_PAIR_TO_VERT: Dict[Tuple[str, str], Tuple[str, int, int]] = {
    ("l1_l2", "l2_l3"): ("L2", 42, 21),
    ("l2_l3", "l3_l4"): ("L3", 43, 22),
    ("l3_l4", "l4_l5"): ("L4", 44, 23),
    ("l4_l5", "l5_s1"): ("L5", 45, 24),
}


# ══════════════════════════════════════════════════════════════════════════════
# NIfTI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _ras_centroid(data: np.ndarray, label: int,
                  affine: np.ndarray) -> Optional[np.ndarray]:
    """RAS-mm centroid of all voxels with the given integer label, or None."""
    mask = (data == label)
    if not mask.any():
        return None
    idx = np.array(np.where(mask), dtype=np.float64).mean(axis=1)  # (3,) ijk
    ras = affine[:3, :3] @ idx + affine[:3, 3]
    return ras


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


def run_cross_validation(sag_spineps, sag_vert, sag_tss, vox_mm, study_id):
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


def isolate_tp_at_tv(sp_data, tp_label, z_min, z_max):
    tp    = (sp_data == tp_label)
    out   = np.zeros_like(tp)
    z_lo  = max(0, z_min - 3)
    z_hi  = min(sp_data.shape[2] - 1, z_max + 3)
    out[:, :, z_lo:z_hi + 1] = tp[:, :, z_lo:z_hi + 1]
    return out


def inferiormost_tp_cc(tp_mask, sacrum_mask=None):
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


def min_dist_3d(mask_a, mask_b, vox_mm):
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
# TV Z-RANGE
# ══════════════════════════════════════════════════════════════════════════════

def get_tv_z_range(sag_vert, sag_tss, tv_label, study_id):
    tss_lbl = VD_TO_TSS_VERT.get(tv_label)
    if tss_lbl is not None and sag_tss is not None:
        tss_mask = (sag_tss == tss_lbl)
        if tss_mask.any():
            zr  = get_z_range_from_mask(tss_mask)
            src = f'TSS label {tss_lbl} ({VERIDAH_NAMES.get(tv_label, str(tv_label))})'
            logger.info(f"  [{study_id}] TV Z range from {src}: {zr}")
            return zr, src
        logger.warning(f"  [{study_id}] TSS label {tss_lbl} empty — falling back to VERIDAH")

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
# SEGMENTAL AXIS
# ══════════════════════════════════════════════════════════════════════════════

def _mask_centroid_mm(mask, vox_mm):
    if not mask.any(): return None
    idx = np.array(np.where(mask), dtype=np.float64)
    return idx.mean(axis=1) * vox_mm


def compute_segmental_axis(sag_tss, sag_vert, tv_label, vox_mm, study_id):
    disc_above_lbl, disc_below_lbl = TV_TO_DISC_LABELS.get(tv_label, (None, None))
    vert_above_tss, vert_below_tss = TV_TO_VERT_FALLBACK.get(tv_label, (None, None))
    vd_above,       vd_below       = TV_TO_VD_FALLBACK.get(tv_label, (None, None))

    sup_pt = inf_pt = None
    sup_src = inf_src = ''

    if disc_above_lbl is not None and sag_tss is not None:
        m = (sag_tss == disc_above_lbl)
        if m.any(): sup_pt = _mask_centroid_mm(m, vox_mm); sup_src = f'TSS disc {disc_above_lbl}'

    if sup_pt is None and vert_above_tss is not None and sag_tss is not None:
        m = (sag_tss == vert_above_tss)
        if m.any(): sup_pt = _mask_centroid_mm(m, vox_mm); sup_src = f'TSS vert {vert_above_tss}'

    if sup_pt is None and vd_above is not None:
        m = (sag_vert == vd_above)
        if m.any(): sup_pt = _mask_centroid_mm(m, vox_mm); sup_src = f'VERIDAH vert {vd_above}'

    if disc_below_lbl is not None and sag_tss is not None:
        m = (sag_tss == disc_below_lbl)
        if m.any(): inf_pt = _mask_centroid_mm(m, vox_mm); inf_src = f'TSS disc {disc_below_lbl}'

    if inf_pt is None and vert_below_tss is not None and sag_tss is not None:
        m = (sag_tss == vert_below_tss)
        if m.any():
            inf_pt  = _mask_centroid_mm(m, vox_mm)
            inf_src = (f'TSS sacrum' if vert_below_tss == TSS_SACRUM
                       else f'TSS vert {vert_below_tss}')

    if inf_pt is None and vd_below is not None:
        m = (sag_vert == vd_below)
        if m.any():
            inf_pt  = _mask_centroid_mm(m, vox_mm)
            inf_src = (f'VERIDAH sacrum' if vd_below == VD_SAC else f'VERIDAH vert {vd_below}')

    if sup_pt is None or inf_pt is None:
        missing = ([' superior'] if sup_pt is None else []) + ([' inferior'] if inf_pt is None else [])
        logger.warning(f"  [{study_id}] Segmental axis: cannot locate {''.join(missing)} — scanner Z")
        return np.array([0., 0., 1.]), 'scanner Z axis (fallback)'

    if 'VERIDAH' in sup_src or 'VERIDAH' in inf_src:
        logger.warning(f"  [{study_id}] Segmental axis using VERIDAH ref ({sup_src} <-> {inf_src})")

    raw  = sup_pt - inf_pt
    norm = float(np.linalg.norm(raw))
    if norm < 1e-6:
        return np.array([0., 0., 1.]), 'scanner Z axis (degenerate centroids)'

    unit = raw / norm
    src  = f'{inf_src} -> {sup_src}'
    logger.info(f"  [{study_id}] Segmental SI axis: {src}  "
                f"v=[{unit[0]:.3f},{unit[1]:.3f},{unit[2]:.3f}]")
    return unit, src


# ══════════════════════════════════════════════════════════════════════════════
# TP HEIGHT
# ══════════════════════════════════════════════════════════════════════════════

def measure_tp_height_principal_axis(tp_mask, vox_mm, segmental_axis):
    if not tp_mask.any():
        return 0.0, segmental_axis.copy(), 0.0, np.eye(3)
    idx    = np.array(np.where(tp_mask), dtype=np.float64).T
    pts_mm = idx * vox_mm
    n_pts  = pts_mm.shape[0]
    if n_pts < 3:
        proj = pts_mm @ segmental_axis
        return (float(proj.max() - proj.min()) if n_pts > 1 else 0.0), \
               segmental_axis.copy(), 1.0, np.eye(3)
    centred = pts_mm - pts_mm.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        proj = pts_mm @ segmental_axis
        return float(proj.max() - proj.min()), segmental_axis.copy(), 1.0, np.eye(3)
    dots    = np.abs(Vt @ segmental_axis)
    best    = int(np.argmax(dots))
    chosen  = Vt[best].copy()
    if np.dot(chosen, segmental_axis) < 0: chosen = -chosen
    cos_align = float(dots[best])
    proj      = pts_mm @ chosen
    return float(proj.max() - proj.min()), chosen, cos_align, Vt


# ══════════════════════════════════════════════════════════════════════════════
# L6 VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

TSS_DISC_LABELS = set(range(91, 101))


def _verify_l6(sag_vert, sag_tss, vox_mm, tv_z, study_id):
    if sag_tss is None:
        return False, "no TSS available — cannot verify L6"
    l6_z_min, l6_z_max = tv_z
    l6_centroid_z = (l6_z_min + l6_z_max) / 2.0
    tss_l5  = (sag_tss == 45)
    tss_sac = (sag_tss == TSS_SACRUM)
    if not tss_sac.any():
        return False, "TSS sacrum (label 50) absent"
    sac_z_sup = float(np.where(tss_sac)[2].max())
    if l6_centroid_z <= sac_z_sup:
        return False, f"L6 centroid z={l6_centroid_z:.0f} <= TSS sacrum sup z={sac_z_sup:.0f}"
    if tss_l5.any():
        l5_z_min = float(np.where(tss_l5)[2].min())
        if l6_centroid_z >= l5_z_min:
            return False, f"L6 centroid z={l6_centroid_z:.0f} >= TSS L5 inf z={l5_z_min:.0f}"
    disc_above_z = disc_below_z = None
    for disc_lbl in TSS_DISC_LABELS:
        disc_mask = (sag_tss == disc_lbl)
        if not disc_mask.any(): continue
        disc_zc = float(np.mean(np.where(disc_mask)[2]))
        if disc_zc > l6_z_max + 2:
            if disc_above_z is None or disc_zc < disc_above_z: disc_above_z = disc_zc
        if disc_zc < l6_z_min - 2 and disc_zc > sac_z_sup:
            if disc_below_z is None or disc_zc > disc_below_z: disc_below_z = disc_zc
    if disc_above_z is None:
        return False, f"no TSS disc above VERIDAH L6 — no L5-L6 disc space"
    if disc_below_z is None:
        return False, f"no TSS disc between L6 inf and sacrum sup — no L6-S1 disc space"
    return True, (f"positional OK (centroid z={l6_centroid_z:.0f}), "
                  f"disc above z={disc_above_z:.0f}, disc below z={disc_below_z:.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2
# ══════════════════════════════════════════════════════════════════════════════

def _extract_bbox(axial_t2w, midpoint, half=BBOX_HALF):
    x0, y0, z0 = int(midpoint[0]), int(midpoint[1]), int(midpoint[2])
    nx, ny, nz  = axial_t2w.shape
    if not (0 <= z0 < nz): return None
    patch = axial_t2w[max(0,x0-half):min(nx,x0+half),
                      max(0,y0-half):min(ny,y0+half), z0].copy()
    return patch if patch.size > 0 else None


def _classify_signal(patch, axial_t2w):
    vals = patch.astype(float).ravel()
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 'Type II', {'reason': 'empty patch', 'valid': False}
    p_mean  = float(np.mean(vals)); p_std = float(np.std(vals))
    cv      = p_std / (p_mean + 1e-6)
    global_fg = axial_t2w[axial_t2w > 0]
    p95     = float(np.percentile(global_fg, 95)) if global_fg.size else 1.0
    dark_thr= P2_DARK_CLEFT_FRAC * p95
    feats   = {'patch_mean': round(p_mean,2), 'patch_std': round(p_std,2),
               'coeff_var': round(cv,4), 'global_p95': round(p95,2),
               'dark_thresh': round(dark_thr,2), 'valid': True}
    if p_mean < dark_thr:
        feats['reason'] = f"mean={p_mean:.1f} < dark_thr={dark_thr:.1f} -> Type II"
        return 'Type II', feats
    elif cv < P2_MIN_STD_RATIO:
        feats['reason'] = f"CV={cv:.3f} < {P2_MIN_STD_RATIO} — uniform bright -> Type III"
        return 'Type III', feats
    else:
        feats['reason'] = "Bright heterogeneous — ambiguous; Type II (conservative)"
        return 'Type II', feats


def phase2_axial(side, tp_label, ax_spineps, ax_tss, ax_t2w, ax_vox_mm):
    out = {'phase2_attempted': True, 'classification': 'Type II',
           'midpoint_vox': None, 'p2_features': None, 'p2_valid': False}
    tp_ax  = (ax_spineps == tp_label)
    sac_ax = (ax_tss     == TSS_SACRUM)
    if not tp_ax.any():
        out['p2_note'] = f"TP label {tp_label} absent in registered SPINEPS"
        return out
    if not sac_ax.any():
        out['p2_note'] = f"Sacrum {TSS_SACRUM} absent in axial TSS"
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
    cls, feats            = _classify_signal(patch, ax_t2w)
    out['classification'] = cls
    out['p2_features']    = feats
    out['p2_valid']       = feats.get('valid', False)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1
# ══════════════════════════════════════════════════════════════════════════════

def phase1_sagittal(side, tp_label, sag_sp, sag_tss, sag_vox_mm,
                    tv_z_range, segmental_axis, segmental_src):
    out = {
        'tp_present': False, 'tp_height_mm': 0.0, 'contact': False,
        'dist_mm': float('inf'), 'tp_vox': None, 'sacrum_vox': None,
        'classification': 'Normal', 'phase1_done': False, 'sacrum_source': None,
        'tp_axis_chosen': None, 'tp_axis_cos_segmental': None,
        'tp_axis_deg_from_segmental': None,
        'tp_segmental_axis': segmental_axis.tolist(),
        'tp_segmental_axis_source': segmental_src,
    }
    tss_sac = (sag_tss == TSS_SACRUM) if sag_tss is not None else None
    if tss_sac is not None and tss_sac.any():
        sac_mask             = tss_sac
        out['sacrum_source'] = f'TSS label {TSS_SACRUM}'
    else:
        sac_mask             = (sag_sp == SP_SACRUM)
        out['sacrum_source'] = 'SPINEPS label 26 (fallback)'

    tp_at_tv = isolate_tp_at_tv(sag_sp, tp_label, *tv_z_range)
    tp_mask  = inferiormost_tp_cc(tp_at_tv, sac_mask if sac_mask.any() else None)
    if not tp_mask.any():
        out['note'] = f"TP label {tp_label} absent at TV level Z={tv_z_range}"
        return out

    out['tp_present'] = True
    tp_z_coords         = np.where(tp_mask)[2]
    out['tp_z_min_vox'] = int(tp_z_coords.min())
    out['tp_z_max_vox'] = int(tp_z_coords.max())
    out['tp_centroid_z_mm'] = round(float(tp_z_coords.mean()) * sag_vox_mm[2], 2)

    h_mm, chosen_axis, cos_align, _ = measure_tp_height_principal_axis(
        tp_mask, sag_vox_mm, segmental_axis)
    out['tp_height_mm']               = round(h_mm, 3)
    out['tp_axis_chosen']             = [round(float(v), 5) for v in chosen_axis]
    out['tp_axis_cos_segmental']      = round(cos_align, 4)
    out['tp_axis_deg_from_segmental'] = round(
        float(np.degrees(np.arccos(np.clip(cos_align, 0., 1.)))), 2)

    if not sac_mask.any():
        return out

    dist_mm, tp_vox, sac_vox = min_dist_3d(tp_mask, sac_mask, sag_vox_mm)
    out['dist_mm']     = round(float(dist_mm), 3)
    out['phase1_done'] = True
    if tp_vox  is not None: out['tp_vox']     = tp_vox.tolist()
    if sac_vox is not None: out['sacrum_vox'] = sac_vox.tolist()

    if dist_mm > CONTACT_DIST_MM:
        out['contact'] = False
        if h_mm >= TP_HEIGHT_MM:
            out['classification'] = 'Type I'
    else:
        out['contact']        = True
        out['classification'] = 'CONTACT_PENDING_P2'
    return out


# ══════════════════════════════════════════════════════════════════════════════
# CASTELLVI TYPE FINALISER
# ══════════════════════════════════════════════════════════════════════════════

def finalise_castellvi(left_result: dict, right_result: dict,
                        p2_available: bool,
                        ax_sp, ax_tss, ax_t2w, ax_vox_mm,
                        confidence_ref: List[str]) -> Tuple[Optional[str], dict, dict]:
    RANK = {'Type I': 1, 'Type II': 2, 'Type III': 3, 'Type IV': 4}

    for side, p1, tp_lbl in (('left',  left_result,  SP_TP_L),
                               ('right', right_result, SP_TP_R)):
        if p1.get('contact') and p2_available:
            p2 = phase2_axial(side, tp_lbl, ax_sp, ax_tss, ax_t2w, ax_vox_mm)
            p1['phase2']         = p2
            p1['classification'] = p2['classification']
        elif p1.get('contact') and not p2_available:
            p1['classification'] = 'Type II'
            p1['phase2']         = {'phase2_attempted': False,
                                     'p2_note': 'Axial data unavailable — Type II'}
            confidence_ref[0]    = 'low'

    l_cls = left_result.get('classification',  'Normal')
    r_cls = right_result.get('classification', 'Normal')
    valid = {l_cls, r_cls} - {'Normal', 'CONTACT_PENDING_P2'}

    if not valid:
        return None, left_result, right_result

    if (l_cls not in ('Normal', 'CONTACT_PENDING_P2') and
            r_cls not in ('Normal', 'CONTACT_PENDING_P2')):
        castellvi = (l_cls + 'b' if l_cls == r_cls else 'Type IV')
    else:
        dominant  = max(valid, key=lambda t: RANK.get(t, 0))
        castellvi = dominant + 'a'

    return castellvi, left_result, right_result


# ══════════════════════════════════════════════════════════════════════════════
# IAN PAN VERTEBRAL ARBITRATION  (v4.4)
# ══════════════════════════════════════════════════════════════════════════════

def apply_ian_pan_tiebreaker(
    alignment_result: AlignmentResult,
    ian_pan_study: dict,
    study_id: str,
    tss_vert_centroids_ras: dict,
    vd_vert_centroids_ras:  dict,
) -> None:
    """
    Infers vertebral body positions from Ian Pan disc peak midpoints
    (world_ras_mm) and scores TSS vs VERIDAH labeling by proximity in RAS mm.

    offset == 0: informational only, no override.
    offset != 0: arbitrates when n_pairs >= IP_MIN_VERT_PAIRS and
                 |margin_mm| >= IP_VERT_MIN_MARGIN_MM.

    Fields written on AlignmentResult (always):
      ip_score_tss_mm, ip_score_vd_mm, ip_vert_margin_mm,
      ip_n_vert_pairs, ip_winner, ip_inferred_verts, ip_tiebreak_applied,
      + legacy disc-level fields (ip_sequence_vote, ip_per_level, etc.)
    """
    if not ian_pan_study:
        return

    agr = ian_pan_study.get("model_agreement", {})
    sv  = agr.get("_sequence_vote", {})

    # ── Extract disc peak positions ───────────────────────────────────────────
    disc_peaks_ras: Dict = {}
    disc_probs:     Dict = {}
    for disc in DISC_NAMES:
        d_lv  = ian_pan_study.get("disc_levels", {}).get(disc, {})
        prob  = float(d_lv.get("peak_prob", 0.0) or 0.0)
        disc_probs[disc] = prob
        coord = d_lv.get("world_ras_mm")
        if coord is not None and prob >= SEQ_VOTE_MIN_PROB:
            disc_peaks_ras[disc] = np.array(coord, dtype=float)
        else:
            disc_peaks_ras[disc] = None

    # ── Infer vertebral body positions ────────────────────────────────────────
    ip_inferred_verts: Dict = {}

    for (disc_above, disc_below), (vert_name, tss_lbl, vd_lbl) in IP_DISC_PAIR_TO_VERT.items():
        pos_a  = disc_peaks_ras.get(disc_above)
        pos_b  = disc_peaks_ras.get(disc_below)
        prob_a = disc_probs.get(disc_above, 0.0)
        prob_b = disc_probs.get(disc_below, 0.0)

        if pos_a is None or pos_b is None:
            ip_inferred_verts[vert_name] = {
                "voted":            False,
                "inferred_pos_ras": None,
                "tss_label":        tss_lbl,
                "vd_label":         vd_lbl,
                "dist_tss_mm":      None,
                "dist_vd_mm":       None,
                "prob_disc_above":  round(prob_a, 3),
                "prob_disc_below":  round(prob_b, 3),
                "skip_reason":      (f"p_above={prob_a:.2f}" if pos_a is None
                                     else f"p_below={prob_b:.2f}") + " < threshold",
            }
            continue

        inferred_pos = (pos_a + pos_b) / 2.0
        tss_ctr = tss_vert_centroids_ras.get(tss_lbl)
        vd_ctr  = vd_vert_centroids_ras.get(vd_lbl)

        dist_tss = float(np.linalg.norm(inferred_pos - tss_ctr)) if tss_ctr is not None else None
        dist_vd  = float(np.linalg.norm(inferred_pos - vd_ctr))  if vd_ctr  is not None else None

        ip_inferred_verts[vert_name] = {
            "voted":            True,
            "inferred_pos_ras": inferred_pos.tolist(),
            "tss_label":        tss_lbl,
            "vd_label":         vd_lbl,
            "dist_tss_mm":      round(dist_tss, 2) if dist_tss is not None else None,
            "dist_vd_mm":       round(dist_vd,  2) if dist_vd  is not None else None,
            "prob_disc_above":  round(prob_a, 3),
            "prob_disc_below":  round(prob_b, 3),
            "closer_model":     ("TSS"     if (dist_tss is not None and dist_vd is not None
                                               and dist_tss <= dist_vd)
                                 else "VERIDAH" if (dist_tss is not None and dist_vd is not None)
                                 else None),
        }

    # ── Aggregate scores ──────────────────────────────────────────────────────
    tss_dists = [v["dist_tss_mm"] for v in ip_inferred_verts.values()
                 if v["voted"] and v["dist_tss_mm"] is not None]
    vd_dists  = [v["dist_vd_mm"]  for v in ip_inferred_verts.values()
                 if v["voted"] and v["dist_vd_mm"]  is not None]

    n_pairs   = min(len(tss_dists), len(vd_dists))
    score_tss = float(np.mean(tss_dists)) if tss_dists else None
    score_vd  = float(np.mean(vd_dists))  if vd_dists  else None

    if score_tss is not None and score_vd is not None:
        margin_mm = score_vd - score_tss   # +ve = TSS closer
        if   margin_mm >  1e-3: ip_winner = "TSS"
        elif margin_mm < -1e-3: ip_winner = "VERIDAH"
        else:                   ip_winner = "tie"
    else:
        margin_mm = None
        ip_winner = "insufficient"

    # ── Legacy disc-vs-disc fields (preserved) ────────────────────────────────
    ip_vote   = sv.get("sequence_vote",   "neutral")
    ip_conf   = sv.get("vote_confidence", "insufficient")
    ip_margin = float(sv.get("margin_mm", 0.0) or 0.0)
    ip_h0     = sv.get("mean_dist_h0_mm")
    ip_h1     = sv.get("mean_dist_h1_mm")
    ip_n      = sv.get("n_levels_voted",  0)

    per_level: Dict = {}
    for disc_name in DISC_NAMES:
        d_ag = agr.get(disc_name, {})
        d_lv = ian_pan_study.get("disc_levels", {}).get(disc_name, {})
        d0   = d_ag.get("dist_to_tss_h0_mm")
        d1   = d_ag.get("dist_to_tss_h1_mm")
        per_level[disc_name] = {
            "peak_prob":  d_lv.get("peak_prob"),
            "entropy":    d_lv.get("entropy"),
            "dist_h0_mm": d0,
            "dist_h1_mm": d1,
            "closer_hyp": ("H0" if (d0 is not None and d1 is not None and d0 <= d1)
                           else "H1" if (d0 is not None and d1 is not None)
                           else None),
        }

    # ── Attach all fields unconditionally ─────────────────────────────────────
    alignment_result.ip_sequence_vote    = ip_vote
    alignment_result.ip_vote_confidence  = ip_conf
    alignment_result.ip_seq_margin_mm    = round(ip_margin, 2)
    alignment_result.ip_mean_dist_h0_mm  = ip_h0
    alignment_result.ip_mean_dist_h1_mm  = ip_h1
    alignment_result.ip_n_levels_voted   = ip_n
    alignment_result.ip_per_level        = per_level
    alignment_result.ip_tiebreak_applied = False
    alignment_result.ip_score_tss_mm     = round(score_tss, 2) if score_tss is not None else None
    alignment_result.ip_score_vd_mm      = round(score_vd,  2) if score_vd  is not None else None
    alignment_result.ip_vert_margin_mm   = round(margin_mm, 2) if margin_mm is not None else None
    alignment_result.ip_n_vert_pairs     = n_pairs
    alignment_result.ip_winner           = ip_winner
    alignment_result.ip_inferred_verts   = ip_inferred_verts

    offset = getattr(alignment_result, "best_offset", 0) or 0

    # ── Logging ────────────────────────────────────────────────────────────────
    if score_tss is not None and score_vd is not None:
        logger.info(
            f"  [{study_id}] Ian Pan vertebral scoring: "
            f"TSS={score_tss:.1f}mm  VERIDAH={score_vd:.1f}mm  "
            f"margin={margin_mm:+.1f}mm -> winner={ip_winner}  "
            f"n={n_pairs} pairs  offset={offset:+d}"
        )
    else:
        logger.info(
            f"  [{study_id}] Ian Pan vertebral scoring: INSUFFICIENT DATA  "
            f"n_pairs={n_pairs}  offset={offset:+d}"
        )

    for vert_name, vd in ip_inferred_verts.items():
        if vd["voted"]:
            dt   = vd["dist_tss_mm"]
            dv   = vd["dist_vd_mm"]
            flag = "TSS" if vd["closer_model"] == "TSS" else "VD"
            logger.info(
                f"    {vert_name}: "
                f"TSS(lbl={vd['tss_label']})={dt:.1f}mm  "
                f"VD(lbl={vd['vd_label']})={dv:.1f}mm  "
                f"-> {flag} closer  "
                f"[p_above={vd['prob_disc_above']:.2f} p_below={vd['prob_disc_below']:.2f}]"
            )
        else:
            logger.info(
                f"    {vert_name}: SKIPPED — {vd.get('skip_reason', 'below threshold')}"
            )

    # ── Override decision ──────────────────────────────────────────────────────
    if offset == 0:
        logger.info(
            f"  [{study_id}] Ian Pan: models agree (offset=0) — "
            f"informational only, no override  "
            f"[TSS={score_tss}mm  VERIDAH={score_vd}mm]"
        )
        return

    if n_pairs < IP_MIN_VERT_PAIRS:
        logger.info(
            f"  [{study_id}] Ian Pan arbitration SKIPPED: "
            f"n_pairs={n_pairs} < {IP_MIN_VERT_PAIRS} required  (offset={offset:+d})"
        )
        return

    if margin_mm is None or abs(margin_mm) < IP_VERT_MIN_MARGIN_MM:
        margin_str = f"{margin_mm:.1f}mm" if margin_mm is not None else "N/A"
        logger.info(
            f"  [{study_id}] Ian Pan arbitration SKIPPED: "
            f"|margin|={margin_str} < {IP_VERT_MIN_MARGIN_MM}mm — ambiguous  "
            f"(offset={offset:+d})"
        )
        return

    old_hyp = alignment_result.preferred_hypothesis

    if ip_winner == "TSS":
        new_hyp = "aligned"
    elif offset > 0:
        new_hyp = "shifted_plus_1"
    else:
        new_hyp = "shifted_minus_1"

    alignment_result.preferred_hypothesis = new_hyp
    alignment_result.confidence           = "moderate"
    alignment_result.ip_tiebreak_applied  = True

    logger.info(
        f"  [{study_id}] !! Ian Pan ARBITRATION: {old_hyp} -> {new_hyp}  "
        f"[winner={ip_winner}  TSS={score_tss:.1f}mm  VERIDAH={score_vd:.1f}mm  "
        f"margin={margin_mm:+.1f}mm  n={n_pairs} pairs  offset={offset:+d}]"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PER-STUDY CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def classify_study(study_id:       str,
                   spineps_dir:    Path,
                   totalspine_dir: Path,
                   registered_dir: Path,
                   nifti_dir:      Path,
                   run_morpho:     bool = True,
                   ian_pan_study:  Optional[dict] = None,
                   ) -> Tuple[dict, Optional[AlignmentResult]]:
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
        'alignment':          None,
        'errors':             [],
    }

    seg_dir    = spineps_dir / 'segmentations' / study_id
    spine_path = seg_dir / f"{study_id}_seg-spine_msk.nii.gz"
    vert_path  = seg_dir / f"{study_id}_seg-vert_msk.nii.gz"
    tss_sag    = totalspine_dir / study_id / 'sagittal' / f"{study_id}_sagittal_labeled.nii.gz"
    tss_ax     = totalspine_dir / study_id / 'axial'    / f"{study_id}_axial_labeled.nii.gz"
    sp_ax      = registered_dir / study_id / f"{study_id}_spineps_reg.nii.gz"

    def _load(path, tag):
        if not path.exists():
            logger.warning(f"  Missing: {path.name}")
            return None, None
        try:
            return load_canonical(path)
        except Exception as exc:
            logger.warning(f"  {tag}: {exc}")
            return None, None

    def _find_t2w(acq):
        sd = nifti_dir / study_id
        if not sd.exists(): return None
        for sub in sorted(sd.iterdir()):
            p = sub / f"sub-{study_id}_acq-{acq}_T2w.nii.gz"
            if p.exists(): return p
        return None

    sag_sp,   sp_nii   = _load(spine_path, 'seg-spine_msk')
    sag_vert, vert_nii = _load(vert_path,  'seg-vert_msk')
    sag_tss,  tss_nii  = _load(tss_sag,    'TSS sagittal')

    if sag_sp   is None: out['errors'].append('Missing SPINEPS seg-spine_msk'); return out, None
    if sag_vert is None: out['errors'].append('Missing SPINEPS seg-vert_msk');  return out, None
    if sag_tss  is None: out['errors'].append('Missing TotalSpineSeg sagittal'); return out, None

    sag_sp   = sag_sp.astype(int)
    sag_vert = sag_vert.astype(int)
    sag_tss  = sag_tss.astype(int)
    vox_mm   = voxel_size_mm(sp_nii)

    # ── Pre-compute RAS centroids for Ian Pan vertebral scoring ───────────────
    tss_vert_centroids_ras: Dict[int, Optional[np.ndarray]] = {
        lbl: _ras_centroid(sag_tss,  lbl, tss_nii.affine)
        for lbl in (41, 42, 43, 44, 45)
    }
    vd_vert_centroids_ras: Dict[int, Optional[np.ndarray]] = {
        lbl: _ras_centroid(sag_vert, lbl, vert_nii.affine)
        for lbl in (20, 21, 22, 23, 24, 25)
    }
    logger.info(
        f"  [{study_id}] RAS centroids: "
        f"TSS={sum(1 for v in tss_vert_centroids_ras.values() if v is not None)}/5  "
        f"VERIDAH={sum(1 for v in vd_vert_centroids_ras.values() if v is not None)}/6"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # ALIGNMENT ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    alignment_result: Optional[AlignmentResult] = None
    try:
        alignment_result = analyse_vertebral_alignment(study_id, sag_tss, sag_vert)
        out['alignment'] = alignment_result.to_dict()
    except Exception as exc:
        logger.error(f"  [{study_id}] Alignment analysis failed: {exc}")
        out['errors'].append(f'alignment: {exc}')

    # ── Ian Pan vertebral arbitration ─────────────────────────────────────────
    if ian_pan_study is not None and alignment_result is not None:
        try:
            apply_ian_pan_tiebreaker(
                alignment_result, ian_pan_study, study_id,
                tss_vert_centroids_ras, vd_vert_centroids_ras,
            )
            out['alignment'] = alignment_result.to_dict()
        except Exception as exc:
            logger.error(f"  [{study_id}] Ian Pan arbitration failed: {exc}")
            out['errors'].append(f'ian_pan_arbitration: {exc}')

    # ── TSS label logging ──────────────────────────────────────────────────────
    tss_unique         = sorted(int(v) for v in np.unique(sag_tss) if v > 0)
    tss_lumbar_present = {lbl: name for lbl, name in TSS_LUMBAR.items() if lbl in tss_unique}
    tss_lumbar_missing = {lbl: name for lbl, name in TSS_LUMBAR.items() if lbl not in tss_unique}
    logger.info(f"  [{study_id}] TSS lumbar labels: {list(tss_lumbar_present.values()) or 'none'}"
                + (f"  MISSING: {list(tss_lumbar_missing.values())}" if tss_lumbar_missing else ""))

    tss_highest_lumbar  = max((lbl for lbl in TSS_LUMBAR if lbl in tss_unique), default=None)
    tss_preferred_tv_vd = TSS_VERT_TO_VD.get(tss_highest_lumbar)
    vert_unique         = sorted(int(v) for v in np.unique(sag_vert) if v > 0)
    logger.info(f"  [{study_id}] TSS TV preference: TSS {tss_highest_lumbar} -> "
                f"VERIDAH {tss_preferred_tv_vd} "
                f"({'present' if tss_preferred_tv_vd in vert_unique else 'ABSENT'})")

    # ── Cross-validation ───────────────────────────────────────────────────────
    xval = run_cross_validation(sag_sp, sag_vert, sag_tss, vox_mm, study_id)
    out['cross_validation'] = xval
    for w in xval.get('warnings', []): out['errors'].append(f'XVAL: {w}')

    # ── Axial data (Phase 2) ───────────────────────────────────────────────────
    ax_tss, _ = _load(tss_ax, 'TSS axial')
    ax_sp,  _ = _load(sp_ax,  'registered SPINEPS axial')
    ax_t2w, ax_vox_mm_val = None, None
    t2w_path = _find_t2w('ax')
    if t2w_path:
        arr, t2w_nii = _load(t2w_path, 'axial T2w')
        if arr is not None:
            ax_t2w        = arr
            ax_vox_mm_val = voxel_size_mm(t2w_nii)

    p2_available = (ax_tss is not None and ax_sp is not None and ax_t2w is not None)
    if not p2_available:
        logger.warning(f"  Phase 2 unavailable — contact cases -> Type II")

    if ax_tss is not None: ax_tss = ax_tss.astype(int)
    if ax_sp  is not None: ax_sp  = ax_sp.astype(int)

    # ══════════════════════════════════════════════════════════════════════════
    # TV IDENTIFICATION — TSS-first
    # ══════════════════════════════════════════════════════════════════════════
    named = [VERIDAH_NAMES[l] for l in vert_unique if l in VERIDAH_NAMES]
    logger.info(f"  [{study_id}] VERIDAH labels: {named}")

    tv_label = tv_name = None

    if tss_preferred_tv_vd is not None and tss_preferred_tv_vd in vert_unique:
        tv_label = tss_preferred_tv_vd
        tv_name  = VERIDAH_NAMES[tv_label]
        logger.info(f"  [{study_id}] TV via TSS GT: {tv_name} "
                    f"(VD {tv_label} <- TSS {tss_highest_lumbar})")
    elif VD_L6 in vert_unique:
        tv_label = VD_L6; tv_name = VERIDAH_NAMES[VD_L6]
        logger.info(f"  [{study_id}] TV: VERIDAH L6 present — will verify")
    else:
        for cand in VERIDAH_TV_SEARCH:
            if cand in vert_unique:
                tv_label = cand; tv_name = VERIDAH_NAMES[cand]
                logger.warning(f"  [{study_id}] TV fallback: {tv_name}")
                break

    if tv_label is None:
        out['errors'].append('No lumbar VERIDAH labels found'); return out, alignment_result

    # ── L6 verification ────────────────────────────────────────────────────────
    l6_verified = False
    if tv_label == VD_L6:
        vd_l6_z = get_z_range_from_mask(sag_vert == VD_L6)
        if vd_l6_z is None:
            out['errors'].append('VERIDAH L6 label present but empty mask')
            return out, alignment_result
        l6_ok, l6_reason = _verify_l6(sag_vert, sag_tss, vox_mm, vd_l6_z, study_id)
        l6_verified = l6_ok
        if not l6_ok:
            logger.warning(f"  [{study_id}] VERIDAH L6 FAILED: {l6_reason} — demoting to L5")
            if tss_preferred_tv_vd is not None and tss_preferred_tv_vd in vert_unique:
                tv_label = tss_preferred_tv_vd; tv_name = VERIDAH_NAMES[tv_label]
            elif VD_L5 in vert_unique:
                tv_label = VD_L5;               tv_name = VERIDAH_NAMES[VD_L5]
            else:
                out['errors'].append('L6 failed and no L5 fallback')
                return out, alignment_result
        else:
            logger.info(f"  [{study_id}] VERIDAH L6 verified -- {l6_reason}")

    # ── TV Z-range + segmental axis ────────────────────────────────────────────
    tv_z, tv_z_src = get_tv_z_range(sag_vert, sag_tss, tv_label, study_id)
    if tv_z is None:
        out['errors'].append(f'TV {tv_name} not found in TSS or VERIDAH mask')
        return out, alignment_result

    seg_axis, seg_src = compute_segmental_axis(sag_tss, sag_vert, tv_label, vox_mm, study_id)

    out['details'] = {
        'tv_label': tv_label, 'tv_name': tv_name,
        'has_l6': tv_label == VD_L6, 'l6_verified': l6_verified if tv_label == VD_L6 else None,
        'tv_z_range': list(tv_z), 'tv_z_source': tv_z_src,
        'segmental_axis': [round(float(v), 5) for v in seg_axis],
        'segmental_axis_source': seg_src,
        'sag_vox_mm': vox_mm.tolist(), 'phase2_available': p2_available,
        'tss_lumbar_labels': tss_lumbar_present,
        'tss_tv_preference': f'TSS {tss_highest_lumbar} -> VERIDAH {tss_preferred_tv_vd}',
        'alignment_hypothesis': (alignment_result.preferred_hypothesis
                                  if alignment_result else 'not_computed'),
    }

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 + 2
    # ══════════════════════════════════════════════════════════════════════════
    confidence_ref = [out['confidence']]

    for side, tp_lbl in (('left', SP_TP_L), ('right', SP_TP_R)):
        try:
            p1 = phase1_sagittal(side, tp_lbl, sag_sp, sag_tss, vox_mm,
                                  tv_z, seg_axis, seg_src)
            logger.info(f"  {side:5s} P1: {p1['classification']:22s} "
                        f"h={p1['tp_height_mm']:.1f}mm  d={p1['dist_mm']:.1f}mm  "
                        f"|cosT|={p1.get('tp_axis_cos_segmental','?')}  "
                        f"z=[{p1.get('tp_z_min_vox','?')},{p1.get('tp_z_max_vox','?')}]")
            out[side] = p1
        except Exception as exc:
            out['errors'].append(f'{side}: {exc}')
            logger.error(f"  {side} failed: {exc}")

    ct, out['left'], out['right'] = finalise_castellvi(
        out.get('left', {}), out.get('right', {}),
        p2_available, ax_sp, ax_tss, ax_t2w, ax_vox_mm_val,
        confidence_ref)
    out['confidence'] = confidence_ref[0]

    if ct:
        out['castellvi_type']  = ct
        out['lstv_detected']   = True
        out['lstv_reason'].append(f"Castellvi {ct} — TP morphology")
        logger.info(f"  + [{study_id}] Castellvi (H0): {ct}")
    else:
        logger.info(f"  - [{study_id}] No Castellvi (H0)")

    # ══════════════════════════════════════════════════════════════════════════
    # ENSEMBLE CLASSIFICATION
    # ══════════════════════════════════════════════════════════════════════════
    if alignment_result is not None:
        alignment_result.castellvi_at_zero    = out.get('castellvi_type')
        alignment_result.lstv_detected_at_zero = out.get('lstv_detected', False)
        alignment_result.phenotype_at_zero    = None

        if alignment_result.best_offset is not None and alignment_result.best_offset != 0:
            try:
                _run_ensemble_offset(
                    study_id, alignment_result,
                    sag_sp, sag_vert, sag_tss, vox_mm,
                    p2_available, ax_sp, ax_tss, ax_t2w, ax_vox_mm_val,
                    tss_highest_lumbar, vert_unique,
                )
            except Exception as exc:
                logger.error(f"  [{study_id}] Ensemble offset classification failed: {exc}")
                logger.debug(traceback.format_exc())
        else:
            alignment_result.castellvi_at_best    = None
            alignment_result.phenotype_at_best    = None
            alignment_result.lstv_detected_at_best = None

    # ══════════════════════════════════════════════════════════════════════════
    # MORPHOMETRICS
    # ══════════════════════════════════════════════════════════════════════════
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
                          f"{direction} (TSS={morpho.lumbar_count_tss}, "
                          f"VERIDAH={morpho.lumbar_count_veridah})")
                out['lstv_reason'].append(reason)

            if phenotype in ('sacralization', 'lumbarization'):
                out['lstv_detected'] = True
                primary = morpho.primary_criteria_met or []
                reason  = (f"Phenotype: {phenotype.upper()} "
                           f"({morpho.phenotype_confidence}) — {'; '.join(primary)}")
                if not any('Phenotype' in r for r in out['lstv_reason']):
                    out['lstv_reason'].append(reason)

            if alignment_result is not None:
                alignment_result.phenotype_at_zero    = phenotype
                alignment_result.lstv_detected_at_zero = out.get('lstv_detected', False)
                if alignment_result.castellvi_at_zero is None:
                    alignment_result.castellvi_at_zero = out.get('castellvi_type')
                out['alignment'] = alignment_result.to_dict()

        except Exception as exc:
            logger.error(f"  [{study_id}] lstv_engine error: {exc}")
            out['errors'].append(f'lstv_engine: {exc}')

    out['pathology_score'] = compute_lstv_pathology_score(
        out, out.get('lstv_morphometrics'))

    # ── Final summary log ──────────────────────────────────────────────────────
    morpho_dict = out.get('lstv_morphometrics') or {}
    probs_dict  = (morpho_dict.get('probabilities') or {})
    sr_dict     = (morpho_dict.get('surgical_relevance') or {})

    ip_str = ''
    if alignment_result and getattr(alignment_result, 'ip_winner', None):
        margin_val = getattr(alignment_result, 'ip_vert_margin_mm', None)
        margin_str = f"{margin_val:+.1f}mm" if margin_val is not None else "N/A"
        ip_str = (f"  ip={alignment_result.ip_winner}({margin_str})"
                  f"{'[ARB]' if alignment_result.ip_tiebreak_applied else ''}")

    if out['lstv_detected']:
        logger.info(
            f"  ++ [{study_id}] LSTV DETECTED  "
            f"Castellvi={out.get('castellvi_type','None')}  "
            f"phenotype={morpho_dict.get('lstv_phenotype','?')}  "
            f"P(sac)={probs_dict.get('p_sacralization',0):.0%}  "
            f"P(lumb)={probs_dict.get('p_lumbarization',0):.0%}  "
            f"surgical_risk={sr_dict.get('wrong_level_risk','?')}  "
            + (f"alignment={alignment_result.preferred_hypothesis} "
               f"[{alignment_result.confidence}]"
               if alignment_result else "alignment=N/A")
            + ip_str
        )
    else:
        logger.info(f"  -- [{study_id}] No LSTV  "
                    f"P(sac)={probs_dict.get('p_sacralization',0):.0%}  "
                    f"P(lumb)={probs_dict.get('p_lumbarization',0):.0%}"
                    + ip_str)

    return out, alignment_result


# ── Ensemble offset helper ────────────────────────────────────────────────────

def _run_ensemble_offset(
        study_id:           str,
        ar:                 AlignmentResult,
        sag_sp:             np.ndarray,
        sag_vert:           np.ndarray,
        sag_tss:            np.ndarray,
        vox_mm:             np.ndarray,
        p2_available:       bool,
        ax_sp, ax_tss, ax_t2w, ax_vox_mm,
        tss_highest_lumbar: Optional[int],
        vert_unique:        List[int],
) -> None:
    offset = ar.best_offset

    from vertebral_alignment import VD_LUMBAR_BASE, VD_LUMBAR_MAX

    at_zero_vd = TSS_VERT_TO_VD.get(tss_highest_lumbar)

    if at_zero_vd is None:
        logger.warning(f"  [{study_id}] Ensemble offset: cannot determine at_zero VD label")
        ar.castellvi_at_best     = None
        ar.lstv_detected_at_best = None
        return

    at_best_vd = at_zero_vd + offset

    if not (VD_LUMBAR_BASE <= at_best_vd <= VD_LUMBAR_MAX):
        logger.warning(
            f"  [{study_id}] Ensemble offset {offset:+d}: computed TV VD{at_best_vd} "
            f"out of valid range [{VD_LUMBAR_BASE},{VD_LUMBAR_MAX}]")
        ar.castellvi_at_best     = None
        ar.lstv_detected_at_best = None
        return

    if at_best_vd not in vert_unique:
        logger.warning(
            f"  [{study_id}] Ensemble offset {offset:+d}: TV VD{at_best_vd} "
            f"({VERIDAH_NAMES.get(at_best_vd,'?')}) not present in VERIDAH")
        ar.castellvi_at_best     = None
        ar.lstv_detected_at_best = None
        return

    at_best_name = VERIDAH_NAMES.get(at_best_vd, f'VD{at_best_vd}')
    logger.info(
        f"  [{study_id}_ens] Ensemble offset={offset:+d}: "
        f"at_zero TV=VD{at_zero_vd}({VERIDAH_NAMES.get(at_zero_vd,'?')}) "
        f"-> at_best TV=VD{at_best_vd}({at_best_name})")

    tv_z_best, _ = get_tv_z_range(sag_vert, sag_tss, at_best_vd,
                                   study_id + '_ens')
    if tv_z_best is None:
        logger.warning(f"  [{study_id}_ens] TV Z-range unavailable for VD{at_best_vd}")
        ar.castellvi_at_best     = None
        ar.lstv_detected_at_best = None
        return

    seg_axis_best, _ = compute_segmental_axis(
        sag_tss, sag_vert, at_best_vd, vox_mm, study_id + '_ens')

    left_best  = phase1_sagittal('left_ens',  SP_TP_L, sag_sp, sag_tss, vox_mm,
                                  tv_z_best, seg_axis_best, 'ens')
    right_best = phase1_sagittal('right_ens', SP_TP_R, sag_sp, sag_tss, vox_mm,
                                  tv_z_best, seg_axis_best, 'ens')

    logger.info(
        f"  left_ens  P1: {left_best['classification']:22s} "
        f"h={left_best['tp_height_mm']:.1f}mm  d={left_best['dist_mm']:.1f}mm  "
        f"z=[{left_best.get('tp_z_min_vox','?')},{left_best.get('tp_z_max_vox','?')}]")
    logger.info(
        f"  right_ens P1: {right_best['classification']:22s} "
        f"h={right_best['tp_height_mm']:.1f}mm  d={right_best['dist_mm']:.1f}mm  "
        f"z=[{right_best.get('tp_z_min_vox','?')},{right_best.get('tp_z_max_vox','?')}]")

    conf_ref = ['high']
    ct_best, left_best, right_best = finalise_castellvi(
        left_best, right_best, p2_available,
        ax_sp, ax_tss, ax_t2w, ax_vox_mm, conf_ref)

    ar.castellvi_at_best     = ct_best
    ar.lstv_detected_at_best = bool(ct_best)

    logger.info(
        f"  [{study_id}_ens] Ensemble Castellvi offset={offset:+d}: "
        f"{ct_best or 'None'}  (TV={at_best_name}, tv_z={tv_z_best})")

    if ar.castellvi_at_zero != ar.castellvi_at_best:
        logger.info(
            f"  [{study_id}_ens] CASTELLVI DISAGREES: "
            f"at_zero={ar.castellvi_at_zero or 'None'} "
            f"vs at_best(offset={offset:+d})={ct_best or 'None'}")


# ══════════════════════════════════════════════════════════════════════════════
# STUDY SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def select_studies_csv(csv_path, top_n, rank_by, valid_ids):
    import pandas as pd
    df = pd.read_csv(csv_path)
    df['study_id'] = df['study_id'].astype(str)
    if valid_ids: df = df[df['study_id'].isin(valid_ids)]
    df = df.sort_values(rank_by, ascending=False).reset_index(drop=True)
    ids  = df.head(top_n)['study_id'].tolist() + df.tail(top_n)['study_id'].tolist()
    seen, result = set(), []
    for sid in ids:
        if sid not in seen: result.append(sid); seen.add(sid)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Hybrid Two-Phase LSTV Castellvi Classifier v4.4 (Ian Pan Vertebral Arbitration)',
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
    parser.add_argument('--ian_pan_coords',   default=None,
                        help='Path to ian_pan_disc_coords.json.  When offset != 0 Ian Pan '
                             'vertebral body positions arbitrate between TSS and VERIDAH.')
    args = parser.parse_args()

    spineps_dir    = Path(args.spineps_dir)
    totalspine_dir = Path(args.totalspine_dir)
    registered_dir = Path(args.registered_dir)
    nifti_dir      = Path(args.nifti_dir)
    output_dir     = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seg_root = spineps_dir / 'segmentations'

    IAN_PAN: Dict[str, dict] = {}
    if args.ian_pan_coords:
        ip_path = Path(args.ian_pan_coords)
        if not ip_path.exists():
            logger.warning(f"--ian_pan_coords file not found: {ip_path}")
        elif not _IAN_PAN_AVAILABLE:
            logger.warning("ian_pan_disc_coords module not importable — arbitration disabled")
        else:
            try:
                IAN_PAN = load_ian_pan_disc_coords(ip_path)
                logger.info(f"Ian Pan disc coords loaded: {len(IAN_PAN)} studies")
            except Exception as exc:
                logger.error(f"Failed to load Ian Pan coords: {exc}")

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

    logger.info(f"Processing {len(study_ids)} studies"
                + (f"  [Ian Pan arbitration: {len(IAN_PAN)} studies loaded]"
                   if IAN_PAN else "  [Ian Pan arbitration: disabled]"))

    results:           List[dict]            = []
    alignment_results: List[AlignmentResult] = []
    errors = 0
    castellvi_counts   = {k: 0 for k in
                          ['Type Ia','Type Ib','Type IIa','Type IIb',
                           'Type IIIa','Type IIIb','Type IV']}
    phenotype_counts:  Dict[str, int] = {}
    axis_deviations:   List[float]    = []

    n_h0 = n_h1 = n_insuf = n_l6 = 0
    n_l6_confirmed = n_l6_rejected = 0
    n_classification_changed = 0
    n_ip_arbitrated = 0

    for sid in study_ids:
        logger.info(f"\n{'='*60}\n[{sid}]")
        try:
            r, ar = classify_study(
                sid, spineps_dir, totalspine_dir, registered_dir, nifti_dir,
                run_morpho=not args.no_morpho,
                ian_pan_study=IAN_PAN.get(sid),
            )
            results.append(r)
            if ar is not None:
                alignment_results.append(ar)

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

            if ar is not None:
                if   ar.preferred_hypothesis == 'aligned':           n_h0   += 1
                elif ar.preferred_hypothesis == 'shifted_plus_1':    n_h1   += 1
                elif ar.preferred_hypothesis == 'insufficient_data': n_insuf += 1

                if 25 in (ar.vd_labels_present or []):
                    n_l6 += 1
                    if ar.preferred_hypothesis == 'shifted_plus_1':  n_l6_confirmed += 1
                    elif ar.preferred_hypothesis == 'aligned':        n_l6_rejected  += 1

                if (ar.castellvi_at_zero != ar.castellvi_at_best or
                        ar.phenotype_at_zero != ar.phenotype_at_best):
                    n_classification_changed += 1

                if getattr(ar, 'ip_tiebreak_applied', False):
                    n_ip_arbitrated += 1

        except Exception as exc:
            logger.error(f"  Unhandled: {exc}")
            logger.debug(traceback.format_exc())
            errors += 1

    cohort_stats: Optional[CohortAlignmentStats] = None
    if alignment_results:
        cohort_stats = compute_cohort_stats(alignment_results)

    lstv_n = sum(1 for r in results if r.get('lstv_detected'))
    n      = max(len(results), 1)
    scores = sorted(
        ((r['study_id'], r.get('pathology_score') or 0) for r in results),
        key=lambda t: t[1], reverse=True)

    p_sac_vals  = []
    p_lumb_vals = []
    wl_risk_counts: Dict[str, int] = {}
    bertolotti_ge50 = high_cert_sac = high_cert_lumb = rel_disc_low = 0

    for r in results:
        morpho = r.get('lstv_morphometrics') or {}
        probs  = morpho.get('probabilities')  or {}
        ps = probs.get('p_sacralization', 0); pl = probs.get('p_lumbarization', 0)
        p_sac_vals.append(ps); p_lumb_vals.append(pl)
        if ps > 0.80: high_cert_sac  += 1
        if pl > 0.80: high_cert_lumb += 1
        sr  = morpho.get('surgical_relevance') or {}
        wlr = sr.get('wrong_level_risk', 'low')
        wl_risk_counts[wlr] = wl_risk_counts.get(wlr, 0) + 1
        if sr.get('bertolotti_probability', 0) >= 0.50: bertolotti_ge50 += 1
        rdr = morpho.get('relative_disc_ratio')
        if rdr is not None and rdr < 0.65: rel_disc_low += 1

    sep = '=' * 70
    logger.info(f"\n{sep}")
    logger.info(f"{'LSTV DETECTION SUMMARY  (v4.4 — IAN PAN VERTEBRAL ARBITRATION)':^70}")
    logger.info(f"{sep}")
    logger.info(f"Studies processed:             {len(results)}")
    logger.info(f"LSTV detected:                 {lstv_n}  ({100*lstv_n/n:.1f}%)")
    logger.info(f"  Sacralization:               {phenotype_counts.get('sacralization',0)}")
    logger.info(f"  Lumbarization:               {phenotype_counts.get('lumbarization',0)}")
    logger.info(f"  Transitional:                {phenotype_counts.get('transitional_indeterminate',0)}")
    logger.info(f"  Normal:                      {phenotype_counts.get('normal',0)}")
    logger.info(f"Errors:                        {errors}")

    logger.info(f"\n-- VERTEBRAL ALIGNMENT ANALYSIS --")
    n_align = len(alignment_results)
    logger.info(f"  Studies with alignment data: {n_align}")
    logger.info(f"  H0 (aligned, TSS=GT):        {n_h0}  ({100*n_h0/max(n_align,1):.1f}%)")
    logger.info(f"  H1 (shifted, SPINEPS=GT):    {n_h1}  ({100*n_h1/max(n_align,1):.1f}%)")
    logger.info(f"  Insufficient data:           {n_insuf}  ({100*n_insuf/max(n_align,1):.1f}%)")
    logger.info(f"  Studies with VERIDAH L6:     {n_l6}  ({100*n_l6/max(n_align,1):.1f}%)")
    if n_l6 > 0:
        logger.info(f"    L6 confirmed off-by-one:   {n_l6_confirmed}  ({100*n_l6_confirmed/n_l6:.1f}%)")
        logger.info(f"    L6 H0 retained (FP):       {n_l6_rejected}  ({100*n_l6_rejected/n_l6:.1f}%)")
    logger.info(f"  Classification changes:      {n_classification_changed}  ({100*n_classification_changed/max(n_align,1):.1f}%)")

    logger.info(f"\n-- IAN PAN VERTEBRAL ARBITRATION --")
    if IAN_PAN:
        n_ip_matched  = sum(1 for ar in alignment_results if getattr(ar, 'ip_winner', None) is not None)
        n_ip_tss      = sum(1 for ar in alignment_results if getattr(ar, 'ip_winner', None) == 'TSS')
        n_ip_vd       = sum(1 for ar in alignment_results if getattr(ar, 'ip_winner', None) == 'VERIDAH')
        n_ip_tie      = sum(1 for ar in alignment_results if getattr(ar, 'ip_winner', None) == 'tie')
        n_ip_insuf    = sum(1 for ar in alignment_results if getattr(ar, 'ip_winner', None) == 'insufficient')
        n_ip_eligible = sum(1 for ar in alignment_results if (getattr(ar, 'best_offset', 0) or 0) != 0)
        logger.info(f"  Ian Pan studies matched:       {n_ip_matched} / {n_align}")
        logger.info(f"  Studies eligible (offset!=0):  {n_ip_eligible}")
        logger.info(f"  IP winner = TSS (aligned):     {n_ip_tss}")
        logger.info(f"  IP winner = VERIDAH (shifted): {n_ip_vd}")
        logger.info(f"  IP winner = tie (ambiguous):   {n_ip_tie}")
        logger.info(f"  IP winner = insufficient:      {n_ip_insuf}")
        logger.info(f"  Arbitrations applied:          {n_ip_arbitrated}")
        margins = [getattr(ar, 'ip_vert_margin_mm', None) for ar in alignment_results
                   if getattr(ar, 'ip_vert_margin_mm', None) is not None]
        if margins:
            logger.info(f"  Mean vert margin (VD-TSS):     {np.mean(margins):+.1f}mm (+ve = TSS closer)")
    else:
        logger.info(f"  Ian Pan arbitration:           DISABLED")

    if cohort_stats:
        logger.info(f"\n-- Alignment Confidence --")
        logger.info(f"  High: {cohort_stats.n_high}  Moderate: {cohort_stats.n_moderate}  "
                    f"Low: {cohort_stats.n_low}  Insufficient: {cohort_stats.n_insufficient}")

    logger.info(f"\n-- Castellvi Type Breakdown --")
    for t, cnt in castellvi_counts.items():
        if cnt: logger.info(f"  {t:12s}: {cnt}")
    total_ct = sum(castellvi_counts.values())
    logger.info(f"  {'TOTAL':12s}: {total_ct}  ({100*total_ct/n:.1f}%)")

    if axis_deviations:
        logger.info(f"\n-- Segmental Axis QA --")
        logger.info(f"  mean={np.mean(axis_deviations):.1f}  median={np.median(axis_deviations):.1f}  "
                    f"max={np.max(axis_deviations):.1f} deg  n={len(axis_deviations)}")
        n_dev = sum(1 for d in axis_deviations if d > 20.)
        logger.info(f"  TPs deviating >20 deg: {n_dev} ({100*n_dev/len(axis_deviations):.1f}%)")

    logger.info(f"\n-- Probability Model Statistics --")
    if p_sac_vals:
        logger.info(f"  P(sacralization):  mean={np.mean(p_sac_vals):.2%}  >80%: {high_cert_sac}")
        logger.info(f"  P(lumbarization):  mean={np.mean(p_lumb_vals):.2%}  >80%: {high_cert_lumb}")

    logger.info(f"\n-- Surgical Risk Distribution --")
    for risk_lvl in ('critical', 'high', 'moderate', 'low-moderate', 'low'):
        cnt = wl_risk_counts.get(risk_lvl, 0)
        if cnt: logger.info(f"  {risk_lvl:14s}: {cnt}")
    logger.info(f"  Bertolotti P>=50%: {bertolotti_ge50}")
    logger.info(f"\n{sep}")

    out_json = output_dir / 'lstv_results.json'
    with open(out_json, 'w') as fh:
        json.dump(results, fh, indent=2, default=str)

    summary = {
        'total': len(results), 'lstv_detected': lstv_n,
        'lstv_rate': round(lstv_n/n, 4), 'errors': errors,
        'castellvi_breakdown': castellvi_counts,
        'phenotype_breakdown': phenotype_counts,
        'alignment_summary': {
            'n_studies_analysed':          len(alignment_results),
            'n_h0_aligned':                n_h0,
            'n_h1_shifted':                n_h1,
            'n_insufficient_data':         n_insuf,
            'off_by_one_rate_pct':         round(100*n_h1/max(len(alignment_results),1), 2),
            'n_vd_l6_present':             n_l6,
            'n_l6_confirmed_shifted':      n_l6_confirmed,
            'n_l6_rejected_H0_retained':   n_l6_rejected,
            'l6_false_positive_rate_pct':  round(100*n_l6_rejected/max(n_l6,1), 2) if n_l6 else None,
            'n_classification_changed':    n_classification_changed,
            'cohort_stats': cohort_stats.to_dict() if cohort_stats else None,
        },
        'ian_pan_summary': {
            'arbitration_enabled':    bool(IAN_PAN),
            'n_studies_with_data':    sum(1 for ar in alignment_results
                                         if getattr(ar, 'ip_winner', None) is not None),
            'n_arbitrations_applied': n_ip_arbitrated,
            'n_winner_tss':           sum(1 for ar in alignment_results
                                         if getattr(ar, 'ip_winner', None) == 'TSS'),
            'n_winner_veridah':       sum(1 for ar in alignment_results
                                         if getattr(ar, 'ip_winner', None) == 'VERIDAH'),
        },
        'probability_stats': {
            'mean_p_sacralization': round(float(np.mean(p_sac_vals)),4) if p_sac_vals else None,
            'mean_p_lumbarization': round(float(np.mean(p_lumb_vals)),4) if p_lumb_vals else None,
            'high_confidence_sac':  high_cert_sac,
            'high_confidence_lumb': high_cert_lumb,
        },
        'surgical_risk_breakdown':     wl_risk_counts,
        'bertolotti_probability_ge50': bertolotti_ge50,
        'top_scores':                  scores[:20],
    }
    with open(output_dir / 'lstv_summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)

    if cohort_stats is not None:
        try:
            write_csv_reports(results, alignment_results, cohort_stats, output_dir)
        except Exception as exc:
            logger.error(f"CSV reporting failed: {exc}")
            logger.debug(traceback.format_exc())

    logger.info(f"Results -> {out_json}")
    logger.info(f"CSV     -> {output_dir}/lstv_per_study.csv  lstv_alignment.csv  lstv_cohort_summary.csv")
    return 0


if __name__ == '__main__':
    sys.exit(main())
