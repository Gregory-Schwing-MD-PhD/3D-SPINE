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
A sacralizing L5 may have Castellvi I-IV AND sacralization phenotype.
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
           MRI: heterogeneous / dark T2 signal at junction (fibrocartilage)
Type III : Complete osseous fusion of TP with sacrum
           IIIa = unilateral; IIIb = bilateral
           MRI: homogeneous bright T2 signal (marrow continuity)
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
  Preferred sacrum source (label 50) for Phase 1 distance measurement.
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
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-7s  %(message)s')
logger = logging.getLogger(__name__)

# ── Phase 2 signal thresholds ─────────────────────────────────────────────────
BBOX_HALF          = 16
P2_DARK_CLEFT_FRAC = 0.55    # mean < 55% of p95 → dark cleft → Type II
P2_MIN_STD_RATIO   = 0.12    # CV < 0.12 → uniform bridge → Type III

# ── Cross-validation thresholds ───────────────────────────────────────────────
XVAL_MIN_DICE      = 0.30
XVAL_MAX_CENTROID  = 20.0   # mm


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

def get_tv_z_range(vert_data: np.ndarray, tv_label: int) -> Optional[Tuple[int, int]]:
    mask = (vert_data == tv_label)
    if not mask.any(): return None
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


def measure_tp_height_mm(tp_mask: np.ndarray, vox_mm: np.ndarray) -> float:
    """
    Craniocaudal TP height = (Z_max − Z_min + 1) × vox_mm[2].
    Spans ALL TP voxels — matches Castellvi's original criterion (1984).
    """
    if not tp_mask.any(): return 0.0
    zc = np.where(tp_mask)[2]
    return float((int(zc.max()) - int(zc.min()) + 1) * vox_mm[2])


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
    Classify axial T2w signal at the TP–sacrum junction.

    MRI classification (Konin & Walz 2010; Nidecker et al. 2018):
      Dark / intermediate signal → fibrocartilaginous pseudo-joint → Type II
      Bright homogeneous signal  → osseous marrow bridge           → Type III
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
                           f"— dark/intermediate signal → fibrocartilage cleft → Type II")
        return 'Type II', feats
    elif cv < P2_MIN_STD_RATIO:
        feats['reason'] = (f"CV={cv:.3f} < {P2_MIN_STD_RATIO} "
                           f"— uniform bright signal → osseous marrow bridge → Type III")
        return 'Type III', feats
    else:
        feats['reason'] = "Bright but heterogeneous — ambiguous; Type II (conservative)"
        return 'Type II', feats


def phase2_axial(side: str, tp_label: int,
                 ax_spineps: np.ndarray, ax_tss: np.ndarray,
                 ax_t2w: np.ndarray, ax_vox_mm: np.ndarray) -> dict:
    """
    Phase 2 using axial T2w signal at the TP–sacrum junction.
    ax_spineps : registered SPINEPS seg-spine labels (43=TP-left, 44=TP-right)
    ax_tss     : native TSS axial labels (50=sacrum; 43/44 = L3/L4, NOT TPs)
    """
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

    midpoint = ((vox_a + vox_b) / 2.0).astype(int)
    out['midpoint_vox']  = midpoint.tolist()
    out['axial_dist_mm'] = round(float(dist_mm), 3)

    patch = _extract_bbox(ax_t2w, midpoint, BBOX_HALF)
    if patch is None:
        out['p2_note'] = 'Bounding box outside axial volume'
        return out

    cls, feats              = _classify_signal(patch, ax_t2w)
    out['classification']   = cls
    out['p2_features']      = feats
    out['p2_valid']         = feats.get('valid', False)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — SAGITTAL GEOMETRIC CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def phase1_sagittal(side: str, tp_label: int,
                    sag_sp: np.ndarray, sag_tss: Optional[np.ndarray],
                    sag_vox_mm: np.ndarray,
                    tv_z_range: Tuple[int, int]) -> dict:
    """
    Phase 1: sagittal geometric Castellvi analysis.
    TP height threshold: ≥ 19 mm (Castellvi et al. 1984).
    Contact threshold: ≤ 2 mm 3D distance to sacrum.
    """
    out = {
        'tp_present': False, 'tp_height_mm': 0.0, 'contact': False,
        'dist_mm': float('inf'), 'tp_vox': None, 'sacrum_vox': None,
        'classification': 'Normal', 'phase1_done': False, 'sacrum_source': None,
    }

    tss_sac = (sag_tss == TSS_SACRUM) if sag_tss is not None else None
    if tss_sac is not None and tss_sac.any():
        sac_mask             = tss_sac
        out['sacrum_source'] = f'TSS label {TSS_SACRUM}'
    else:
        sac_mask = (sag_sp == SP_SACRUM)
        out['sacrum_source'] = 'SPINEPS label 26 (fallback)'

    tp_at_tv = isolate_tp_at_tv(sag_sp, tp_label, *tv_z_range)
    tp_mask  = inferiormost_tp_cc(tp_at_tv, sac_mask if sac_mask.any() else None)

    if not tp_mask.any():
        out['note'] = f"TP label {tp_label} absent at TV level"
        return out

    out['tp_present']   = True
    out['tp_height_mm'] = measure_tp_height_mm(tp_mask, sag_vox_mm)
    out['tp_z_min_vox'] = int(np.where(tp_mask)[2].min())
    out['tp_z_max_vox'] = int(np.where(tp_mask)[2].max())

    if not sac_mask.any():
        return out

    dist_mm, tp_vox, sac_vox = min_dist_3d(tp_mask, sac_mask, sag_vox_mm)
    out['dist_mm']    = round(float(dist_mm), 3)
    out['phase1_done']= True
    if tp_vox  is not None: out['tp_vox']     = tp_vox.tolist()
    if sac_vox is not None: out['sacrum_vox'] = sac_vox.tolist()

    if dist_mm > CONTACT_DIST_MM:
        out['contact']        = False
        if out['tp_height_mm'] >= TP_HEIGHT_MM:
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
    ax_tss, ax_tss_nii = _load(tss_ax, 'TSS axial')
    ax_sp,  _          = _load(sp_ax,  'registered SPINEPS axial')
    ax_t2w, ax_vox_mm  = None, None
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

    tv_z = get_tv_z_range(sag_vert, tv_label)
    if tv_z is None:
        out['errors'].append(f'TV label {tv_name} empty in VERIDAH mask'); return out

    out['details'] = {
        'tv_label':         tv_label,
        'tv_name':          tv_name,
        'has_l6':           tv_label == VD_L6,
        'tv_z_range':       list(tv_z),
        'sag_vox_mm':       vox_mm.tolist(),
        'phase2_available': p2_available,
        'tp_source':        'seg-spine_msk labels 43 (L) / 44 (R)',
        'sacrum_source':    'TSS label 50 (preferred) / SPINEPS 26 (fallback)',
        'label_note':       'TSS 43/44 = L3/L4 vertebrae — TP always from seg-spine_msk',
        'tss_lumbar_labels':tss_lumbar_present,
    }
    logger.info(f"  [{study_id}] TV={tv_name}  z=[{tv_z[0]},{tv_z[1]}]")

    # ── Phase 1 + 2 per side ───────────────────────────────────────────────────
    for side, tp_lbl in (('left', SP_TP_L), ('right', SP_TP_R)):
        try:
            p1 = phase1_sagittal(side, tp_lbl, sag_sp, sag_tss, vox_mm, tv_z)
            logger.info(
                f"  {side:5s} P1: {p1['classification']:22s} "
                f"h={p1['tp_height_mm']:.1f}mm  "
                f"d={p1['dist_mm']:.1f}mm  "
                f"z=[{p1.get('tp_z_min_vox','?')},{p1.get('tp_z_max_vox','?')}]  "
                f"sac={p1.get('sacrum_source','?')}"
            )

            if p1['contact'] and p2_available:
                p2 = phase2_axial(side, tp_lbl, ax_sp, ax_tss, ax_t2w, ax_vox_mm)
                p1['phase2']        = p2
                p1['classification']= p2['classification']
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
    # Run regardless of Castellvi — count anomaly and disc metrics may detect LSTV
    if run_morpho:
        try:
            masks  = load_lstv_masks(study_id, spineps_dir, totalspine_dir)
            morpho = analyze_lstv(masks, castellvi_result=out)
            out['lstv_morphometrics'] = morpho.to_dict()

            # ── Count-anomaly and phenotype-based LSTV flags ────────────────────
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
                # Only add if not already captured by count anomaly
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
    if out['lstv_detected']:
        ph_str = ''
        morpho_dict = out.get('lstv_morphometrics') or {}
        if morpho_dict.get('lstv_phenotype'):
            ph_str = f"  phenotype={morpho_dict['lstv_phenotype']} ({morpho_dict.get('phenotype_confidence','')})"
        logger.info(
            f"  ✓✓ [{study_id}] LSTV DETECTED  "
            f"Castellvi={out.get('castellvi_type','None')}  "
            f"{ph_str}  "
            f"reasons={out['lstv_reason']}"
        )
    else:
        logger.info(f"  ✗✗ [{study_id}] No LSTV (Castellvi=None, count=5, normal phenotype)")

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

    results: List[dict] = []
    errors              = 0
    castellvi_counts    = {k: 0 for k in
                           ['Type Ia','Type Ib','Type IIa','Type IIb',
                            'Type IIIa','Type IIIb','Type IV']}
    phenotype_counts: Dict[str, int] = {}

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
        except Exception as exc:
            logger.error(f"  Unhandled: {exc}")
            logger.debug(traceback.format_exc())
            errors += 1

    lstv_n = sum(1 for r in results if r.get('lstv_detected'))
    scores = sorted(
        ((r['study_id'], r.get('pathology_score') or 0) for r in results),
        key=lambda t: t[1], reverse=True,
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"Studies:        {len(results)}")
    logger.info(f"LSTV detected:  {lstv_n}  ({100*lstv_n/max(len(results),1):.1f}%)")
    logger.info(f"Errors:         {errors}")
    logger.info("Castellvi breakdown:")
    for t, n in castellvi_counts.items():
        if n: logger.info(f"  {t}: {n}")
    logger.info("Phenotype breakdown:")
    for t, n in phenotype_counts.items():
        logger.info(f"  {t}: {n}")
    logger.info("Top-10 pathology scores:")
    for sid, sc in scores[:10]:
        r_match = next((r for r in results if r['study_id'] == sid), {})
        ph = (r_match.get('lstv_morphometrics') or {}).get('lstv_phenotype', '?')
        ct = r_match.get('castellvi_type', 'None')
        logger.info(f"  {sid}: score={sc:.1f}  phenotype={ph}  castellvi={ct}")

    out_json = output_dir / 'lstv_results.json'
    with open(out_json, 'w') as fh:
        json.dump(results, fh, indent=2, default=str)

    summary = {
        'total':               len(results),
        'lstv_detected':       lstv_n,
        'lstv_rate':           round(lstv_n / max(len(results), 1), 4),
        'errors':              errors,
        'castellvi_breakdown': castellvi_counts,
        'phenotype_breakdown': phenotype_counts,
        'top_scores':          scores[:20],
    }
    with open(output_dir / 'lstv_summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.info(f"\nResults → {out_json}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
