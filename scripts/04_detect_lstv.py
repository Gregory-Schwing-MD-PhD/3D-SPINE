#!/usr/bin/env python3
"""
04_detect_lstv.py — Hybrid Two-Phase LSTV Castellvi Classifier
===============================================================

Label reference (from READMEs — do NOT confuse between tools):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPINEPS seg-spine_msk.nii.gz  (subregion/semantic mask):
  43 = Costal_Process_Left  (transverse process left  ← TP source)
  44 = Costal_Process_Right (transverse process right ← TP source)
  26 = Sacrum

SPINEPS seg-vert_msk.nii.gz  (VERIDAH instance mask):
  20=L1  21=L2  22=L3  23=L4  24=L5  25=L6  26=Sacrum
  (used ONLY for transitional vertebra identification)

TotalSpineSeg step2_output (sagittal + axial labeled):
  41=L1  42=L2  43=L3  44=L4  45=L5   ← vertebrae (43/44 ≠ SPINEPS TP!)
  50 = sacrum                          ← preferred sacrum source
  (TSS 43/44 are L3/L4 vertebrae, NOT transverse processes)

registered_dir / study_id / *_spineps_reg.nii.gz:
  Registered SPINEPS spine (subregion) labels in axial T2w space.
  Same label scheme as seg-spine_msk: 43=TP-left, 44=TP-right.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cross-validation performed at load time:
  • SPINEPS sacrum (label 26 from seg-spine_msk) vs TSS sacrum (label 50)
    — Dice overlap logged; warns if < 0.30
  • VERIDAH L5 (label 24 from seg-vert_msk) vs TSS L5 (label 45)
    — Centroid distance logged; warns if > 20 mm
  These checks catch mis-registered or mis-identified masks early.
"""

import argparse
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import binary_dilation, distance_transform_edt, label as cc_label

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# SPINEPS seg-spine_msk subregion labels
TP_LEFT_LABEL      = 43   # Costal_Process_Left  (transverse process)
TP_RIGHT_LABEL     = 44   # Costal_Process_Right
SPINEPS_SACRUM     = 26   # Sacrum in seg-spine_msk AND seg-vert_msk

# VERIDAH seg-vert_msk instance labels
VERIDAH_L5         = 24
VERIDAH_L6         = 25
VERIDAH_LUMBAR     = [25, 24, 23, 22, 21, 20]  # L6 first so L6 is preferred TV
VERIDAH_NAMES      = {20:'L1', 21:'L2', 22:'L3', 23:'L4', 24:'L5', 25:'L6'}

# TotalSpineSeg step2_output labels
# IMPORTANT: TSS 43=L3, 44=L4 — completely different meaning from SPINEPS 43/44!
TSS_L5_LABEL       = 45   # vertebrae_L5 in TotalSpineSeg
TSS_SACRUM_LABEL   = 50   # sacrum in TotalSpineSeg
TSS_SACRUM_LABELS  = {TSS_SACRUM_LABEL}

# Classifier thresholds
TP_HEIGHT_MM       = 19.0
CONTACT_DIST_MM    = 2.0
BBOX_HALF          = 16
P2_DARK_CLEFT_FRAC = 0.55
P2_MIN_STD_RATIO   = 0.12

# Cross-validation thresholds
XVAL_MIN_DICE      = 0.30   # warn if SPINEPS/TSS sacrum Dice < this
XVAL_MAX_CENTROID  = 20.0   # warn if L5 centroid distance > this mm


# ============================================================================
# STUDY SELECTION
# ============================================================================

def select_studies(csv_path: Path, top_n: int, rank_by: str,
                   valid_ids) -> list:
    if not csv_path.exists():
        raise FileNotFoundError(f"Uncertainty CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df['study_id'] = df['study_id'].astype(str)
    if valid_ids is not None:
        before = len(df)
        df = df[df['study_id'].isin(valid_ids)]
        logger.info(f"Filtered to {len(df)} studies via valid_ids "
                    f"({before - len(df)} excluded)")
    if rank_by not in df.columns:
        raise ValueError(f"Column '{rank_by}' not in CSV. "
                         f"Available: {', '.join(df.columns)}")
    df_sorted  = df.sort_values(rank_by, ascending=False).reset_index(drop=True)
    top_ids    = df_sorted.head(top_n)['study_id'].tolist()
    bottom_ids = df_sorted.tail(top_n)['study_id'].tolist()
    seen, selected = set(), []
    for sid in top_ids + bottom_ids:
        if sid not in seen:
            selected.append(sid); seen.add(sid)
    logger.info(f"Rank by: {rank_by}  Top {top_n}: {top_ids}  Bottom {top_n}: {bottom_ids}")
    return selected


# ============================================================================
# NIfTI HELPERS
# ============================================================================

def load_canonical(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """Load NIfTI, reorient to RAS canonical, robustly reduce to 3D.

    Handles 4D volumes:
      - Strip trailing size-1 axes first.
      - If still 4D (multiple timepoints/echoes), select first volume on axis 3.
    Raises ValueError if cannot reduce to exactly 3D.
    """
    nii  = nib.load(str(path))
    nii  = nib.as_closest_canonical(nii)
    data = nii.get_fdata()

    # Strip trailing size-1 dims
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]

    # Still 4D → multiple volumes; take first along axis 3 (time/echo)
    if data.ndim == 4:
        logger.debug(f"  4D volume {path.name} shape={data.shape}; selecting volume 0")
        data = data[..., 0]

    if data.ndim != 3:
        raise ValueError(
            f"Cannot reduce {path.name} to 3D: shape after squeeze = {data.shape}"
        )
    return data, nii


def voxel_size_mm(nii: nib.Nifti1Image) -> np.ndarray:
    return np.abs(np.array(nii.header.get_zooms()[:3], dtype=float))


# ============================================================================
# CROSS-VALIDATION HELPERS
# ============================================================================

def dice_coefficient(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Dice similarity between two boolean 3D masks."""
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = (a & b).sum()
    denom = a.sum() + b.sum()
    if denom == 0:
        return float('nan')
    return float(2 * intersection / denom)


def centroid_mm(mask: np.ndarray, vox_mm: np.ndarray) -> Optional[np.ndarray]:
    """Return physical-space centroid of a binary mask (in mm), or None."""
    coords = np.array(np.where(mask))
    if coords.size == 0:
        return None
    return coords.mean(axis=1) * vox_mm


def run_cross_validation(sag_spineps: np.ndarray,
                          sag_vert: np.ndarray,
                          sag_tss: np.ndarray,
                          vox_mm: np.ndarray,
                          study_id: str) -> dict:
    """
    Cross-validate SPINEPS and TotalSpineSeg label consistency.

    Checks:
      1. Sacrum: SPINEPS seg-spine label 26 vs TSS label 50
         → Dice overlap should be > XVAL_MIN_DICE
      2. L5 vertebra: VERIDAH label 24 vs TSS label 45
         → Centroid distance should be < XVAL_MAX_CENTROID mm

    Returns dict with keys: sacrum_dice, l5_centroid_dist_mm, warnings[]
    """
    xval = {'sacrum_dice': None, 'l5_centroid_dist_mm': None, 'warnings': []}

    # ── 1. Sacrum Dice ───────────────────────────────────────────────────────
    spineps_sac = (sag_spineps == SPINEPS_SACRUM)
    tss_sac     = (sag_tss     == TSS_SACRUM_LABEL)

    if not spineps_sac.any():
        xval['warnings'].append('SPINEPS sacrum (label 26) absent from seg-spine_msk')
    if not tss_sac.any():
        xval['warnings'].append('TSS sacrum (label 50) absent from sagittal TSS mask')

    if spineps_sac.any() and tss_sac.any():
        dice = dice_coefficient(spineps_sac, tss_sac)
        xval['sacrum_dice'] = round(dice, 4)
        if dice < XVAL_MIN_DICE:
            msg = (f'⚠ Sacrum Dice={dice:.3f} < {XVAL_MIN_DICE} — '
                   f'SPINEPS and TSS sacrum masks disagree significantly')
            logger.warning(f'  [{study_id}] {msg}')
            xval['warnings'].append(msg)
        else:
            logger.info(f'  [{study_id}] Sacrum Dice={dice:.3f} ✓')

    # ── 2. L5 centroid distance ──────────────────────────────────────────────
    veridah_l5 = (sag_vert == VERIDAH_L5)
    tss_l5     = (sag_tss  == TSS_L5_LABEL)

    if not veridah_l5.any():
        xval['warnings'].append('VERIDAH L5 (label 24) absent from seg-vert_msk')
    if not tss_l5.any():
        xval['warnings'].append('TSS L5 (label 45) absent from sagittal TSS mask')

    if veridah_l5.any() and tss_l5.any():
        c_ver = centroid_mm(veridah_l5, vox_mm)
        c_tss = centroid_mm(tss_l5,     vox_mm)
        if c_ver is not None and c_tss is not None:
            dist = float(np.linalg.norm(c_ver - c_tss))
            xval['l5_centroid_dist_mm'] = round(dist, 2)
            if dist > XVAL_MAX_CENTROID:
                msg = (f'⚠ L5 centroid distance={dist:.1f}mm > {XVAL_MAX_CENTROID}mm — '
                       f'VERIDAH and TSS L5 labels disagree')
                logger.warning(f'  [{study_id}] {msg}')
                xval['warnings'].append(msg)
            else:
                logger.info(f'  [{study_id}] L5 centroid dist={dist:.1f}mm ✓')

    # ── 3. Sanity: confirm TP labels in seg-spine_msk ───────────────────────
    # TSS labels 43/44 mean L3/L4 — they must NOT be used as TP source.
    # Confirm our TP comes from sag_spineps (seg-spine_msk), not sag_tss.
    tss_labels_present = set(np.unique(sag_tss).tolist())
    if 43 in tss_labels_present or 44 in tss_labels_present:
        # This is expected (TSS L3=43, L4=44) — just note for clarity
        logger.debug(f'  [{study_id}] TSS has labels 43/44 (L3/L4 vertebrae) — '
                     f'TP source correctly remains seg-spine_msk')

    spineps_labels = set(np.unique(sag_spineps).tolist())
    if TP_LEFT_LABEL not in spineps_labels and TP_RIGHT_LABEL not in spineps_labels:
        msg = (f'⚠ TP labels {TP_LEFT_LABEL}/{TP_RIGHT_LABEL} absent from seg-spine_msk '
               f'(present: {sorted(spineps_labels)[:15]})')
        logger.warning(f'  [{study_id}] {msg}')
        xval['warnings'].append(msg)

    return xval


# ============================================================================
# MASK OPERATIONS
# ============================================================================

def get_tv_z_range(vert_data, tv_label):
    mask = vert_data == tv_label
    if not mask.any():
        return None
    z = np.where(mask)[2]
    return int(z.min()), int(z.max())


def isolate_tp_at_tv(subreg_data, tp_label, z_min, z_max):
    """Extract TP voxels from seg-spine_msk restricted to TV z-range."""
    tp_full = subreg_data == tp_label
    iso     = np.zeros_like(tp_full)
    z_lo    = max(z_min, 0)
    z_hi    = min(z_max, subreg_data.shape[2] - 1)
    iso[:, :, z_lo:z_hi + 1] = tp_full[:, :, z_lo:z_hi + 1]
    return iso


def inferiormost_tp_cc(tp_mask3d, sacrum_mask3d=None):
    if not tp_mask3d.any():
        return np.zeros_like(tp_mask3d, dtype=bool)
    labeled, n = cc_label(tp_mask3d)
    if n <= 1:
        return tp_mask3d.astype(bool)
    sac_z_min = None
    if sacrum_mask3d is not None and sacrum_mask3d.any():
        sac_z_min = int(np.where(sacrum_mask3d)[2].min())
    cc_info = []
    for i in range(1, n + 1):
        comp = (labeled == i)
        zc   = np.where(comp)[2]
        cc_info.append((float(zc.mean()), int(zc.max()), comp))
    cc_info.sort(key=lambda t: t[0])
    if sac_z_min is not None:
        cands = [c for _, zm, c in cc_info if zm < sac_z_min]
        if cands:
            return cands[0].astype(bool)
    return cc_info[0][2].astype(bool)


def measure_tp_height_mm(tp_mask, vox_mm):
    if not tp_mask.any():
        return 0.0
    z = np.where(tp_mask)[2]
    return float((z.max() - z.min() + 1) * vox_mm[2])


def min_dist_3d(mask_a, mask_b, vox_mm):
    if not mask_a.any() or not mask_b.any():
        return float('inf'), None, None
    dist_to_b = distance_transform_edt(~mask_b, sampling=vox_mm)
    dist_at_a = np.where(mask_a, dist_to_b, np.inf)
    flat_idx  = int(np.argmin(dist_at_a))
    vox_a     = np.array(np.unravel_index(flat_idx, mask_a.shape))
    dist_mm   = float(dist_to_b[tuple(vox_a)])
    z_lo = max(0, int(vox_a[2]) - 20)
    z_hi = min(mask_b.shape[2], int(vox_a[2]) + 20)
    sub  = mask_b[:, :, z_lo:z_hi]
    if sub.any():
        coords = np.array(np.where(sub)); coords[2] += z_lo
    else:
        coords = np.array(np.where(mask_b))
    d2    = ((coords.T * vox_mm - vox_a * vox_mm) ** 2).sum(axis=1)
    vox_b = coords[:, int(np.argmin(d2))]
    return dist_mm, vox_a, vox_b


# ============================================================================
# PHASE 2 — AXIAL
# ============================================================================

def extract_axial_bbox(axial_t2w, midpoint_vox, half=BBOX_HALF):
    x0, y0, z0 = int(midpoint_vox[0]), int(midpoint_vox[1]), int(midpoint_vox[2])
    nx, ny, nz = axial_t2w.shape
    if not (0 <= z0 < nz):
        return None
    patch = axial_t2w[max(0, x0-half):min(nx, x0+half),
                      max(0, y0-half):min(ny, y0+half), z0].copy()
    return patch if patch.size > 0 else None


def classify_cleft_from_bbox(patch, axial_t2w):
    vals = patch.astype(float).ravel()
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 'Type II', {'reason': 'empty patch', 'valid': False}
    patch_mean = float(np.mean(vals))
    patch_std  = float(np.std(vals))
    cv         = patch_std / (patch_mean + 1e-6)
    global_p95 = float(np.percentile(axial_t2w[axial_t2w > 0], 95))
    dark_thresh = P2_DARK_CLEFT_FRAC * global_p95
    features = {
        'patch_mean':  round(patch_mean, 2),
        'patch_std':   round(patch_std, 2),
        'patch_min':   round(float(np.min(vals)), 2),
        'patch_max':   round(float(np.max(vals)), 2),
        'coeff_var':   round(cv, 4),
        'global_p95':  round(global_p95, 2),
        'dark_thresh': round(dark_thresh, 2),
        'valid':       True,
    }
    if patch_mean < dark_thresh:
        features['reason'] = f'mean({patch_mean:.1f}) < dark_thresh({dark_thresh:.1f})'
        return 'Type II', features
    elif cv < P2_MIN_STD_RATIO:
        features['reason'] = f'CV({cv:.3f}) < {P2_MIN_STD_RATIO} → uniform bridge'
        return 'Type III', features
    else:
        features['reason'] = 'Ambiguous: bright but heterogeneous → Type II (conservative)'
        return 'Type II', features


def phase2_axial(side, tp_label, axial_spineps_data, axial_tss_data,
                 axial_t2w, axial_vox_mm):
    """
    Phase 2 axial classification.

    axial_spineps_data: registered SPINEPS seg-spine labels in axial space.
                        TP_LEFT_LABEL=43, TP_RIGHT_LABEL=44 are transverse processes.
    axial_tss_data:     native TotalSpineSeg axial labels.
                        TSS_SACRUM_LABEL=50 is the sacrum.
                        NOTE: TSS labels 43/44 here are L3/L4 vertebrae — not used.
    """
    result = {'phase2_attempted': True, 'classification': 'Type II',
              'midpoint_vox': None, 'bbox_patch_shape': None,
              'p2_features': None, 'p2_valid': False}

    # TP from registered SPINEPS (seg-spine labels: 43=TP-left, 44=TP-right)
    tp_ax  = (axial_spineps_data == tp_label)
    # Sacrum from native TSS (label 50)
    sac_ax = np.isin(axial_tss_data, list(TSS_SACRUM_LABELS))

    if not tp_ax.any():
        result['p2_note'] = (f'TP label {tp_label} (SPINEPS costal process) '
                             f'absent in registered axial SPINEPS mask')
        return result
    if not sac_ax.any():
        result['p2_note'] = (f'Sacrum label {TSS_SACRUM_LABEL} absent in '
                             f'native axial TotalSpineSeg mask')
        return result

    dist_mm, vox_a, vox_b = min_dist_3d(tp_ax, sac_ax, axial_vox_mm)
    if vox_a is None or vox_b is None:
        result['p2_note'] = 'min_dist_3d returned None'
        return result

    midpoint = ((vox_a + vox_b) / 2.0).astype(int)
    result['midpoint_vox']  = midpoint.tolist()
    result['axial_dist_mm'] = round(dist_mm, 3)

    patch = extract_axial_bbox(axial_t2w, midpoint, BBOX_HALF)
    if patch is None:
        result['p2_note'] = 'Bounding box outside axial volume bounds'
        return result

    result['bbox_patch_shape'] = list(patch.shape)
    classification, features   = classify_cleft_from_bbox(patch, axial_t2w)
    result['classification']   = classification
    result['p2_features']      = features
    result['p2_valid']         = features.get('valid', False)
    return result


# ============================================================================
# PHASE 1 — SAGITTAL
# ============================================================================

def phase1_sagittal(side, tp_label, sag_spineps, sag_tss,
                    sag_vox_mm, tv_z_range):
    """
    Phase 1 sagittal classification.

    sag_spineps: SPINEPS seg-spine_msk (subregion labels).
                 TP source: labels 43 (left) / 44 (right) = costal/transverse processes.
                 Sacrum fallback: label 26.
    sag_tss:     TotalSpineSeg sagittal labeled.
                 Preferred sacrum: label 50.
                 (Labels 43/44 in sag_tss are L3/L4 vertebrae — NOT used here.)
    """
    result = {'tp_present': False, 'tp_height_mm': 0.0, 'contact': False,
              'dist_mm': float('inf'), 'tp_vox': None, 'sacrum_vox': None,
              'classification': 'Normal', 'phase1_done': False,
              'sacrum_source': None}

    # Sacrum: prefer TSS label 50, fall back to SPINEPS label 26
    tss_sac = np.isin(sag_tss, list(TSS_SACRUM_LABELS)) if sag_tss is not None else None
    if tss_sac is not None and tss_sac.any():
        sac_mask = tss_sac
        result['sacrum_source'] = f'TSS label {TSS_SACRUM_LABEL}'
    else:
        sac_mask = (sag_spineps == SPINEPS_SACRUM)
        result['sacrum_source'] = 'SPINEPS label 26 (fallback)'
        if sac_mask.any():
            logger.warning(f'    {side}: using SPINEPS sacrum fallback (TSS sacrum absent)')
        else:
            result['note'] = 'Sacrum absent in both TSS and SPINEPS masks'

    # TP from seg-spine_msk (NOT from TSS — TSS 43/44 = L3/L4 vertebrae)
    tp_at_tv = isolate_tp_at_tv(sag_spineps, tp_label, *tv_z_range)
    tp_mask  = inferiormost_tp_cc(tp_at_tv, sac_mask if sac_mask.any() else None)

    if not tp_mask.any():
        result['note'] = (f'TP label {tp_label} (SPINEPS costal process) '
                          f'absent at TV level in seg-spine_msk')
        return result

    result['tp_present']   = True
    result['tp_height_mm'] = measure_tp_height_mm(tp_mask, sag_vox_mm)

    if not sac_mask.any():
        return result

    dist_mm, tp_vox, sac_vox = min_dist_3d(tp_mask, sac_mask, sag_vox_mm)
    result['dist_mm']     = round(dist_mm, 3)
    result['phase1_done'] = True
    if tp_vox  is not None: result['tp_vox']     = tp_vox.tolist()
    if sac_vox is not None: result['sacrum_vox'] = sac_vox.tolist()

    if dist_mm > CONTACT_DIST_MM:
        if result['tp_height_mm'] > TP_HEIGHT_MM:
            result['classification'] = 'Type I'
        result['contact'] = False
        return result

    result['contact']        = True
    result['classification'] = 'CONTACT_PENDING_P2'
    return result


# ============================================================================
# PER-STUDY CLASSIFIER
# ============================================================================

def classify_study(study_id, spineps_dir, totalspine_dir, registered_dir, nifti_dir):
    out = {'study_id': study_id, 'lstv_detected': False, 'castellvi_type': None,
           'confidence': 'high', 'left': {}, 'right': {}, 'details': {},
           'cross_validation': {}, 'errors': []}

    seg_dir    = spineps_dir / 'segmentations' / study_id
    spine_mask = seg_dir / f"{study_id}_seg-spine_msk.nii.gz"   # subregion labels (TP source)
    vert_mask  = seg_dir / f"{study_id}_seg-vert_msk.nii.gz"    # VERIDAH instance labels
    tss_sag    = totalspine_dir / study_id / 'sagittal' / f"{study_id}_sagittal_labeled.nii.gz"
    tss_ax     = totalspine_dir / study_id / 'axial'    / f"{study_id}_axial_labeled.nii.gz"
    spineps_ax = registered_dir / study_id / f"{study_id}_spineps_reg.nii.gz"

    def _find_t2w(acq):
        study_d = nifti_dir / study_id
        if not study_d.exists(): return None
        for sd in sorted(study_d.iterdir()):
            p = sd / f"sub-{study_id}_acq-{acq}_T2w.nii.gz"
            if p.exists(): return p
        return None

    def _load(path, label):
        if not path.exists():
            logger.warning(f"  Missing: {path.name}")
            return None, None
        try:
            return load_canonical(path)
        except Exception as e:
            logger.warning(f"  Cannot load {label}: {e}")
            return None, None

    sag_spineps, sag_sp_nii = _load(spine_mask, 'SPINEPS seg-spine_msk (subregion)')
    sag_vert,    _          = _load(vert_mask,  'SPINEPS seg-vert_msk (VERIDAH)')
    sag_tss,     _          = _load(tss_sag,    'TotalSpineSeg sagittal labeled')

    if sag_spineps is None:
        out['errors'].append('Missing SPINEPS seg-spine_msk (subregion mask)'); return out
    if sag_vert is None:
        out['errors'].append('Missing SPINEPS seg-vert_msk (VERIDAH mask)'); return out
    if sag_tss is None:
        out['errors'].append('Missing TotalSpineSeg sagittal labeled mask'); return out

    sag_spineps = sag_spineps.astype(int)
    sag_vert    = sag_vert.astype(int)
    sag_tss     = sag_tss.astype(int)
    sag_vox_mm  = voxel_size_mm(sag_sp_nii)

    # ── Cross-validation ─────────────────────────────────────────────────────
    xval = run_cross_validation(sag_spineps, sag_vert, sag_tss, sag_vox_mm, study_id)
    out['cross_validation'] = xval
    for w in xval.get('warnings', []):
        out['errors'].append(f'XVAL: {w}')

    # ── Axial data ───────────────────────────────────────────────────────────
    ax_tss,     ax_tss_nii = _load(tss_ax,    'TotalSpineSeg axial labeled')
    ax_spineps, ax_sp_nii  = _load(spineps_ax, 'registered SPINEPS axial (seg-spine labels)')
    ax_t2w, ax_vox_mm = None, None
    ax_t2w_path = _find_t2w('ax')
    if ax_t2w_path:
        ax_t2w_arr, ax_t2w_nii = _load(ax_t2w_path, 'axial T2w')
        if ax_t2w_arr is not None:
            ax_t2w    = ax_t2w_arr
            ax_vox_mm = voxel_size_mm(ax_t2w_nii)

    phase2_available = (ax_tss is not None and ax_spineps is not None and ax_t2w is not None)
    if not phase2_available:
        logger.warning(f"  Phase 2 inputs incomplete — contact cases fall back to Type II")

    if ax_tss     is not None: ax_tss     = ax_tss.astype(int)
    if ax_spineps is not None: ax_spineps = ax_spineps.astype(int)

    # ── Transitional vertebra from VERIDAH seg-vert_msk ──────────────────────
    unique_vert = sorted(np.unique(sag_vert[sag_vert > 0]).tolist())
    named = [VERIDAH_NAMES.get(l, str(l)) for l in unique_vert if l in VERIDAH_NAMES]
    logger.info(f"  VERIDAH labels: {named}")

    tv_label, tv_name = None, None
    for candidate in VERIDAH_LUMBAR:
        if candidate in unique_vert:
            tv_label = candidate
            tv_name  = VERIDAH_NAMES.get(candidate, str(candidate))
            break

    if tv_label is None:
        out['errors'].append(f'No lumbar VERIDAH labels in seg-vert_msk. Present: {unique_vert}')
        return out

    tv_z_range = get_tv_z_range(sag_vert, tv_label)
    if tv_z_range is None:
        out['errors'].append(f'TV label {tv_name} absent in VERIDAH mask')
        return out

    out['details'] = {
        'tv_label': tv_label, 'tv_name': tv_name,
        'has_l6': tv_label == VERIDAH_L6, 'tv_z_range': list(tv_z_range),
        'sag_vox_mm': sag_vox_mm.tolist(), 'phase2_available': phase2_available,
        'tss_sacrum_labels': sorted(TSS_SACRUM_LABELS),
        'tp_source': 'seg-spine_msk labels 43 (L) / 44 (R)',
        'sacrum_source_sag': 'TSS label 50 (preferred) / SPINEPS label 26 (fallback)',
        'sacrum_source_ax':  'TSS axial label 50',
        'label_warning': ('TSS labels 43/44 = L3/L4 vertebrae — '
                          'TP source is always seg-spine_msk, never TSS'),
    }
    logger.info(f"  TV={tv_name}  z=[{tv_z_range[0]},{tv_z_range[1]}]")

    # ── Phase 1 + 2 per side ─────────────────────────────────────────────────
    for side, tp_label in (('left', TP_LEFT_LABEL), ('right', TP_RIGHT_LABEL)):
        try:
            p1 = phase1_sagittal(side, tp_label, sag_spineps, sag_tss,
                                 sag_vox_mm, tv_z_range)
            logger.info(f"  {side:5s} P1: {p1['classification']:22s} "
                        f"h={p1['tp_height_mm']:.1f}mm  d={p1['dist_mm']:.1f}mm  "
                        f"sac={p1.get('sacrum_source','?')}")

            if p1['contact'] and phase2_available:
                p2 = phase2_axial(side, tp_label, ax_spineps, ax_tss,
                                  ax_t2w, ax_vox_mm)
                p1['phase2']         = p2
                p1['classification'] = p2['classification']
                logger.info(f"  {side:5s} P2: {p2['classification']}  "
                            f"bbox={p2.get('bbox_patch_shape')}  valid={p2.get('p2_valid')}")
            elif p1['contact'] and not phase2_available:
                p1['classification'] = 'Type II'
                p1['phase2']         = {'phase2_attempted': False,
                                        'p2_note': 'Axial data unavailable'}
            out[side] = p1

        except Exception as e:
            out['errors'].append(f'{side}: {e}')
            logger.error(f"  {side} failed: {e}")
            logger.debug(traceback.format_exc())

    # ── Final Castellvi classification ────────────────────────────────────────
    left_cls  = out['left'].get('classification',  'Normal')
    right_cls = out['right'].get('classification', 'Normal')
    types     = {left_cls, right_cls} - {'Normal', 'CONTACT_PENDING_P2'}

    if not types:
        logger.info(f"  ✗ No LSTV detected")
    else:
        out['lstv_detected'] = True
        rank = {'Type I': 1, 'Type II': 2, 'Type III': 3, 'Type IV': 4}
        if (left_cls  not in ('Normal', 'CONTACT_PENDING_P2') and
            right_cls not in ('Normal', 'CONTACT_PENDING_P2')):
            out['castellvi_type'] = ('Type I' if left_cls == right_cls == 'Type I'
                                     else 'Type IV')
        else:
            out['castellvi_type'] = max(types, key=lambda t: rank.get(t, 0))
        logger.info(f"  ✓ LSTV: {out['castellvi_type']}")

    return out


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Hybrid Two-Phase LSTV Castellvi Classifier'
    )
    parser.add_argument('--spineps_dir',    required=True)
    parser.add_argument('--totalspine_dir', required=True)
    parser.add_argument('--registered_dir', required=True)
    parser.add_argument('--nifti_dir',      required=True)
    parser.add_argument('--output_dir',     required=True)
    parser.add_argument('--uncertainty_csv', default=None)
    parser.add_argument('--valid_ids',       default=None)
    parser.add_argument('--top_n',           type=int, default=None)
    parser.add_argument('--rank_by',         default='l5_s1_confidence')
    parser.add_argument('--all',             action='store_true')
    parser.add_argument('--study_id',        default=None)
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
        logger.info(f"ALL mode: {len(study_ids)} studies from {seg_root}")
    else:
        if not args.uncertainty_csv or args.top_n is None:
            parser.error("--uncertainty_csv and --top_n required unless --all or --study_id")
        valid_ids = None
        if args.valid_ids:
            valid_ids = set(str(x) for x in np.load(args.valid_ids))
            logger.info(f"Loaded {len(valid_ids)} valid study IDs")
        study_ids = select_studies(Path(args.uncertainty_csv), args.top_n,
                                   args.rank_by, valid_ids)
        study_ids = [s for s in study_ids if (seg_root / s).is_dir()]
        logger.info(f"Selective mode: {len(study_ids)} studies with SPINEPS segs")

    results = []
    errors  = 0
    castellvi_counts = {'Type I': 0, 'Type II': 0, 'Type III': 0, 'Type IV': 0}

    for sid in study_ids:
        logger.info(f"\n[{sid}]")
        try:
            r = classify_study(sid, spineps_dir, totalspine_dir, registered_dir, nifti_dir)
            results.append(r)
            if r.get('errors'):
                errors += 1
            ct = r.get('castellvi_type')
            if ct in castellvi_counts:
                castellvi_counts[ct] += 1
        except Exception as e:
            logger.error(f"  Unhandled: {e}")
            logger.debug(traceback.format_exc())
            errors += 1

    lstv_n = sum(1 for r in results if r.get('lstv_detected'))
    logger.info('\n' + '=' * 70)
    logger.info(f"Total:          {len(results)}")
    logger.info(f"LSTV detected:  {lstv_n}")
    logger.info(f"Errors:         {errors}")
    for t, n in castellvi_counts.items():
        logger.info(f"  {t}: {n}")

    with open(output_dir / 'lstv_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    summary = {
        'total_studies': len(results), 'lstv_detected': lstv_n,
        'lstv_rate': round(lstv_n / max(len(results), 1), 4),
        'error_count': errors, 'castellvi_breakdown': castellvi_counts,
    }
    with open(output_dir / 'lstv_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults → {output_dir}/lstv_results.json")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
