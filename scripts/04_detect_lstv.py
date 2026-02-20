#!/usr/bin/env python3
"""
04_detect_lstv.py — LSTV Morphological Castellvi Classifier (SPINEPS-only)
===========================================================================
Classifies lumbosacral transitional vertebrae using SPINEPS segmentation
with VERIDAH vertebra labeling. All measurements are performed directly in
sagittal T2w space — no registration step required.

VERIDAH label scheme (instance mask, seg-vert_msk.nii.gz)
----------------------------------------------------------
  20 = L1,  21 = L2,  22 = L3,  23 = L4,  24 = L5,  25 = L6

SPINEPS semantic label scheme (seg-spine_msk.nii.gz)
-----------------------------------------------------
  43 = Costal_Process_Left  (= lumbar transverse process left)
  44 = Costal_Process_Right (= lumbar transverse process right)
  26 = Sacrum

NOTE: SPINEPS does NOT produce a separate seg-subreg file.
      seg-spine_msk IS the subregion mask (labels 41-100 + 26 sacrum).
      Both 'subreg' and 'semantic' keys point to the same file.
"""

import argparse
import json
import logging
import math
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, distance_transform_edt, label as cc_label

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS — VERIDAH vertebra instance labels
# ============================================================================

L5_LABEL     = 24   # VERIDAH
L6_LABEL     = 25   # VERIDAH
SACRUM_LABEL = 26   # SPINEPS semantic / sub-region

TP_LEFT_LABEL  = 43  # SPINEPS costal/transverse process left
TP_RIGHT_LABEL = 44  # SPINEPS costal/transverse process right

TP_HEIGHT_MM        = 19.0
CONTACT_DIST_MM     = 2.0
CONTACT_DILATION_MM = 3.0

LOGISTIC_INTERCEPT   =  0.5
LOGISTIC_W_UNC_MEAN  = -5.0
LOGISTIC_W_UNC_STD   = -3.0
LOGISTIC_W_UNC_HIGH  = -4.0
LOGISTIC_W_TP_HEIGHT =  1.0

CONF_HIGH_MARGIN     = 0.30
CONF_MODERATE_MARGIN = 0.15


# ============================================================================
# NIfTI HELPERS
# ============================================================================

def load_canonical(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """Load NIfTI in RAS canonical orientation. Squeeze 4D → 3D."""
    nii  = nib.load(str(path))
    nii  = nib.as_closest_canonical(nii)
    data = nii.get_fdata()
    if data.ndim == 4:
        data = data[..., 0]
    return data, nii


def voxel_size_mm(nii: nib.Nifti1Image) -> np.ndarray:
    return np.abs(np.array(nii.header.get_zooms()[:3], dtype=float))


# ============================================================================
# TV LEVEL FROM VERIDAH INSTANCE MASK
# ============================================================================

def get_tv_z_range(vert_data: np.ndarray,
                   tv_label: int) -> Optional[Tuple[int, int]]:
    """Return (z_min, z_max) of the TV vertebra instance in sagittal space."""
    mask = vert_data == tv_label
    if not mask.any():
        return None
    z = np.where(mask)[2]
    return int(z.min()), int(z.max())


# ============================================================================
# TP ISOLATION
# ============================================================================

def isolate_tp_at_tv(subreg_data: np.ndarray,
                     tp_label: int,
                     z_min: int,
                     z_max: int) -> np.ndarray:
    """Extract TP voxels within the TV z-range."""
    tp_full = subreg_data == tp_label
    iso     = np.zeros_like(tp_full)
    z_lo    = max(z_min, 0)
    z_hi    = min(z_max, subreg_data.shape[2] - 1)
    iso[:, :, z_lo:z_hi + 1] = tp_full[:, :, z_lo:z_hi + 1]
    return iso


def inferiormost_tp_cc(tp_mask3d: np.ndarray,
                       sacrum_mask3d: Optional[np.ndarray]) -> np.ndarray:
    """
    From a 3-D TP label mask (may contain blobs at multiple levels), isolate
    the inferiormost blob above the sacrum — anatomy-driven, not label-driven.
    """
    if not tp_mask3d.any():
        return np.zeros_like(tp_mask3d, dtype=bool)
    labeled, n = cc_label(tp_mask3d)
    if n == 0:
        return np.zeros_like(tp_mask3d, dtype=bool)
    if n == 1:
        return tp_mask3d.astype(bool)

    sac_z_min = None
    if sacrum_mask3d is not None and sacrum_mask3d.any():
        sac_z_min = int(np.where(sacrum_mask3d)[2].min())

    cc_info = []
    for i in range(1, n + 1):
        comp     = (labeled == i)
        z_coords = np.where(comp)[2]
        cc_info.append((float(z_coords.mean()), int(z_coords.max()), comp))

    cc_info.sort(key=lambda t: t[0])  # ascending z_centroid = most inferior first

    if sac_z_min is not None:
        candidates = [(zc, zm, c) for zc, zm, c in cc_info if zm < sac_z_min]
        if candidates:
            return candidates[0][2].astype(bool)

    return cc_info[0][2].astype(bool)


# ============================================================================
# MORPHOMETRICS
# ============================================================================

def measure_tp_height_mm(tp_mask: np.ndarray,
                          vox_mm: np.ndarray) -> float:
    """S-I extent of TP mask in mm (axis 2 = I-S in RAS canonical)."""
    if not tp_mask.any():
        return 0.0
    z = np.where(tp_mask)[2]
    return float((z.max() - z.min() + 1) * vox_mm[2])


def min_dist_3d(mask_a: np.ndarray,
                mask_b: np.ndarray,
                vox_mm: np.ndarray) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    True 3-D Euclidean minimum distance between two binary masks (mm).
    Returns (dist_mm, vox_in_a, vox_in_b).
    """
    if not mask_a.any() or not mask_b.any():
        return float('inf'), None, None

    dist_to_b = distance_transform_edt(~mask_b, sampling=vox_mm)
    dist_at_a = np.where(mask_a, dist_to_b, np.inf)
    flat_idx  = int(np.argmin(dist_at_a))
    vox_a     = np.array(np.unravel_index(flat_idx, mask_a.shape))
    dist_mm   = float(dist_to_b[tuple(vox_a)])

    # Nearest sacrum voxel — z-bounded brute force for speed
    z_lo = max(0, int(vox_a[2]) - 20)
    z_hi = min(mask_b.shape[2], int(vox_a[2]) + 20)
    sub  = mask_b[:, :, z_lo:z_hi]
    if sub.any():
        coords    = np.array(np.where(sub))
        coords[2] += z_lo
        phys_a    = vox_a * vox_mm
        phys_b    = coords.T * vox_mm
        d2        = ((phys_b - phys_a) ** 2).sum(axis=1)
        vox_b     = coords[:, int(np.argmin(d2))]
    else:
        coords = np.array(np.where(mask_b))
        phys_a = vox_a * vox_mm
        phys_b = coords.T * vox_mm
        d2     = ((phys_b - phys_a) ** 2).sum(axis=1)
        vox_b  = coords[:, int(np.argmin(d2))]

    return dist_mm, vox_a, vox_b


# ============================================================================
# UNCERTAINTY FEATURES
# ============================================================================

def build_contact_zone(mask_a: np.ndarray,
                       mask_b: np.ndarray,
                       vox_mm: np.ndarray,
                       radius_mm: float) -> np.ndarray:
    radius_vox = np.maximum(np.round(radius_mm / vox_mm).astype(int), 1)
    struct     = np.ones(2 * radius_vox + 1, dtype=bool)
    return (binary_dilation(mask_a, structure=struct) &
            binary_dilation(mask_b, structure=struct))


def extract_uncertainty_features(unc_map: Optional[np.ndarray],
                                  contact_zone: np.ndarray) -> dict:
    base = {
        'unc_mean': None, 'unc_std': None, 'unc_high_frac': None,
        'n_voxels': 0, 'valid': False,
    }
    if unc_map is None or not contact_zone.any():
        return base
    vals = unc_map[contact_zone]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return base
    base['unc_mean']      = float(np.mean(vals))
    base['unc_std']       = float(np.std(vals))
    base['unc_high_frac'] = float(np.mean(vals > 0.3))
    base['n_voxels']      = int(len(vals))
    base['valid']         = True
    return base


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))


def compute_p_type_iii(unc_mean: float, unc_std: float,
                        unc_high_frac: float, tp_height_mm: float) -> float:
    score = (LOGISTIC_INTERCEPT
             + LOGISTIC_W_UNC_MEAN  * unc_mean
             + LOGISTIC_W_UNC_STD   * unc_std
             + LOGISTIC_W_UNC_HIGH  * unc_high_frac
             + LOGISTIC_W_TP_HEIGHT * (tp_height_mm / 30.0))
    return sigmoid(score)


def probability_to_confidence(p_type_iii: float) -> str:
    margin = abs(p_type_iii - 0.5)
    if margin > CONF_HIGH_MARGIN:
        return 'high'
    if margin > CONF_MODERATE_MARGIN:
        return 'moderate'
    return 'low'


# ============================================================================
# PER-SIDE CLASSIFICATION
# ============================================================================

def classify_side(side: str,
                  tp_label: int,
                  subreg_data: np.ndarray,
                  sacrum_mask: np.ndarray,
                  vox_mm: np.ndarray,
                  tv_z_range: Tuple[int, int],
                  unc_map: Optional[np.ndarray]) -> dict:

    result = {
        'tp_present':     False,
        'tp_height_mm':   0.0,
        'contact':        False,
        'dist_mm':        float('inf'),
        'tp_vox':         None,
        'p_type_ii':      None,
        'p_type_iii':     None,
        'confidence':     None,
        'unc_features':   None,
        'classification': 'Normal',
    }

    z_min, z_max = tv_z_range

    # Step 1: isolate TP at TV z-range then pick inferiormost blob
    tp_at_tv = isolate_tp_at_tv(subreg_data, tp_label, z_min, z_max)
    tp_mask  = inferiormost_tp_cc(tp_at_tv, sacrum_mask)

    if not tp_mask.any():
        result['note'] = 'TP label absent at TV level'
        return result

    result['tp_present']   = True
    result['tp_height_mm'] = measure_tp_height_mm(tp_mask, vox_mm)

    if not sacrum_mask.any():
        result['note'] = 'Sacrum label absent'
        return result

    dist_mm, tp_vox, _ = min_dist_3d(tp_mask, sacrum_mask, vox_mm)
    result['dist_mm'] = round(dist_mm, 2)
    if tp_vox is not None:
        result['tp_vox'] = tp_vox.tolist()

    # Type I: enlarged but no contact
    if dist_mm > CONTACT_DIST_MM:
        if result['tp_height_mm'] > TP_HEIGHT_MM:
            result['classification'] = 'Type I'
        return result

    # Contact: Type II vs III via uncertainty
    result['contact']    = True
    contact_zone         = build_contact_zone(tp_mask, sacrum_mask, vox_mm, CONTACT_DILATION_MM)
    unc_features         = extract_uncertainty_features(unc_map, contact_zone)
    result['unc_features'] = unc_features

    if unc_features['valid']:
        p3 = compute_p_type_iii(
            unc_features['unc_mean'],
            unc_features['unc_std'],
            unc_features['unc_high_frac'],
            result['tp_height_mm'],
        )
        result['p_type_iii']     = round(p3, 4)
        result['p_type_ii']      = round(1.0 - p3, 4)
        result['confidence']     = probability_to_confidence(p3)
        result['classification'] = 'Type III' if p3 >= 0.5 else 'Type II'
    else:
        result['classification'] = 'Type II'
        result['note']           = 'No uncertainty map — defaulted to Type II'

    return result


# ============================================================================
# FILE DISCOVERY
# ============================================================================

def find_spineps_files(spineps_dir: Path, study_id: str) -> dict:
    seg_dir = spineps_dir / 'segmentations' / study_id
    # SPINEPS does NOT produce a separate seg-subreg file.
    # seg-spine_msk IS the subregion mask — both keys point to the same file.
    spine_mask = seg_dir / f"{study_id}_seg-spine_msk.nii.gz"
    return {
        'subreg':   spine_mask,   # ← seg-spine IS the subreg mask
        'instance': seg_dir / f"{study_id}_seg-vert_msk.nii.gz",
        'semantic': spine_mask,   # ← same file, kept for fallback logic below
        'unc':      seg_dir / f"{study_id}_unc.nii.gz",
    }


# ============================================================================
# PER-STUDY
# ============================================================================

def classify_study(study_id: str,
                   spineps_dir: Path) -> dict:
    out = {
        'study_id':       study_id,
        'lstv_detected':  False,
        'castellvi_type': None,
        'confidence':     'high',
        'left':           {},
        'right':          {},
        'details':        {},
        'errors':         [],
    }

    files = find_spineps_files(spineps_dir, study_id)

    def _load(key, label):
        p = files[key]
        if not p.exists():
            logger.warning(f"  [{study_id}] Missing: {p.name}")
            return None, None
        try:
            return load_canonical(p)
        except Exception as e:
            logger.warning(f"  [{study_id}] Cannot load {label}: {e}")
            return None, None

    # Sub-region mask = seg-spine mask (has TP labels 43/44 and sacrum 26)
    subreg_data, subreg_nii = _load('subreg',   'spine/subreg mask')
    # VERIDAH instance mask has per-vertebra labels (L1=20 … L6=25)
    vert_data,   vert_nii   = _load('instance', 'instance/VERIDAH mask')
    # Uncertainty map for II vs III discrimination
    unc_data,    _          = _load('unc',       'uncertainty map')

    if subreg_data is None:
        out['errors'].append('Missing seg-spine mask (needed for TP labels 43/44 and sacrum 26)')
        return out
    if vert_data is None:
        out['errors'].append('Missing VERIDAH instance mask (needed for level labels)')
        return out

    subreg_data = subreg_data.astype(int)
    vert_data   = vert_data.astype(int)
    vox_mm      = voxel_size_mm(subreg_nii)

    # Log what labels are actually present in the spine mask
    unique_subreg = sorted(np.unique(subreg_data[subreg_data > 0]).tolist())
    logger.info(f"  [{study_id}] Spine mask labels present: {unique_subreg}")

    # Sacrum from spine/subreg mask (label 26)
    sacrum_mask = (subreg_data == SACRUM_LABEL)
    if not sacrum_mask.any():
        logger.warning(f"  [{study_id}] Sacrum label (26) not found in spine mask")

    # Target vertebra: L6 if VERIDAH found it, else L5
    unique_vert = sorted(np.unique(vert_data[vert_data > 0]).tolist())
    logger.info(f"  [{study_id}] VERIDAH instance labels present: {unique_vert}")

    VERIDAH_NAMES = {20:'L1',21:'L2',22:'L3',23:'L4',24:'L5',25:'L6',26:'Sacrum(inst)'}
    named = [VERIDAH_NAMES.get(l, str(l)) for l in unique_vert]
    logger.info(f"  [{study_id}] VERIDAH named: {named}")

    # Pick TV: L6 > L5 > inferiormost lumbar label found
    # VERIDAH sometimes only labels up to L4 on short FOV scans — use
    # whatever the inferiormost lumbar vertebra is rather than erroring out.
    LUMBAR_LABELS = [25, 24, 23, 22, 21, 20]  # L6 down to L1, pick highest present
    tv_label, tv_name = None, None
    for candidate in LUMBAR_LABELS:
        if candidate in unique_vert:
            tv_label = candidate
            tv_name  = VERIDAH_NAMES.get(candidate, str(candidate))
            break

    if tv_label is None:
        msg = f'No lumbar labels (20-25) found. Labels present: {unique_vert} ({named}).'
        logger.error(f"  [{study_id}] ✗ {msg}")
        out['errors'].append(msg)
        return out

    if tv_label not in (L5_LABEL, L6_LABEL):
        logger.warning(f"  [{study_id}] ⚠ L5/L6 absent — using inferiormost label {tv_name} ({tv_label}) as TV")

    tv_z_range = get_tv_z_range(vert_data, tv_label)

    out['details'] = {
        'tv_label':            tv_label,
        'tv_name':             tv_name,
        'has_l6':              tv_label == L6_LABEL,
        'sacrum_present':      bool(sacrum_mask.any()),
        'tv_z_range':          list(tv_z_range),
        'vox_mm':              vox_mm.tolist(),
        'spine_mask_labels':   unique_subreg,
    }

    logger.info(f"  [{study_id}] TV={tv_name} z=[{tv_z_range[0]},{tv_z_range[1]}]"
                f"  sacrum={'yes' if sacrum_mask.any() else 'NO'}")

    # Classify each side
    for side, tp_label in (('left', TP_LEFT_LABEL), ('right', TP_RIGHT_LABEL)):
        try:
            r = classify_side(
                side, tp_label,
                subreg_data, sacrum_mask, vox_mm, tv_z_range,
                unc_data,
            )
            out[side] = r
            logger.info(
                f"  [{study_id}] {side:5s}: {r['classification']:8s}"
                f"  h={r['tp_height_mm']:.1f}mm  d={r['dist_mm']:.1f}mm"
            )
        except Exception as e:
            out['errors'].append(f'{side}: {e}')
            logger.error(f"  [{study_id}] {side} failed: {e}")
            logger.debug(traceback.format_exc())

    # Overall Castellvi type
    left_cls  = out.get('left',  {}).get('classification', 'Normal')
    right_cls = out.get('right', {}).get('classification', 'Normal')
    types     = {left_cls, right_cls} - {'Normal'}

    if not types:
        logger.info(f"  [{study_id}] ✗ No LSTV detected")
    else:
        out['lstv_detected'] = True
        rank = {'Type I': 1, 'Type II': 2, 'Type III': 3, 'Type IV': 4}

        if left_cls != 'Normal' and right_cls != 'Normal':
            if left_cls == right_cls == 'Type I':
                out['castellvi_type'] = 'Type I'
            else:
                out['castellvi_type'] = 'Type IV'
        else:
            out['castellvi_type'] = max(types, key=lambda t: rank.get(t, 0))

        conf_order = {'high': 3, 'moderate': 2, 'low': 1, None: 3}
        out['confidence'] = min(
            (out.get('left',  {}).get('confidence'),
             out.get('right', {}).get('confidence')),
            key=lambda c: conf_order.get(c, 3)
        ) or 'high'

        logger.info(f"  [{study_id}] ✓ LSTV: {out['castellvi_type']} ({out['confidence']})")

    return out


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='LSTV Castellvi Classifier (SPINEPS-only)')
    parser.add_argument('--spineps_dir', required=True,
                        help='Root output of 02_run_spineps.py  '
                             '(contains segmentations/{study_id}/)')
    parser.add_argument('--output_dir',  required=True)
    parser.add_argument('--study_id',    default=None,
                        help='Single study (omit for batch)')
    parser.add_argument('--mode',        default='prod',
                        choices=['trial', 'prod'])
    args = parser.parse_args()

    spineps_dir = Path(args.spineps_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seg_root = spineps_dir / 'segmentations'
    if not seg_root.exists():
        logger.error(f"segmentations/ not found under {spineps_dir}")
        return 1

    if args.study_id:
        study_ids = [args.study_id]
    else:
        study_ids = sorted(d.name for d in seg_root.iterdir() if d.is_dir())
        if args.mode == 'trial':
            study_ids = study_ids[:5]

    logger.info(f"Processing {len(study_ids)} studies")

    results          = []
    errors           = 0
    low_conf         = 0
    castellvi_counts = {'Type I': 0, 'Type II': 0, 'Type III': 0, 'Type IV': 0}

    for sid in study_ids:
        logger.info(f"\n[{sid}]")
        try:
            r = classify_study(sid, spineps_dir)
            results.append(r)
            if r.get('errors'):
                errors += 1
            if r.get('confidence') == 'low':
                low_conf += 1
            ct = r.get('castellvi_type')
            if ct in castellvi_counts:
                castellvi_counts[ct] += 1
        except Exception as e:
            logger.error(f"  [{sid}] Unhandled: {e}")
            logger.debug(traceback.format_exc())
            errors += 1

    lstv_n = sum(1 for r in results if r.get('lstv_detected'))

    logger.info('\n' + '=' * 70)
    logger.info('LSTV DETECTION SUMMARY')
    logger.info('=' * 70)
    logger.info(f"Total studies:         {len(results)}")
    logger.info(f"LSTV detected:         {lstv_n} ({100*lstv_n/max(len(results),1):.1f}%)")
    logger.info(f"Errors / incomplete:   {errors}")
    logger.info(f"Low confidence cases:  {low_conf}")
    logger.info('')
    logger.info('Castellvi Breakdown:')
    for t, n in castellvi_counts.items():
        logger.info(f"  {t}: {n}")

    results_path = output_dir / 'lstv_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults -> {results_path}")

    summary = {
        'total_studies':        len(results),
        'lstv_detected':        lstv_n,
        'lstv_rate':            round(lstv_n / max(len(results), 1), 4),
        'error_count':          errors,
        'low_confidence_cases': low_conf,
        'castellvi_breakdown':  castellvi_counts,
        'note': (
            'All measurements in sagittal SPINEPS space. '
            'VERIDAH labels used for vertebral level identification. '
            'seg-spine_msk used as subreg source (SPINEPS does not emit a separate subreg file). '
            'Type II/III probabilities use a logistic scoring function. '
            'Replace with CalibratedClassifierCV once ground truth available.'
        ),
    }
    summary_path = output_dir / 'lstv_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary -> {summary_path}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
