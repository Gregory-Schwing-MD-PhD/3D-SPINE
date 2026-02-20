#!/usr/bin/env python3
"""
04_detect_lstv.py — LSTV Morphological Castellvi Classifier
============================================================
Classifies lumbosacral transitional vertebrae using registered segmentations
from 03b_register_to_axial.py. All volumes are in a common axial reference
frame so no affine mapping is needed — distances are computed directly in
world mm using the shared affine.

Input files per study (all in axial space after registration)
-------------------------------------------------------------
  registered_dir/{study_id}/{study_id}_spineps_reg.nii.gz
    Labels:  43 = Left TP/costal process
             44 = Right TP/costal process
  registered_dir/{study_id}/{study_id}_tss_reg.nii.gz
    Labels:  41-45 = L1-L5, 46 = L6, 50 = Sacrum
  registered_dir/{study_id}/{study_id}_spineps_unc_reg.nii.gz
  registered_dir/{study_id}/{study_id}_tss_unc_reg.nii.gz

Classification (per side)
--------------------------
1.  TV = L6 if present in TSS, else L5.
2.  Isolate TP voxels at TV z-range (all in same grid now).
3.  Measure TP S-I height (mm) from voxel size.
4.  Measure TP-to-sacrum minimum distance (mm) via EDT in world space.
5.  Decision:
      dist > 2.0 mm, height > 19 mm  → Type I
      dist > 2.0 mm, height ≤ 19 mm  → Normal
      dist ≤ 2.0 mm                  → extract uncertainty features,
                                        compute P(Type III) via logistic score
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
from scipy.ndimage import binary_dilation, distance_transform_edt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

L5_LABEL        = 45
L6_LABEL        = 46
SACRUM_LABEL    = 50
TP_LEFT_LABEL   = 43
TP_RIGHT_LABEL  = 44

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
    nii = nib.load(str(path))
    nii = nib.as_closest_canonical(nii)
    data = nii.get_fdata()
    if data.ndim == 4:
        data = data[..., 0]
    return data, nii


def voxel_size_mm(nii: nib.Nifti1Image) -> np.ndarray:
    return np.abs(np.array(nii.header.get_zooms()[:3], dtype=float))


# ============================================================================
# MORPHOMETRICS (all in shared axial space)
# ============================================================================

def get_tv_z_range(tss_data: np.ndarray,
                   tv_label: int) -> Optional[Tuple[int, int]]:
    mask = tss_data == tv_label
    if not mask.any():
        return None
    z = np.where(mask)[2]
    return int(z.min()), int(z.max())


def isolate_tp_at_tv(spineps_data: np.ndarray,
                     tp_label: int,
                     z_min: int, z_max: int) -> np.ndarray:
    """Extract TP voxels within the TV z range (same grid — no mapping needed)."""
    tp_full = spineps_data == tp_label
    iso = np.zeros_like(tp_full)
    z_lo = max(z_min, 0)
    z_hi = min(z_max, spineps_data.shape[2] - 1)
    iso[:, :, z_lo:z_hi + 1] = tp_full[:, :, z_lo:z_hi + 1]
    return iso


def measure_tp_height_mm(tp_mask: np.ndarray,
                         vox_mm: np.ndarray) -> float:
    """S-I extent of TP mask in mm (axis 2 = I-S in RAS canonical)."""
    if not tp_mask.any():
        return 0.0
    z = np.where(tp_mask)[2]
    return float((z.max() - z.min() + 1) * vox_mm[2])


def measure_min_dist_mm(mask_a: np.ndarray,
                        mask_b: np.ndarray,
                        vox_mm: np.ndarray) -> float:
    """
    Minimum Euclidean distance (mm) between two binary masks.
    Uses EDT on mask_b with voxel spacing, then samples at mask_a locations.
    """
    if not mask_a.any() or not mask_b.any():
        return float('inf')
    edt = distance_transform_edt(~mask_b, sampling=vox_mm)
    return float(edt[mask_a].min())


# ============================================================================
# UNCERTAINTY FEATURES
# ============================================================================

def build_contact_zone(mask_a: np.ndarray,
                       mask_b: np.ndarray,
                       vox_mm: np.ndarray,
                       radius_mm: float) -> np.ndarray:
    radius_vox = np.maximum(np.round(radius_mm / vox_mm).astype(int), 1)
    struct = np.ones(2 * radius_vox + 1, dtype=bool)
    return (binary_dilation(mask_a, structure=struct) &
            binary_dilation(mask_b, structure=struct))


def extract_uncertainty_features(
    unc_map: Optional[np.ndarray],
    contact_zone: np.ndarray,
    source_label: str,
) -> dict:
    base = {
        'unc_mean':      None,
        'unc_std':       None,
        'unc_high_frac': None,
        'n_voxels':      0,
        'valid':         False,
        'source':        source_label,
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


def compute_p_type_iii(
    unc_mean: float,
    unc_std: float,
    unc_high_frac: float,
    tp_height_mm: float,
) -> float:
    score = (
        LOGISTIC_INTERCEPT
        + LOGISTIC_W_UNC_MEAN  * unc_mean
        + LOGISTIC_W_UNC_STD   * unc_std
        + LOGISTIC_W_UNC_HIGH  * unc_high_frac
        + LOGISTIC_W_TP_HEIGHT * (tp_height_mm / 30.0)
    )
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

def classify_side(
    side: str,
    tp_label: int,
    spineps_data: np.ndarray,
    tss_data: np.ndarray,
    vox_mm: np.ndarray,
    tv_z_range: Tuple[int, int],
    spineps_unc: Optional[np.ndarray],
    tss_unc: Optional[np.ndarray],
) -> dict:
    result = {
        'tp_present':    False,
        'tp_height_mm':  0.0,
        'contact':       False,
        'dist_mm':       float('inf'),
        'p_type_ii':     None,
        'p_type_iii':    None,
        'confidence':    None,
        'unc_features':  None,
        'classification': 'Normal',
    }

    z_min, z_max = tv_z_range
    tp_mask = isolate_tp_at_tv(spineps_data, tp_label, z_min, z_max)

    if not tp_mask.any():
        result['note'] = 'TP label absent at TV level'
        return result

    result['tp_present']   = True
    result['tp_height_mm'] = measure_tp_height_mm(tp_mask, vox_mm)

    sacrum_mask = tss_data == SACRUM_LABEL
    if not sacrum_mask.any():
        result['note'] = 'Sacrum label absent'
        return result

    dist = measure_min_dist_mm(tp_mask, sacrum_mask, vox_mm)
    result['dist_mm'] = round(dist, 2)

    # --- Type I: no contact, but enlarged TP ---------------------------------
    if dist > CONTACT_DIST_MM:
        if result['tp_height_mm'] > TP_HEIGHT_MM:
            result['classification'] = 'Type I'
        return result

    # --- Contact: Type II vs III via uncertainty features --------------------
    result['contact'] = True
    contact_zone = build_contact_zone(tp_mask, sacrum_mask, vox_mm, CONTACT_DILATION_MM)

    # Prefer TSS uncertainty (higher resolution), fall back to SPINEPS
    unc_features = extract_uncertainty_features(tss_unc, contact_zone, 'tss')
    if not unc_features['valid']:
        unc_features = extract_uncertainty_features(spineps_unc, contact_zone, 'spineps')

    result['unc_features'] = unc_features

    if unc_features['valid']:
        p3 = compute_p_type_iii(
            unc_features['unc_mean'],
            unc_features['unc_std'],
            unc_features['unc_high_frac'],
            result['tp_height_mm'],
        )
        result['p_type_iii']  = round(p3, 4)
        result['p_type_ii']   = round(1.0 - p3, 4)
        result['confidence']  = probability_to_confidence(p3)
        result['classification'] = 'Type III' if p3 >= 0.5 else 'Type II'
    else:
        # No uncertainty available — default to Type II (pseudo-arthrosis)
        result['classification'] = 'Type II'
        result['note'] = 'No uncertainty map — defaulted to Type II'

    return result


# ============================================================================
# PER-STUDY
# ============================================================================

def classify_study(
    study_id: str,
    registered_dir: Path,
) -> dict:
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

    reg = registered_dir / study_id

    def load(name, label):
        p = reg / f"{study_id}_{name}"
        if not p.exists():
            logger.warning(f"  [{study_id}] Missing: {p.name}")
            return None, None
        try:
            return load_canonical(p)
        except Exception as e:
            logger.warning(f"  [{study_id}] Cannot load {label}: {e}")
            return None, None

    spineps_data, spineps_nii = load('spineps_reg.nii.gz',     'SPINEPS seg')
    tss_data,     tss_nii     = load('tss_reg.nii.gz',         'TSS seg')
    spineps_unc,  _           = load('spineps_unc_reg.nii.gz', 'SPINEPS unc')
    tss_unc,      _           = load('tss_unc_reg.nii.gz',     'TSS unc')

    if spineps_data is None or tss_data is None:
        out['errors'].append('Missing required segmentation(s)')
        return out

    spineps_data = spineps_data.astype(int)
    tss_data     = tss_data.astype(int)
    vox_mm       = voxel_size_mm(spineps_nii)  # both on same grid now

    # Target vertebra
    has_l6   = L6_LABEL in np.unique(tss_data)
    tv_label = L6_LABEL if has_l6 else L5_LABEL
    tv_name  = 'L6' if has_l6 else 'L5'

    tv_z_range = get_tv_z_range(tss_data, tv_label)
    if tv_z_range is None:
        out['errors'].append(f'TV label {tv_name} not found in TSS segmentation')
        return out

    sacrum_present = (SACRUM_LABEL in np.unique(tss_data))

    out['details'] = {
        'tv_label':      tv_label,
        'tv_name':       tv_name,
        'has_l6':        has_l6,
        'sacrum_present': sacrum_present,
        'tv_z_range':    list(tv_z_range),
    }

    # Classify each side
    for side, tp_label in (('left', TP_LEFT_LABEL), ('right', TP_RIGHT_LABEL)):
        try:
            r = classify_side(
                side, tp_label,
                spineps_data, tss_data, vox_mm, tv_z_range,
                spineps_unc if spineps_unc is not None else None,
                tss_unc     if tss_unc     is not None else None,
            )
            out[side] = r
            logger.info(
                f"  [{study_id}] {side:5s}: {r['classification']:8s}"
                f"  h={r['tp_height_mm']:.1f}mm  d={r['dist_mm']:.1f}mm"
            )
        except Exception as e:
            out['errors'].append(f'{side}: {e}')
            logger.error(f"  [{study_id}] {side} failed: {e}")

    # Overall Castellvi type
    types = {
        out.get('left',  {}).get('classification'),
        out.get('right', {}).get('classification'),
    }
    types.discard(None)
    types.discard('Normal')

    if not types:
        logger.info(f"  [{study_id}] ✗ No LSTV detected")
    else:
        out['lstv_detected'] = True
        # Bilateral → Type IV; unilateral → use the positive side
        if len(types) == 2 and types != {'Type I'} and types != {'Type I', 'Normal'}:
            left_cls  = out.get('left',  {}).get('classification', 'Normal')
            right_cls = out.get('right', {}).get('classification', 'Normal')
            if left_cls != 'Normal' and right_cls != 'Normal':
                out['castellvi_type'] = 'Type IV'
            else:
                out['castellvi_type'] = next(
                    c for c in (left_cls, right_cls) if c != 'Normal'
                )
        else:
            out['castellvi_type'] = max(
                types,
                key=lambda t: {'Type I': 1, 'Type II': 2,
                               'Type III': 3, 'Type IV': 4}.get(t, 0)
            )

        # Overall confidence = minimum of both sides
        conf_order = {'high': 3, 'moderate': 2, 'low': 1, None: 3}
        left_conf  = out.get('left',  {}).get('confidence')
        right_conf = out.get('right', {}).get('confidence')
        out['confidence'] = min(
            (left_conf, right_conf),
            key=lambda c: conf_order.get(c, 3)
        ) or 'high'

        logger.info(f"  [{study_id}] ✓ LSTV: {out['castellvi_type']} ({out['confidence']})")

    return out


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='LSTV Castellvi Classifier')
    parser.add_argument('--registered_dir', required=True,
                        help='Output of 03b_register_to_axial.py')
    parser.add_argument('--output_dir',     required=True)
    parser.add_argument('--study_id',       default=None,
                        help='Single study (omit for batch)')
    parser.add_argument('--mode',           default='prod',
                        choices=['trial', 'prod'])
    args = parser.parse_args()

    registered_dir = Path(args.registered_dir)
    output_dir     = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.study_id:
        study_ids = [args.study_id]
    else:
        study_ids = sorted(d.name for d in registered_dir.iterdir() if d.is_dir())
        if args.mode == 'trial':
            study_ids = study_ids[:5]

    logger.info(f"Found {len(study_ids)} studies to process")

    results = []
    errors  = 0
    low_conf = 0
    castellvi_counts = {'Type I': 0, 'Type II': 0, 'Type III': 0, 'Type IV': 0}

    for sid in study_ids:
        logger.info(f"\n[{sid}]")
        try:
            r = classify_study(sid, registered_dir)
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
            'All measurements in registered axial space. '
            'Type II/III probabilities use a logistic scoring function with '
            'clinically-grounded priors. Replace with CalibratedClassifierCV '
            'once ground truth Castellvi labels are available.'
        ),
    }
    summary_path = output_dir / 'lstv_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary -> {summary_path}")


if __name__ == '__main__':
    main()
