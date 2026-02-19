#!/usr/bin/env python3
"""
LSTV Detector - Morphological Castellvi Classification
with Probabilistic Uncertainty Integration

Combines SPINEPS and TotalSpineSeg outputs to detect and classify
lumbosacral transitional vertebrae (LSTV) using purely morphological
criteria, assessed independently for each side (Left / Right).

Input files per study_id
------------------------
SPINEPS:
  spineps_dir/segmentations/{study_id}/{study_id}_seg-spine_msk.nii.gz
    - Label 43: Left transverse / costal process
    - Label 44: Right transverse / costal process
  spineps_dir/segmentations/{study_id}/{study_id}_unc.nii.gz
    - Normalised Shannon entropy [0, 1], float32

TotalSpineSeg:
  totalspine_dir/{study_id}/sagittal/{study_id}_sagittal_labeled.nii.gz
    - Labels 41-45: L1-L5; 46: L6 (only in lumbarisation); 50: Sacrum
    - Labels 91-100: intervertebral discs
  totalspine_dir/{study_id}/sagittal/{study_id}_sagittal_unc.nii.gz
    - Normalised Shannon entropy [0, 1], float32

Classification logic (per side)
---------------------------------
1.  Identify Target Vertebra (TV): L6 if label 46 present, else L5 (45).
2.  Isolate TV-level TP voxels from SPINEPS using the TV's S-I z-range in
    the TotalSpineSeg mask.
3.  Measure S-I height of the isolated TP mask (mm).
4.  Compute minimum distance between isolated TP mask and Sacrum mask.
5.  Decision tree:
      distance > 2.0 mm  ->  height > 19 mm  ->  Type I
                         ->  height <= 19 mm  ->  Normal
      distance <= 2.0 mm ->  extract uncertainty feature vector from
                             contact zone and compute P(Type III)
                             via logistic scoring.

Uncertainty Feature Vector (contact zone)
-------------------------------------------
Rather than a single mean uncertainty threshold, we extract:
  - unc_mean      : mean entropy -> overall model confidence
  - unc_std       : std dev -> heterogeneity (pseudo-arthrosis is patchy)
  - unc_high_frac : fraction of voxels with unc > 0.3 -> extent of confusion

Logistic scoring converts these to P(Type III | contact zone features):
  - Low uniform uncertainty -> solid bone -> Type III (bony fusion)
  - High or heterogeneous uncertainty -> fibrocartilage -> Type II

Output JSON per study includes:
  - castellvi_type    : hard classification
  - confidence        : high / moderate / low
  - p_type_ii / p_type_iii : calibrated probabilities (contact cases only)

Usage:
    python 04_detect_lstv.py \
        --spineps_dir    results/spineps \
        --totalspine_dir results/totalspineseg \
        --output_dir     results/lstv_detection \
        [--debug]
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

LUMBAR_LABELS   = {41: 'L1', 42: 'L2', 43: 'L3', 44: 'L4', 45: 'L5', 46: 'L6'}
DISC_LABELS     = {91: 'T12-L1', 92: 'L1-L2', 93: 'L2-L3', 94: 'L3-L4',
                   95: 'L4-L5', 100: 'L5-S'}
SACRUM_LABEL    = 50
L5_LABEL        = 45
L6_LABEL        = 46
TP_LEFT_LABEL   = 43
TP_RIGHT_LABEL  = 44

TP_HEIGHT_MM        = 19.0
CONTACT_DIST_MM     = 2.0
CONTACT_DILATION_MM = 3.0

# Logistic scoring weights for P(Type III | contact zone features).
# Negative weight = evidence AGAINST fusion (toward Type II).
# Retrain with CalibratedClassifierCV once ground truth labels are available.
LOGISTIC_INTERCEPT   =  0.5
LOGISTIC_W_UNC_MEAN  = -5.0   # high mean entropy -> pseudo-arthrosis
LOGISTIC_W_UNC_STD   = -3.0   # high heterogeneity -> irregular fibrocartilage
LOGISTIC_W_UNC_HIGH  = -4.0   # many confused voxels -> no clean bone signal
LOGISTIC_W_TP_HEIGHT =  1.0   # larger TP more likely to fuse completely

CONF_HIGH_MARGIN     = 0.30
CONF_MODERATE_MARGIN = 0.15


# ============================================================================
# NIfTI HELPERS
# ============================================================================

def load_canonical(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    nii = nib.load(str(path))
    nii = nib.as_closest_canonical(nii)
    return nii.get_fdata(), nii


def voxel_size_mm(nii: nib.Nifti1Image) -> np.ndarray:
    return np.abs(np.array(nii.header.get_zooms()[:3], dtype=float))


# ============================================================================
# MORPHOMETRICS
# ============================================================================

def get_tv_z_range(tss_data: np.ndarray, tv_label: int) -> Optional[Tuple[int, int]]:
    tv_mask = tss_data == tv_label
    if not tv_mask.any():
        return None
    z_coords = np.where(tv_mask)[2]
    return int(z_coords.min()), int(z_coords.max())


def isolate_tp_at_tv(spineps_data, tp_label, z_min, z_max):
    tp_full = spineps_data == tp_label
    z_max_safe = min(z_max, spineps_data.shape[2] - 1)
    z_min_safe = max(z_min, 0)
    isolated = np.zeros_like(tp_full)
    isolated[:, :, z_min_safe:z_max_safe + 1] = tp_full[:, :, z_min_safe:z_max_safe + 1]
    return isolated


def measure_si_height_mm(mask, vox_mm):
    if not mask.any():
        return 0.0
    z_coords = np.where(mask)[2]
    return float((z_coords.max() - z_coords.min()) * vox_mm[2])


def min_distance_mm(mask_a, mask_b, vox_mm):
    if not mask_a.any() or not mask_b.any():
        return float('inf')
    dist_from_b = distance_transform_edt(~mask_b, sampling=vox_mm)
    return float(dist_from_b[mask_a].min())


def build_contact_zone(mask_a, mask_b, vox_mm, radius_mm):
    radius_vox = np.maximum(np.round(radius_mm / vox_mm).astype(int), 1)
    struct = np.ones(2 * radius_vox + 1, dtype=bool)
    return binary_dilation(mask_a, structure=struct) & binary_dilation(mask_b, structure=struct)


# ============================================================================
# UNCERTAINTY FEATURE EXTRACTION
# ============================================================================

def extract_uncertainty_features(unc_data, zone_mask, high_thresh=0.30):
    """
    Extract a feature vector from uncertainty map within zone_mask.

    unc_mean      : mean entropy — overall model confidence in the zone
    unc_std       : std dev — heterogeneity; pseudo-arthrosis is patchy,
                    bony fusion is uniform
    unc_high_frac : fraction of voxels with unc > high_thresh — extent
                    of model confusion; higher = more tissue ambiguity
    n_voxels      : zone size
    valid         : False if zone empty or all-NaN
    """
    vals = unc_data[zone_mask]
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return {'unc_mean': float('nan'), 'unc_std': float('nan'),
                'unc_high_frac': float('nan'), 'n_voxels': 0, 'valid': False}
    return {
        'unc_mean':      float(np.mean(vals)),
        'unc_std':       float(np.std(vals)),
        'unc_high_frac': float(np.mean(vals > high_thresh)),
        'n_voxels':      int(vals.size),
        'valid':         True,
    }


def merge_uncertainty_features(features_tss, features_spineps):
    """TSS is primary; SPINEPS fills in as fallback."""
    if features_tss.get('valid'):
        merged = dict(features_tss)
        merged['source'] = 'tss'
    elif features_spineps.get('valid'):
        merged = dict(features_spineps)
        merged['source'] = 'spineps'
    else:
        merged = {'unc_mean': float('nan'), 'unc_std': float('nan'),
                  'unc_high_frac': float('nan'), 'n_voxels': 0,
                  'valid': False, 'source': 'none'}
    return merged


# ============================================================================
# PROBABILISTIC TYPE II vs III
# ============================================================================

def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def compute_p_type_iii(unc_mean, unc_std, unc_high_frac, tp_height_mm):
    """
    P(Type III bony fusion | contact zone features) via logistic scoring.

    Returns probability in [0,1]:
      -> 1.0: very likely solid bony fusion (Type III)
      -> 0.0: very likely pseudo-arthrosis (Type II)

    NaN features are replaced with maximally uncertain values (conservative,
    biases toward Type II which is the safer default for missed fusion).
    """
    if math.isnan(unc_mean):      unc_mean = 0.5
    if math.isnan(unc_std):       unc_std = 0.3
    if math.isnan(unc_high_frac): unc_high_frac = 0.5

    score = (
        LOGISTIC_INTERCEPT
        + LOGISTIC_W_UNC_MEAN  * unc_mean
        + LOGISTIC_W_UNC_STD   * unc_std
        + LOGISTIC_W_UNC_HIGH  * unc_high_frac
        + LOGISTIC_W_TP_HEIGHT * (tp_height_mm / 30.0)
    )
    return _sigmoid(score)


def probability_to_confidence(p_type_iii):
    margin = abs(p_type_iii - 0.5)
    if margin > CONF_HIGH_MARGIN:     return 'high'
    if margin > CONF_MODERATE_MARGIN: return 'moderate'
    return 'low'


def classify_contact_zone(unc_features, tp_height_mm):
    """
    Given uncertainty features for a confirmed TP-sacrum contact case,
    return classification, probabilities, and confidence.
    """
    if not unc_features.get('valid'):
        return {
            'classification': 'Type II',
            'p_type_ii':      0.65,
            'p_type_iii':     0.35,
            'confidence':     'low',
            'unc_features':   unc_features,
            'note':           'No uncertainty data; conservatively defaulted to Type II',
        }

    p3 = compute_p_type_iii(
        unc_mean      = unc_features['unc_mean'],
        unc_std       = unc_features['unc_std'],
        unc_high_frac = unc_features['unc_high_frac'],
        tp_height_mm  = tp_height_mm,
    )
    p2 = 1.0 - p3

    return {
        'classification': 'Type III' if p3 >= 0.5 else 'Type II',
        'p_type_ii':      round(p2, 4),
        'p_type_iii':     round(p3, 4),
        'confidence':     probability_to_confidence(p3),
        'unc_features':   {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in unc_features.items()},
    }


# ============================================================================
# PER-SIDE CLASSIFICATION
# ============================================================================

def classify_side(tp_isolated, sacrum_mask, spineps_unc, tss_unc,
                  vox_mm_spineps, vox_mm_tss, debug):
    result = {
        'tp_present':    bool(tp_isolated.any()),
        'tp_height_mm':  0.0,
        'contact':       False,
        'dist_mm':       float('inf'),
        'p_type_ii':     None,
        'p_type_iii':    None,
        'confidence':    None,
        'unc_features':  None,
        'classification':'Normal',
    }

    if not tp_isolated.any():
        return result

    height_mm = measure_si_height_mm(tp_isolated, vox_mm_spineps)
    result['tp_height_mm'] = round(height_mm, 2)

    if not sacrum_mask.any():
        result['dist_mm'] = float('inf')
        result['classification'] = 'Type I' if height_mm > TP_HEIGHT_MM else 'Normal'
        return result

    dist_mm = min_distance_mm(tp_isolated, sacrum_mask, vox_mm_tss)
    result['dist_mm'] = round(dist_mm, 2)
    result['contact'] = dist_mm <= CONTACT_DIST_MM

    if not result['contact']:
        result['classification'] = 'Type I' if height_mm > TP_HEIGHT_MM else 'Normal'
        return result

    # --- Contact case: probabilistic Type II vs III ---
    contact_zone = build_contact_zone(
        tp_isolated, sacrum_mask, vox_mm_tss, CONTACT_DILATION_MM
    )

    features_tss     = extract_uncertainty_features(tss_unc, contact_zone)
    features_spineps = extract_uncertainty_features(spineps_unc, contact_zone)
    merged           = merge_uncertainty_features(features_tss, features_spineps)

    if debug:
        merged['tss_features']     = features_tss
        merged['spineps_features'] = features_spineps

    cr = classify_contact_zone(merged, height_mm)

    result['classification'] = cr['classification']
    result['p_type_ii']      = cr['p_type_ii']
    result['p_type_iii']     = cr['p_type_iii']
    result['confidence']     = cr['confidence']
    result['unc_features']   = cr['unc_features']
    if 'note' in cr:
        result['note'] = cr['note']

    return result


# ============================================================================
# CASTELLVI ASSEMBLY
# ============================================================================

CASTELLVI_TABLE = {
    ('Normal',  'Normal'):   None,
    ('Type I',  'Normal'):   'Type I',
    ('Normal',  'Type I'):   'Type I',
    ('Type I',  'Type I'):   'Type I',
    ('Type II', 'Normal'):   'Type II',
    ('Normal',  'Type II'):  'Type II',
    ('Type II', 'Type II'):  'Type II',
    ('Type III','Normal'):   'Type III',
    ('Normal',  'Type III'): 'Type III',
    ('Type III','Type III'): 'Type III',
    ('Type II', 'Type III'): 'Type IV',
    ('Type III','Type II'):  'Type IV',
    ('Type I',  'Type II'):  'Type II',
    ('Type II', 'Type I'):   'Type II',
    ('Type I',  'Type III'): 'Type III',
    ('Type III','Type I'):   'Type III',
}


def assemble_castellvi(left_cls, right_cls):
    return CASTELLVI_TABLE.get((left_cls, right_cls), None)


def assemble_overall_confidence(left, right):
    rank = {'high': 2, 'moderate': 1, 'low': 0}
    scores = [rank[s['confidence']] for s in (left, right)
              if s.get('contact') and s.get('confidence')]
    if not scores:
        return 'high'
    return {2: 'high', 1: 'moderate', 0: 'low'}[min(scores)]


# ============================================================================
# DETECTOR
# ============================================================================

class LSTVDetector:
    def __init__(self, spineps_dir, totalspine_dir, debug=False):
        self.spineps_dir    = Path(spineps_dir)
        self.totalspine_dir = Path(totalspine_dir)
        self.debug          = debug

    def _spineps_mask(self, sid):
        return self.spineps_dir / 'segmentations' / sid / f"{sid}_seg-spine_msk.nii.gz"

    def _spineps_unc(self, sid):
        return self.spineps_dir / 'segmentations' / sid / f"{sid}_unc.nii.gz"

    def _tss_mask(self, sid):
        return self.totalspine_dir / sid / 'sagittal' / f"{sid}_sagittal_labeled.nii.gz"

    def _tss_unc(self, sid):
        return self.totalspine_dir / sid / 'sagittal' / f"{sid}_sagittal_unc.nii.gz"

    def _load_unc(self, path, ref_shape):
        """Load uncertainty map, returning NaN array if missing or unreadable."""
        if path.exists():
            try:
                data, _ = load_canonical(path)
                return data.astype(np.float32)
            except Exception as e:
                logger.warning(f"  Could not load uncertainty map {path}: {e}")
        else:
            logger.warning(f"  Uncertainty map not found: {path}")
        return np.full(ref_shape, float('nan'), dtype=np.float32)

    def classify_study(self, study_id):
        result = {
            'study_id':       study_id,
            'lstv_detected':  False,
            'castellvi_type': None,
            'confidence':     None,
            'left':           {},
            'right':          {},
            'details':        {},
            'errors':         [],
        }

        # TSS mask (required)
        tss_mask_path = self._tss_mask(study_id)
        if not tss_mask_path.exists():
            result['errors'].append(f"TSS mask not found: {tss_mask_path}")
            return result
        try:
            tss_data, tss_nii = load_canonical(tss_mask_path)
            tss_data = tss_data.astype(int)
        except Exception as e:
            result['errors'].append(f"Failed to load TSS mask: {e}")
            return result
        vox_tss = voxel_size_mm(tss_nii)

        # SPINEPS mask (required)
        spineps_mask_path = self._spineps_mask(study_id)
        if not spineps_mask_path.exists():
            result['errors'].append(f"SPINEPS mask not found: {spineps_mask_path}")
            return result
        try:
            spineps_data, spineps_nii = load_canonical(spineps_mask_path)
            spineps_data = spineps_data.astype(int)
        except Exception as e:
            result['errors'].append(f"Failed to load SPINEPS mask: {e}")
            return result
        vox_spineps = voxel_size_mm(spineps_nii)

        # Uncertainty maps (optional — NaN array if missing)
        tss_unc_data     = self._load_unc(self._tss_unc(study_id),     tss_data.shape)
        spineps_unc_data = self._load_unc(self._spineps_unc(study_id), spineps_data.shape)

        # Target vertebra
        unique_labels = set(np.unique(tss_data).tolist())
        tv_label = L6_LABEL if L6_LABEL in unique_labels else L5_LABEL
        result['details'].update({
            'tv_label':       tv_label,
            'tv_name':        LUMBAR_LABELS.get(tv_label, '?'),
            'has_l6':         L6_LABEL in unique_labels,
            'sacrum_present': SACRUM_LABEL in unique_labels,
        })

        tv_z_range = get_tv_z_range(tss_data, tv_label)
        if tv_z_range is None:
            result['errors'].append(f"TV label {tv_label} not found in TSS mask")
            return result
        z_min, z_max = tv_z_range
        result['details']['tv_z_range'] = [z_min, z_max]

        sacrum_mask = (tss_data == SACRUM_LABEL)

        # Per-side
        for side, tp_label in [('left', TP_LEFT_LABEL), ('right', TP_RIGHT_LABEL)]:
            tp_isolated = isolate_tp_at_tv(spineps_data, tp_label, z_min, z_max)
            side_result = classify_side(
                tp_isolated    = tp_isolated,
                sacrum_mask    = sacrum_mask,
                spineps_unc    = spineps_unc_data,
                tss_unc        = tss_unc_data,
                vox_mm_spineps = vox_spineps,
                vox_mm_tss     = vox_tss,
                debug          = self.debug,
            )
            result[side] = side_result

            prob_str = ''
            if side_result.get('p_type_iii') is not None:
                prob_str = (f"  P(II)={side_result['p_type_ii']:.2f}"
                            f"  P(III)={side_result['p_type_iii']:.2f}"
                            f"  [{side_result['confidence']}]")
            logger.info(
                f"  [{study_id}] {side:5s}: {side_result['classification']:8s}"
                f"  h={side_result['tp_height_mm']:.1f}mm"
                f"  d={side_result['dist_mm']:.1f}mm"
                + prob_str
            )

        # Final assembly
        left_cls  = result['left'].get('classification', 'Normal')
        right_cls = result['right'].get('classification', 'Normal')
        final     = assemble_castellvi(left_cls, right_cls)

        result['castellvi_type'] = final
        result['lstv_detected']  = final is not None
        result['confidence']     = assemble_overall_confidence(result['left'], result['right'])

        return result

    def detect_all_studies(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        spineps_seg_dir = self.spineps_dir / 'segmentations'
        if not spineps_seg_dir.exists():
            logger.error(f"SPINEPS segmentation dir not found: {spineps_seg_dir}")
            return []

        study_dirs  = sorted([d for d in spineps_seg_dir.iterdir() if d.is_dir()])
        results     = []
        lstv_count  = 0
        error_count = 0
        low_conf    = 0

        logger.info(f"Found {len(study_dirs)} studies to process")

        for study_dir in study_dirs:
            study_id = study_dir.name
            logger.info(f"\n[{study_id}]")
            try:
                r = self.classify_study(study_id)
                results.append(r)
                if r['errors']:
                    error_count += 1
                    for err in r['errors']:
                        logger.warning(f"  ⚠  {err}")
                if r['lstv_detected']:
                    lstv_count += 1
                    logger.info(f"  ✓ LSTV: {r['castellvi_type']}  (confidence: {r['confidence']})")
                else:
                    logger.info(f"  ✗ No LSTV detected")
                if r.get('confidence') == 'low':
                    low_conf += 1
            except Exception as e:
                logger.error(f"  Unhandled error for {study_id}: {e}")
                logger.debug(traceback.format_exc())
                results.append({'study_id': study_id, 'lstv_detected': False,
                                 'castellvi_type': None, 'confidence': None, 'errors': [str(e)]})
                error_count += 1

        results_file = output_dir / 'lstv_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        breakdown = {t: 0 for t in ('Type I', 'Type II', 'Type III', 'Type IV')}
        for r in results:
            if r.get('castellvi_type') in breakdown:
                breakdown[r['castellvi_type']] += 1

        summary = {
            'total_studies':        len(results),
            'lstv_detected':        lstv_count,
            'lstv_rate':            lstv_count / len(results) if results else 0.0,
            'error_count':          error_count,
            'low_confidence_cases': low_conf,
            'castellvi_breakdown':  breakdown,
            'note': (
                'Type II/III probabilities use a logistic scoring function with '
                'clinically-grounded priors. Replace with CalibratedClassifierCV '
                'once ground truth Castellvi labels are available.'
            ),
        }

        summary_file = output_dir / 'lstv_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info("\n" + "=" * 70)
        logger.info("LSTV DETECTION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total studies:         {summary['total_studies']}")
        logger.info(f"LSTV detected:         {summary['lstv_detected']} ({summary['lstv_rate']:.1%})")
        logger.info(f"Errors / incomplete:   {summary['error_count']}")
        logger.info(f"Low confidence cases:  {summary['low_confidence_cases']}")
        logger.info("")
        logger.info("Castellvi Breakdown:")
        for t, c in summary['castellvi_breakdown'].items():
            logger.info(f"  {t}: {c}")
        logger.info(f"\nResults -> {results_file}")
        logger.info(f"Summary -> {summary_file}")

        return results


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Morphological LSTV Detection and Castellvi Classification'
    )
    parser.add_argument('--spineps_dir',    required=True)
    parser.add_argument('--totalspine_dir', required=True)
    parser.add_argument('--output_dir',     required=True)
    parser.add_argument('--debug', action='store_true',
                        help='Include full uncertainty feature vectors and per-model '
                             'breakdown in JSON output for threshold analysis.')
    args = parser.parse_args()

    LSTVDetector(
        spineps_dir    = args.spineps_dir,
        totalspine_dir = args.totalspine_dir,
        debug          = args.debug,
    ).detect_all_studies(args.output_dir)


if __name__ == '__main__':
    main()
