#!/usr/bin/env python3
"""
LSTV Overlay Visualizer

Visualizes SPINEPS and TotalSpineSeg outputs for one study (or all studies
in batch mode), including both uncertainty maps and the contact zone used
by 04_detect_lstv.py to decide between Castellvi Type II and III.

Output: one 3x3 panel PNG per study saved to --output_dir.

  Row 0 | Original image      | SPINEPS instance mask  | TotalSpineSeg labeled mask
  Row 1 | SPINEPS uncertainty | TSS uncertainty         | Contact zone
  Row 2 | Isolated TPs at TV  | SPINEPS semantic (all)  | Classification summary

All NIfTIs are reoriented to canonical RAS (nib.as_closest_canonical) before
any slicing so axis 0 = L-R, axis 2 = I-S -- identical to 04_detect_lstv.py.

Usage -- single study:
    python visualize_overlay.py \
        --study_id       1020394063 \
        --spineps_dir    results/spineps \
        --totalspine_dir results/totalspineseg \
        --output_dir     results/lstv_viz \
        [--lstv_json     results/lstv_detection/lstv_results.json] \
        [--slice         N]

Usage -- batch (all studies found under spineps_dir/segmentations/):
    python visualize_overlay.py \
        --spineps_dir    results/spineps \
        --totalspine_dir results/totalspineseg \
        --output_dir     results/lstv_viz \
        [--lstv_json     results/lstv_detection/lstv_results.json]
"""

import argparse
import json
import logging
import traceback
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS -- kept in sync with 04_detect_lstv.py
# ============================================================================

TP_HEIGHT_MM        = 19.0
CONTACT_DIST_MM     = 2.0
CONTACT_DILATION_MM = 3.0
TP_LEFT_LABEL       = 43
TP_RIGHT_LABEL      = 44
SACRUM_LABEL        = 50
L5_LABEL            = 45
L6_LABEL            = 46

# TotalSpineSeg display colours: label -> ([R,G,B], display name)
TSS_LABEL_COLORS = {
    41:  ([0.20, 0.40, 0.80], 'L1'),
    42:  ([0.25, 0.50, 0.85], 'L2'),
    43:  ([0.30, 0.60, 0.90], 'L3'),
    44:  ([0.35, 0.70, 0.95], 'L4'),
    45:  ([0.40, 0.80, 1.00], 'L5'),
    46:  ([0.00, 0.30, 0.70], 'L6'),
    50:  ([1.00, 0.55, 0.00], 'Sacrum'),
    95:  ([0.90, 0.20, 0.90], 'L4-L5'),
    100: ([1.00, 0.00, 0.60], 'L5-S'),
}


# ============================================================================
# NIfTI HELPERS -- mirror 04_detect_lstv.py exactly so geometry is identical
# ============================================================================

def load_canonical(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """Load NIfTI and reorient to RAS canonical. axis2 = I-S."""
    nii = nib.load(str(path))
    nii = nib.as_closest_canonical(nii)
    return nii.get_fdata(), nii


def voxel_size_mm(nii: nib.Nifti1Image) -> np.ndarray:
    return np.abs(np.array(nii.header.get_zooms()[:3], dtype=float))


def get_tv_z_range(tss_data: np.ndarray, tv_label: int) -> Optional[Tuple[int, int]]:
    tv_mask = tss_data == tv_label
    if not tv_mask.any():
        return None
    z = np.where(tv_mask)[2]
    return int(z.min()), int(z.max())


def isolate_tp_at_tv(spineps_data: np.ndarray, tp_label: int,
                     z_min: int, z_max: int) -> np.ndarray:
    tp_full = spineps_data == tp_label
    iso = np.zeros_like(tp_full)
    z_lo = max(z_min, 0)
    z_hi = min(z_max, spineps_data.shape[2] - 1)
    iso[:, :, z_lo:z_hi + 1] = tp_full[:, :, z_lo:z_hi + 1]
    return iso


def build_contact_zone(mask_a: np.ndarray, mask_b: np.ndarray,
                       vox_mm: np.ndarray, radius_mm: float) -> np.ndarray:
    radius_vox = np.maximum(np.round(radius_mm / vox_mm).astype(int), 1)
    struct = np.ones(2 * radius_vox + 1, dtype=bool)
    return binary_dilation(mask_a, structure=struct) & \
           binary_dilation(mask_b, structure=struct)


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

def norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def overlay_mask(ax, mask2d: np.ndarray, color_rgb, alpha: float = 0.65):
    """Paint a boolean 2-D mask as a solid-colour RGBA layer."""
    if not mask2d.any():
        return
    rgba = np.zeros((*mask2d.shape, 4), dtype=float)
    rgba[mask2d] = [*color_rgb, alpha]
    ax.imshow(rgba.T, origin='lower')


def _pick_best_sag_slice(vol3d: np.ndarray,
                         highlight: Optional[np.ndarray] = None,
                         override: Optional[int] = None) -> int:
    if override is not None:
        return min(override, vol3d.shape[0] - 1)
    if highlight is not None and highlight.any():
        return int(np.argmax(highlight.sum(axis=(1, 2))))
    return vol3d.shape[0] // 2


def _sag(vol: Optional[np.ndarray], sl: int) -> np.ndarray:
    if vol is None:
        return np.zeros((1, 1))
    return vol[min(sl, vol.shape[0] - 1), :, :]


# ============================================================================
# PANEL FUNCTIONS
# ============================================================================

def _panel_original(ax, img_sl, sl_idx):
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower')
    ax.set_title(f'Original Image  (sag slice {sl_idx})', fontsize=11)
    ax.axis('off')


def _panel_spineps_instance(ax, img_sl, inst_sl):
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower', alpha=0.7)
    overlay_mask(ax, (inst_sl >= 19)  & (inst_sl <= 28),  [0.20, 0.40, 1.00], 0.50)
    overlay_mask(ax, (inst_sl >= 119) & (inst_sl <= 126), [0.00, 0.85, 0.00], 0.70)
    overlay_mask(ax, inst_sl >= 200,                       [1.00, 1.00, 0.00], 0.50)
    ax.legend(handles=[
        mpatches.Patch(color=[0.20, 0.40, 1.00], label='Vertebrae'),
        mpatches.Patch(color=[0.00, 0.85, 0.00], label='Discs'),
        mpatches.Patch(color=[1.00, 1.00, 0.00], label='Endplates'),
    ], loc='lower right', fontsize=7, framealpha=0.6)
    ax.set_title('SPINEPS Instance Mask', fontsize=11)
    ax.axis('off')


def _panel_tss_labeled(ax, img_sl, tss_sl):
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower', alpha=0.65)
    patches = []
    for label, (color, name) in TSS_LABEL_COLORS.items():
        m = tss_sl == label
        if m.any():
            overlay_mask(ax, m, color, 0.65)
            patches.append(mpatches.Patch(color=color, label=name))
    if patches:
        ax.legend(handles=patches, loc='lower right', fontsize=7, framealpha=0.6)
    ax.set_title('TotalSpineSeg Labeled Mask', fontsize=11)
    ax.axis('off')


def _panel_uncertainty(ax, unc_sl, title):
    im = ax.imshow(norm(unc_sl).T, cmap='hot', origin='lower', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Entropy [0-1]')
    ax.set_title(title, fontsize=11)
    ax.axis('off')


def _panel_contact_zone(ax, img_sl, cz_left, cz_right,
                        tp_left_sl, tp_right_sl, sacrum_sl):
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower', alpha=0.65)
    overlay_mask(ax, sacrum_sl,   [1.00, 0.55, 0.00], 0.25)
    overlay_mask(ax, tp_left_sl,  [1.00, 0.10, 0.10], 0.20)
    overlay_mask(ax, tp_right_sl, [0.00, 0.80, 1.00], 0.20)
    overlay_mask(ax, cz_left,  [1.00, 0.00, 1.00], 0.85)
    overlay_mask(ax, cz_right, [0.00, 1.00, 0.50], 0.85)
    ax.legend(handles=[
        mpatches.Patch(color=[1.00, 0.00, 1.00], label='Left contact zone'),
        mpatches.Patch(color=[0.00, 1.00, 0.50], label='Right contact zone'),
    ], loc='lower right', fontsize=7, framealpha=0.6)
    ax.set_title('Contact Zones\n(Type II/III uncertainty decision)', fontsize=11)
    ax.axis('off')


def _panel_tp_isolated(ax, img_sl, tp_left_sl, tp_right_sl, sacrum_sl, tv_name, sl_idx):
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower', alpha=0.7)
    overlay_mask(ax, tp_left_sl,  [1.00, 0.10, 0.10], 0.80)
    overlay_mask(ax, tp_right_sl, [0.00, 0.80, 1.00], 0.80)
    overlay_mask(ax, sacrum_sl,   [1.00, 0.55, 0.00], 0.45)
    ax.legend(handles=[
        mpatches.Patch(color=[1.00, 0.10, 0.10], label='Left TP (TV)'),
        mpatches.Patch(color=[0.00, 0.80, 1.00], label='Right TP (TV)'),
        mpatches.Patch(color=[1.00, 0.55, 0.00], label='Sacrum'),
    ], loc='lower right', fontsize=7, framealpha=0.6)
    ax.set_title(f'TV-level TPs Isolated\n(TV = {tv_name},  slice {sl_idx})', fontsize=11)
    ax.axis('off')


def _panel_spineps_semantic(ax, img_sl, sem_sl):
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower', alpha=0.7)
    costal = (sem_sl == TP_LEFT_LABEL) | (sem_sl == TP_RIGHT_LABEL)
    overlay_mask(ax, sem_sl == TP_LEFT_LABEL,  [1.00, 0.10, 0.10], 0.80)
    overlay_mask(ax, sem_sl == TP_RIGHT_LABEL, [0.00, 0.80, 1.00], 0.80)
    overlay_mask(ax, (sem_sl > 0) & ~costal,   [0.00, 1.00, 1.00], 0.22)
    ax.legend(handles=[
        mpatches.Patch(color=[1.00, 0.10, 0.10], label='Left TP (label 43)'),
        mpatches.Patch(color=[0.00, 0.80, 1.00], label='Right TP (label 44)'),
    ], loc='lower right', fontsize=7, framealpha=0.6)
    ax.set_title('SPINEPS Semantic Mask\n(all TPs, pre-isolation)', fontsize=11)
    ax.axis('off')


def _panel_summary(ax, study_id: str, result: Optional[dict], tv_name: str):
    """
    Classification summary panel -- updated for 04_detect_lstv.py v2 output
    which uses probabilistic uncertainty scoring instead of a single threshold.

    New fields consumed:
      result[side]['p_type_ii']    : float
      result[side]['p_type_iii']   : float
      result[side]['confidence']   : 'high' | 'moderate' | 'low'
      result[side]['unc_features'] : dict with unc_mean, unc_std, unc_high_frac, source
      result['confidence']         : overall confidence
    """
    ax.axis('off')
    ax.set_facecolor('#1a1a2e')

    lines = [f'Study : {study_id}', f'TV    : {tv_name}', '']

    if result is None:
        lines += ['No detection results available.', '',
                  'Run 04_detect_lstv.py and pass', '--lstv_json to annotate this panel.']
    else:
        ct           = result.get('castellvi_type') or 'None (no LSTV)'
        detected     = result.get('lstv_detected', False)
        overall_conf = result.get('confidence', '')
        lines += [
            f'Castellvi : {ct}',
            f'LSTV      : {"YES  !" if detected else "No"}',
            f'Confidence: {overall_conf}',
            '',
        ]

        for side in ('left', 'right'):
            sd = result.get(side, {})
            if not sd:
                continue

            lines += [
                f'{"Left" if side == "left" else "Right"} side:',
                f'  Class     : {sd.get("classification", "?")}',
                f'  TP height : {sd.get("tp_height_mm", 0):.1f} mm  (thresh {TP_HEIGHT_MM} mm)',
                f'  TP-Sacrum : {sd.get("dist_mm", float("inf")):.1f} mm  (thresh {CONTACT_DIST_MM} mm)',
            ]

            # Probabilistic output (only present for contact cases)
            p2 = sd.get('p_type_ii')
            p3 = sd.get('p_type_iii')
            if p2 is not None and p3 is not None:
                lines.append(
                    f'  P(II)={p2:.2f}  P(III)={p3:.2f}  [{sd.get("confidence", "")}]'
                )

            # Uncertainty feature vector
            unc = sd.get('unc_features') or {}
            if unc and unc.get('unc_mean') is not None:
                try:
                    lines.append(
                        f'  unc mean={unc["unc_mean"]:.3f}'
                        f'  std={unc.get("unc_std", float("nan")):.3f}'
                        f'  hi_frac={unc.get("unc_high_frac", float("nan")):.3f}'
                        f'  [{unc.get("source", "")}]'
                    )
                except (TypeError, ValueError):
                    pass  # NaN formatting edge case

            if sd.get('note'):
                lines.append(f'  NOTE: {sd["note"]}')

            lines.append('')

        if result.get('errors'):
            lines += ['Errors:'] + [f'  {e}' for e in result['errors']]

    ax.text(0.05, 0.97, '\n'.join(lines),
            transform=ax.transAxes, va='top', ha='left',
            fontsize=8.5, family='monospace', color='white')
    ax.set_title('Classification Summary', fontsize=11)


def _unavailable(ax, label):
    ax.text(0.5, 0.5, f'{label}\nnot available',
            ha='center', va='center', color='grey', fontsize=10)
    ax.axis('off')


# ============================================================================
# CORE VISUALIZER
# ============================================================================

def visualize_study(
    study_id: str,
    spineps_dir: Path,
    totalspine_dir: Path,
    output_dir: Path,
    slice_override: Optional[int] = None,
    result: Optional[dict] = None,
):
    """Build and save the 3x3 panel PNG for one study."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{study_id}_lstv_overlay.png"

    # ---- Resolve file paths -------------------------------------------------
    sp_sem_path   = spineps_dir / 'segmentations' / study_id / f"{study_id}_seg-spine_msk.nii.gz"
    sp_inst_path  = spineps_dir / 'segmentations' / study_id / f"{study_id}_seg-vert_msk.nii.gz"
    sp_unc_path   = spineps_dir / 'segmentations' / study_id / f"{study_id}_unc.nii.gz"
    tss_mask_path = totalspine_dir / study_id / 'sagittal' / f"{study_id}_sagittal_labeled.nii.gz"
    tss_unc_path  = totalspine_dir / study_id / 'sagittal' / f"{study_id}_sagittal_unc.nii.gz"

    # ---- Load everything in canonical orientation ----------------------------
    def try_load(path, label):
        if path.exists():
            try:
                return load_canonical(path)
            except Exception as e:
                logger.warning(f"  [{study_id}] Cannot load {label}: {e}")
        else:
            logger.warning(f"  [{study_id}] Missing: {path.name}")
        return None, None

    sp_sem_data,  sp_sem_nii = try_load(sp_sem_path,   'SPINEPS semantic')
    sp_inst_data, _          = try_load(sp_inst_path,  'SPINEPS instance')
    sp_unc_data,  _          = try_load(sp_unc_path,   'SPINEPS uncertainty')
    tss_data,     tss_nii    = try_load(tss_mask_path, 'TSS labeled')
    tss_unc_data, _          = try_load(tss_unc_path,  'TSS uncertainty')

    if sp_sem_data is None and tss_data is None:
        logger.error(f"  [{study_id}] No usable data -- skipping")
        return

    bg_data = sp_sem_data if sp_sem_data is not None else tss_data
    bg_nii  = sp_sem_nii  if sp_sem_nii  is not None else tss_nii

    if sp_sem_data  is not None: sp_sem_data  = sp_sem_data.astype(int)
    if sp_inst_data is not None: sp_inst_data = sp_inst_data.astype(int)
    if tss_data     is not None: tss_data     = tss_data.astype(int)
    if sp_unc_data  is not None: sp_unc_data  = sp_unc_data.astype(np.float32)
    if tss_unc_data is not None: tss_unc_data = tss_unc_data.astype(np.float32)

    vox_bg  = voxel_size_mm(bg_nii)
    vox_tss = voxel_size_mm(tss_nii) if tss_nii is not None else vox_bg

    # ---- Target Vertebra -----------------------------------------------------
    tv_label = (L6_LABEL
                if tss_data is not None and L6_LABEL in np.unique(tss_data)
                else L5_LABEL)
    tv_name  = 'L6' if tv_label == L6_LABEL else 'L5'

    # ---- Build 3-D masks -----------------------------------------------------
    z_range = get_tv_z_range(tss_data, tv_label) if tss_data is not None else None
    zeros3  = np.zeros(bg_data.shape, dtype=bool)

    if z_range is not None and sp_sem_data is not None:
        z_min, z_max = z_range
        tp_left_3d  = isolate_tp_at_tv(sp_sem_data, TP_LEFT_LABEL,  z_min, z_max)
        tp_right_3d = isolate_tp_at_tv(sp_sem_data, TP_RIGHT_LABEL, z_min, z_max)
    else:
        tp_left_3d = tp_right_3d = zeros3

    sacrum_3d = (tss_data == SACRUM_LABEL) if tss_data is not None else zeros3

    def _cz(tp_mask):
        if tp_mask.any() and sacrum_3d.any():
            common = tuple(min(a, b) for a, b in zip(tp_mask.shape, sacrum_3d.shape))
            tp_c   = tp_mask[:common[0],   :common[1], :common[2]]
            sac_c  = sacrum_3d[:common[0], :common[1], :common[2]]
            cz     = build_contact_zone(tp_c, sac_c, vox_tss, CONTACT_DILATION_MM)
            out    = np.zeros(bg_data.shape, dtype=bool)
            out[:cz.shape[0], :cz.shape[1], :cz.shape[2]] = cz
            return out
        return zeros3

    cz_left_3d  = _cz(tp_left_3d)
    cz_right_3d = _cz(tp_right_3d)
    cz_combined = cz_left_3d | cz_right_3d

    # ---- Choose display slices -----------------------------------------------
    sl_main = _pick_best_sag_slice(bg_data, override=slice_override)
    sl_cz   = _pick_best_sag_slice(bg_data, highlight=cz_combined,
                                   override=slice_override)

    def s(vol, sl=None):
        return _sag(vol, sl if sl is not None else sl_main)

    # ---- Assemble figure -----------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(21, 21))
    fig.patch.set_facecolor('#0d0d1a')
    for ax in axes.flat:
        ax.set_facecolor('#0d0d1a')

    castellvi = result.get('castellvi_type') if result else None
    conf_str  = f"  confidence: {result.get('confidence','')}" if result else ''
    fig.suptitle(
        f"Study {study_id}   |   Castellvi: {castellvi or 'N/A'}{conf_str}"
        f"   |   TV: {tv_name}",
        fontsize=15, color='white', y=0.998,
    )

    # Row 0: anatomy
    _panel_original(axes[0, 0], s(bg_data), sl_main)

    if sp_inst_data is not None:
        _panel_spineps_instance(axes[0, 1], s(bg_data), s(sp_inst_data))
    else:
        _unavailable(axes[0, 1], 'SPINEPS instance mask')

    if tss_data is not None:
        _panel_tss_labeled(axes[0, 2], s(bg_data), s(tss_data))
    else:
        _unavailable(axes[0, 2], 'TotalSpineSeg labeled mask')

    # Row 1: uncertainty + contact zone
    if sp_unc_data is not None:
        _panel_uncertainty(axes[1, 0], s(sp_unc_data),
                           'SPINEPS Uncertainty\n(Normalised Shannon Entropy)')
    else:
        _unavailable(axes[1, 0], 'SPINEPS uncertainty')

    if tss_unc_data is not None:
        _panel_uncertainty(axes[1, 1], s(tss_unc_data),
                           'TotalSpineSeg Uncertainty\n(Normalised Shannon Entropy)')
    else:
        _unavailable(axes[1, 1], 'TSS uncertainty')

    _panel_contact_zone(
        axes[1, 2],
        s(bg_data, sl_cz),
        cz_left_3d[min(sl_cz,  cz_left_3d.shape[0]-1),  :, :],
        cz_right_3d[min(sl_cz, cz_right_3d.shape[0]-1), :, :],
        tp_left_3d[min(sl_cz,  tp_left_3d.shape[0]-1),  :, :],
        tp_right_3d[min(sl_cz, tp_right_3d.shape[0]-1), :, :],
        sacrum_3d[min(sl_cz,   sacrum_3d.shape[0]-1),   :, :],
    )

    # Row 2: TPs + semantic + summary
    _panel_tp_isolated(
        axes[2, 0], s(bg_data),
        tp_left_3d[min(sl_main,  tp_left_3d.shape[0]-1),  :, :],
        tp_right_3d[min(sl_main, tp_right_3d.shape[0]-1), :, :],
        sacrum_3d[min(sl_main,   sacrum_3d.shape[0]-1),   :, :],
        tv_name, sl_main,
    )

    if sp_sem_data is not None:
        _panel_spineps_semantic(axes[2, 1], s(bg_data), s(sp_sem_data))
    else:
        _unavailable(axes[2, 1], 'SPINEPS semantic mask')

    _panel_summary(axes[2, 2], study_id, result, tv_name)

    plt.tight_layout(rect=[0, 0, 1, 0.997])
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"  [{study_id}] OK  {out_path}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='LSTV Overlay Visualizer')
    parser.add_argument('--spineps_dir',    required=True)
    parser.add_argument('--totalspine_dir', required=True)
    parser.add_argument('--output_dir',     required=True)
    parser.add_argument('--study_id',       default=None,
                        help='Single study ID (omit for batch mode)')
    parser.add_argument('--lstv_json',      default=None,
                        help='lstv_results.json from 04_detect_lstv.py')
    parser.add_argument('--slice',          type=int, default=None,
                        help='Override sagittal slice index (default: mid-volume)')
    args = parser.parse_args()

    spineps_dir    = Path(args.spineps_dir)
    totalspine_dir = Path(args.totalspine_dir)
    output_dir     = Path(args.output_dir)

    results_by_id = {}
    if args.lstv_json:
        json_path = Path(args.lstv_json)
        if json_path.exists():
            with open(json_path) as f:
                results_by_id = {r['study_id']: r for r in json.load(f)}
            logger.info(f"Loaded {len(results_by_id)} detection results")
        else:
            logger.warning(f"lstv_json not found: {json_path}")

    if args.study_id:
        study_ids = [args.study_id]
    else:
        seg_dir = spineps_dir / 'segmentations'
        if not seg_dir.exists():
            logger.error(f"SPINEPS segmentations dir not found: {seg_dir}")
            return 1
        study_ids = sorted(d.name for d in seg_dir.iterdir() if d.is_dir())
        logger.info(f"Batch mode: {len(study_ids)} studies")

    errors = 0
    for sid in study_ids:
        try:
            visualize_study(
                study_id       = sid,
                spineps_dir    = spineps_dir,
                totalspine_dir = totalspine_dir,
                output_dir     = output_dir,
                slice_override = args.slice,
                result         = results_by_id.get(sid),
            )
        except Exception as e:
            logger.error(f"  [{sid}] Failed: {e}")
            logger.debug(traceback.format_exc())
            errors += 1

    logger.info(
        f"Done. {len(study_ids)-errors}/{len(study_ids)} PNGs saved -> {output_dir}"
    )
    return 0 if errors == 0 else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
