#!/usr/bin/env python3
"""
05_visualize_overlay.py — LSTV Overlay Visualizer v4
=====================================================
Panel layout:

  Row 0 | Axial T2w at TV      | SPINEPS TPs — axial (dilated for visibility) | TSS labels — axial
  Row 1 | Sagittal TSS confirm | TSS uncertainty — sagittal (native, pre-reg) | Contact zones — axial
  Row 2 | Coronal TPs          | Sagittal TPs + sacrum                        | Classification summary

Key design decisions in v4
--------------------------
* Row 2 coronal and sagittal panels use the ORIGINAL sagittal T2w background
  with the ORIGINAL (pre-registration) SPINEPS labels. This avoids the squished
  appearance that results from reslicing the axial volume coronally/sagittally
  (axial has only ~30-50 slices in I-S so coronal/sagittal reslices are tiny).

* TP masks in the axial panel (0,1) are dilated by 2 voxels for display only
  so they are actually visible. The dilation is NOT used for any measurements.

* TSS uncertainty is shown in its native sagittal orientation (pre-registration)
  alongside the sagittal T2w. The registered axial uncertainty had severe
  artifacts (diagonal stripe pattern from sagittal entropy resampled to axial).

* TSS axial panel (0,2) keeps the resampled labels at 0.45 alpha so anatomy
  shows through. Uses native axial TSS if available.
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
# CONSTANTS
# ============================================================================

TP_HEIGHT_MM        = 19.0
CONTACT_DIST_MM     = 2.0
CONTACT_DILATION_MM = 3.0
TP_LEFT_LABEL       = 43
TP_RIGHT_LABEL      = 44
SACRUM_LABEL        = 50
L5_LABEL            = 45
L6_LABEL            = 46

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

# Voxel dilation radius for TP display in axial (display only, not measurement)
DISPLAY_DILATION_VOXELS = 2


# ============================================================================
# NIfTI HELPERS
# ============================================================================

def load_canonical(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    nii = nib.load(str(path))
    nii = nib.as_closest_canonical(nii)
    data = nii.get_fdata()
    if data.ndim == 4:
        data = data[..., 0]
    return data, nii


def voxel_size_mm(nii: nib.Nifti1Image) -> np.ndarray:
    return np.abs(np.array(nii.header.get_zooms()[:3], dtype=float))


def get_tv_z_range(tss_data: np.ndarray,
                   tv_label: int) -> Optional[Tuple[int, int]]:
    mask = tss_data == tv_label
    if not mask.any():
        return None
    z = np.where(mask)[2]
    return int(z.min()), int(z.max())


def isolate_tp_at_tv(data: np.ndarray, tp_label: int,
                     z_min: int, z_max: int) -> np.ndarray:
    tp_full = data == tp_label
    iso = np.zeros_like(tp_full)
    z_lo = max(z_min, 0)
    z_hi = min(z_max, data.shape[2] - 1)
    iso[:, :, z_lo:z_hi + 1] = tp_full[:, :, z_lo:z_hi + 1]
    return iso


def build_contact_zone(mask_a: np.ndarray, mask_b: np.ndarray,
                       vox_mm: np.ndarray, radius_mm: float) -> np.ndarray:
    radius_vox = np.maximum(np.round(radius_mm / vox_mm).astype(int), 1)
    struct = np.ones(2 * radius_vox + 1, dtype=bool)
    return (binary_dilation(mask_a, structure=struct) &
            binary_dilation(mask_b, structure=struct))


def dilate_for_display(mask: np.ndarray, voxels: int = 2) -> np.ndarray:
    """Dilate a binary mask for display only — do not use for measurements."""
    if not mask.any() or voxels < 1:
        return mask
    struct = np.ones((voxels * 2 + 1,) * 3, dtype=bool)
    return binary_dilation(mask, structure=struct)


# ============================================================================
# FILE DISCOVERY
# ============================================================================

def find_t2w(nifti_dir: Path, study_id: str, acq: str) -> Optional[Path]:
    study_dir = nifti_dir / study_id
    if not study_dir.exists():
        return None
    for series_dir in sorted(study_dir.iterdir()):
        p = series_dir / f"sub-{study_id}_acq-{acq}_T2w.nii.gz"
        if p.exists():
            return p
    return None


def find_original_spineps_seg(spineps_dir: Path, study_id: str) -> Optional[Path]:
    p = spineps_dir / 'segmentations' / study_id / f"{study_id}_seg-spine_msk.nii.gz"
    return p if p.exists() else None


def find_original_tss_sag(totalspine_dir: Path, study_id: str) -> Optional[Path]:
    p = totalspine_dir / study_id / 'sagittal' / f"{study_id}_sagittal_labeled.nii.gz"
    return p if p.exists() else None


def find_original_tss_sag_unc(totalspine_dir: Path, study_id: str) -> Optional[Path]:
    p = totalspine_dir / study_id / 'sagittal' / f"{study_id}_sagittal_unc.nii.gz"
    return p if p.exists() else None


def find_native_axial_tss(totalspine_dir: Path, study_id: str) -> Optional[Path]:
    p = totalspine_dir / study_id / 'axial' / f"{study_id}_axial_labeled.nii.gz"
    return p if p.exists() else None


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

def norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def overlay_mask(ax, mask2d: np.ndarray, color_rgb, alpha: float = 0.65):
    if not mask2d.any():
        return
    rgba = np.zeros((*mask2d.shape, 4), dtype=float)
    rgba[mask2d] = [*color_rgb, alpha]
    ax.imshow(rgba.transpose(1, 0, 2), origin='lower')


def _best_z(vol: np.ndarray, highlight: Optional[np.ndarray] = None,
            override: Optional[int] = None) -> int:
    """Best axial (z) index. Axis 2 = I-S in RAS canonical."""
    if override is not None:
        return min(override, vol.shape[2] - 1)
    if highlight is not None and highlight.any():
        return int(np.argmax(highlight.sum(axis=(0, 1))))
    return vol.shape[2] // 2


def _best_y(vol: np.ndarray, highlight: Optional[np.ndarray] = None) -> int:
    """Best coronal (y) index. Axis 1 = P-A in RAS canonical."""
    if highlight is not None and highlight.any():
        return int(np.argmax(highlight.sum(axis=(0, 2))))
    return vol.shape[1] // 2


def _best_x(vol: np.ndarray, highlight: Optional[np.ndarray] = None) -> int:
    """Best sagittal (x) index. Axis 0 = L-R in RAS canonical."""
    if highlight is not None and highlight.any():
        return int(np.argmax(highlight.sum(axis=(1, 2))))
    return vol.shape[0] // 2


def _ax(vol: Optional[np.ndarray], z: int) -> np.ndarray:
    if vol is None:
        return np.zeros((1, 1))
    return vol[:, :, min(z, vol.shape[2] - 1)]


def _cor(vol: Optional[np.ndarray], y: int) -> np.ndarray:
    if vol is None:
        return np.zeros((1, 1))
    return vol[:, min(y, vol.shape[1] - 1), :]


def _sag(vol: Optional[np.ndarray], x: int) -> np.ndarray:
    if vol is None:
        return np.zeros((1, 1))
    return vol[min(x, vol.shape[0] - 1), :, :]


def _unavailable(ax, label: str):
    ax.text(0.5, 0.5, f'{label}\nnot available',
            ha='center', va='center', color='grey', fontsize=10)
    ax.axis('off')


# ============================================================================
# PANEL FUNCTIONS
# ============================================================================

def _panel_axial_t2w(ax, img_sl, z_idx: int):
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower')
    ax.set_title(f'Axial T2w at TV level  (z={z_idx})', fontsize=11)
    ax.axis('off')


def _panel_axial_tps(ax, img_sl, tp_left_sl, tp_right_sl, sacrum_sl):
    """
    Axial TP overlay. TPs are dilated for visibility (display only).
    Note in title makes clear this is a display dilation.
    """
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower', alpha=0.75)
    overlay_mask(ax, sacrum_sl,   [1.00, 0.55, 0.00], 0.35)
    overlay_mask(ax, tp_left_sl,  [1.00, 0.10, 0.10], 0.85)
    overlay_mask(ax, tp_right_sl, [0.00, 0.80, 1.00], 0.85)
    ax.legend(handles=[
        mpatches.Patch(color=[1.00, 0.10, 0.10], label='Left TP (dilated)'),
        mpatches.Patch(color=[0.00, 0.80, 1.00], label='Right TP (dilated)'),
        mpatches.Patch(color=[1.00, 0.55, 0.00], label='Sacrum'),
    ], loc='lower right', fontsize=7, framealpha=0.6)
    ax.set_title(
        'SPINEPS TPs — Axial\n(masks dilated 2 vox for visibility; not used for measurement)',
        fontsize=10
    )
    ax.axis('off')


def _panel_tss_axial(ax, img_sl, tss_sl, native: bool):
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower', alpha=0.75)
    patches = []
    for label, (color, name) in TSS_LABEL_COLORS.items():
        m = tss_sl == label
        if m.any():
            overlay_mask(ax, m, color, 0.45)
            patches.append(mpatches.Patch(color=color, label=name))
    if patches:
        ax.legend(handles=patches, loc='lower right', fontsize=7, framealpha=0.6)
    src = 'native axial' if native else 'resampled from sag'
    ax.set_title(f'TSS Labels — Axial\n({src})', fontsize=11)
    ax.axis('off')


def _panel_sagittal_tss_confirm(ax, sag_img_sl, tss_sag_sl, tv_name: str, x_idx: int):
    """
    Sagittal T2w + original TSS labels (pre-registration).
    TV and sacrum highlighted; other levels faint.
    """
    ax.imshow(norm(sag_img_sl).T, cmap='gray', origin='lower', alpha=0.80)
    patches = []
    for label, (color, name) in TSS_LABEL_COLORS.items():
        m = tss_sag_sl == label
        if not m.any():
            continue
        alpha = 0.70 if label in (L5_LABEL, L6_LABEL, SACRUM_LABEL) else 0.28
        overlay_mask(ax, m, color, alpha)
        patches.append(mpatches.Patch(color=color, label=name))
    if patches:
        ax.legend(handles=patches, loc='lower right', fontsize=7, framealpha=0.6)
    ax.set_title(
        f'Sagittal TSS — Level Confirmation\n(TV={tv_name} highlighted)',
        fontsize=11
    )
    ax.axis('off')


def _panel_tss_uncertainty_sagittal(ax, sag_img_sl, unc_sl):
    """
    TSS uncertainty in its native sagittal orientation alongside the anatomy.
    Far more interpretable than the resampled axial version.
    """
    ax.imshow(norm(sag_img_sl).T, cmap='gray', origin='lower', alpha=0.55)
    # Uncertainty as hot overlay on top
    unc_norm = norm(unc_sl)
    unc_rgba = plt.cm.hot(unc_norm.T)
    unc_rgba[..., 3] = np.where(unc_norm.T > 0.05, 0.70, 0.0)
    ax.imshow(unc_rgba, origin='lower')
    ax.set_title(
        'TSS Uncertainty — Sagittal (native)\n(Normalised Shannon Entropy)',
        fontsize=11
    )
    ax.axis('off')


def _panel_contact_zone_axial(ax, img_sl, cz_left, cz_right,
                               tp_left_sl, tp_right_sl, sacrum_sl):
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower', alpha=0.75)
    overlay_mask(ax, sacrum_sl,   [1.00, 0.55, 0.00], 0.25)
    overlay_mask(ax, tp_left_sl,  [1.00, 0.10, 0.10], 0.20)
    overlay_mask(ax, tp_right_sl, [0.00, 0.80, 1.00], 0.20)
    overlay_mask(ax, cz_left,     [1.00, 0.00, 1.00], 0.85)
    overlay_mask(ax, cz_right,    [0.00, 1.00, 0.50], 0.85)
    ax.legend(handles=[
        mpatches.Patch(color=[1.00, 0.00, 1.00], label='Left contact zone'),
        mpatches.Patch(color=[0.00, 1.00, 0.50], label='Right contact zone'),
    ], loc='lower right', fontsize=7, framealpha=0.6)
    ax.set_title('Contact Zones — Axial\n(Type II/III decision region)', fontsize=11)
    ax.axis('off')


def _panel_coronal_tps_sag(ax, sag_img, sag_spineps, sag_tss,
                            tv_name: str, y_idx: int):
    """
    Coronal slice from the ORIGINAL SAGITTAL volume.
    Avoids the squished appearance from reslicing the axial volume coronally.
    Uses original SPINEPS and TSS labels (pre-registration) in sagittal space.
    """
    img_sl = _cor(sag_img, y_idx)
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower', alpha=0.75)

    if sag_spineps is not None:
        tp_left_sl  = _cor(sag_spineps == TP_LEFT_LABEL,  y_idx)
        tp_right_sl = _cor(sag_spineps == TP_RIGHT_LABEL, y_idx)
        overlay_mask(ax, tp_left_sl,  [1.00, 0.10, 0.10], 0.80)
        overlay_mask(ax, tp_right_sl, [0.00, 0.80, 1.00], 0.80)

    if sag_tss is not None:
        sacrum_sl = _cor(sag_tss == SACRUM_LABEL, y_idx)
        overlay_mask(ax, sacrum_sl, [1.00, 0.55, 0.00], 0.45)

    ax.legend(handles=[
        mpatches.Patch(color=[1.00, 0.10, 0.10], label='Left TP'),
        mpatches.Patch(color=[0.00, 0.80, 1.00], label='Right TP'),
        mpatches.Patch(color=[1.00, 0.55, 0.00], label='Sacrum'),
    ], loc='lower right', fontsize=7, framealpha=0.6)
    ax.set_title(
        f'TPs — Coronal  (TV={tv_name}, y={y_idx})\n(original sag space)',
        fontsize=11
    )
    ax.axis('off')


def _panel_sagittal_tps_sag(ax, sag_img, sag_spineps, sag_tss,
                              tv_name: str, x_idx: int):
    """
    Sagittal slice from the ORIGINAL SAGITTAL volume.
    Shows TP height and sacrum relationship clearly.
    """
    img_sl = _sag(sag_img, x_idx)
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower', alpha=0.75)

    if sag_spineps is not None:
        tp_left_sl  = _sag(sag_spineps == TP_LEFT_LABEL,  x_idx)
        tp_right_sl = _sag(sag_spineps == TP_RIGHT_LABEL, x_idx)
        overlay_mask(ax, tp_left_sl,  [1.00, 0.10, 0.10], 0.80)
        overlay_mask(ax, tp_right_sl, [0.00, 0.80, 1.00], 0.80)

    if sag_tss is not None:
        sacrum_sl = _sag(sag_tss == SACRUM_LABEL, x_idx)
        overlay_mask(ax, sacrum_sl, [1.00, 0.55, 0.00], 0.45)

    ax.legend(handles=[
        mpatches.Patch(color=[1.00, 0.10, 0.10], label='Left TP'),
        mpatches.Patch(color=[0.00, 0.80, 1.00], label='Right TP'),
        mpatches.Patch(color=[1.00, 0.55, 0.00], label='Sacrum'),
    ], loc='lower right', fontsize=7, framealpha=0.6)
    ax.set_title(
        f'TPs — Sagittal  (TV={tv_name}, x={x_idx})\n(original sag space — height context)',
        fontsize=11
    )
    ax.axis('off')


def _panel_summary(ax, study_id: str, result: Optional[dict], tv_name: str):
    ax.axis('off')
    ax.set_facecolor('#1a1a2e')
    lines = [f'Study : {study_id}', f'TV    : {tv_name}', '']

    if result is None:
        lines += ['No detection results available.', '',
                  'Run 04_detect_lstv.py and pass', '--lstv_json to annotate.']
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
                f'  TP height : {sd.get("tp_height_mm", 0):.1f} mm'
                f'  (thresh {TP_HEIGHT_MM} mm)',
                f'  TP-Sacrum : {sd.get("dist_mm", float("inf")):.1f} mm'
                f'  (thresh {CONTACT_DIST_MM} mm)',
            ]
            p2 = sd.get('p_type_ii')
            p3 = sd.get('p_type_iii')
            if p2 is not None and p3 is not None:
                lines.append(
                    f'  P(II)={p2:.2f}  P(III)={p3:.2f}'
                    f'  [{sd.get("confidence", "")}]'
                )
            unc = sd.get('unc_features') or {}
            if unc and unc.get('unc_mean') is not None:
                try:
                    lines.append(
                        f'  unc_mean={unc["unc_mean"]:.3f}'
                        f'  std={unc.get("unc_std", float("nan")):.3f}'
                        f'  hi_frac={unc.get("unc_high_frac", float("nan")):.3f}'
                        f'  [{unc.get("source", "")}]'
                    )
                except (TypeError, ValueError):
                    pass
            if sd.get('note'):
                lines.append(f'  NOTE: {sd["note"]}')
            lines.append('')

        if result.get('errors'):
            lines += ['Errors:'] + [f'  {e}' for e in result['errors']]

    ax.text(0.05, 0.97, '\n'.join(lines),
            transform=ax.transAxes, va='top', ha='left',
            fontsize=8.5, family='monospace', color='white')
    ax.set_title('Classification Summary', fontsize=11)


# ============================================================================
# CORE VISUALIZER
# ============================================================================

def visualize_study(
    study_id: str,
    registered_dir: Path,
    nifti_dir: Path,
    spineps_dir: Path,
    totalspine_dir: Path,
    output_dir: Path,
    result: Optional[dict] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{study_id}_lstv_overlay.png"

    reg = registered_dir / study_id

    def try_load(path, label):
        p = Path(path) if path is not None else None
        if p is not None and p.exists():
            try:
                return load_canonical(p)
            except Exception as e:
                logger.warning(f"  [{study_id}] Cannot load {label}: {e}")
        elif p is not None:
            logger.warning(f"  [{study_id}] Missing: {p.name}")
        return None, None

    # --- Registered (axial space) ---
    spineps_reg, spineps_nii = try_load(
        reg / f"{study_id}_spineps_reg.nii.gz", 'SPINEPS reg')
    tss_reg,     tss_reg_nii = try_load(
        reg / f"{study_id}_tss_reg.nii.gz",     'TSS reg')

    # --- Original sagittal space (for coronal/sagittal panels and TSS confirm) ---
    sag_t2w_path  = find_t2w(nifti_dir, study_id, 'sag')
    sag_bg, sag_nii = try_load(sag_t2w_path, 'Sagittal T2w')

    orig_spineps_path = find_original_spineps_seg(spineps_dir, study_id)
    orig_spineps, _   = try_load(orig_spineps_path, 'SPINEPS orig sag')

    orig_tss_sag_path = find_original_tss_sag(totalspine_dir, study_id)
    orig_tss_sag, _   = try_load(orig_tss_sag_path, 'TSS orig sag')

    tss_sag_unc_path  = find_original_tss_sag_unc(totalspine_dir, study_id)
    tss_sag_unc, _    = try_load(tss_sag_unc_path, 'TSS sag uncertainty')

    # --- Axial T2w background for row 0 ---
    axial_t2w_path  = find_t2w(nifti_dir, study_id, 'ax')
    ax_bg, ax_bg_nii = try_load(axial_t2w_path, 'Axial T2w')
    if ax_bg is None:
        ax_bg, ax_bg_nii = spineps_reg, spineps_nii

    # --- Native axial TSS if available ---
    native_tss_path = find_native_axial_tss(totalspine_dir, study_id)
    tss_native, _   = try_load(native_tss_path, 'TSS native axial')
    using_native_tss = tss_native is not None

    if ax_bg is None:
        logger.error(f"  [{study_id}] No axial background — skipping")
        return

    # Cast types
    if spineps_reg   is not None: spineps_reg   = spineps_reg.astype(int)
    if tss_reg       is not None: tss_reg       = tss_reg.astype(int)
    if orig_spineps  is not None: orig_spineps  = orig_spineps.astype(int)
    if orig_tss_sag  is not None: orig_tss_sag  = orig_tss_sag.astype(int)
    if tss_native    is not None: tss_native    = tss_native.astype(int)
    if tss_sag_unc   is not None: tss_sag_unc   = tss_sag_unc.astype(np.float32)

    vox_mm_ax  = voxel_size_mm(ax_bg_nii)
    vox_mm_sag = voxel_size_mm(sag_nii) if sag_nii is not None else vox_mm_ax

    # ---- Target Vertebra (use registered TSS in axial space) ---------------
    tss_for_labels = tss_reg if tss_reg is not None else tss_native
    tv_label = (L6_LABEL
                if tss_for_labels is not None and L6_LABEL in np.unique(tss_for_labels)
                else L5_LABEL)
    tv_name  = 'L6' if tv_label == L6_LABEL else 'L5'

    # ---- TV z range in axial space -----------------------------------------
    z_range = get_tv_z_range(tss_for_labels, tv_label) if tss_for_labels is not None else None
    zeros3_ax  = np.zeros(ax_bg.shape, dtype=bool)
    zeros3_sag = np.zeros(sag_bg.shape if sag_bg is not None else ax_bg.shape, dtype=bool)

    if z_range is not None and spineps_reg is not None:
        z_min, z_max = z_range
        z_tv = (z_min + z_max) // 2
        tp_left_ax  = isolate_tp_at_tv(spineps_reg, TP_LEFT_LABEL,  z_min, z_max)
        tp_right_ax = isolate_tp_at_tv(spineps_reg, TP_RIGHT_LABEL, z_min, z_max)
    else:
        z_tv = ax_bg.shape[2] // 2
        tp_left_ax = tp_right_ax = zeros3_ax

    sacrum_ax = ((tss_for_labels == SACRUM_LABEL)
                 if tss_for_labels is not None else zeros3_ax)

    # Contact zones (axial space)
    def _cz(tp_mask):
        if tp_mask.any() and sacrum_ax.any():
            return build_contact_zone(tp_mask, sacrum_ax, vox_mm_ax, CONTACT_DILATION_MM)
        return zeros3_ax

    cz_left_ax  = _cz(tp_left_ax)
    cz_right_ax = _cz(tp_right_ax)
    cz_combined = cz_left_ax | cz_right_ax
    tp_combined_ax = tp_left_ax | tp_right_ax

    # TV z range in sagittal space (for coronal/sagittal slice selection)
    tp_combined_sag = zeros3_sag
    if orig_spineps is not None:
        tp_combined_sag = (
            (orig_spineps == TP_LEFT_LABEL) | (orig_spineps == TP_RIGHT_LABEL)
        )

    # ---- Slice indices -----------------------------------------------------
    z_ax  = z_tv
    z_cz  = _best_z(ax_bg, highlight=cz_combined) if cz_combined.any() else z_ax

    # Coronal and sagittal from the SAGITTAL volume
    sag_ref = sag_bg if sag_bg is not None else ax_bg
    y_cor = _best_y(sag_ref, highlight=tp_combined_sag if tp_combined_sag.any() else None)
    x_sag = _best_x(sag_ref, highlight=tp_combined_sag if tp_combined_sag.any() else None)

    # Sagittal midline x for TSS confirm panel
    x_mid_sag = (orig_tss_sag.shape[0] // 2) if orig_tss_sag is not None else (
                  sag_bg.shape[0] // 2 if sag_bg is not None else 0)

    # TSS display for axial panel
    tss_display = tss_native if using_native_tss else tss_reg

    # Dilated TP masks for axial display only
    tp_left_ax_disp  = dilate_for_display(tp_left_ax,  DISPLAY_DILATION_VOXELS)
    tp_right_ax_disp = dilate_for_display(tp_right_ax, DISPLAY_DILATION_VOXELS)

    # ---- Figure ------------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(21, 21))
    fig.patch.set_facecolor('#0d0d1a')
    for ax in axes.flat:
        ax.set_facecolor('#0d0d1a')

    castellvi = result.get('castellvi_type') if result else None
    conf_str  = f"  conf: {result.get('confidence','')}" if result else ''
    fig.suptitle(
        f"Study {study_id}   |   Castellvi: {castellvi or 'N/A'}{conf_str}"
        f"   |   TV: {tv_name}",
        fontsize=15, color='white', y=0.998,
    )

    # ---- Row 0: Axial anatomy ----------------------------------------------
    _panel_axial_t2w(axes[0, 0], _ax(ax_bg, z_ax), z_ax)

    if spineps_reg is not None:
        _panel_axial_tps(
            axes[0, 1],
            _ax(ax_bg,            z_ax),
            _ax(tp_left_ax_disp,  z_ax),
            _ax(tp_right_ax_disp, z_ax),
            _ax(sacrum_ax,        z_ax),
        )
    else:
        _unavailable(axes[0, 1], 'SPINEPS reg not available')

    if tss_display is not None:
        bg_sl  = _ax(ax_bg, z_ax)
        tss_sl = _ax(tss_display, min(z_ax, tss_display.shape[2] - 1))
        if tss_sl.shape != bg_sl.shape:
            sy = min(tss_sl.shape[0], bg_sl.shape[0])
            sx = min(tss_sl.shape[1], bg_sl.shape[1])
            tss_sl_use = np.zeros_like(bg_sl)
            tss_sl_use[:sy, :sx] = tss_sl[:sy, :sx]
        else:
            tss_sl_use = tss_sl
        _panel_tss_axial(axes[0, 2], bg_sl, tss_sl_use, using_native_tss)
    else:
        _unavailable(axes[0, 2], 'TSS labels not available')

    # ---- Row 1: Sagittal confirm + uncertainty + contact zones -------------
    if orig_tss_sag is not None and sag_bg is not None:
        _panel_sagittal_tss_confirm(
            axes[1, 0],
            _sag(sag_bg,      x_mid_sag),
            _sag(orig_tss_sag, x_mid_sag),
            tv_name, x_mid_sag,
        )
    elif orig_tss_sag is not None:
        _panel_sagittal_tss_confirm(
            axes[1, 0],
            _sag(orig_tss_sag, x_mid_sag).astype(float),
            _sag(orig_tss_sag, x_mid_sag),
            tv_name, x_mid_sag,
        )
    else:
        _unavailable(axes[1, 0], 'Sagittal TSS not found')

    if tss_sag_unc is not None and sag_bg is not None:
        _panel_tss_uncertainty_sagittal(
            axes[1, 1],
            _sag(sag_bg,     x_mid_sag),
            _sag(tss_sag_unc, x_mid_sag),
        )
    elif tss_sag_unc is not None:
        # No T2w background — show uncertainty alone
        im = axes[1, 1].imshow(norm(_sag(tss_sag_unc, x_mid_sag)).T,
                               cmap='hot', origin='lower')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        axes[1, 1].set_title('TSS Uncertainty — Sagittal', fontsize=11)
        axes[1, 1].axis('off')
    else:
        _unavailable(axes[1, 1], 'TSS uncertainty not available')

    _panel_contact_zone_axial(
        axes[1, 2],
        _ax(ax_bg,          z_cz),
        _ax(cz_left_ax,     z_cz),
        _ax(cz_right_ax,    z_cz),
        _ax(tp_left_ax,     z_cz),   # undilated for contact zone accuracy
        _ax(tp_right_ax,    z_cz),
        _ax(sacrum_ax,      z_cz),
    )

    # ---- Row 2: Coronal + sagittal from original sag space -----------------
    if sag_bg is not None:
        _panel_coronal_tps_sag(
            axes[2, 0], sag_bg, orig_spineps, orig_tss_sag, tv_name, y_cor
        )
        _panel_sagittal_tps_sag(
            axes[2, 1], sag_bg, orig_spineps, orig_tss_sag, tv_name, x_sag
        )
    else:
        _unavailable(axes[2, 0], 'Sagittal T2w not found')
        _unavailable(axes[2, 1], 'Sagittal T2w not found')

    _panel_summary(axes[2, 2], study_id, result, tv_name)

    plt.tight_layout(rect=[0, 0, 1, 0.997])
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"  [{study_id}] OK -> {out_path}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='LSTV Overlay Visualizer v4')
    parser.add_argument('--registered_dir',  required=True)
    parser.add_argument('--nifti_dir',       required=True)
    parser.add_argument('--spineps_dir',     required=True,
                        help='Original SPINEPS output (for pre-registration labels)')
    parser.add_argument('--totalspine_dir',  required=True)
    parser.add_argument('--output_dir',      required=True)
    parser.add_argument('--study_id',        default=None)
    parser.add_argument('--lstv_json',       default=None)
    args = parser.parse_args()

    registered_dir = Path(args.registered_dir)
    nifti_dir      = Path(args.nifti_dir)
    spineps_dir    = Path(args.spineps_dir)
    totalspine_dir = Path(args.totalspine_dir)
    output_dir     = Path(args.output_dir)

    results_by_id = {}
    if args.lstv_json:
        p = Path(args.lstv_json)
        if p.exists():
            with open(p) as f:
                results_by_id = {r['study_id']: r for r in json.load(f)}
            logger.info(f"Loaded {len(results_by_id)} detection results")

    if args.study_id:
        study_ids = [args.study_id]
    else:
        study_ids = sorted(d.name for d in registered_dir.iterdir() if d.is_dir())
        logger.info(f"Batch mode: {len(study_ids)} studies")

    errors = 0
    for sid in study_ids:
        try:
            visualize_study(
                study_id       = sid,
                registered_dir = registered_dir,
                nifti_dir      = nifti_dir,
                spineps_dir    = spineps_dir,
                totalspine_dir = totalspine_dir,
                output_dir     = output_dir,
                result         = results_by_id.get(sid),
            )
        except Exception as e:
            logger.error(f"  [{sid}] Failed: {e}")
            logger.debug(traceback.format_exc())
            errors += 1

    logger.info(f"Done. {len(study_ids)-errors}/{len(study_ids)} PNGs -> {output_dir}")


if __name__ == '__main__':
    main()
