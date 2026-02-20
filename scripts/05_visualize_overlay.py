#!/usr/bin/env python3
"""
05_visualize_overlay.py — LSTV Overlay Visualizer (registered space, v3)
=========================================================================
Panel layout:

  Row 0 | Axial T2w at TV level  | SPINEPS TPs — axial        | TSS labels — axial
  Row 1 | Sagittal T2w + TSS     | TSS uncertainty — axial    | Contact zones — axial
        | (confirms L5/L6 call)  |                             |
  Row 2 | Coronal TPs + sacrum   | Sagittal TP context        | Classification summary

Design decisions
----------------
* SPINEPS uncertainty is dropped — the rigid registration resamples it poorly
  (line artifacts). TSS uncertainty resamples acceptably because it was
  produced at isotropic-ish resolution.
* The sagittal panel (1,0) uses the *original* (pre-registration) TSS labeled
  mask overlaid on the original sagittal T2w so the L5/L6 identification can
  be confirmed visually without registration error.
* TSS axial panel (0,2) prefers a *native* axial TSS output if one exists at
  totalspine_dir/{study_id}/axial/{study_id}_axial_labeled.nii.gz,
  otherwise falls back to the registered sagittal TSS with reduced opacity
  and a "(resampled)" note in the title.
* TSS label overlays use 0.45 alpha so background anatomy always shows through.
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

# Labels to always show in sagittal confirmation panel
SAG_CONFIRM_LABELS = {45, 46, 50, 95, 100}


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


def isolate_tp_at_tv(spineps_data: np.ndarray, tp_label: int,
                     z_min: int, z_max: int) -> np.ndarray:
    tp_full = spineps_data == tp_label
    iso = np.zeros_like(tp_full)
    iso[:, :, max(z_min, 0):min(z_max, spineps_data.shape[2] - 1) + 1] = \
        tp_full[:, :, max(z_min, 0):min(z_max, spineps_data.shape[2] - 1) + 1]
    return iso


def build_contact_zone(mask_a: np.ndarray, mask_b: np.ndarray,
                       vox_mm: np.ndarray, radius_mm: float) -> np.ndarray:
    radius_vox = np.maximum(np.round(radius_mm / vox_mm).astype(int), 1)
    struct = np.ones(2 * radius_vox + 1, dtype=bool)
    return (binary_dilation(mask_a, structure=struct) &
            binary_dilation(mask_b, structure=struct))


# ============================================================================
# FILE DISCOVERY
# ============================================================================

def find_axial_t2w(nifti_dir: Path, study_id: str) -> Optional[Path]:
    study_dir = nifti_dir / study_id
    if not study_dir.exists():
        return None
    for series_dir in sorted(study_dir.iterdir()):
        p = series_dir / f"sub-{study_id}_acq-ax_T2w.nii.gz"
        if p.exists():
            return p
    return None


def find_sagittal_t2w(nifti_dir: Path, study_id: str) -> Optional[Path]:
    study_dir = nifti_dir / study_id
    if not study_dir.exists():
        return None
    for series_dir in sorted(study_dir.iterdir()):
        p = series_dir / f"sub-{study_id}_acq-sag_T2w.nii.gz"
        if p.exists():
            return p
    return None


def find_native_axial_tss(totalspine_dir: Path, study_id: str) -> Optional[Path]:
    """Check for TotalSpineSeg run natively on the axial T2w."""
    p = totalspine_dir / study_id / 'axial' / f"{study_id}_axial_labeled.nii.gz"
    return p if p.exists() else None


def find_original_tss_sag(totalspine_dir: Path, study_id: str) -> Optional[Path]:
    """Original (pre-registration) sagittal TSS label map."""
    p = totalspine_dir / study_id / 'sagittal' / f"{study_id}_sagittal_labeled.nii.gz"
    return p if p.exists() else None


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

def norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def overlay_mask(ax, mask2d: np.ndarray, color_rgb, alpha: float = 0.55):
    if not mask2d.any():
        return
    rgba = np.zeros((*mask2d.shape, 4), dtype=float)
    rgba[mask2d] = [*color_rgb, alpha]
    ax.imshow(rgba.transpose(1, 0, 2), origin='lower')


def _best_axial_z(vol: np.ndarray, highlight: Optional[np.ndarray] = None,
                  override: Optional[int] = None) -> int:
    if override is not None:
        return min(override, vol.shape[2] - 1)
    if highlight is not None and highlight.any():
        return int(np.argmax(highlight.sum(axis=(0, 1))))
    return vol.shape[2] // 2


def _best_coronal_y(vol: np.ndarray, highlight: Optional[np.ndarray] = None) -> int:
    if highlight is not None and highlight.any():
        return int(np.argmax(highlight.sum(axis=(0, 2))))
    return vol.shape[1] // 2


def _best_sagittal_x(vol: np.ndarray, highlight: Optional[np.ndarray] = None) -> int:
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
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower', alpha=0.75)
    overlay_mask(ax, sacrum_sl,   [1.00, 0.55, 0.00], 0.35)
    overlay_mask(ax, tp_left_sl,  [1.00, 0.10, 0.10], 0.80)
    overlay_mask(ax, tp_right_sl, [0.00, 0.80, 1.00], 0.80)
    ax.legend(handles=[
        mpatches.Patch(color=[1.00, 0.10, 0.10], label='Left TP'),
        mpatches.Patch(color=[0.00, 0.80, 1.00], label='Right TP'),
        mpatches.Patch(color=[1.00, 0.55, 0.00], label='Sacrum'),
    ], loc='lower right', fontsize=7, framealpha=0.6)
    ax.set_title('SPINEPS TPs — Axial\n(LSTV assessment plane)', fontsize=11)
    ax.axis('off')


def _panel_tss_axial(ax, img_sl, tss_sl, native: bool):
    """TSS labeled overlay on axial slice. Lower alpha so anatomy shows through."""
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower', alpha=0.75)
    patches = []
    for label, (color, name) in TSS_LABEL_COLORS.items():
        m = tss_sl == label
        if m.any():
            overlay_mask(ax, m, color, 0.45)   # reduced alpha — anatomy visible
            patches.append(mpatches.Patch(color=color, label=name))
    if patches:
        ax.legend(handles=patches, loc='lower right', fontsize=7, framealpha=0.6)
    src = 'native axial' if native else 'resampled from sag'
    ax.set_title(f'TSS Labels — Axial\n({src})', fontsize=11)
    ax.axis('off')


def _panel_sagittal_tss_confirm(ax, sag_img_sl, tss_sag_sl, tv_name: str, x_idx: int):
    """
    Sagittal T2w with original TSS labels overlaid.
    Shows only the lower lumbar + sacrum labels (L4-S1) at higher alpha so the
    radiologist can confirm which level was called as TV.
    """
    ax.imshow(norm(sag_img_sl).T, cmap='gray', origin='lower', alpha=0.80)
    patches = []
    for label, (color, name) in TSS_LABEL_COLORS.items():
        m = tss_sag_sl == label
        if not m.any():
            continue
        # Highlight TV and sacrum brightly; others faint
        if label in (L5_LABEL, L6_LABEL, SACRUM_LABEL):
            overlay_mask(ax, m, color, 0.70)
        else:
            overlay_mask(ax, m, color, 0.28)
        patches.append(mpatches.Patch(color=color, label=name))
    if patches:
        ax.legend(handles=patches, loc='lower right', fontsize=7, framealpha=0.6)
    ax.set_title(
        f'Sagittal TSS — Level Confirmation\n(TV={tv_name} highlighted,  x={x_idx})',
        fontsize=11
    )
    ax.axis('off')


def _panel_tss_uncertainty_axial(ax, unc_sl, z_idx: int):
    im = ax.imshow(norm(unc_sl).T, cmap='hot', origin='lower', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Entropy [0-1]')
    ax.set_title(f'TSS Uncertainty — Axial  (z={z_idx})\n(Normalised Shannon Entropy)',
                 fontsize=11)
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


def _panel_coronal_tps(ax, img_sl, tp_left_sl, tp_right_sl, sacrum_sl,
                       tv_name: str, y_idx: int):
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower', alpha=0.75)
    overlay_mask(ax, tp_left_sl,  [1.00, 0.10, 0.10], 0.80)
    overlay_mask(ax, tp_right_sl, [0.00, 0.80, 1.00], 0.80)
    overlay_mask(ax, sacrum_sl,   [1.00, 0.55, 0.00], 0.40)
    ax.legend(handles=[
        mpatches.Patch(color=[1.00, 0.10, 0.10], label='Left TP'),
        mpatches.Patch(color=[0.00, 0.80, 1.00], label='Right TP'),
        mpatches.Patch(color=[1.00, 0.55, 0.00], label='Sacrum'),
    ], loc='lower right', fontsize=7, framealpha=0.6)
    ax.set_title(f'TPs — Coronal  (TV={tv_name}, y={y_idx})', fontsize=11)
    ax.axis('off')


def _panel_sagittal_tps(ax, img_sl, tp_left_sl, tp_right_sl, sacrum_sl,
                        tv_name: str, x_idx: int):
    """Sagittal slice showing TP and sacrum context — useful for height measurement."""
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower', alpha=0.75)
    overlay_mask(ax, tp_left_sl,  [1.00, 0.10, 0.10], 0.70)
    overlay_mask(ax, tp_right_sl, [0.00, 0.80, 1.00], 0.70)
    overlay_mask(ax, sacrum_sl,   [1.00, 0.55, 0.00], 0.40)
    ax.legend(handles=[
        mpatches.Patch(color=[1.00, 0.10, 0.10], label='Left TP'),
        mpatches.Patch(color=[0.00, 0.80, 1.00], label='Right TP'),
        mpatches.Patch(color=[1.00, 0.55, 0.00], label='Sacrum'),
    ], loc='lower right', fontsize=7, framealpha=0.6)
    ax.set_title(f'TPs — Sagittal  (TV={tv_name}, x={x_idx})\n(TP height context)',
                 fontsize=11)
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
    totalspine_dir: Path,
    output_dir: Path,
    result: Optional[dict] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{study_id}_lstv_overlay.png"

    reg = registered_dir / study_id

    def try_load(path, label):
        if path is not None and Path(path).exists():
            try:
                return load_canonical(Path(path))
            except Exception as e:
                logger.warning(f"  [{study_id}] Cannot load {label}: {e}")
        else:
            if path is not None:
                logger.warning(f"  [{study_id}] Missing: {Path(path).name}")
        return None, None

    # Registered volumes (axial space)
    spineps_reg, spineps_nii = try_load(
        reg / f"{study_id}_spineps_reg.nii.gz", 'SPINEPS reg')
    tss_reg,     tss_reg_nii = try_load(
        reg / f"{study_id}_tss_reg.nii.gz",     'TSS reg')
    tss_unc_reg, _           = try_load(
        reg / f"{study_id}_tss_unc_reg.nii.gz", 'TSS unc reg')

    # Native axial TSS (preferred for display if it exists)
    native_tss_path = find_native_axial_tss(totalspine_dir, study_id)
    tss_native, tss_native_nii = try_load(native_tss_path, 'TSS native axial')
    using_native_tss = tss_native is not None

    # Original sagittal TSS for level-confirmation panel
    orig_tss_sag_path = find_original_tss_sag(totalspine_dir, study_id)
    tss_orig_sag, tss_orig_sag_nii = try_load(orig_tss_sag_path, 'TSS orig sag')

    # T2w backgrounds
    axial_t2w_path = find_axial_t2w(nifti_dir, study_id)
    ax_bg, ax_bg_nii = try_load(axial_t2w_path, 'Axial T2w')

    sag_t2w_path = find_sagittal_t2w(nifti_dir, study_id)
    sag_bg, _ = try_load(sag_t2w_path, 'Sagittal T2w')

    # Fallback background
    if ax_bg is None:
        ax_bg, ax_bg_nii = spineps_reg, spineps_nii
    if ax_bg is None:
        logger.error(f"  [{study_id}] No axial background — skipping")
        return

    # TSS display layer (prefer native axial, fall back to registered)
    tss_display = tss_native if using_native_tss else tss_reg

    # Cast types
    if spineps_reg is not None: spineps_reg = spineps_reg.astype(int)
    if tss_reg     is not None: tss_reg     = tss_reg.astype(int)
    if tss_display is not None: tss_display = tss_display.astype(int)
    if tss_orig_sag is not None: tss_orig_sag = tss_orig_sag.astype(int)
    if tss_unc_reg is not None: tss_unc_reg = tss_unc_reg.astype(np.float32)

    vox_mm = voxel_size_mm(ax_bg_nii)

    # ---- Target Vertebra ---------------------------------------------------
    # Use the registered TSS for label detection (most reliable)
    tss_for_labels = tss_reg if tss_reg is not None else tss_display
    tv_label = (L6_LABEL
                if tss_for_labels is not None and L6_LABEL in np.unique(tss_for_labels)
                else L5_LABEL)
    tv_name  = 'L6' if tv_label == L6_LABEL else 'L5'

    # ---- TV z range -------------------------------------------------------
    z_range = get_tv_z_range(tss_for_labels, tv_label) if tss_for_labels is not None else None
    zeros3  = np.zeros(ax_bg.shape, dtype=bool)

    if z_range is not None and spineps_reg is not None:
        z_min, z_max = z_range
        z_tv = (z_min + z_max) // 2
        tp_left_3d  = isolate_tp_at_tv(spineps_reg, TP_LEFT_LABEL,  z_min, z_max)
        tp_right_3d = isolate_tp_at_tv(spineps_reg, TP_RIGHT_LABEL, z_min, z_max)
    else:
        z_tv = ax_bg.shape[2] // 2
        tp_left_3d = tp_right_3d = zeros3

    sacrum_3d = ((tss_for_labels == SACRUM_LABEL)
                 if tss_for_labels is not None else zeros3)

    def _cz(tp_mask):
        if tp_mask.any() and sacrum_3d.any():
            return build_contact_zone(tp_mask, sacrum_3d, vox_mm, CONTACT_DILATION_MM)
        return zeros3

    cz_left_3d  = _cz(tp_left_3d)
    cz_right_3d = _cz(tp_right_3d)
    cz_combined = cz_left_3d | cz_right_3d
    tp_combined = tp_left_3d | tp_right_3d

    # ---- Slice selection ---------------------------------------------------
    z_ax  = z_tv
    z_cz  = _best_axial_z(ax_bg, highlight=cz_combined) if cz_combined.any() else z_ax
    y_cor = _best_coronal_y(ax_bg, highlight=tp_combined)
    x_sag = _best_sagittal_x(ax_bg, highlight=tp_combined)

    # For sagittal panels use the mid-x of the background (spine midline)
    x_mid = ax_bg.shape[0] // 2

    # ---- Figure -----------------------------------------------------------
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

    # Row 0: Axial anatomy
    _panel_axial_t2w(axes[0, 0], _ax(ax_bg, z_ax), z_ax)

    if spineps_reg is not None:
        _panel_axial_tps(
            axes[0, 1],
            _ax(ax_bg, z_ax),
            _ax(tp_left_3d,  z_ax),
            _ax(tp_right_3d, z_ax),
            _ax(sacrum_3d,   z_ax),
        )
    else:
        _unavailable(axes[0, 1], 'SPINEPS seg not available')

    if tss_display is not None:
        # Resize tss_display slice to ax_bg shape if needed (native axial may differ)
        tss_sl = _ax(tss_display, min(z_ax, tss_display.shape[2] - 1))
        bg_sl  = _ax(ax_bg, z_ax)
        if tss_sl.shape != bg_sl.shape:
            # Pad or crop to match — simple crop to minimum
            sy = min(tss_sl.shape[0], bg_sl.shape[0])
            sx = min(tss_sl.shape[1], bg_sl.shape[1])
            tss_sl_use = np.zeros_like(bg_sl)
            tss_sl_use[:sy, :sx] = tss_sl[:sy, :sx]
        else:
            tss_sl_use = tss_sl
        _panel_tss_axial(axes[0, 2], bg_sl, tss_sl_use, using_native_tss)
    else:
        _unavailable(axes[0, 2], 'TSS labels not available')

    # Row 1: Sagittal confirmation + uncertainty + contact zones
    if tss_orig_sag is not None and sag_bg is not None:
        # Find best sagittal x in original sag space — just use mid
        x_sag_orig = tss_orig_sag.shape[0] // 2
        _panel_sagittal_tss_confirm(
            axes[1, 0],
            _sag(sag_bg, x_sag_orig),
            _sag(tss_orig_sag, x_sag_orig),
            tv_name, x_sag_orig,
        )
    elif tss_orig_sag is not None:
        # No sagittal T2w but have TSS sag labels — use as background
        x_sag_orig = tss_orig_sag.shape[0] // 2
        _panel_sagittal_tss_confirm(
            axes[1, 0],
            _sag(tss_orig_sag, x_sag_orig).astype(float),
            _sag(tss_orig_sag, x_sag_orig),
            tv_name, x_sag_orig,
        )
    else:
        _unavailable(axes[1, 0], 'Original sagittal TSS\nnot found')

    if tss_unc_reg is not None:
        _panel_tss_uncertainty_axial(axes[1, 1], _ax(tss_unc_reg, z_ax), z_ax)
    else:
        _unavailable(axes[1, 1], 'TSS uncertainty not available\n(SPINEPS uncertainty omitted\ndue to resampling artifacts)')

    _panel_contact_zone_axial(
        axes[1, 2],
        _ax(ax_bg, z_cz),
        _ax(cz_left_3d,  z_cz),
        _ax(cz_right_3d, z_cz),
        _ax(tp_left_3d,  z_cz),
        _ax(tp_right_3d, z_cz),
        _ax(sacrum_3d,   z_cz),
    )

    # Row 2: Coronal TPs + sagittal TPs + summary
    _panel_coronal_tps(
        axes[2, 0],
        _cor(ax_bg, y_cor),
        _cor(tp_left_3d,  y_cor),
        _cor(tp_right_3d, y_cor),
        _cor(sacrum_3d,   y_cor),
        tv_name, y_cor,
    )

    _panel_sagittal_tps(
        axes[2, 1],
        _sag(ax_bg, x_sag),
        _sag(tp_left_3d,  x_sag),
        _sag(tp_right_3d, x_sag),
        _sag(sacrum_3d,   x_sag),
        tv_name, x_sag,
    )

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
    parser = argparse.ArgumentParser(description='LSTV Overlay Visualizer v3')
    parser.add_argument('--registered_dir',  required=True)
    parser.add_argument('--nifti_dir',       required=True)
    parser.add_argument('--totalspine_dir',  required=True,
                        help='For original sagittal TSS labels + native axial TSS if present')
    parser.add_argument('--output_dir',      required=True)
    parser.add_argument('--study_id',        default=None)
    parser.add_argument('--lstv_json',       default=None)
    args = parser.parse_args()

    registered_dir = Path(args.registered_dir)
    nifti_dir      = Path(args.nifti_dir)
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
