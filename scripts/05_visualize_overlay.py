#!/usr/bin/env python3
"""
05_visualize_overlay.py — LSTV Overlay Visualizer (post-registration)
======================================================================
Visualizes registered segmentations from 03b_register_to_axial.py.
All volumes are in the common axial reference frame.

Panel layout (3 rows x 3 columns):

  Row 0 | Axial T2w at L5     | SPINEPS TP overlay (axial) | TSS labeled (axial)
  Row 1 | SPINEPS uncertainty | TSS uncertainty             | Contact zones (axial)
  Row 2 | Sagittal T2w        | SPINEPS TPs (coronal)       | Classification summary

Key change from the sagittal-only version:
  - TPs are shown in the AXIAL plane at the TV level, where they appear
    as lateral projections and their relationship to the sacrum is clear.
  - A coronal view is added (row 2, middle) for anatomical context.
  - All masks are from the registered outputs so spatial alignment is exact.

Usage — single study:
    python 05_visualize_overlay.py \
        --registered_dir results/registered \
        --nifti_dir      results/nifti \
        --output_dir     results/lstv_viz \
        --study_id       100206310 \
        [--lstv_json     results/lstv_detection/lstv_results.json]

Usage — batch:
    python 05_visualize_overlay.py \
        --registered_dir results/registered \
        --nifti_dir      results/nifti \
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
# CONSTANTS — keep in sync with 04_detect_lstv.py
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


# ============================================================================
# NIfTI HELPERS
# ============================================================================

def load_canonical(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """Load NIfTI, reorient to RAS canonical. Squeeze 4D → 3D."""
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


def isolate_tp_at_tv(spineps_data: np.ndarray,
                     tp_label: int,
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
    return (binary_dilation(mask_a, structure=struct) &
            binary_dilation(mask_b, structure=struct))


# ============================================================================
# T2w AXIAL DISCOVERY
# ============================================================================

def find_axial_t2w(nifti_dir: Path, study_id: str) -> Optional[Path]:
    """Locate the axial T2w NIfTI for a study."""
    study_dir = nifti_dir / study_id
    if not study_dir.exists():
        return None
    for series_dir in sorted(study_dir.iterdir()):
        p = series_dir / f"sub-{study_id}_acq-ax_T2w.nii.gz"
        if p.exists():
            return p
    return None


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
    ax.imshow(rgba.transpose(1, 0, 2), origin='lower')


def _pick_best_axial_slice(vol3d: np.ndarray,
                           highlight: Optional[np.ndarray] = None,
                           override: Optional[int] = None) -> int:
    """
    In RAS canonical, axis 2 = I-S, so axial slices are vol[:, :, z].
    Pick the z with the most highlight voxels, or the mid-z if no highlight.
    """
    if override is not None:
        return min(override, vol3d.shape[2] - 1)
    if highlight is not None and highlight.any():
        return int(np.argmax(highlight.sum(axis=(0, 1))))
    return vol3d.shape[2] // 2


def _pick_best_coronal_slice(vol3d: np.ndarray,
                              highlight: Optional[np.ndarray] = None,
                              override: Optional[int] = None) -> int:
    """
    In RAS canonical, axis 1 = A-P, so coronal slices are vol[:, y, :].
    Pick the y with the most highlight voxels, or mid-y.
    """
    if override is not None:
        return min(override, vol3d.shape[1] - 1)
    if highlight is not None and highlight.any():
        return int(np.argmax(highlight.sum(axis=(0, 2))))
    return vol3d.shape[1] // 2


def _pick_best_sagittal_slice(vol3d: np.ndarray,
                               highlight: Optional[np.ndarray] = None,
                               override: Optional[int] = None) -> int:
    """Axis 0 = L-R. Pick x with most highlight voxels, or mid-x."""
    if override is not None:
        return min(override, vol3d.shape[0] - 1)
    if highlight is not None and highlight.any():
        return int(np.argmax(highlight.sum(axis=(1, 2))))
    return vol3d.shape[0] // 2


def _axial(vol: Optional[np.ndarray], z: int) -> np.ndarray:
    if vol is None:
        return np.zeros((1, 1))
    return vol[:, :, min(z, vol.shape[2] - 1)]


def _coronal(vol: Optional[np.ndarray], y: int) -> np.ndarray:
    if vol is None:
        return np.zeros((1, 1))
    return vol[:, min(y, vol.shape[1] - 1), :]


def _sagittal(vol: Optional[np.ndarray], x: int) -> np.ndarray:
    if vol is None:
        return np.zeros((1, 1))
    return vol[min(x, vol.shape[0] - 1), :, :]


# ============================================================================
# PANEL FUNCTIONS
# ============================================================================

def _panel_axial_t2w(ax, img_sl, z_idx):
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower')
    ax.set_title(f'Axial T2w at TV level  (z={z_idx})', fontsize=11)
    ax.axis('off')


def _panel_axial_tps(ax, img_sl, tp_left_sl, tp_right_sl, sacrum_sl):
    """Axial view showing TP–sacrum relationship — the key LSTV plane."""
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower', alpha=0.7)
    overlay_mask(ax, sacrum_sl,   [1.00, 0.55, 0.00], 0.40)
    overlay_mask(ax, tp_left_sl,  [1.00, 0.10, 0.10], 0.80)
    overlay_mask(ax, tp_right_sl, [0.00, 0.80, 1.00], 0.80)
    ax.legend(handles=[
        mpatches.Patch(color=[1.00, 0.10, 0.10], label='Left TP'),
        mpatches.Patch(color=[0.00, 0.80, 1.00], label='Right TP'),
        mpatches.Patch(color=[1.00, 0.55, 0.00], label='Sacrum'),
    ], loc='lower right', fontsize=7, framealpha=0.6)
    ax.set_title('TPs at TV level — Axial\n(LSTV assessment plane)', fontsize=11)
    ax.axis('off')


def _panel_tss_axial(ax, img_sl, tss_sl):
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower', alpha=0.65)
    patches = []
    for label, (color, name) in TSS_LABEL_COLORS.items():
        m = tss_sl == label
        if m.any():
            overlay_mask(ax, m, color, 0.65)
            patches.append(mpatches.Patch(color=color, label=name))
    if patches:
        ax.legend(handles=patches, loc='lower right', fontsize=7, framealpha=0.6)
    ax.set_title('TSS Labeled Mask — Axial', fontsize=11)
    ax.axis('off')


def _panel_uncertainty(ax, unc_sl, title):
    im = ax.imshow(norm(unc_sl).T, cmap='hot', origin='lower', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Entropy [0-1]')
    ax.set_title(title, fontsize=11)
    ax.axis('off')


def _panel_contact_zone_axial(ax, img_sl, cz_left, cz_right,
                               tp_left_sl, tp_right_sl, sacrum_sl):
    """Axial view of contact zones — most diagnostic for Type II vs III."""
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower', alpha=0.65)
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


def _panel_sagittal_preview(ax, img_sl, x_idx):
    """Sagittal registered preview — anatomical context."""
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower')
    ax.set_title(f'Registered Sag Preview  (x={x_idx})', fontsize=11)
    ax.axis('off')


def _panel_coronal_tps(ax, img_sl, tp_left_sl, tp_right_sl, sacrum_sl, tv_name, y_idx):
    """Coronal view — both TPs and sacrum visible simultaneously."""
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower', alpha=0.7)
    overlay_mask(ax, tp_left_sl,  [1.00, 0.10, 0.10], 0.80)
    overlay_mask(ax, tp_right_sl, [0.00, 0.80, 1.00], 0.80)
    overlay_mask(ax, sacrum_sl,   [1.00, 0.55, 0.00], 0.45)
    ax.legend(handles=[
        mpatches.Patch(color=[1.00, 0.10, 0.10], label='Left TP'),
        mpatches.Patch(color=[0.00, 0.80, 1.00], label='Right TP'),
        mpatches.Patch(color=[1.00, 0.55, 0.00], label='Sacrum'),
    ], loc='lower right', fontsize=7, framealpha=0.6)
    ax.set_title(f'TPs at TV ({tv_name}) — Coronal  (y={y_idx})', fontsize=11)
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


def _unavailable(ax, label):
    ax.text(0.5, 0.5, f'{label}\nnot available',
            ha='center', va='center', color='grey', fontsize=10)
    ax.axis('off')


# ============================================================================
# CORE VISUALIZER
# ============================================================================

def visualize_study(
    study_id: str,
    registered_dir: Path,
    nifti_dir: Path,
    output_dir: Path,
    result: Optional[dict] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{study_id}_lstv_overlay.png"

    reg = registered_dir / study_id

    def try_load(path, label):
        if path.exists():
            try:
                return load_canonical(path)
            except Exception as e:
                logger.warning(f"  [{study_id}] Cannot load {label}: {e}")
        else:
            logger.warning(f"  [{study_id}] Missing: {path.name}")
        return None, None

    spineps_data, spineps_nii = try_load(
        reg / f"{study_id}_spineps_reg.nii.gz", 'SPINEPS reg')
    tss_data,     tss_nii     = try_load(
        reg / f"{study_id}_tss_reg.nii.gz",     'TSS reg')
    spineps_unc,  _           = try_load(
        reg / f"{study_id}_spineps_unc_reg.nii.gz", 'SPINEPS unc reg')
    tss_unc,      _           = try_load(
        reg / f"{study_id}_tss_unc_reg.nii.gz",     'TSS unc reg')
    sag_preview,  _           = try_load(
        reg / f"{study_id}_sag_preview.nii.gz",     'Sag preview')

    # Axial T2w as background (best quality axial image)
    axial_t2w_path = find_axial_t2w(nifti_dir, study_id)
    if axial_t2w_path:
        bg_data, bg_nii = try_load(axial_t2w_path, 'Axial T2w')
    else:
        bg_data, bg_nii = None, None

    # Fall back to SPINEPS reg if no axial T2w
    if bg_data is None:
        bg_data, bg_nii = spineps_data, spineps_nii

    if bg_data is None and tss_data is None:
        logger.error(f"  [{study_id}] No usable data — skipping")
        return

    if spineps_data is not None:
        spineps_data = spineps_data.astype(int)
    if tss_data is not None:
        tss_data = tss_data.astype(int)
    if spineps_unc is not None:
        spineps_unc = spineps_unc.astype(np.float32)
    if tss_unc is not None:
        tss_unc = tss_unc.astype(np.float32)

    vox_mm = voxel_size_mm(bg_nii) if bg_nii is not None else np.array([1., 1., 1.])

    # ---- Target Vertebra ---------------------------------------------------
    tv_label = (L6_LABEL
                if tss_data is not None and L6_LABEL in np.unique(tss_data)
                else L5_LABEL)
    tv_name  = 'L6' if tv_label == L6_LABEL else 'L5'

    # ---- TV z range -------------------------------------------------------
    z_range = get_tv_z_range(tss_data, tv_label) if tss_data is not None else None
    zeros3  = np.zeros(bg_data.shape, dtype=bool)

    if z_range is not None and spineps_data is not None:
        z_min, z_max = z_range
        # Pick representative axial z for display (middle of TV range)
        z_tv = (z_min + z_max) // 2
        tp_left_3d  = isolate_tp_at_tv(spineps_data, TP_LEFT_LABEL,  z_min, z_max)
        tp_right_3d = isolate_tp_at_tv(spineps_data, TP_RIGHT_LABEL, z_min, z_max)
    else:
        z_tv = bg_data.shape[2] // 2
        tp_left_3d = tp_right_3d = zeros3

    sacrum_3d = (tss_data == SACRUM_LABEL) if tss_data is not None else zeros3

    # Contact zones
    def _cz(tp_mask):
        if tp_mask.any() and sacrum_3d.any():
            return build_contact_zone(tp_mask, sacrum_3d, vox_mm, CONTACT_DILATION_MM)
        return zeros3

    cz_left_3d  = _cz(tp_left_3d)
    cz_right_3d = _cz(tp_right_3d)

    # ---- Choose display slices -------------------------------------------
    # Axial: z at middle of TV range (where TPs attach)
    z_ax = z_tv

    # Coronal: y slice with most TP voxels
    tp_combined = tp_left_3d | tp_right_3d
    y_cor = _pick_best_coronal_slice(bg_data, highlight=tp_combined)

    # Sagittal: mid-x or most-TP x
    x_sag = _pick_best_sagittal_slice(bg_data, highlight=tp_combined)

    # Contact zone axial: z slice with most contact voxels
    cz_combined = cz_left_3d | cz_right_3d
    z_cz = _pick_best_axial_slice(bg_data, highlight=cz_combined) if cz_combined.any() else z_ax

    def ax_sl(vol, z=None):
        return _axial(vol, z if z is not None else z_ax)

    def cor_sl(vol, y=None):
        return _coronal(vol, y if y is not None else y_cor)

    def sag_sl(vol, x=None):
        return _sagittal(vol, x if x is not None else x_sag)

    # ---- Assemble figure -------------------------------------------------
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
    _panel_axial_t2w(axes[0, 0], ax_sl(bg_data, z_ax), z_ax)

    if spineps_data is not None:
        _panel_axial_tps(
            axes[0, 1],
            ax_sl(bg_data, z_ax),
            ax_sl(tp_left_3d,  z_ax),
            ax_sl(tp_right_3d, z_ax),
            ax_sl(sacrum_3d,   z_ax),
        )
    else:
        _unavailable(axes[0, 1], 'SPINEPS seg not available')

    if tss_data is not None:
        _panel_tss_axial(axes[0, 2], ax_sl(bg_data, z_ax), ax_sl(tss_data, z_ax))
    else:
        _unavailable(axes[0, 2], 'TSS seg not available')

    # Row 1: Uncertainty + contact zone
    if spineps_unc is not None:
        _panel_uncertainty(axes[1, 0], ax_sl(spineps_unc, z_ax),
                           'SPINEPS Uncertainty (axial)\n(Normalised Shannon Entropy)')
    else:
        _unavailable(axes[1, 0], 'SPINEPS uncertainty')

    if tss_unc is not None:
        _panel_uncertainty(axes[1, 1], ax_sl(tss_unc, z_ax),
                           'TSS Uncertainty (axial)\n(Normalised Shannon Entropy)')
    else:
        _unavailable(axes[1, 1], 'TSS uncertainty')

    _panel_contact_zone_axial(
        axes[1, 2],
        ax_sl(bg_data, z_cz),
        ax_sl(cz_left_3d,  z_cz),
        ax_sl(cz_right_3d, z_cz),
        ax_sl(tp_left_3d,  z_cz),
        ax_sl(tp_right_3d, z_cz),
        ax_sl(sacrum_3d,   z_cz),
    )

    # Row 2: Sagittal preview + coronal TPs + summary
    if sag_preview is not None:
        _panel_sagittal_preview(axes[2, 0], sag_sl(sag_preview, x_sag), x_sag)
    elif bg_data is not None:
        _panel_sagittal_preview(axes[2, 0], sag_sl(bg_data, x_sag), x_sag)
    else:
        _unavailable(axes[2, 0], 'Sagittal preview')

    if spineps_data is not None:
        _panel_coronal_tps(
            axes[2, 1],
            cor_sl(bg_data, y_cor),
            cor_sl(tp_left_3d,  y_cor),
            cor_sl(tp_right_3d, y_cor),
            cor_sl(sacrum_3d,   y_cor),
            tv_name, y_cor,
        )
    else:
        _unavailable(axes[2, 1], 'SPINEPS seg not available')

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
    parser = argparse.ArgumentParser(description='LSTV Overlay Visualizer (registered)')
    parser.add_argument('--registered_dir', required=True,
                        help='Output of 03b_register_to_axial.py')
    parser.add_argument('--nifti_dir',      required=True,
                        help='NIfTI directory (for axial T2w background)')
    parser.add_argument('--output_dir',     required=True)
    parser.add_argument('--study_id',       default=None)
    parser.add_argument('--lstv_json',      default=None,
                        help='lstv_results.json from 04_detect_lstv.py')
    args = parser.parse_args()

    registered_dir = Path(args.registered_dir)
    nifti_dir      = Path(args.nifti_dir)
    output_dir     = Path(args.output_dir)

    results_by_id = {}
    if args.lstv_json:
        json_path = Path(args.lstv_json)
        if json_path.exists():
            with open(json_path) as f:
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
                output_dir     = output_dir,
                result         = results_by_id.get(sid),
            )
        except Exception as e:
            logger.error(f"  [{sid}] Failed: {e}")
            logger.debug(traceback.format_exc())
            errors += 1

    logger.info(
        f"Done. {len(study_ids)-errors}/{len(study_ids)} PNGs -> {output_dir}"
    )


if __name__ == '__main__':
    main()
