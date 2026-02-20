#!/usr/bin/env python3
"""
05_visualize_overlay.py — LSTV Overlay Visualizer v11 (SPINEPS-only)
=====================================================================
Produces a 3×3 panel overlay image per study using SPINEPS segmentation
and VERIDAH vertebra labels only. No registration step, no TotalSpineSeg.
All geometry computed in native sagittal T2w space.

VERIDAH instance label scheme (seg-vert_msk.nii.gz)
----------------------------------------------------
  20=L1  21=L2  22=L3  23=L4  24=L5  25=L6  (26=Sacrum instance, if present)

SPINEPS spine/subreg label scheme (seg-spine_msk.nii.gz)
---------------------------------------------------------
  43=Costal_Process_Left  44=Costal_Process_Right  26=Sacrum

NOTE: SPINEPS does NOT produce a separate seg-subreg file.
      seg-spine_msk IS the subregion mask. Both 'subreg' and 'semantic'
      keys in find_spineps_files() point to the same seg-spine file.

Layout
------
  [0,0]  Left  TP sagittal — max craniocaudal height  (Type I check)
  [0,1]  Right TP sagittal — max craniocaudal height  (Type I check)
  [0,2]  Sagittal at TP blob midpoint — VERIDAH labels + TPs

  [1,0]  Left  TP sagittal at min-dist slice  (Type II/III) + gap ruler
  [1,1]  Right TP sagittal at min-dist slice  (Type II/III) + gap ruler
  [1,2]  Sagittal at min-dist z — VERIDAH + dilated TP + sacrum

  [2,0]  Midline sagittal — VERIDAH level confirmation
  [2,1]  Midline sagittal — all VERIDAH + all SPINEPS masks
  [2,2]  Classification summary text
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
from scipy.ndimage import binary_dilation, distance_transform_edt, label as cc_label

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

TP_HEIGHT_MM    = 19.0
CONTACT_DIST_MM = 2.0
TP_LEFT_LABEL   = 43
TP_RIGHT_LABEL  = 44
SACRUM_LABEL    = 26
L5_LABEL        = 24   # VERIDAH
L6_LABEL        = 25   # VERIDAH

VERIDAH_LABEL_COLORS = {
    20: ([0.15, 0.35, 0.75], 'L1'),
    21: ([0.20, 0.45, 0.80], 'L2'),
    22: ([0.25, 0.55, 0.85], 'L3'),
    23: ([0.30, 0.65, 0.90], 'L4'),
    24: ([0.40, 0.80, 1.00], 'L5'),
    25: ([0.00, 0.25, 0.65], 'L6'),
    26: ([1.00, 0.55, 0.00], 'Sacrum'),
}
SPINEPS_TP_COLORS = {
    TP_LEFT_LABEL:  ([1.00, 0.10, 0.10], 'Left TP'),
    TP_RIGHT_LABEL: ([0.00, 0.80, 1.00], 'Right TP'),
}
SACRUM_COLOR = [1.00, 0.55, 0.00]
DISPLAY_DILATION_VOXELS = 2
LINE_TV      = 'cyan'
LINE_MINDIST = '#FF8C00'


# ============================================================================
# NIfTI / ARRAY HELPERS
# ============================================================================

def load_canonical(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    nii  = nib.load(str(path))
    nii  = nib.as_closest_canonical(nii)
    data = nii.get_fdata()
    if data.ndim == 4:
        data = data[..., 0]
    return data, nii


def voxel_size_mm(nii: nib.Nifti1Image) -> np.ndarray:
    return np.abs(np.array(nii.header.get_zooms()[:3], dtype=float))


def _sag_sl(vol: Optional[np.ndarray], x: int) -> np.ndarray:
    if vol is None:
        return np.zeros((1, 1))
    return vol[min(x, vol.shape[0] - 1), :, :]


def norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def overlay_mask(ax, mask2d: np.ndarray, color_rgb, alpha: float = 0.65):
    if not mask2d.any():
        return
    rgba = np.zeros((*mask2d.shape, 4), dtype=float)
    rgba[mask2d] = [*color_rgb, alpha]
    ax.imshow(rgba.transpose(1, 0, 2), origin='lower')


def dilate_for_display(mask: np.ndarray, voxels: int = 2) -> np.ndarray:
    if not mask.any() or voxels < 1:
        return mask
    struct = np.ones((voxels * 2 + 1,) * mask.ndim, dtype=bool)
    return binary_dilation(mask, structure=struct)


def largest_cc_2d(mask2d: np.ndarray) -> np.ndarray:
    if not mask2d.any():
        return np.zeros_like(mask2d, dtype=bool)
    labeled, n = cc_label(mask2d)
    if n == 0:
        return np.zeros_like(mask2d, dtype=bool)
    sizes  = [(labeled == i).sum() for i in range(1, n + 1)]
    return (labeled == (int(np.argmax(sizes)) + 1))


def _unavailable(ax, label: str):
    ax.set_facecolor('#0d0d1a')
    ax.text(0.5, 0.5, f'{label}\nnot available',
            ha='center', va='center', color='#888888', fontsize=10,
            transform=ax.transAxes)
    ax.axis('off')


def _hline(ax, z: Optional[float], color: str, label: str = ''):
    if z is None:
        return
    ax.axhline(y=float(z), color=color, linewidth=1.6, linestyle='--', alpha=0.95)
    if label:
        ax.text(3, float(z) + 1, label, color=color, fontsize=7,
                va='bottom', fontweight='bold')


# ============================================================================
# MASK OPERATIONS
# ============================================================================

def get_tv_z_range(vert_data: np.ndarray,
                   tv_label: int) -> Optional[Tuple[int, int]]:
    mask = vert_data == tv_label
    if not mask.any():
        return None
    z = np.where(mask)[2]
    return int(z.min()), int(z.max())


def isolate_tp_at_tv(subreg_data: np.ndarray,
                     tp_label: int,
                     z_min: int, z_max: int) -> np.ndarray:
    tp_full = subreg_data == tp_label
    iso     = np.zeros_like(tp_full)
    z_lo    = max(z_min, 0)
    z_hi    = min(z_max, subreg_data.shape[2] - 1)
    iso[:, :, z_lo:z_hi + 1] = tp_full[:, :, z_lo:z_hi + 1]
    return iso


def inferiormost_tp_cc(tp_mask3d: np.ndarray,
                       sacrum_mask3d: Optional[np.ndarray]) -> np.ndarray:
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
        comp     = (labeled == i)
        z_coords = np.where(comp)[2]
        cc_info.append((float(z_coords.mean()), int(z_coords.max()), comp))
    cc_info.sort(key=lambda t: t[0])

    if sac_z_min is not None:
        candidates = [(zc, zm, c) for zc, zm, c in cc_info if zm < sac_z_min]
        if candidates:
            return candidates[0][2].astype(bool)
    return cc_info[0][2].astype(bool)


def min_dist_3d(mask_a: np.ndarray, mask_b: np.ndarray,
                vox_mm: np.ndarray) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
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
        coords    = np.array(np.where(sub))
        coords[2] += z_lo
    else:
        coords = np.array(np.where(mask_b))
    phys_a = vox_a * vox_mm
    phys_b = coords.T * vox_mm
    d2     = ((phys_b - phys_a) ** 2).sum(axis=1)
    vox_b  = coords[:, int(np.argmin(d2))]
    return dist_mm, vox_a, vox_b


def best_x_for_tp_height(tp_l5_3d: np.ndarray,
                          vox_z_mm: float) -> Tuple[int, float]:
    if not tp_l5_3d.any():
        return tp_l5_3d.shape[0] // 2, 0.0
    best_x, best_span = tp_l5_3d.shape[0] // 2, 0.0
    for x in range(tp_l5_3d.shape[0]):
        col = tp_l5_3d[x]
        if not col.any():
            continue
        zc = np.where(col.any(axis=0))[0]
        if zc.size < 2:
            continue
        span = (zc.max() - zc.min()) * vox_z_mm
        if span > best_span:
            best_span = span
            best_x    = x
    return best_x, best_span


def tp_blob_z_midpoint(tp_3d: np.ndarray) -> int:
    if not tp_3d.any():
        return tp_3d.shape[2] // 2
    zc = np.where(tp_3d)[2]
    return int((zc.min() + zc.max()) // 2)


# ============================================================================
# RULERS
# ============================================================================

def _draw_height_ruler(ax, mask2d: np.ndarray, vox_z_mm: float,
                        color: str = 'yellow') -> float:
    lcc = largest_cc_2d(mask2d)
    if not lcc.any():
        return 0.0
    zc = np.where(lcc.any(axis=0))[0]
    if zc.size < 2:
        return 0.0
    z_lo, z_hi = int(zc.min()), int(zc.max())
    span_mm    = (z_hi - z_lo) * vox_z_mm
    mid_z      = zc[len(zc) // 2]
    col_at     = lcc[:, mid_z]
    x_mid      = int(np.where(col_at)[0].mean()) if col_at.any() else lcc.shape[0] // 2
    tick       = max(3, int(lcc.shape[0] * 0.025))
    ax.plot([x_mid, x_mid], [z_lo, z_hi], color=color, lw=1.8, alpha=0.95)
    for z_end in (z_lo, z_hi):
        ax.plot([x_mid - tick, x_mid + tick], [z_end, z_end],
                color=color, lw=1.8, alpha=0.95)
    ax.text(x_mid + tick + 2, (z_lo + z_hi) / 2,
            f'{span_mm:.1f} mm', color=color, fontsize=8,
            va='center', fontweight='bold')
    return span_mm


def _draw_gap_ruler(ax, tp2d: np.ndarray, sac2d: np.ndarray,
                    vox_z_mm: float, color: str = '#FF8C00') -> float:
    tp_lcc  = largest_cc_2d(tp2d)
    sac_lcc = largest_cc_2d(sac2d)
    if not tp_lcc.any() or not sac_lcc.any():
        return float('inf')
    tp_zc  = np.where(tp_lcc.any(axis=0))[0]
    sac_zc = np.where(sac_lcc.any(axis=0))[0]
    if tp_zc.size == 0 or sac_zc.size == 0:
        return float('inf')
    z_tp_inf  = int(tp_zc.min())
    z_sac_sup = int(sac_zc.max())
    gap_mm    = (z_tp_inf - z_sac_sup) * vox_z_mm
    tp_xc     = int(np.where(tp_lcc.any(axis=1))[0].mean())
    sac_xc    = int(np.where(sac_lcc.any(axis=1))[0].mean())
    x_ruler   = (tp_xc + sac_xc) // 2
    tick      = max(3, int(tp2d.shape[0] * 0.025))
    if z_sac_sup < z_tp_inf:
        ax.plot([x_ruler, x_ruler], [z_sac_sup, z_tp_inf],
                color=color, lw=1.8, alpha=0.95)
        for z_end in (z_sac_sup, z_tp_inf):
            ax.plot([x_ruler - tick, x_ruler + tick], [z_end, z_end],
                    color=color, lw=1.8, alpha=0.95)
        lbl = f'{gap_mm:.1f} mm gap'
    else:
        z_mid = (z_tp_inf + z_sac_sup) // 2
        ax.plot([x_ruler - tick, x_ruler + tick], [z_mid, z_mid],
                color=color, lw=2.0, alpha=0.95)
        lbl = 'overlap'
    ax.text(x_ruler + tick + 2, (z_sac_sup + z_tp_inf) / 2,
            lbl, color=color, fontsize=8, va='center', fontweight='bold')
    return gap_mm


# ============================================================================
# PANEL FUNCTIONS
# ============================================================================

def _panel_height(ax, sag_img, tp_l5, tp_other, subreg_data,
                  side_name, x_idx, span_mm, vox_z_mm, z_ref, tv_name):
    """Row 0 — Type I craniocaudal height check."""
    color_this  = [1.00, 0.10, 0.10] if side_name == 'Left' else [0.00, 0.80, 1.00]
    color_other = [0.00, 0.80, 1.00] if side_name == 'Left' else [1.00, 0.10, 0.10]
    ax.imshow(norm(_sag_sl(sag_img, x_idx)).T, cmap='gray', origin='lower', alpha=0.80)
    if tp_other is not None and tp_other.any():
        overlay_mask(ax, _sag_sl(tp_other, x_idx), color_other, 0.22)
    if tp_l5 is not None and tp_l5.any():
        this_sl = _sag_sl(tp_l5, x_idx)
        overlay_mask(ax, this_sl, color_this, 0.85)
        _draw_height_ruler(ax, this_sl, vox_z_mm, color='yellow')
    if subreg_data is not None:
        overlay_mask(ax, _sag_sl(subreg_data == SACRUM_LABEL, x_idx), SACRUM_COLOR, 0.45)
    _hline(ax, z_ref, LINE_TV, 'ref →')
    ax.legend(handles=[
        mpatches.Patch(color=color_this,   label=f'{side_name} TP'),
        mpatches.Patch(color=color_other,  label=f'{"R" if side_name=="Left" else "L"} TP'),
        mpatches.Patch(color=SACRUM_COLOR, label='Sacrum'),
    ], loc='lower right', fontsize=6, framealpha=0.55)
    flag = '✓' if span_mm < TP_HEIGHT_MM else f'✗ ≥{TP_HEIGHT_MM:.0f}mm → Type I'
    ax.set_title(f'Type I — {side_name} TP  x={x_idx}\n'
                 f'Height: {span_mm:.1f} mm  {flag}', fontsize=10, color='white')
    ax.axis('off')


def _panel_sag_labels(ax, sag_img, vert_data, subreg_data,
                       tp_left, tp_right, x_idx, z_ref, subtitle=''):
    """Generic VERIDAH labels + TP overlay on a sagittal slice."""
    ax.imshow(norm(_sag_sl(sag_img, x_idx)).T, cmap='gray', origin='lower', alpha=0.78)
    patches = []
    if vert_data is not None:
        for label, (color, name) in VERIDAH_LABEL_COLORS.items():
            m = _sag_sl(vert_data == label, x_idx)
            if m.any():
                overlay_mask(ax, m, color, 0.40)
                patches.append(mpatches.Patch(color=color, label=name))
    if subreg_data is not None:
        sac_sl = _sag_sl(subreg_data == SACRUM_LABEL, x_idx)
        if sac_sl.any() and not any(p.get_label() == 'Sacrum' for p in patches):
            overlay_mask(ax, sac_sl, SACRUM_COLOR, 0.45)
            patches.append(mpatches.Patch(color=SACRUM_COLOR, label='Sacrum'))
    for mask, color, name in [
        (tp_left,  [1.00, 0.10, 0.10], 'Left TP'),
        (tp_right, [0.00, 0.80, 1.00], 'Right TP'),
    ]:
        if mask is not None and mask.any():
            sl = dilate_for_display(_sag_sl(mask, x_idx), DISPLAY_DILATION_VOXELS)
            overlay_mask(ax, sl, color, 0.80)
            patches.append(mpatches.Patch(color=color, label=name))
    if patches:
        ax.legend(handles=patches, loc='lower right', fontsize=6, framealpha=0.55)
    _hline(ax, z_ref, LINE_TV)
    title = f'VERIDAH Labels + TPs  x={x_idx}'
    if subtitle:
        title += f'\n{subtitle}'
    ax.set_title(title, fontsize=10, color='white')
    ax.axis('off')


def _panel_proximity(ax, sag_img, tp_l5, tp_other, subreg_data,
                     side_name, x_idx, dist_mm, vox_z_mm, z_ref, tv_name):
    """Row 1 — Type II/III proximity check."""
    color_this  = [1.00, 0.10, 0.10] if side_name == 'Left' else [0.00, 0.80, 1.00]
    color_other = [0.00, 0.80, 1.00] if side_name == 'Left' else [1.00, 0.10, 0.10]
    ax.imshow(norm(_sag_sl(sag_img, x_idx)).T, cmap='gray', origin='lower', alpha=0.80)
    this_sl = (_sag_sl(tp_l5, x_idx) if (tp_l5 is not None and tp_l5.any())
               else np.zeros((1, 1), bool))
    sac_sl  = (_sag_sl(subreg_data == SACRUM_LABEL, x_idx) if subreg_data is not None
               else np.zeros((1, 1), bool))
    if tp_other is not None and tp_other.any():
        overlay_mask(ax, _sag_sl(tp_other, x_idx), color_other, 0.22)
    if tp_l5 is not None and tp_l5.any():
        overlay_mask(ax, this_sl, color_this, 0.85)
    if subreg_data is not None:
        overlay_mask(ax, sac_sl, SACRUM_COLOR, 0.60)
    _draw_gap_ruler(ax, this_sl, sac_sl, vox_z_mm, color=LINE_MINDIST)
    _hline(ax, z_ref, LINE_MINDIST, 'min-dist →')
    ax.legend(handles=[
        mpatches.Patch(color=color_this,   label=f'{side_name} TP'),
        mpatches.Patch(color=color_other,  label=f'{"R" if side_name=="Left" else "L"} TP'),
        mpatches.Patch(color=SACRUM_COLOR, label='Sacrum'),
    ], loc='lower right', fontsize=6, framealpha=0.55)
    dist_str = f'{dist_mm:.1f} mm' if np.isfinite(dist_mm) else 'N/A'
    contact  = np.isfinite(dist_mm) and dist_mm <= CONTACT_DIST_MM
    flag     = '✗ contact → II/III' if contact else '✓ no contact'
    ax.set_title(f'Type II/III — {side_name} TP  x={x_idx}\n'
                 f'TP–Sacrum: {dist_str}  {flag}', fontsize=10, color='white')
    ax.axis('off')


def _panel_level_confirm(ax, sag_img, vert_data, subreg_data, x_mid, tv_name, z_tv):
    """[2,0] VERIDAH level confirmation, L5/L6 highlighted."""
    ax.imshow(norm(_sag_sl(sag_img, x_mid)).T, cmap='gray', origin='lower', alpha=0.80)
    patches = []
    if vert_data is not None:
        for label, (color, name) in VERIDAH_LABEL_COLORS.items():
            m = _sag_sl(vert_data == label, x_mid)
            if not m.any():
                continue
            alpha = 0.70 if label in (L5_LABEL, L6_LABEL, SACRUM_LABEL) else 0.28
            overlay_mask(ax, m, color, alpha)
            patches.append(mpatches.Patch(color=color, label=name))
    if subreg_data is not None:
        sac_sl = _sag_sl(subreg_data == SACRUM_LABEL, x_mid)
        if sac_sl.any() and not any(p.get_label() == 'Sacrum' for p in patches):
            overlay_mask(ax, sac_sl, SACRUM_COLOR, 0.50)
            patches.append(mpatches.Patch(color=SACRUM_COLOR, label='Sacrum'))
    if patches:
        ax.legend(handles=patches, loc='lower right', fontsize=6, framealpha=0.55)
    _hline(ax, z_tv, LINE_TV, 'TV mid')
    ax.set_title(f'Level Confirm  (TV={tv_name})\n'
                 'L5/L6 + Sacrum highlighted', fontsize=10, color='white')
    ax.axis('off')


def _panel_all_masks(ax, sag_img, vert_data, subreg_data, x_mid):
    """[2,1] All VERIDAH + SPINEPS masks on midline sagittal."""
    ax.imshow(norm(_sag_sl(sag_img, x_mid)).T, cmap='gray', origin='lower', alpha=0.75)
    patches = []
    if vert_data is not None:
        for label, (color, name) in VERIDAH_LABEL_COLORS.items():
            m = _sag_sl(vert_data == label, x_mid)
            if m.any():
                overlay_mask(ax, m, color, 0.35)
                patches.append(mpatches.Patch(color=color, label=f'VERIDAH {name}'))
    if subreg_data is not None:
        for label, (color, name) in SPINEPS_TP_COLORS.items():
            m = _sag_sl(subreg_data == label, x_mid)
            if m.any():
                overlay_mask(ax, dilate_for_display(m, DISPLAY_DILATION_VOXELS), color, 0.75)
                patches.append(mpatches.Patch(color=color, label=f'SPINEPS {name}'))
        sac_sl = _sag_sl(subreg_data == SACRUM_LABEL, x_mid)
        if sac_sl.any():
            overlay_mask(ax, sac_sl, SACRUM_COLOR, 0.45)
            patches.append(mpatches.Patch(color=SACRUM_COLOR, label='SPINEPS Sacrum'))
    if patches:
        ax.legend(handles=patches, loc='lower right', fontsize=6, framealpha=0.55)
    ax.set_title('All Masks — Sagittal midline\nVERIDAH labels + SPINEPS TPs & Sacrum',
                 fontsize=10, color='white')
    ax.axis('off')


def _panel_summary(ax, study_id, result, tv_name, span_L, span_R, dist_L, dist_R):
    ax.axis('off')
    ax.set_facecolor('#1a1a2e')

    def _d(v):
        return f'{v:.1f}' if np.isfinite(v) else 'N/A'

    lines = [
        f'Study  : {study_id}',
        f'TV     : {tv_name}',
        f'Method : SPINEPS + VERIDAH (sagittal space)',
        f'Subreg : seg-spine_msk (SPINEPS has no separate subreg file)',
        '',
        '─── Measured ──────────────────────────────',
        f'  Left  TP height : {_d(span_L):>6} mm  (thresh {TP_HEIGHT_MM:.0f} mm)',
        f'  Right TP height : {_d(span_R):>6} mm  (thresh {TP_HEIGHT_MM:.0f} mm)',
        f'  Left  TP–Sacrum : {_d(dist_L):>6} mm  (thresh {CONTACT_DIST_MM:.0f} mm)',
        f'  Right TP–Sacrum : {_d(dist_R):>6} mm  (thresh {CONTACT_DIST_MM:.0f} mm)',
        '',
    ]
    if result is None:
        lines += ['─── Classifier output ─────────────────────',
                  '  (run 04_detect_lstv.py + --lstv_json)']
    else:
        ct       = result.get('castellvi_type') or 'None'
        detected = result.get('lstv_detected', False)
        lines += [
            '─── Classifier output ─────────────────────',
            f'  Castellvi : {ct}',
            f'  LSTV      : {"YES  !" if detected else "No"}',
            '',
        ]
        for side in ('left', 'right'):
            sd = result.get(side) or {}
            if not sd:
                continue
            lines += [
                f'  {"Left " if side=="left" else "Right"} → {sd.get("classification","?")}',
                f'    height : {sd.get("tp_height_mm", 0.0):.1f} mm',
                f'    dist   : {_d(sd.get("dist_mm", float("inf")))} mm',
            ]
            if sd.get('note'):
                lines.append(f'    NOTE: {sd["note"]}')
            lines.append('')
        if result.get('errors'):
            lines += ['  Errors:'] + [f'    {e}' for e in result['errors']]

    ax.text(0.05, 0.97, '\n'.join(lines),
            transform=ax.transAxes, va='top', ha='left',
            fontsize=10.5, family='monospace', color='white', linespacing=1.45)
    ax.set_title('Classification Summary', fontsize=13, color='white')


# ============================================================================
# FILE DISCOVERY
# ============================================================================

def find_t2w_sag(nifti_dir: Path, study_id: str) -> Optional[Path]:
    study_dir = nifti_dir / study_id
    if not study_dir.exists():
        return None
    for series_dir in sorted(study_dir.iterdir()):
        p = series_dir / f"sub-{study_id}_acq-sag_T2w.nii.gz"
        if p.exists():
            return p
    return None


def find_spineps_files(spineps_dir: Path, study_id: str) -> dict:
    seg_dir = spineps_dir / 'segmentations' / study_id
    # SPINEPS does NOT produce a separate seg-subreg file.
    # seg-spine_msk IS the subregion mask — both keys point to the same file.
    spine_mask = seg_dir / f"{study_id}_seg-spine_msk.nii.gz"
    return {
        'subreg':   spine_mask,   # ← seg-spine IS the subreg mask
        'instance': seg_dir / f"{study_id}_seg-vert_msk.nii.gz",
        'semantic': spine_mask,   # ← same file
    }


# ============================================================================
# CORE VISUALIZER
# ============================================================================

def visualize_study(study_id: str,
                    spineps_dir: Path,
                    nifti_dir: Path,
                    output_dir: Path,
                    result: Optional[dict] = None) -> Optional[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{study_id}_lstv_overlay.png"

    files = find_spineps_files(spineps_dir, study_id)

    def _load(key, label):
        p = files.get(key)
        if p and p.exists():
            try:
                return load_canonical(p)
            except Exception as e:
                logger.warning(f"  [{study_id}] Cannot load {label}: {e}")
        elif p:
            logger.warning(f"  [{study_id}] Missing: {p.name}")
        return None, None

    subreg_data, subreg_nii = _load('subreg',   'spine/subreg mask')
    vert_data,   _          = _load('instance', 'VERIDAH instance mask')

    sag_bg, sag_nii = None, None
    sag_path = find_t2w_sag(nifti_dir, study_id)
    if sag_path:
        try:
            sag_bg, sag_nii = load_canonical(sag_path)
        except Exception as e:
            logger.warning(f"  [{study_id}] Cannot load sag T2w: {e}")

    if subreg_data is None:
        logger.error(f"  [{study_id}] No spine/subreg mask — skipping")
        return None
    if vert_data is None:
        logger.error(f"  [{study_id}] No VERIDAH instance mask — skipping")
        return None
    if sag_bg is None:
        sag_bg, sag_nii = subreg_data.astype(float), subreg_nii

    subreg_data = subreg_data.astype(int)
    vert_data   = vert_data.astype(int)
    vox_mm      = voxel_size_mm(subreg_nii)

    # Pick TV: L6 > L5 > inferiormost lumbar label (matches 04_detect_lstv.py logic)
    VERIDAH_NAMES = {20:'L1',21:'L2',22:'L3',23:'L4',24:'L5',25:'L6'}
    LUMBAR_LABELS = [25, 24, 23, 22, 21, 20]
    unique_vert   = set(np.unique(vert_data).tolist())
    tv_label, tv_name = None, None
    for candidate in LUMBAR_LABELS:
        if candidate in unique_vert:
            tv_label = candidate
            tv_name  = VERIDAH_NAMES.get(candidate, str(candidate))
            break

    if tv_label is None:
        logger.error(f"  [{study_id}] No lumbar labels found — skipping")
        return None

    z_range = get_tv_z_range(vert_data, tv_label)
    if z_range is None:
        logger.error(f"  [{study_id}] TV label {tv_name} not found in mask — skipping")
        return None
    z_min_tv, z_max_tv = z_range
    z_tv_mid = (z_min_tv + z_max_tv) // 2

    sacrum_mask = (subreg_data == SACRUM_LABEL)

    def _isolate(tp_label):
        tp_at_tv = isolate_tp_at_tv(subreg_data, tp_label, z_min_tv, z_max_tv)
        return inferiormost_tp_cc(tp_at_tv, sacrum_mask if sacrum_mask.any() else None)

    tp_l5_left  = _isolate(TP_LEFT_LABEL)
    tp_l5_right = _isolate(TP_RIGHT_LABEL)

    dist_L, tp_vox_L, _ = min_dist_3d(tp_l5_left,  sacrum_mask, vox_mm)
    dist_R, tp_vox_R, _ = min_dist_3d(tp_l5_right, sacrum_mask, vox_mm)

    x_left,  span_L = best_x_for_tp_height(tp_l5_left,  vox_mm[2])
    x_right, span_R = best_x_for_tp_height(tp_l5_right, vox_mm[2])

    z_mid_L = tp_blob_z_midpoint(tp_l5_left)  if tp_l5_left.any()  else z_tv_mid
    z_mid_R = tp_blob_z_midpoint(tp_l5_right) if tp_l5_right.any() else z_tv_mid
    z_row0  = (z_mid_L + z_mid_R) // 2

    z_md_L  = int(tp_vox_L[2]) if tp_vox_L is not None else z_tv_mid
    z_md_R  = int(tp_vox_R[2]) if tp_vox_R is not None else z_tv_mid
    z_md    = z_md_L if dist_L <= dist_R else z_md_R

    x_mid = sag_bg.shape[0] // 2

    logger.info(f"  [{study_id}] TV={tv_name} z_tv={z_tv_mid} "
                f"dist_L={dist_L:.1f}mm dist_R={dist_R:.1f}mm "
                f"span_L={span_L:.1f}mm span_R={span_R:.1f}mm")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(21, 21))
    fig.patch.set_facecolor('#0d0d1a')
    for a in axes.flat:
        a.set_facecolor('#0d0d1a')

    castellvi = result.get('castellvi_type') if result else None
    fig.suptitle(
        f"Study {study_id}   |   Castellvi: {castellvi or 'N/A'}   |   "
        f"TV: {tv_name}   |   SPINEPS + VERIDAH",
        fontsize=15, color='white', y=0.999,
    )

    # ROW 0 — Type I
    _panel_height(axes[0, 0], sag_bg, tp_l5_left,  tp_l5_right, subreg_data,
                  'Left',  x_left,  span_L, vox_mm[2], z_row0, tv_name)
    _panel_height(axes[0, 1], sag_bg, tp_l5_right, tp_l5_left,  subreg_data,
                  'Right', x_right, span_R, vox_mm[2], z_row0, tv_name)
    _panel_sag_labels(axes[0, 2], sag_bg, vert_data, subreg_data,
                      tp_l5_left, tp_l5_right, x_left, z_row0,
                      subtitle=f'TP blob midpoint  z={z_row0}')

    # ROW 1 — Type II/III
    _panel_proximity(axes[1, 0], sag_bg, tp_l5_left,  tp_l5_right, subreg_data,
                     'Left',  x_left,  dist_L, vox_mm[2], z_md, tv_name)
    _panel_proximity(axes[1, 1], sag_bg, tp_l5_right, tp_l5_left,  subreg_data,
                     'Right', x_right, dist_R, vox_mm[2], z_md, tv_name)
    dist_label = (f'L={dist_L:.1f}mm  R={dist_R:.1f}mm'
                  if (np.isfinite(dist_L) or np.isfinite(dist_R)) else 'no masks')
    _panel_sag_labels(axes[1, 2], sag_bg, vert_data, subreg_data,
                      tp_l5_left, tp_l5_right, x_left, z_md,
                      subtitle=f'Min-dist slice [{dist_label}]  [Type II/III]')

    # ROW 2 — Context
    _panel_level_confirm(axes[2, 0], sag_bg, vert_data, subreg_data,
                         x_mid, tv_name, z_tv_mid)
    _panel_all_masks(axes[2, 1], sag_bg, vert_data, subreg_data, x_mid)
    _panel_summary(axes[2, 2], study_id, result, tv_name,
                   span_L, span_R, dist_L, dist_R)

    plt.tight_layout(rect=[0, 0, 1, 0.998])
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"  [{study_id}] OK -> {out_path}")
    return out_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='LSTV Overlay Visualizer v11 (SPINEPS-only)')
    parser.add_argument('--spineps_dir', required=True)
    parser.add_argument('--nifti_dir',   required=True)
    parser.add_argument('--output_dir',  required=True)
    parser.add_argument('--study_id',    default=None)
    parser.add_argument('--lstv_json',   default=None)
    args = parser.parse_args()

    spineps_dir = Path(args.spineps_dir)
    nifti_dir   = Path(args.nifti_dir)
    output_dir  = Path(args.output_dir)

    results_by_id = {}
    if args.lstv_json:
        p = Path(args.lstv_json)
        if p.exists():
            with open(p) as f:
                results_by_id = {r['study_id']: r for r in json.load(f)}
            logger.info(f"Loaded {len(results_by_id)} detection results")

    seg_root = spineps_dir / 'segmentations'
    if args.study_id:
        study_ids = [args.study_id]
    else:
        study_ids = sorted(d.name for d in seg_root.iterdir() if d.is_dir())
        logger.info(f"Batch mode: {len(study_ids)} studies")

    errors = 0
    for sid in study_ids:
        try:
            visualize_study(
                study_id    = sid,
                spineps_dir = spineps_dir,
                nifti_dir   = nifti_dir,
                output_dir  = output_dir,
                result      = results_by_id.get(sid),
            )
        except Exception as e:
            logger.error(f"  [{sid}] Failed: {e}")
            logger.debug(traceback.format_exc())
            errors += 1

    logger.info(f"Done. {len(study_ids)-errors}/{len(study_ids)} PNGs -> {output_dir}")


if __name__ == '__main__':
    main()
