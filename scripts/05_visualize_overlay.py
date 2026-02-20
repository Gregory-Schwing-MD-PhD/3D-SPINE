#!/usr/bin/env python3
"""
05_visualize_overlay.py — LSTV Overlay Visualizer v9
=====================================================
Fixes over v8
─────────────────────────────────────────────────────────────────────────────
1. ROW 0 AXIAL SLICE — now taken at the z-midpoint of the TP blob actually
   being measured (largest CC of the L5 TP mask), not mean(z_tv, z_md).
   The reference line in the sagittal panels also reflects this z.

2. ROW 1 AXIAL SLICE — confirmed at z_md_combined (min-dist slice).

3. REFERENCE LINES — simplified to ONE dashed line per sagittal panel:
   - Row 0 panels: single cyan line at the axial z shown in [0,2]
   - Row 1 panels: single orange line at the axial z shown in [1,2]
   No more confusing double lines.

4. SUMMARY TEXT — larger font (10.5 pt instead of 8.2 pt).

Layout (unchanged from v8)
─────────────────────────────────────────────────────────────────────────────
  [0,0]  Left  TP sagittal — max craniocaudal height  (Type I check)
  [0,1]  Right TP sagittal — max craniocaudal height  (Type I check)
  [0,2]  TSS Labels axial at z_tp_mid (midpoint of measured TP blob)

  [1,0]  Left  TP sagittal at z_md_L  (Type II/III check) + gap ruler
  [1,1]  Right TP sagittal at z_md_R  (Type II/III check) + gap ruler
  [1,2]  TSS Labels axial at z_md_combined + dilated TP + sacrum

  [2,0]  Sagittal TSS level confirmation (midline)
  [2,1]  Axial T2w + ALL TSS + ALL SPINEPS masks at TV mid
  [2,2]  Classification summary (measured heights + min dists)
─────────────────────────────────────────────────────────────────────────────
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

TP_HEIGHT_MM     = 19.0
CONTACT_DIST_MM  = 2.0
TP_LEFT_LABEL    = 43
TP_RIGHT_LABEL   = 44
SACRUM_LABEL     = 50
L5_LABEL         = 45
L6_LABEL         = 46

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

# SPINEPS label colours for [2,1] combined panel
SPINEPS_LABEL_COLORS = {
    TP_LEFT_LABEL:  ([1.00, 0.10, 0.10], 'Left TP'),
    TP_RIGHT_LABEL: ([0.00, 0.80, 1.00], 'Right TP'),
    SACRUM_LABEL:   ([1.00, 0.55, 0.00], 'Sacrum'),
}

DISPLAY_DILATION_VOXELS = 2

LINE_TV      = 'cyan'
LINE_MINDIST = '#FF8C00'


# ============================================================================
# NIfTI HELPERS
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
    iso     = np.zeros_like(tp_full)
    z_lo    = max(z_min, 0)
    z_hi    = min(z_max, data.shape[2] - 1)
    iso[:, :, z_lo:z_hi + 1] = tp_full[:, :, z_lo:z_hi + 1]
    return iso


def dilate_for_display(mask: np.ndarray, voxels: int = 2) -> np.ndarray:
    if not mask.any() or voxels < 1:
        return mask
    struct = np.ones((voxels * 2 + 1,) * mask.ndim, dtype=bool)
    return binary_dilation(mask, structure=struct)


# ============================================================================
# CONNECTED COMPONENT HELPERS
# ============================================================================

def largest_cc_2d(mask2d: np.ndarray) -> np.ndarray:
    """Largest CC of a 2-D boolean mask."""
    if not mask2d.any():
        return np.zeros_like(mask2d, dtype=bool)
    labeled, n = cc_label(mask2d)
    if n == 0:
        return np.zeros_like(mask2d, dtype=bool)
    sizes  = [(labeled == i).sum() for i in range(1, n + 1)]
    best_i = int(np.argmax(sizes)) + 1
    return labeled == best_i


def inferiormost_tp_cc(tp_mask3d: np.ndarray,
                        sacrum_mask3d) -> np.ndarray:
    """
    Isolate the L5 TP blob from a 3-D TP label mask containing blobs at
    multiple spinal levels (SPINEPS uses the same label for all TPs).

    1. Find all 3-D CCs.
    2. Exclude any CC whose z_max >= sacrum z_min (overlaps/inferior to sacrum).
    3. Pick the CC with the LOWEST z-centroid among survivors (most caudal = L5).
    4. Fallback: globally lowest-z CC if sacrum filter removes everything.
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

    # Sort by z_centroid ascending (most inferior / lowest z first)
    cc_info.sort(key=lambda t: t[0])

    if sac_z_min is not None:
        candidates = [(zc, zm, c) for zc, zm, c in cc_info if zm < sac_z_min]
        if candidates:
            return candidates[0][2].astype(bool)

    return cc_info[0][2].astype(bool)


def tp_blob_z_midpoint(tp_l5_3d: np.ndarray) -> int:
    """
    Return the z-index at the midpoint of the L5 TP blob's craniocaudal extent.
    Used to pick the correct axial slice for Row 0.
    """
    if not tp_l5_3d.any():
        return tp_l5_3d.shape[2] // 2
    zc = np.where(tp_l5_3d)[2]
    return int((zc.min() + zc.max()) // 2)


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


def _ax_sl(vol: Optional[np.ndarray], z: int) -> np.ndarray:
    if vol is None:
        return np.zeros((1, 1))
    return vol[:, :, min(z, vol.shape[2] - 1)]


def _sag_sl(vol: Optional[np.ndarray], x: int) -> np.ndarray:
    if vol is None:
        return np.zeros((1, 1))
    return vol[min(x, vol.shape[0] - 1), :, :]


def _unavailable(ax, label: str):
    ax.set_facecolor('#0d0d1a')
    ax.text(0.5, 0.5, f'{label}\nnot available',
            ha='center', va='center', color='#888888', fontsize=10,
            transform=ax.transAxes)
    ax.axis('off')


def _hline(ax, z_sag: Optional[float], color: str, label: str = ''):
    """Draw one horizontal reference line in a sagittal panel."""
    if z_sag is None:
        return
    y = float(z_sag)
    ax.axhline(y=y, color=color, linewidth=1.6, linestyle='--', alpha=0.95)
    if label:
        ax.text(3, y + 1, label, color=color, fontsize=7, va='bottom',
                fontweight='bold')


# ============================================================================
# AFFINE CONVERSION
# ============================================================================

def ax_z_to_sag_z(ax_nii:  Optional[nib.Nifti1Image],
                   sag_nii: Optional[nib.Nifti1Image],
                   z_ax: int) -> Optional[float]:
    if ax_nii is None or sag_nii is None:
        return None
    try:
        cx    = ax_nii.shape[0] / 2.0
        cy    = ax_nii.shape[1] / 2.0
        world = nib.affines.apply_affine(ax_nii.affine,
                                          np.array([[cx, cy, z_ax]]))[0]
        vx    = nib.affines.apply_affine(np.linalg.inv(sag_nii.affine),
                                          world[np.newaxis])[0]
        return float(vx[2])
    except Exception as e:
        logger.debug(f"ax_z_to_sag_z: {e}")
        return None


def sag_z_to_ax_z(sag_nii: Optional[nib.Nifti1Image],
                   ax_nii:  Optional[nib.Nifti1Image],
                   z_sag:   int) -> Optional[int]:
    """Convert a sagittal volume z-index to the nearest axial volume z-index."""
    if sag_nii is None or ax_nii is None:
        return None
    try:
        cx    = sag_nii.shape[0] / 2.0
        cy    = sag_nii.shape[1] / 2.0
        world = nib.affines.apply_affine(sag_nii.affine,
                                          np.array([[cx, cy, z_sag]]))[0]
        vx    = nib.affines.apply_affine(np.linalg.inv(ax_nii.affine),
                                          world[np.newaxis])[0]
        z_ax  = int(round(float(vx[2])))
        return max(0, min(z_ax, ax_nii.shape[2] - 1))
    except Exception as e:
        logger.debug(f"sag_z_to_ax_z: {e}")
        return None


# ============================================================================
# SLICE SELECTION: L5 TP isolation + max craniocaudal height  (Type I)
# ============================================================================

def isolate_l5_tp_sag(orig_spineps: Optional[np.ndarray],
                       orig_tss_sag: Optional[np.ndarray],
                       tp_label:     int,
                       tv_label:     int) -> np.ndarray:
    """
    Return a 3-D boolean mask containing ONLY the L5 TP blob in sagittal space.
    Uses inferiormost_tp_cc() — anatomy-driven, not label-range-driven.
    """
    zeros = np.zeros((1, 1, 1), dtype=bool)
    if orig_spineps is None:
        return zeros
    tp_full = (orig_spineps == tp_label).astype(bool)
    if not tp_full.any():
        return tp_full

    # Sacrum in sagittal space
    sacrum_sag = None
    if orig_tss_sag is not None:
        s = (orig_tss_sag == SACRUM_LABEL)
        if s.any():
            sacrum_sag = s

    return inferiormost_tp_cc(tp_full, sacrum_sag)


def best_x_for_tp_height(tp_l5_3d: np.ndarray,
                          vox_z_mm: float) -> Tuple[int, float]:
    """
    Given the already-isolated L5 TP mask (3-D sagittal), find the x-slice
    with maximum craniocaudal z-span.  Returns (best_x, max_span_mm).
    """
    if not tp_l5_3d.any():
        return tp_l5_3d.shape[0] // 2, 0.0

    best_x, best_span = tp_l5_3d.shape[0] // 2, 0.0
    for x in range(tp_l5_3d.shape[0]):
        col = tp_l5_3d[x]           # (Y, Z)
        if not col.any():
            continue
        zc   = np.where(col.any(axis=0))[0]
        if zc.size < 2:
            continue
        span = (zc.max() - zc.min()) * vox_z_mm
        if span > best_span:
            best_span = span
            best_x    = x

    return best_x, best_span


# ============================================================================
# SLICE SELECTION: min distance TP ↔ sacrum  (Type II/III)
# ============================================================================

def min_dist_z_for_tp_sacrum(tp_mask:     np.ndarray,
                               sacrum_mask: np.ndarray,
                               vox_mm:      np.ndarray) -> Tuple[int, float]:
    if not tp_mask.any() or not sacrum_mask.any():
        return 0, float('inf')
    dist  = distance_transform_edt(~sacrum_mask, sampling=vox_mm)
    nz    = tp_mask.shape[2]
    per_z = np.full(nz, np.inf)
    for z in range(nz):
        sl = tp_mask[:, :, z]
        if sl.any():
            per_z[z] = dist[:, :, z][sl].min()
    z_best = int(np.argmin(per_z))
    return z_best, float(per_z[z_best])


# ============================================================================
# TSS AXIAL SLICE HELPER
# ============================================================================

def _tss_axial_slice(ax_bg: np.ndarray,
                      tss:   Optional[np.ndarray],
                      z:     int) -> Optional[np.ndarray]:
    if tss is None:
        return None
    z  = min(z, tss.shape[2] - 1)
    sl = tss[:, :, z]
    bg = ax_bg[:, :, min(z, ax_bg.shape[2] - 1)]
    if sl.shape != bg.shape:
        out = np.zeros(bg.shape, dtype=sl.dtype)
        sy  = min(sl.shape[0], bg.shape[0])
        sx  = min(sl.shape[1], bg.shape[1])
        out[:sy, :sx] = sl[:sy, :sx]
        return out
    return sl


# ============================================================================
# RULER HELPERS
# ============================================================================

def _draw_height_ruler(ax, mask2d: np.ndarray, vox_z_mm: float,
                        color: str = 'yellow') -> float:
    """
    Draw a vertical bracket spanning z_min→z_max of the LARGEST CONNECTED
    COMPONENT of mask2d (shape X×Z).
    Returns measured span in mm (0.0 if mask is empty).
    """
    lcc = largest_cc_2d(mask2d)
    if not lcc.any():
        return 0.0

    zc = np.where(lcc.any(axis=0))[0]
    if zc.size < 2:
        return 0.0
    z_lo, z_hi = int(zc.min()), int(zc.max())
    span_mm    = (z_hi - z_lo) * vox_z_mm

    mid_z   = zc[len(zc) // 2]
    col_at  = lcc[:, mid_z]
    x_mid   = int(np.where(col_at)[0].mean()) if col_at.any() else lcc.shape[0] // 2
    tick    = max(3, int(lcc.shape[0] * 0.025))
    offset  = tick + 2

    ax.plot([x_mid, x_mid], [z_lo, z_hi], color=color, lw=1.8, alpha=0.95)
    for z_end in (z_lo, z_hi):
        ax.plot([x_mid - tick, x_mid + tick], [z_end, z_end],
                color=color, lw=1.8, alpha=0.95)
    ax.text(x_mid + offset, (z_lo + z_hi) / 2,
            f'{span_mm:.1f} mm',
            color=color, fontsize=8, va='center', fontweight='bold')
    return span_mm


def _draw_gap_ruler(ax,
                    tp_mask2d:     np.ndarray,
                    sacrum_mask2d: np.ndarray,
                    vox_z_mm:      float,
                    color:         str = '#FF8C00') -> float:
    """
    Draw a vertical gap ruler from the INFERIOR edge of the TP's largest CC
    to the SUPERIOR edge of the sacrum's largest CC in the same sagittal slice.
    Returns the gap in mm.
    """
    tp_lcc  = largest_cc_2d(tp_mask2d)
    sac_lcc = largest_cc_2d(sacrum_mask2d)

    if not tp_lcc.any() or not sac_lcc.any():
        return float('inf')

    tp_zc  = np.where(tp_lcc.any(axis=0))[0]
    sac_zc = np.where(sac_lcc.any(axis=0))[0]

    if tp_zc.size == 0 or sac_zc.size == 0:
        return float('inf')

    z_tp_inf  = int(tp_zc.min())
    z_sac_sup = int(sac_zc.max())

    gap_mm = (z_tp_inf - z_sac_sup) * vox_z_mm

    tp_xc   = int(np.where(tp_lcc.any(axis=1))[0].mean())
    sac_xc  = int(np.where(sac_lcc.any(axis=1))[0].mean())
    x_ruler = (tp_xc + sac_xc) // 2
    tick    = max(3, int(tp_mask2d.shape[0] * 0.025))

    if z_sac_sup < z_tp_inf:
        ax.plot([x_ruler, x_ruler], [z_sac_sup, z_tp_inf],
                color=color, lw=1.8, alpha=0.95)
        for z_end in (z_sac_sup, z_tp_inf):
            ax.plot([x_ruler - tick, x_ruler + tick], [z_end, z_end],
                    color=color, lw=1.8, alpha=0.95)
        label_mm = f'{gap_mm:.1f} mm gap'
    else:
        z_mid = (z_tp_inf + z_sac_sup) // 2
        ax.plot([x_ruler - tick, x_ruler + tick], [z_mid, z_mid],
                color=color, lw=2.0, alpha=0.95)
        label_mm = 'overlap'

    ax.text(x_ruler + tick + 2, (z_sac_sup + z_tp_inf) / 2,
            label_mm, color=color, fontsize=8, va='center', fontweight='bold')
    return gap_mm


# ============================================================================
# PANEL FUNCTIONS
# ============================================================================

def _panel_sag_tp_height(ax,
                          sag_img:       Optional[np.ndarray],
                          tp_l5_mask:    Optional[np.ndarray],
                          tp_other_mask: Optional[np.ndarray],
                          sag_tss:       Optional[np.ndarray],
                          side_name:     str,
                          x_idx:         int,
                          span_mm:       float,
                          vox_z_mm:      float,
                          z_axial_sag:   Optional[float],   # ONE line: axial z mapped to sag space
                          tv_name:       str):
    """
    Row 0 — Type I check.
    Single cyan reference line at the axial z shown in [0,2] (TP blob midpoint).
    """
    color_this  = [1.00, 0.10, 0.10] if side_name == 'Left' else [0.00, 0.80, 1.00]
    color_other = [0.00, 0.80, 1.00] if side_name == 'Left' else [1.00, 0.10, 0.10]

    ax.imshow(norm(_sag_sl(sag_img, x_idx)).T, cmap='gray', origin='lower', alpha=0.80)

    if tp_other_mask is not None and tp_other_mask.any():
        overlay_mask(ax, _sag_sl(tp_other_mask, x_idx), color_other, 0.22)

    if tp_l5_mask is not None and tp_l5_mask.any():
        this_sl = _sag_sl(tp_l5_mask, x_idx)
        overlay_mask(ax, this_sl, color_this, 0.85)
        _draw_height_ruler(ax, this_sl, vox_z_mm, color='yellow')

    if sag_tss is not None:
        overlay_mask(ax, _sag_sl(sag_tss == SACRUM_LABEL, x_idx),
                     [1.00, 0.55, 0.00], 0.45)

    # ONE reference line — the axial slice shown in [0,2]
    _hline(ax, z_axial_sag, LINE_TV, label='axial slice →')

    ax.legend(handles=[
        mpatches.Patch(color=color_this,         label=f'{side_name} TP'),
        mpatches.Patch(color=color_other,        label=f'{"R" if side_name=="Left" else "L"} TP (faint)'),
        mpatches.Patch(color=[1.00, 0.55, 0.00], label='Sacrum'),
        mpatches.Patch(color=LINE_TV,            label='axial slice'),
    ], loc='lower right', fontsize=6, framealpha=0.55)

    flag = '✓' if span_mm < TP_HEIGHT_MM else '✗ ≥19 mm → Type I'
    ax.set_title(f'Type I check — {side_name} TP  (x={x_idx})\n'
                 f'Height: {span_mm:.1f} mm  {flag}',
                 fontsize=10, color='white')
    ax.axis('off')


def _panel_tss_axial_with_tps(ax,
                                img_sl:      np.ndarray,
                                tss_sl:      Optional[np.ndarray],
                                tp_left_sl:  Optional[np.ndarray],
                                tp_right_sl: Optional[np.ndarray],
                                native:      bool,
                                z_idx:       int,
                                subtitle:    str = ''):
    """
    TSS label overlay + dilated TP masks projected onto axial.
    Used for [0,2] and [1,2].
    """
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower', alpha=0.78)
    patches = []

    if tss_sl is not None:
        for label, (color, name) in TSS_LABEL_COLORS.items():
            m = tss_sl == label
            if m.any():
                overlay_mask(ax, m, color, 0.45)
                patches.append(mpatches.Patch(color=color, label=name))

    if tp_left_sl is not None and tp_left_sl.any():
        tp_l_disp = dilate_for_display(tp_left_sl, DISPLAY_DILATION_VOXELS)
        overlay_mask(ax, tp_l_disp, [1.00, 0.10, 0.10], 0.80)
        patches.append(mpatches.Patch(color=[1.00, 0.10, 0.10], label='Left TP'))
    if tp_right_sl is not None and tp_right_sl.any():
        tp_r_disp = dilate_for_display(tp_right_sl, DISPLAY_DILATION_VOXELS)
        overlay_mask(ax, tp_r_disp, [0.00, 0.80, 1.00], 0.80)
        patches.append(mpatches.Patch(color=[0.00, 0.80, 1.00], label='Right TP'))

    if patches:
        ax.legend(handles=patches, loc='lower right', fontsize=6, framealpha=0.55)

    src   = 'native' if native else 'resampled'
    title = f'TSS Labels + TPs — Axial  z={z_idx}  ({src})'
    if subtitle:
        title += f'\n{subtitle}'
    ax.set_title(title, fontsize=10, color='white')
    ax.axis('off')


def _panel_sag_tp_proximity(ax,
                              sag_img:       Optional[np.ndarray],
                              sag_spineps:   Optional[np.ndarray],
                              tp_other_mask: Optional[np.ndarray],
                              sag_tss:       Optional[np.ndarray],
                              side_name:     str,
                              x_idx:         int,
                              dist_mm:       float,
                              vox_z_mm:      float,
                              z_axial_sag:   Optional[float],   # ONE line: z_md shown in [1,2]
                              tv_name:       str):
    """
    Row 1 — Type II/III check.
    Single orange reference line at the axial z shown in [1,2] (min-dist slice).
    """
    color_this  = [1.00, 0.10, 0.10] if side_name == 'Left' else [0.00, 0.80, 1.00]
    color_other = [0.00, 0.80, 1.00] if side_name == 'Left' else [1.00, 0.10, 0.10]

    ax.imshow(norm(_sag_sl(sag_img, x_idx)).T, cmap='gray', origin='lower', alpha=0.80)

    this_sl   = _sag_sl(sag_spineps, x_idx) if (sag_spineps is not None and sag_spineps.any()) else np.zeros((1,1), bool)
    sacrum_sl = _sag_sl(sag_tss == SACRUM_LABEL, x_idx) if sag_tss is not None else np.zeros((1,1), bool)

    if tp_other_mask is not None and tp_other_mask.any():
        overlay_mask(ax, _sag_sl(tp_other_mask, x_idx), color_other, 0.22)
    if sag_spineps is not None and sag_spineps.any():
        overlay_mask(ax, this_sl, color_this, 0.85)
    if sag_tss is not None:
        overlay_mask(ax, sacrum_sl, [1.00, 0.55, 0.00], 0.60)

    _draw_gap_ruler(ax, this_sl, sacrum_sl, vox_z_mm, color=LINE_MINDIST)

    # ONE reference line — the axial slice shown in [1,2]
    _hline(ax, z_axial_sag, LINE_MINDIST, label='axial slice →')

    ax.legend(handles=[
        mpatches.Patch(color=color_this,         label=f'{side_name} TP'),
        mpatches.Patch(color=color_other,        label=f'{"R" if side_name=="Left" else "L"} TP (faint)'),
        mpatches.Patch(color=[1.00, 0.55, 0.00], label='Sacrum'),
        mpatches.Patch(color=LINE_MINDIST,       label='axial slice (min-dist)'),
    ], loc='lower right', fontsize=6, framealpha=0.55)

    dist_str = f'{dist_mm:.1f} mm' if np.isfinite(dist_mm) else 'N/A'
    contact  = np.isfinite(dist_mm) and dist_mm <= CONTACT_DIST_MM
    flag     = '✗ contact → II/III' if contact else '✓ no contact'
    ax.set_title(f'Type II/III check — {side_name} TP  (x={x_idx})\n'
                 f'TP–Sacrum min dist: {dist_str}  {flag}',
                 fontsize=10, color='white')
    ax.axis('off')


def _panel_sag_tss_confirm(ax,
                             sag_img_sl: np.ndarray,
                             tss_sag_sl: np.ndarray,
                             tv_name:    str,
                             z_tv_sag:   Optional[float]):
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
        ax.legend(handles=patches, loc='lower right', fontsize=6, framealpha=0.55)
    _hline(ax, z_tv_sag, LINE_TV, label='TV mid')
    ax.set_title(f'Sagittal TSS — Level Confirm  (TV={tv_name})\n'
                 'L5/L6 + Sacrum highlighted',
                 fontsize=10, color='white')
    ax.axis('off')


def _panel_axial_all_masks(ax,
                             img_sl:      np.ndarray,
                             tss_sl:      Optional[np.ndarray],
                             spineps_sl:  Optional[np.ndarray],
                             z_idx:       int):
    """
    [2,1] — axial T2w with ALL TSS label colours + ALL SPINEPS masks overlaid.
    """
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower', alpha=0.75)
    patches = []

    if tss_sl is not None:
        for label, (color, name) in TSS_LABEL_COLORS.items():
            m = tss_sl == label
            if m.any():
                overlay_mask(ax, m, color, 0.40)
                patches.append(mpatches.Patch(color=color, label=f'TSS {name}'))

    if spineps_sl is not None:
        for label, (color, name) in SPINEPS_LABEL_COLORS.items():
            m = spineps_sl == label
            if m.any():
                m_disp = dilate_for_display(m, DISPLAY_DILATION_VOXELS)
                overlay_mask(ax, m_disp, color, 0.75)
                patches.append(mpatches.Patch(color=color, label=f'SPINEPS {name}'))

    if patches:
        ax.legend(handles=patches, loc='lower right', fontsize=6, framealpha=0.55)

    ax.set_title(f'All Masks — Axial TV mid  z={z_idx}\n'
                 'TSS labels + SPINEPS TPs & Sacrum',
                 fontsize=10, color='white')
    ax.axis('off')


def _panel_summary(ax,
                    study_id:      str,
                    result:        Optional[dict],
                    tv_name:       str,
                    span_left_mm:  float,
                    span_right_mm: float,
                    dist_L_mm:     float,
                    dist_R_mm:     float):
    ax.axis('off')
    ax.set_facecolor('#1a1a2e')

    def _d(v: float) -> str:
        return f'{v:.1f}' if np.isfinite(v) else 'N/A'

    lines = [
        f'Study  : {study_id}',
        f'TV     : {tv_name}',
        '',
        '─── Measured (from masks) ─────────────────',
        f'  Left  TP height : {_d(span_left_mm):>6} mm  (thresh {TP_HEIGHT_MM:.0f} mm)',
        f'  Right TP height : {_d(span_right_mm):>6} mm  (thresh {TP_HEIGHT_MM:.0f} mm)',
        f'  Left  TP–Sacrum : {_d(dist_L_mm):>6} mm  (thresh {CONTACT_DIST_MM:.0f} mm)',
        f'  Right TP–Sacrum : {_d(dist_R_mm):>6} mm  (thresh {CONTACT_DIST_MM:.0f} mm)',
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
                f'    classifier height : {sd.get("tp_height_mm",0.0):.1f} mm',
                f'    classifier dist   : {_d(sd.get("dist_mm", float("inf")))} mm',
            ]
            if sd.get('note'):
                lines.append(f'    NOTE: {sd["note"]}')
            lines.append('')
        if result.get('errors'):
            lines += ['  Errors:'] + [f'    {e}' for e in result['errors']]

    # Larger font for readability
    ax.text(0.05, 0.97, '\n'.join(lines),
            transform=ax.transAxes, va='top', ha='left',
            fontsize=10.5, family='monospace', color='white', linespacing=1.45)
    ax.set_title('Classification Summary', fontsize=13, color='white')


# ============================================================================
# CORE VISUALIZER
# ============================================================================

def visualize_study(
    study_id:       str,
    registered_dir: Path,
    nifti_dir:      Path,
    spineps_dir:    Path,
    totalspine_dir: Path,
    output_dir:     Path,
    result:         Optional[dict] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{study_id}_lstv_overlay.png"
    reg      = registered_dir / study_id

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

    # ── Load ─────────────────────────────────────────────────────────────────
    spineps_reg, spineps_nii = try_load(reg / f"{study_id}_spineps_reg.nii.gz", 'SPINEPS reg')
    tss_reg, _               = try_load(reg / f"{study_id}_tss_reg.nii.gz",     'TSS reg')
    sag_bg,  sag_nii         = try_load(find_t2w(nifti_dir, study_id, 'sag'),   'Sag T2w')
    orig_spineps, _          = try_load(find_original_spineps_seg(spineps_dir, study_id), 'SPINEPS orig')
    orig_tss_sag, _          = try_load(find_original_tss_sag(totalspine_dir, study_id),  'TSS orig sag')
    ax_bg, ax_bg_nii         = try_load(find_t2w(nifti_dir, study_id, 'ax'),    'Axial T2w')
    if ax_bg is None:
        ax_bg, ax_bg_nii     = spineps_reg, spineps_nii
    tss_native, _            = try_load(find_native_axial_tss(totalspine_dir, study_id), 'TSS native axial')
    using_native             = tss_native is not None

    if ax_bg is None:
        logger.error(f"  [{study_id}] No axial background — skipping")
        return

    # ── Type casts ────────────────────────────────────────────────────────────
    if spineps_reg  is not None: spineps_reg  = spineps_reg.astype(int)
    if tss_reg      is not None: tss_reg      = tss_reg.astype(int)
    if orig_spineps is not None: orig_spineps = orig_spineps.astype(int)
    if orig_tss_sag is not None: orig_tss_sag = orig_tss_sag.astype(int)
    if tss_native   is not None: tss_native   = tss_native.astype(int)

    vox_ax  = voxel_size_mm(ax_bg_nii)
    vox_sag = voxel_size_mm(sag_nii) if sag_nii is not None else vox_ax

    # ── Target vertebra ───────────────────────────────────────────────────────
    tss_labels = tss_reg if tss_reg is not None else tss_native
    tv_label   = (L6_LABEL
                  if tss_labels is not None and L6_LABEL in np.unique(tss_labels)
                  else L5_LABEL)
    tv_name    = 'L6' if tv_label == L6_LABEL else 'L5'

    # ── Axial masks ───────────────────────────────────────────────────────────
    zeros   = np.zeros(ax_bg.shape, dtype=bool)
    z_range = get_tv_z_range(tss_labels, tv_label) if tss_labels is not None else None

    if z_range is not None and spineps_reg is not None:
        z_min_tv, z_max_tv = z_range
        z_tv        = (z_min_tv + z_max_tv) // 2
        tp_left_ax  = isolate_tp_at_tv(spineps_reg, TP_LEFT_LABEL,  z_min_tv, z_max_tv)
        tp_right_ax = isolate_tp_at_tv(spineps_reg, TP_RIGHT_LABEL, z_min_tv, z_max_tv)
    else:
        z_tv = ax_bg.shape[2] // 2
        tp_left_ax = tp_right_ax = zeros

    sacrum_ax = (tss_labels == SACRUM_LABEL) if tss_labels is not None else zeros

    # ── Min-dist z per side ───────────────────────────────────────────────────
    z_md_L, dist_L = min_dist_z_for_tp_sacrum(tp_left_ax,  sacrum_ax, vox_ax)
    z_md_R, dist_R = min_dist_z_for_tp_sacrum(tp_right_ax, sacrum_ax, vox_ax)
    if np.isfinite(dist_L) or np.isfinite(dist_R):
        z_md_combined = z_md_L if dist_L <= dist_R else z_md_R
    else:
        z_md_combined = z_tv

    logger.info(f"  [{study_id}] z_tv={z_tv}  "
                f"z_md_L={z_md_L}({dist_L:.1f}mm)  z_md_R={z_md_R}({dist_R:.1f}mm)")

    # ── Isolate L5-only TP blobs in sagittal space ───────────────────────────
    tp_l5_left_sag  = isolate_l5_tp_sag(orig_spineps, orig_tss_sag,
                                          TP_LEFT_LABEL,  tv_label)
    tp_l5_right_sag = isolate_l5_tp_sag(orig_spineps, orig_tss_sag,
                                          TP_RIGHT_LABEL, tv_label)

    logger.info(f"  [{study_id}] L5-left voxels={tp_l5_left_sag.sum()}  "
                f"L5-right voxels={tp_l5_right_sag.sum()}")

    # ── Max-height sagittal slices per side ───────────────────────────────────
    x_left,  span_L = best_x_for_tp_height(tp_l5_left_sag,  vox_sag[2])
    x_right, span_R = best_x_for_tp_height(tp_l5_right_sag, vox_sag[2])

    logger.info(f"  [{study_id}] x_left={x_left}({span_L:.1f}mm)  "
                f"x_right={x_right}({span_R:.1f}mm)")

    # ── Row 0 axial z: midpoint of the combined TP blobs in sagittal space ───
    # Convert sagittal z-midpoint of each L5 TP blob to axial z
    z_tp_mid_sag_L = tp_blob_z_midpoint(tp_l5_left_sag)
    z_tp_mid_sag_R = tp_blob_z_midpoint(tp_l5_right_sag)
    # Average the two midpoints (or use whichever exists)
    if tp_l5_left_sag.any() and tp_l5_right_sag.any():
        z_tp_mid_sag = (z_tp_mid_sag_L + z_tp_mid_sag_R) // 2
    elif tp_l5_left_sag.any():
        z_tp_mid_sag = z_tp_mid_sag_L
    elif tp_l5_right_sag.any():
        z_tp_mid_sag = z_tp_mid_sag_R
    else:
        z_tp_mid_sag = None

    # Convert sag-space z to axial-space z for [0,2] slice
    if z_tp_mid_sag is not None and sag_nii is not None:
        z_row0_ax = sag_z_to_ax_z(sag_nii, ax_bg_nii, z_tp_mid_sag)
    else:
        z_row0_ax = z_tv  # fallback

    if z_row0_ax is None:
        z_row0_ax = z_tv

    logger.info(f"  [{study_id}] z_tp_mid_sag={z_tp_mid_sag}  z_row0_ax={z_row0_ax}  z_md_combined={z_md_combined}")

    # ── Midline x for TSS confirm ─────────────────────────────────────────────
    x_mid = (orig_tss_sag.shape[0] // 2 if orig_tss_sag is not None
             else sag_bg.shape[0] // 2   if sag_bg       is not None else 0)

    # ── Affine conversions for sagittal reference lines ───────────────────────
    # Row 0 panels get ONE line: z_row0_ax in sag space
    z_row0_sag = ax_z_to_sag_z(ax_bg_nii, sag_nii, z_row0_ax)
    # Row 1 panels get ONE line: z_md_combined in sag space
    z_md_c_sag = ax_z_to_sag_z(ax_bg_nii, sag_nii, z_md_combined)
    # TV mid for [2,0] confirm panel
    z_tv_sag   = ax_z_to_sag_z(ax_bg_nii, sag_nii, z_tv)

    # ── TSS display volume ────────────────────────────────────────────────────
    tss_disp = tss_native if using_native else tss_reg

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(21, 21))
    fig.patch.set_facecolor('#0d0d1a')
    for a in axes.flat:
        a.set_facecolor('#0d0d1a')

    castellvi = result.get('castellvi_type') if result else None
    fig.suptitle(
        f"Study {study_id}   |   Castellvi: {castellvi or 'N/A'}   |   TV: {tv_name}",
        fontsize=15, color='white', y=0.999,
    )

    # ── ROW 0 — Type I ───────────────────────────────────────────────────────

    if sag_bg is not None:
        _panel_sag_tp_height(
            axes[0, 0], sag_bg, tp_l5_left_sag, tp_l5_right_sag, orig_tss_sag,
            'Left', x_left, span_L, vox_sag[2],
            z_axial_sag=z_row0_sag,   # ONE line
            tv_name=tv_name)
        _panel_sag_tp_height(
            axes[0, 1], sag_bg, tp_l5_right_sag, tp_l5_left_sag, orig_tss_sag,
            'Right', x_right, span_R, vox_sag[2],
            z_axial_sag=z_row0_sag,   # ONE line
            tv_name=tv_name)
    else:
        _unavailable(axes[0, 0], 'Sagittal T2w not found')
        _unavailable(axes[0, 1], 'Sagittal T2w not found')

    # [0,2] TSS labels + TP projections at z_row0_ax (TP blob midpoint)
    tss_sl_row0 = _tss_axial_slice(ax_bg, tss_disp, z_row0_ax)
    _panel_tss_axial_with_tps(
        axes[0, 2],
        _ax_sl(ax_bg, z_row0_ax),
        tss_sl_row0,
        _ax_sl(tp_left_ax,  z_row0_ax),
        _ax_sl(tp_right_ax, z_row0_ax),
        using_native, z_row0_ax,
        subtitle=f'TP blob midpoint  z={z_row0_ax}',
    )

    # ── ROW 1 — Type II/III ──────────────────────────────────────────────────

    if sag_bg is not None:
        _panel_sag_tp_proximity(
            axes[1, 0], sag_bg, tp_l5_left_sag, tp_l5_right_sag, orig_tss_sag,
            'Left', x_left, dist_L, vox_sag[2],
            z_axial_sag=z_md_c_sag,   # ONE line
            tv_name=tv_name)
        _panel_sag_tp_proximity(
            axes[1, 1], sag_bg, tp_l5_right_sag, tp_l5_left_sag, orig_tss_sag,
            'Right', x_right, dist_R, vox_sag[2],
            z_axial_sag=z_md_c_sag,   # ONE line
            tv_name=tv_name)
    else:
        _unavailable(axes[1, 0], 'Sagittal T2w not found')
        _unavailable(axes[1, 1], 'Sagittal T2w not found')

    # [1,2] TSS labels + dilated TP + sacrum at z_md_combined
    tss_sl_md  = _tss_axial_slice(ax_bg, tss_disp, z_md_combined)
    dist_label = (f'L={dist_L:.1f}mm  R={dist_R:.1f}mm'
                  if (np.isfinite(dist_L) or np.isfinite(dist_R)) else 'no masks')
    _panel_tss_axial_with_tps(
        axes[1, 2],
        _ax_sl(ax_bg, z_md_combined),
        tss_sl_md,
        _ax_sl(tp_left_ax,  z_md_combined),
        _ax_sl(tp_right_ax, z_md_combined),
        using_native, z_md_combined,
        subtitle=f'Min-dist slice  [{dist_label}]  [Type II/III]',
    )

    # ── ROW 2 — Context ───────────────────────────────────────────────────────

    # [2,0] Sagittal TSS level confirm — one line at TV mid
    if orig_tss_sag is not None:
        sag_bg_sl = (_sag_sl(sag_bg, x_mid) if sag_bg is not None
                     else _sag_sl(orig_tss_sag, x_mid).astype(float))
        _panel_sag_tss_confirm(
            axes[2, 0], sag_bg_sl, _sag_sl(orig_tss_sag, x_mid),
            tv_name, z_tv_sag)
    else:
        _unavailable(axes[2, 0], 'Sagittal TSS not found')

    # [2,1] All masks axial at TV mid
    tss_sl_tv     = _tss_axial_slice(ax_bg, tss_disp, z_tv)
    spineps_sl_tv = _ax_sl(spineps_reg, z_tv) if spineps_reg is not None else None
    _panel_axial_all_masks(
        axes[2, 1],
        _ax_sl(ax_bg, z_tv),
        tss_sl_tv,
        spineps_sl_tv,
        z_tv,
    )

    # [2,2] Summary — larger font set inside _panel_summary
    _panel_summary(
        axes[2, 2], study_id, result, tv_name,
        span_left_mm=span_L, span_right_mm=span_R,
        dist_L_mm=dist_L, dist_R_mm=dist_R,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.998])
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"  [{study_id}] OK -> {out_path}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='LSTV Overlay Visualizer v9')
    parser.add_argument('--registered_dir',  required=True)
    parser.add_argument('--nifti_dir',       required=True)
    parser.add_argument('--spineps_dir',     required=True)
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
