#!/usr/bin/env python3
"""
06_visualize_3d.py — Interactive 3D Spine Segmentation Viewer
=============================================================
Generates a self-contained HTML file with an interactive 3D render of:
  • Vertebral bodies      (VERIDAH seg-vert_msk — L1 through L6, colour-coded)
  • Transverse processes  (SPINEPS seg-spine_msk labels 43=Left, 44=Right)
  • Sacrum               (TotalSpineSeg label 50, fallback SPINEPS label 26)
  • Spinal canal         (SPINEPS seg-spine_msk label 61, optional)

Method: Marching Cubes (scikit-image) → Plotly Mesh3d traces → standalone HTML.
No server needed — open the .html directly in any browser.

Study selection mirrors 05_visualize_overlay.py exactly:
  --study_id        single study (highest priority)
  --all             every study with SPINEPS segmentation
  --uncertainty_csv + --valid_ids + --top_n + --rank_by
                    top-N / bottom-N from Ian Pan uncertainty CSV (default)

Label reference:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  seg-spine_msk (subregion):  43=TP-Left  44=TP-Right  26=Sacrum(fb)
                               61=Spinal_Canal  60=Spinal_Cord
  seg-vert_msk  (VERIDAH):    20=L1 21=L2 22=L3 23=L4 24=L5 25=L6
  TotalSpineSeg sagittal:     50=Sacrum (preferred)  45=L5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage
-----
  # Single study
  python 06_visualize_3d.py \
      --spineps_dir results/spineps --totalspine_dir results/totalspineseg \
      --output_dir results/lstv_3d --study_id 60612428

  # Top/bottom N by Ian Pan confidence (mirrors 05)
  python 06_visualize_3d.py \
      --spineps_dir results/spineps --totalspine_dir results/totalspineseg \
      --output_dir results/lstv_3d \
      --uncertainty_csv results/epistemic_uncertainty/lstv_uncertainty_metrics.csv \
      --valid_ids models/valid_id.npy \
      --top_n 1 --rank_by l5_s1_confidence

  # Every study
  python 06_visualize_3d.py \
      --spineps_dir results/spineps --totalspine_dir results/totalspineseg \
      --output_dir results/lstv_3d --all

Dependencies
------------
  pip install scikit-image plotly nibabel numpy scipy pandas
"""

import argparse
import json
import logging
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.measure import marching_cubes

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# SPINEPS seg-spine_msk subregion labels
TP_LEFT_LABEL    = 43
TP_RIGHT_LABEL   = 44
SPINEPS_SACRUM   = 26
SPINEPS_CANAL    = 61
SPINEPS_CORD     = 60

# VERIDAH seg-vert_msk instance labels
VERIDAH_NAMES    = {20: 'L1', 21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5', 25: 'L6'}
VERIDAH_LUMBAR   = [25, 24, 23, 22, 21, 20]

# TotalSpineSeg labels
TSS_SACRUM_LABEL = 50

# Colour palette — chosen for clarity on dark background
COLOURS = {
    'L1':       '#1e6fa8',
    'L2':       '#2389cc',
    'L3':       '#29a3e8',
    'L4':       '#52bef5',
    'L5':       '#85d4ff',
    'L6':       '#aae3ff',
    'TP_Left':  '#ff3333',
    'TP_Right': '#00d4ff',
    'Sacrum':   '#ff8c00',
    'Canal':    '#00ffb3',
    'Cord':     '#ffe066',
}

OPACITY = {
    'vertebra': 0.45,
    'tp':       0.90,
    'sacrum':   0.72,
    'canal':    0.18,
    'cord':     0.55,
}

# Ian Pan disc levels — same list as 05_visualize_overlay.py
IAN_PAN_LEVELS = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']


# ============================================================================
# STUDY SELECTION — mirrors 05_visualize_overlay.py select_studies()
# ============================================================================

def select_studies(csv_path: Path, top_n: int, rank_by: str,
                   valid_ids) -> list:
    """
    Return study IDs for the top-N and bottom-N rows in the uncertainty CSV,
    filtered to valid_ids if provided.  Identical logic to script 05.
    """
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
        raise ValueError(
            f"Column '{rank_by}' not in CSV. "
            f"Available: {', '.join(df.columns)}"
        )

    df_sorted  = df.sort_values(rank_by, ascending=False).reset_index(drop=True)
    top_ids    = df_sorted.head(top_n)['study_id'].tolist()
    bottom_ids = df_sorted.tail(top_n)['study_id'].tolist()

    seen, selected = set(), []
    for sid in top_ids + bottom_ids:
        if sid not in seen:
            selected.append(sid)
            seen.add(sid)

    logger.info(
        f"Rank by: {rank_by}  "
        f"Top {top_n}: {top_ids}  "
        f"Bottom {top_n}: {bottom_ids}"
    )
    return selected


# ============================================================================
# NIBABEL HELPERS
# ============================================================================

def load_canonical(path: Path):
    nii  = nib.load(str(path))
    nii  = nib.as_closest_canonical(nii)
    data = nii.get_fdata()
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim == 4:
        logger.debug(f"  4D {path.name} → selecting volume 0")
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Cannot reduce {path.name} to 3D: {data.shape}")
    return data, nii


def voxel_size_mm(nii):
    return np.abs(np.array(nii.header.get_zooms()[:3], dtype=float))


def vox_to_mm(vertices, vox_mm, origin_mm=None):
    v = vertices * vox_mm[np.newaxis, :]
    if origin_mm is not None:
        v -= origin_mm[np.newaxis, :]
    return v


# ============================================================================
# MARCHING CUBES → PLOTLY MESH
# ============================================================================

def mask_to_mesh3d(
    mask: np.ndarray,
    vox_mm: np.ndarray,
    name: str,
    colour: str,
    opacity: float,
    step: int = 2,
    smooth_sigma: float = 1.0,
    fill_holes: bool = True,
    origin_mm: np.ndarray = None,
) -> go.Mesh3d | None:
    """
    Convert a binary 3D mask to a Plotly Mesh3d trace via marching cubes.

    Parameters
    ----------
    step         : voxel subsampling step (2 = every other voxel, faster + smoother)
    smooth_sigma : Gaussian sigma for pre-smoothing the binary mask (helps MC)
    fill_holes   : fill internal holes before meshing
    origin_mm    : subtract this offset so all meshes share a common origin
    """
    if not mask.any():
        logger.warning(f"  '{name}': mask is empty — skipping")
        return None

    if fill_holes:
        mask = binary_fill_holes(mask)

    mask_sub = mask[::step, ::step, ::step]
    vox_sub  = vox_mm * step

    vol = (gaussian_filter(mask_sub.astype(float), sigma=smooth_sigma)
           if smooth_sigma > 0 else mask_sub.astype(float))

    # Pad by 1 voxel so MC can close surface at borders
    vol = np.pad(vol, 1, mode='constant', constant_values=0)

    try:
        verts, faces, _, _ = marching_cubes(vol, level=0.5, spacing=(1, 1, 1))
    except Exception as e:
        logger.warning(f"  '{name}': marching cubes failed: {e}")
        return None

    verts -= 1.0          # undo padding offset
    verts_mm = vox_to_mm(verts, vox_sub, origin_mm)

    x, y, z = verts_mm[:, 0], verts_mm[:, 1], verts_mm[:, 2]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    return go.Mesh3d(
        x=x.tolist(), y=y.tolist(), z=z.tolist(),
        i=i.tolist(), j=j.tolist(), k=k.tolist(),
        color=colour,
        opacity=opacity,
        name=name,
        showlegend=True,
        flatshading=False,
        lighting=dict(
            ambient=0.35,
            diffuse=0.75,
            specular=0.35,
            roughness=0.6,
            fresnel=0.2,
        ),
        lightposition=dict(x=100, y=200, z=150),
        hoverinfo='name',
        showscale=False,
    )


# ============================================================================
# PER-STUDY 3D BUILDER
# ============================================================================

def build_3d_figure(
    study_id: str,
    spineps_dir: Path,
    totalspine_dir: Path,
    step: int = 2,
    smooth: float = 1.5,
    show_canal: bool = False,
    show_cord: bool = False,
    lstv_result: dict = None,
    uncertainty_row: dict = None,
) -> go.Figure | None:

    seg_dir    = spineps_dir / 'segmentations' / study_id
    spine_mask = seg_dir / f"{study_id}_seg-spine_msk.nii.gz"
    vert_mask  = seg_dir / f"{study_id}_seg-vert_msk.nii.gz"
    tss_sag    = (totalspine_dir / study_id / 'sagittal'
                  / f"{study_id}_sagittal_labeled.nii.gz")

    def _load(path, label):
        if not path.exists():
            logger.warning(f"  Missing: {path.name}")
            return None, None
        try:
            return load_canonical(path)
        except Exception as e:
            logger.warning(f"  Cannot load {label}: {e}")
            return None, None

    sag_sp,   nii_ref = _load(spine_mask, 'seg-spine_msk')
    sag_vert, _       = _load(vert_mask,  'seg-vert_msk')
    sag_tss,  _       = _load(tss_sag,    'TSS sagittal')

    if sag_sp is None or nii_ref is None:
        logger.error(f"  [{study_id}] Missing seg-spine_msk — cannot build 3D")
        return None
    if sag_vert is None:
        logger.error(f"  [{study_id}] Missing seg-vert_msk — cannot build 3D")
        return None

    sag_sp   = sag_sp.astype(int)
    sag_vert = sag_vert.astype(int)
    if sag_tss is not None:
        sag_tss = sag_tss.astype(int)

    vox_mm = voxel_size_mm(nii_ref)
    logger.info(f"  Voxel size: {vox_mm} mm  Volume: {sag_sp.shape}")

    # Centre all meshes on the centroid of the vertebral column
    col_mask = sag_vert > 0
    origin_mm = (np.array(np.where(col_mask)).mean(axis=1) * vox_mm
                 if col_mask.any()
                 else np.array(sag_sp.shape) / 2 * vox_mm)

    traces        = []
    summary_lines = []

    # ── LSTV annotation ──────────────────────────────────────────────────────
    castellvi = 'N/A'
    tv_name   = 'N/A'
    if lstv_result:
        castellvi = lstv_result.get('castellvi_type') or 'None'
        tv_name   = lstv_result.get('details', {}).get('tv_name', 'N/A')

    # ── Ian Pan uncertainty annotation ───────────────────────────────────────
    ian_str = None
    if uncertainty_row:
        l5s1_conf = uncertainty_row.get('l5_s1_confidence', float('nan'))
        l5s1_entr = uncertainty_row.get('l5_s1_entropy',    float('nan'))
        ian_str   = (f"L5-S1 conf={l5s1_conf:.3f}  "
                     f"entropy={l5s1_entr:.3f}")
        # Collect all level confidences for the summary box
        for lvl in IAN_PAN_LEVELS:
            v = uncertainty_row.get(f'{lvl}_confidence', float('nan'))
            if not np.isnan(v):
                summary_lines.append(
                    f"Ian {lvl.upper()}: conf={v:.3f}")

    # ── Sacrum ───────────────────────────────────────────────────────────────
    if sag_tss is not None and (sag_tss == TSS_SACRUM_LABEL).any():
        sac_mask  = (sag_tss == TSS_SACRUM_LABEL)
        sac_label = 'Sacrum (TSS 50)'
    else:
        sac_mask  = (sag_sp == SPINEPS_SACRUM)
        sac_label = 'Sacrum (SPINEPS 26)'
        if not sac_mask.any():
            logger.warning("  Sacrum absent in both masks")

    if sac_mask.any():
        t = mask_to_mesh3d(sac_mask, vox_mm, sac_label,
                           COLOURS['Sacrum'], OPACITY['sacrum'],
                           step=step, smooth_sigma=smooth,
                           origin_mm=origin_mm)
        if t:
            traces.append(t)
        vol_cm3 = sac_mask.sum() * vox_mm.prod() / 1000
        summary_lines.append(f"Sacrum volume: {vol_cm3:.1f} cm³")

    # ── Transverse processes ─────────────────────────────────────────────────
    for tp_label, tp_name, tp_col in (
        (TP_LEFT_LABEL,  'TP Left (43)',  COLOURS['TP_Left']),
        (TP_RIGHT_LABEL, 'TP Right (44)', COLOURS['TP_Right']),
    ):
        tp_mask = (sag_sp == tp_label)
        if tp_mask.any():
            t = mask_to_mesh3d(
                tp_mask, vox_mm, tp_name, tp_col, OPACITY['tp'],
                step=max(1, step - 1),         # finer step for small structures
                smooth_sigma=max(0.5, smooth - 0.5),
                fill_holes=False,
                origin_mm=origin_mm,
            )
            if t:
                traces.append(t)
            vol_cm3 = tp_mask.sum() * vox_mm.prod() / 1000
            summary_lines.append(f"{tp_name} vol: {vol_cm3:.2f} cm³")
        else:
            logger.warning(f"  {tp_name}: not found in seg-spine_msk")
            summary_lines.append(f"{tp_name}: absent")

    # ── Vertebral bodies (VERIDAH) ───────────────────────────────────────────
    present_vertebrae = []
    for label in sorted(VERIDAH_NAMES.keys()):
        name  = VERIDAH_NAMES[label]
        vmask = (sag_vert == label)
        if not vmask.any():
            continue
        present_vertebrae.append(name)
        t = mask_to_mesh3d(vmask, vox_mm, name,
                           COLOURS.get(name, '#aaaaaa'), OPACITY['vertebra'],
                           step=step, smooth_sigma=smooth,
                           origin_mm=origin_mm)
        if t:
            traces.append(t)
    logger.info(f"  Vertebrae present: {present_vertebrae}")

    # ── Spinal canal (optional) ──────────────────────────────────────────────
    if show_canal:
        canal_mask = (sag_sp == SPINEPS_CANAL)
        if canal_mask.any():
            t = mask_to_mesh3d(canal_mask, vox_mm, 'Spinal Canal (61)',
                               COLOURS['Canal'], OPACITY['canal'],
                               step=step, smooth_sigma=smooth,
                               origin_mm=origin_mm)
            if t:
                traces.append(t)

    if show_cord:
        cord_mask = (sag_sp == SPINEPS_CORD)
        if cord_mask.any():
            t = mask_to_mesh3d(cord_mask, vox_mm, 'Spinal Cord (60)',
                               COLOURS['Cord'], OPACITY['cord'],
                               step=max(1, step - 1), smooth_sigma=smooth,
                               origin_mm=origin_mm)
            if t:
                traces.append(t)

    if not traces:
        logger.error(f"  [{study_id}] No meshes generated")
        return None

    # ── Titles / annotation strings ───────────────────────────────────────────
    title_str = (
        f"<b>Study {study_id}</b>  ·  Castellvi: <b>{castellvi}</b>"
        f"  ·  TV: <b>{tv_name}</b>"
        f"  ·  Vertebrae: {', '.join(present_vertebrae)}"
    )
    if ian_str:
        title_str += f"  ·  Ian Pan {ian_str}"

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = go.Figure(data=traces)

    # Annotation: controls + LSTV metrics box
    annotations = [
        dict(
            text=(
                '<b>Controls:</b> Left-drag = rotate  ·  '
                'Right-drag / scroll = zoom  ·  '
                'Middle-drag = pan  ·  '
                'Legend click = toggle  ·  '
                'Double-click = isolate<br>'
                '<span style="color:#ff3333">■</span> TP Left  '
                '<span style="color:#00d4ff">■</span> TP Right  '
                '<span style="color:#ff8c00">■</span> Sacrum  '
                '<span style="color:#85d4ff">■</span> Vertebrae'
            ),
            xref='paper', yref='paper',
            x=0.5, y=-0.01, xanchor='center', yanchor='top',
            showarrow=False,
            font=dict(size=11, color='#8888aa'),
            align='center',
        ),
    ]
    if summary_lines:
        annotations.append(dict(
            text='<b>LSTV Metrics</b><br>' + '<br>'.join(summary_lines),
            xref='paper', yref='paper',
            x=0.99, y=0.98,
            xanchor='right', yanchor='top',
            showarrow=False,
            font=dict(size=11, color='#e8e8f0', family='monospace'),
            bgcolor='rgba(13,13,26,0.85)',
            bordercolor='#2a2a4a',
            borderwidth=1,
            align='left',
        ))

    fig.update_layout(
        title=dict(text=title_str,
                   font=dict(size=15, color='#e8e8f0'), x=0.01),
        paper_bgcolor='#0d0d1a',
        plot_bgcolor='#0d0d1a',
        scene=dict(
            bgcolor='#0d0d1a',
            xaxis=dict(
                title='X (mm)', showgrid=True, gridcolor='#1a1a3e',
                showbackground=True, backgroundcolor='#0d0d1a',
                tickfont=dict(color='#8888aa'), titlefont=dict(color='#8888aa'),
                zeroline=False,
            ),
            yaxis=dict(
                title='Y (mm)', showgrid=True, gridcolor='#1a1a3e',
                showbackground=True, backgroundcolor='#0d0d1a',
                tickfont=dict(color='#8888aa'), titlefont=dict(color='#8888aa'),
                zeroline=False,
            ),
            zaxis=dict(
                title='Z (mm)', showgrid=True, gridcolor='#1a1a3e',
                showbackground=True, backgroundcolor='#0d0d1a',
                tickfont=dict(color='#8888aa'), titlefont=dict(color='#8888aa'),
                zeroline=False,
            ),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.6, y=0.0, z=0.3),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        legend=dict(
            font=dict(color='#e8e8f0', size=12),
            bgcolor='rgba(13,13,26,0.85)',
            bordercolor='#2a2a4a',
            borderwidth=1,
            x=0.01, y=0.98,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        annotations=annotations,
    )

    return fig


# ============================================================================
# HTML WRAPPER — adds a view-preset toolbar above the plotly div
# ============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>3D Spine — {study_id}</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@500;700&display=swap');

    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --bg:      #0d0d1a;
      --surface: #13132a;
      --border:  #2a2a4a;
      --text:    #e8e8f0;
      --muted:   #6666aa;
      --red:     #ff3333;
      --cyan:    #00d4ff;
      --amber:   #ff8c00;
      --blue:    #3a86ff;
    }}

    html, body {{
      background: var(--bg);
      color: var(--text);
      font-family: 'JetBrains Mono', monospace;
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }}

    header {{
      display: flex;
      align-items: center;
      gap: 20px;
      padding: 10px 20px;
      border-bottom: 1px solid var(--border);
      background: var(--surface);
      flex-shrink: 0;
      flex-wrap: wrap;
    }}

    header h1 {{
      font-family: 'Syne', sans-serif;
      font-size: 1rem;
      font-weight: 700;
      color: var(--text);
      letter-spacing: 0.04em;
      white-space: nowrap;
    }}

    .badge {{
      display: inline-block;
      padding: 2px 10px;
      border-radius: 20px;
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.06em;
    }}
    .badge-castellvi {{ background: #ff8c00; color: #0d0d1a; }}
    .badge-tv         {{ background: #1e6fa8; color: #fff; }}
    .badge-study      {{ background: #2a2a4a; color: var(--muted); }}
    .badge-ian        {{ background: #1a3a2a; color: #2dc653; border: 1px solid #2dc653; }}

    .toolbar {{
      display: flex;
      gap: 8px;
      align-items: center;
      margin-left: auto;
    }}

    .toolbar span {{
      font-size: 0.68rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-right: 4px;
    }}

    button {{
      background: var(--bg);
      border: 1px solid var(--border);
      color: var(--text);
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.72rem;
      padding: 5px 12px;
      border-radius: 4px;
      cursor: pointer;
      transition: all 0.15s;
    }}
    button:hover {{
      background: var(--border);
      border-color: var(--muted);
    }}
    button.active {{
      background: var(--blue);
      border-color: var(--blue);
      color: #fff;
    }}

    .legend-strip {{
      display: flex;
      gap: 16px;
      align-items: center;
      padding: 6px 20px;
      background: var(--bg);
      border-bottom: 1px solid var(--border);
      flex-shrink: 0;
      flex-wrap: wrap;
    }}
    .legend-strip .item {{
      display: flex;
      align-items: center;
      gap: 5px;
      font-size: 0.70rem;
      color: var(--muted);
    }}
    .legend-strip .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 2px;
      flex-shrink: 0;
    }}

    #plot-container {{
      flex: 1;
      min-height: 0;
    }}

    #plot-container .js-plotly-plot,
    #plot-container .plot-container {{
      height: 100% !important;
    }}
  </style>
</head>
<body>
  <header>
    <h1>3D SPINE SEGMENTATION</h1>
    <span class="badge badge-study">Study {study_id}</span>
    <span class="badge badge-castellvi">Castellvi {castellvi}</span>
    <span class="badge badge-tv">TV: {tv_name}</span>
    {ian_badge}
    <div class="toolbar">
      <span>View preset</span>
      <button onclick="setView('lateral')"   id="btn-lateral">Lateral</button>
      <button onclick="setView('posterior')" id="btn-posterior">Posterior</button>
      <button onclick="setView('axial')"     id="btn-axial">Axial</button>
      <button onclick="setView('oblique')"   id="btn-oblique" class="active">Oblique</button>
    </div>
  </header>

  <div class="legend-strip">
    <div class="item"><div class="swatch" style="background:#ff3333"></div> TP Left (seg-spine 43)</div>
    <div class="item"><div class="swatch" style="background:#00d4ff"></div> TP Right (seg-spine 44)</div>
    <div class="item"><div class="swatch" style="background:#ff8c00"></div> Sacrum (TSS 50)</div>
    <div class="item"><div class="swatch" style="background:#1e6fa8;opacity:0.6"></div> L1</div>
    <div class="item"><div class="swatch" style="background:#29a3e8;opacity:0.6"></div> L3</div>
    <div class="item"><div class="swatch" style="background:#85d4ff;opacity:0.6"></div> L5</div>
    <div class="item" style="color:#444466;margin-left:auto;font-size:0.65rem">
      Left-drag=rotate · Scroll=zoom · Legend-click=toggle · Dbl-click=isolate
    </div>
  </div>

  <div id="plot-container">{plotly_div}</div>

  <script>
    const VIEWS = {{
      lateral:   {{ eye: {{ x: 2.2, y: 0.0, z: 0.0 }}, up: {{ x: 0, y: 0, z: 1 }} }},
      posterior: {{ eye: {{ x: 0.0, y: 2.2, z: 0.0 }}, up: {{ x: 0, y: 0, z: 1 }} }},
      axial:     {{ eye: {{ x: 0.0, y: 0.0, z: 2.8 }}, up: {{ x: 0, y: 1, z: 0 }} }},
      oblique:   {{ eye: {{ x: 1.6, y: 0.8, z: 0.4 }}, up: {{ x: 0, y: 0, z: 1 }} }},
    }};

    function setView(name) {{
      const plotDiv = document.querySelector('#plot-container .js-plotly-plot');
      if (!plotDiv) return;
      const v = VIEWS[name];
      Plotly.relayout(plotDiv, {{
        'scene.camera.eye': v.eye,
        'scene.camera.up':  v.up,
      }});
      document.querySelectorAll('.toolbar button').forEach(b => b.classList.remove('active'));
      document.getElementById('btn-' + name).classList.add('active');
    }}

    window.addEventListener('resize', () => {{
      const plotDiv = document.querySelector('#plot-container .js-plotly-plot');
      if (plotDiv) Plotly.Plots.resize(plotDiv);
    }});
  </script>
</body>
</html>"""


# ============================================================================
# SAVE
# ============================================================================

def save_html(fig: go.Figure, study_id: str, output_dir: Path,
              castellvi: str, tv_name: str,
              uncertainty_row: dict = None) -> Path:
    """Embed the plotly figure into the custom HTML wrapper."""
    from plotly.io import to_html

    plotly_div = to_html(
        fig,
        full_html=False,
        include_plotlyjs='cdn',
        config=dict(
            responsive=True,
            displayModeBar=True,
            modeBarButtonsToRemove=['toImage'],
            displaylogo=False,
        ),
    )

    # Build optional Ian Pan badge
    ian_badge = ''
    if uncertainty_row:
        conf = uncertainty_row.get('l5_s1_confidence', float('nan'))
        if not np.isnan(conf):
            ian_badge = (f'<span class="badge badge-ian">'
                         f'Ian Pan L5-S1: {conf:.3f}</span>')

    html = HTML_TEMPLATE.format(
        study_id=study_id,
        castellvi=castellvi or 'N/A',
        tv_name=tv_name or 'N/A',
        ian_badge=ian_badge,
        plotly_div=plotly_div,
    )

    out_path = output_dir / f"{study_id}_3d_spine.html"
    out_path.write_text(html, encoding='utf-8')
    size_mb = out_path.stat().st_size / 1e6
    logger.info(f"  Saved → {out_path}  ({size_mb:.1f} MB)")
    return out_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Interactive 3D Spine Segmentation Viewer — '
                    'study selection mirrors 05_visualize_overlay.py'
    )
    # --- paths ---
    parser.add_argument('--spineps_dir',     required=True)
    parser.add_argument('--totalspine_dir',  required=True)
    parser.add_argument('--output_dir',      required=True)

    # --- study selection (mirrors 05) ---
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--study_id', default=None,
                       help='Single study ID (highest priority)')
    group.add_argument('--all', action='store_true',
                       help='Render every study with SPINEPS segmentation')

    parser.add_argument('--uncertainty_csv', default=None,
                        help='Ian Pan uncertainty CSV (required for selective mode)')
    parser.add_argument('--valid_ids', default=None,
                        help='Path to valid_id.npy for filtering')
    parser.add_argument('--top_n', type=int, default=None,
                        help='Studies from each end of the ranking')
    parser.add_argument('--rank_by', default='l5_s1_confidence',
                        help='CSV column to rank by (default: l5_s1_confidence)')

    # --- annotation ---
    parser.add_argument('--lstv_json', default=None,
                        help='lstv_results.json from 04_detect_lstv.py for annotations')

    # --- rendering ---
    parser.add_argument('--step',   type=int,   default=2,
                        help='Marching cubes subsampling step (1=full res; default 2)')
    parser.add_argument('--smooth', type=float, default=1.5,
                        help='Gaussian pre-smoothing sigma (default 1.5)')
    parser.add_argument('--show_canal', action='store_true',
                        help='Also render spinal canal (label 61, translucent)')
    parser.add_argument('--show_cord',  action='store_true',
                        help='Also render spinal cord (label 60)')

    args = parser.parse_args()

    spineps_dir    = Path(args.spineps_dir)
    totalspine_dir = Path(args.totalspine_dir)
    output_dir     = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seg_root = spineps_dir / 'segmentations'

    # ── Load LSTV results ────────────────────────────────────────────────────
    results_by_id = {}
    if args.lstv_json:
        p = Path(args.lstv_json)
        if p.exists():
            with open(p) as f:
                results_by_id = {str(r['study_id']): r for r in json.load(f)}
            logger.info(f"Loaded {len(results_by_id)} LSTV results for annotation")

    # ── Load uncertainty CSV (used in selective mode + for HTML badges) ──────
    uncertainty_by_id = {}
    csv_path = Path(args.uncertainty_csv) if args.uncertainty_csv else None
    if csv_path and csv_path.exists():
        df_unc = pd.read_csv(csv_path)
        df_unc['study_id'] = df_unc['study_id'].astype(str)
        uncertainty_by_id  = {r['study_id']: r
                               for r in df_unc.to_dict('records')}
        logger.info(
            f"Loaded Ian Pan uncertainty for {len(uncertainty_by_id)} studies")

    # ── Study selection (identical logic to 05_visualize_overlay.py) ─────────
    if args.study_id:
        study_ids = [args.study_id]
        logger.info(f"Single-study mode: {args.study_id}")

    elif args.all:
        study_ids = sorted(d.name for d in seg_root.iterdir() if d.is_dir())
        logger.info(f"ALL mode: {len(study_ids)} studies")

    else:
        # Selective mode — requires uncertainty CSV + top_n
        if not args.uncertainty_csv or args.top_n is None:
            parser.error(
                "--uncertainty_csv and --top_n are required unless "
                "--all or --study_id is given"
            )
        valid_ids = None
        if args.valid_ids:
            valid_ids = set(str(x) for x in np.load(args.valid_ids))

        study_ids = select_studies(csv_path, args.top_n, args.rank_by, valid_ids)
        # Keep only studies that actually have a segmentation directory
        study_ids = [s for s in study_ids if (seg_root / s).is_dir()]
        logger.info(f"Selective mode: {len(study_ids)} studies after directory check")

    # ── Render ───────────────────────────────────────────────────────────────
    ok = 0
    for sid in study_ids:
        logger.info(f"\n[{sid}]")
        try:
            lstv_result     = results_by_id.get(sid)
            uncertainty_row = uncertainty_by_id.get(sid)

            fig = build_3d_figure(
                study_id       = sid,
                spineps_dir    = spineps_dir,
                totalspine_dir = totalspine_dir,
                step           = args.step,
                smooth         = args.smooth,
                show_canal     = args.show_canal,
                show_cord      = args.show_cord,
                lstv_result    = lstv_result,
                uncertainty_row= uncertainty_row,
            )
            if fig is None:
                continue

            castellvi = (lstv_result or {}).get('castellvi_type') or 'N/A'
            tv_name   = ((lstv_result or {})
                         .get('details', {}).get('tv_name', 'N/A'))

            save_html(fig, sid, output_dir, castellvi, tv_name,
                      uncertainty_row=uncertainty_row)
            ok += 1

        except Exception as e:
            logger.error(f"  [{sid}] Failed: {e}")
            logger.debug(traceback.format_exc())

    logger.info(f"\nDone. {ok}/{len(study_ids)} HTML files → {output_dir}")


if __name__ == '__main__':
    main()
