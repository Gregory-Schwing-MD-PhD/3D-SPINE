#!/usr/bin/env python3
"""
06_html_report.py — Hybrid Pipeline LSTV Summary Report
========================================================
Produces a self-contained HTML report from the lstv_results.json output
of the hybrid two-phase 04_detect_lstv.py.

New fields handled vs. the previous SPINEPS-only report:
  - result.details.phase2_available
  - result.{left|right}.phase2.phase2_attempted
  - result.{left|right}.phase2.p2_valid
  - result.{left|right}.phase2.axial_dist_mm
  - result.{left|right}.phase2.p2_features.{patch_mean, coeff_var, reason}
  - result.{left|right}.phase2.midpoint_vox

Usage
-----
  python 06_html_report.py \
      --lstv_json   results/lstv_detection/lstv_results.json \
      --image_dir   results/lstv_visualization \
      --output_html results/lstv_report.html \
      [--n_reps 3]
"""

import argparse
import base64
import json
import logging
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

CASTELLVI_ORDER  = ['Type I', 'Type II', 'Type III', 'Type IV']
CASTELLVI_COLORS = {
    'Type I':   '#3a86ff',
    'Type II':  '#ff9f1c',
    'Type III': '#e63946',
    'Type IV':  '#9d0208',
    'None':     '#444466',
    'N/A':      '#444466',
}
CONFIDENCE_COLORS = {
    'high':     '#2dc653',
    'moderate': '#f4a261',
    'low':      '#e76f51',
}
CONFIDENCE_RANK = {'high': 3, 'moderate': 2, 'low': 1}
P2_VALID_COLOR  = '#2dc653'
P2_FAIL_COLOR   = '#888888'


# ============================================================================
# STATS HELPERS
# ============================================================================

def _finite(v):
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _mean(vals):
    v = [x for x in vals if _finite(x)]
    return sum(v) / len(v) if v else None


def _fmt(v, unit='mm', decimals=1):
    if v is None:
        return 'N/A'
    try:
        return f'{float(v):.{decimals}f} {unit}'.strip()
    except (TypeError, ValueError):
        return 'N/A'


def side_vals(result, key):
    out = []
    for side in ('left', 'right'):
        sd = result.get(side) or {}
        v  = sd.get(key)
        if _finite(v):
            out.append(float(v))
    return out


def p2_side_vals(result, key):
    """Pull a metric from both sides' phase2 sub-dict."""
    out = []
    for side in ('left', 'right'):
        p2 = (result.get(side) or {}).get('phase2') or {}
        v  = p2.get(key)
        if _finite(v):
            out.append(float(v))
    return out


def compute_stats(results):
    total       = len(results)
    lstv_count  = sum(1 for r in results if r.get('lstv_detected'))
    error_count = sum(1 for r in results if r.get('errors'))
    l6_count    = sum(1 for r in results if r.get('details', {}).get('has_l6'))
    p2_avail    = sum(1 for r in results if r.get('details', {}).get('phase2_available'))
    p2_valid_ct = 0
    for r in results:
        for side in ('left', 'right'):
            if (r.get(side) or {}).get('phase2', {}).get('p2_valid'):
                p2_valid_ct += 1
                break

    by_class = defaultdict(lambda: {
        'count': 0, 'heights': [], 'dists': [], 'ax_dists': [], 'tv_names': []
    })

    for r in results:
        ct = r.get('castellvi_type') or 'None'
        by_class[ct]['count'] += 1
        by_class[ct]['heights'].extend(side_vals(r, 'tp_height_mm'))
        by_class[ct]['dists'].extend(side_vals(r, 'dist_mm'))
        by_class[ct]['ax_dists'].extend(p2_side_vals(r, 'axial_dist_mm'))
        tv = r.get('details', {}).get('tv_name', '')
        if tv:
            by_class[ct]['tv_names'].append(tv)

    morpho = {}
    for ct, d in by_class.items():
        morpho[ct] = {
            'mean_h':   _mean(d['heights']),
            'mean_d':   _mean(d['dists']),
            'mean_axd': _mean(d['ax_dists']),
            'min_h':    min(d['heights'])   if d['heights']   else None,
            'max_h':    max(d['heights'])   if d['heights']   else None,
            'min_d':    min(d['dists'])     if d['dists']     else None,
            'max_d':    max(d['dists'])     if d['dists']     else None,
            'l6_frac':  (sum(1 for t in d['tv_names'] if t == 'L6') /
                         len(d['tv_names'])) if d['tv_names'] else None,
        }

    return {
        'overall': {
            'total':         total,
            'lstv_count':    lstv_count,
            'error_count':   error_count,
            'lstv_rate':     100 * lstv_count / max(total, 1),
            'l6_count':      l6_count,
            'p2_avail':      p2_avail,
            'p2_valid_count': p2_valid_ct,
        },
        'by_class': dict(by_class),
        'morpho':   morpho,
    }


def pick_representatives(results, image_dir, n_reps=3):
    by_class = defaultdict(list)
    for r in results:
        ct       = r.get('castellvi_type') or 'None'
        img_path = image_dir / f"{r.get('study_id','')}_lstv_overlay.png"
        if img_path.exists():
            by_class[ct].append(r)

    reps = {}
    for ct, group in by_class.items():
        def _key(r):
            conf = CONFIDENCE_RANK.get(r.get('confidence', 'low'), 0)
            dist = min(
                r.get('left',  {}).get('dist_mm', 999) or 999,
                r.get('right', {}).get('dist_mm', 999) or 999,
            )
            return (-conf, dist)
        group.sort(key=_key)
        reps[ct] = group[:n_reps]
    return reps


# ============================================================================
# IMAGE EMBED
# ============================================================================

def img_b64(path: Path) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


# ============================================================================
# CSS
# ============================================================================

CSS = """
:root {
  --bg:      #0d0d1a;
  --surface: #161628;
  --border:  #2a2a4a;
  --text:    #e8e8f0;
  --muted:   #8888aa;
  --accent:  #3a86ff;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--text);
  font-family: 'Segoe UI', system-ui, sans-serif;
  font-size: 14px; padding: 28px;
  max-width: 1700px; margin: 0 auto;
}
h1 { font-size: 1.9rem; color: var(--accent); margin-bottom: 4px; }
h2 { font-size: 1.25rem; color: var(--accent); margin: 32px 0 12px;
     border-bottom: 1px solid var(--border); padding-bottom: 6px; }
h3 { font-size: 1.05rem; color: var(--text); margin-bottom: 8px; }
.subtitle { color: var(--muted); margin-bottom: 20px; }

.stats-bar {
  display: flex; gap: 18px; flex-wrap: wrap;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 18px; margin-bottom: 28px;
}
.stat { display: flex; flex-direction: column; min-width: 90px; }
.stat .val { font-size: 1.55rem; font-weight: 700; color: var(--accent); }
.stat .lbl { font-size: 0.70rem; color: var(--muted); text-transform: uppercase; letter-spacing:.04em; }

.dist-grid { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 24px; }
.dist-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 16px; min-width: 160px; flex: 1;
}
.dist-card .ct-name  { font-weight: 700; font-size: 1rem; margin-bottom: 4px; }
.dist-card .ct-count { font-size: 2rem; font-weight: 700; }
.dist-card .ct-pct   { color: var(--muted); font-size: 0.85rem; }
.dist-card .ct-bar   { height: 5px; border-radius: 3px; margin-top: 10px; }

table.morpho {
  width: 100%; border-collapse: collapse; margin-bottom: 28px; font-size: 0.87rem;
}
table.morpho th {
  text-align: left; color: var(--muted); padding: 6px 10px;
  border-bottom: 2px solid var(--border); font-weight: 600; white-space: nowrap;
}
table.morpho td { padding: 8px 10px; border-bottom: 1px solid #1e1e36; }
table.morpho tr:last-child td { border-bottom: none; }
.ct-badge {
  display: inline-block; padding: 2px 10px; border-radius: 10px;
  font-size: 0.78rem; font-weight: 600; color: #fff;
}

.rep-section { margin-bottom: 36px; }
.rep-class-header { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
.rep-grid { display: flex; gap: 14px; flex-wrap: wrap; }
.rep-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; overflow: hidden; flex: 1; min-width: 320px; max-width: 580px;
}
.rep-card img  { width: 100%; height: auto; display: block; }
.rep-card .rep-meta {
  padding: 10px 12px; font-size: 0.82rem; color: var(--muted);
}
.rep-card .rep-meta strong { color: var(--text); }

table.results {
  width: 100%; border-collapse: collapse; font-size: 0.84rem; margin-bottom: 32px;
}
table.results th {
  text-align: left; color: var(--muted); padding: 6px 8px;
  border-bottom: 2px solid var(--border); font-weight: 600;
  position: sticky; top: 0; background: var(--bg); z-index: 1;
}
table.results td { padding: 6px 8px; border-bottom: 1px solid #1e1e36; }
table.results tr:hover td { background: #1a1a2e; }
.badge {
  display: inline-block; padding: 2px 8px; border-radius: 10px;
  font-size: 0.75rem; font-weight: 600; color: #fff;
}
.p2-cell { font-size: 0.78rem; color: var(--muted); }
footer {
  text-align: center; color: var(--muted); font-size: 0.8rem;
  margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--border);
}
"""


def _badge(text, color):
    return f'<span class="badge" style="background:{color}">{text}</span>'


# ============================================================================
# HTML SECTIONS
# ============================================================================

def section_summary(stats):
    o    = stats['overall']
    html = '<div class="stats-bar">'
    items = [
        (o['total'],                        'Total Studies'),
        (o['lstv_count'],                   'LSTV Detected'),
        (f"{o['lstv_rate']:.1f}%",          'LSTV Rate'),
        (o['error_count'],                  'Errors/Incomplete'),
        (o['l6_count'],                     'L6 Present'),
        (o['p2_avail'],                     'Phase 2 Available'),
        (o['p2_valid_count'],               'Phase 2 Valid Sides'),
    ]
    for val, lbl in items:
        html += (f'<div class="stat"><span class="val">{val}</span>'
                 f'<span class="lbl">{lbl}</span></div>')
    html += '</div>'
    return html


def section_distribution(stats):
    by_class = stats['by_class']
    total    = stats['overall']['total']
    html     = '<h2>Castellvi Class Distribution</h2><div class="dist-grid">'
    for ct in CASTELLVI_ORDER + ['None']:
        d = by_class.get(ct)
        if not d:
            continue
        count = d['count']
        pct   = 100 * count / max(total, 1)
        color = CASTELLVI_COLORS.get(ct, '#444466')
        html += f"""
<div class="dist-card">
  <div class="ct-name" style="color:{color}">{ct}</div>
  <div class="ct-count" style="color:{color}">{count}</div>
  <div class="ct-pct">{pct:.1f}% of all studies</div>
  <div class="ct-bar" style="background:{color};width:{min(pct,100):.0f}%"></div>
</div>"""
    html += '</div>'
    return html


def section_morphometrics(stats):
    morpho   = stats['morpho']
    by_class = stats['by_class']
    html     = '<h2>Average Morphometrics by Castellvi Class</h2>'
    html += '''<table class="morpho">
<thead><tr>
  <th>Class</th><th>N</th>
  <th>P1 TP height mean (range) mm</th>
  <th>P1 TP–Sacrum dist mean (range) mm</th>
  <th>P2 Axial dist mean mm</th>
  <th>L6 rate</th>
</tr></thead><tbody>'''

    for ct in CASTELLVI_ORDER + ['None']:
        m = morpho.get(ct)
        d = by_class.get(ct)
        if not m or not d:
            continue
        color = CASTELLVI_COLORS.get(ct, '#444466')

        def _rng(mn, mx):
            return f'{mn:.1f}–{mx:.1f}' if mn is not None else 'N/A'

        mean_h  = f'{m["mean_h"]:.1f} ({_rng(m["min_h"],m["max_h"])})' if m['mean_h'] else 'N/A'
        mean_d  = f'{m["mean_d"]:.1f} ({_rng(m["min_d"],m["max_d"])})' if m['mean_d'] else 'N/A'
        mean_ax = f'{m["mean_axd"]:.2f}' if m['mean_axd'] else 'N/A'
        l6_frac = f'{100*m["l6_frac"]:.0f}%' if m['l6_frac'] is not None else 'N/A'

        html += f'''<tr>
  <td><span class="ct-badge" style="background:{color}">{ct}</span></td>
  <td>{d["count"]}</td>
  <td>{mean_h}</td>
  <td>{mean_d}</td>
  <td>{mean_ax}</td>
  <td>{l6_frac}</td>
</tr>'''
    html += '</tbody></table>'
    return html


def section_representatives(reps, image_dir):
    html = '<h2>Representative Cases by Castellvi Class</h2>'
    for ct in CASTELLVI_ORDER + ['None']:
        cases = reps.get(ct)
        if not cases:
            continue
        color = CASTELLVI_COLORS.get(ct, '#444466')
        html += f'''<div class="rep-section">
  <div class="rep-class-header">
    {_badge(ct, color)}
    <h3>{ct} — {len(cases)} representative{"s" if len(cases)>1 else ""}</h3>
  </div>
  <div class="rep-grid">'''
        for r in cases:
            sid      = r.get('study_id', '?')
            conf     = r.get('confidence', 'N/A')
            tv       = r.get('details', {}).get('tv_name', '?')
            dist_L   = r.get('left',  {}).get('dist_mm')
            dist_R   = r.get('right', {}).get('dist_mm')
            h_L      = r.get('left',  {}).get('tp_height_mm')
            h_R      = r.get('right', {}).get('tp_height_mm')
            p2a_L    = (r.get('left',  {}).get('phase2') or {})
            p2a_R    = (r.get('right', {}).get('phase2') or {})
            p2v_L    = '✓' if p2a_L.get('p2_valid') else '—'
            p2v_R    = '✓' if p2a_R.get('p2_valid') else '—'
            conf_col = CONFIDENCE_COLORS.get(conf, '#888')
            img_path = image_dir / f"{sid}_lstv_overlay.png"
            b64      = img_b64(img_path)
            html += f'''<div class="rep-card">
  <img src="data:image/png;base64,{b64}" alt="overlay {sid}" loading="lazy">
  <div class="rep-meta">
    <strong>{sid}</strong> &nbsp;
    {_badge(ct, color)} {_badge(conf, conf_col)}
    &nbsp; TV: {tv}<br>
    Ht L/R: {_fmt(h_L)}/{_fmt(h_R)} &nbsp;|&nbsp;
    P1-dist L/R: {_fmt(dist_L)}/{_fmt(dist_R)}<br>
    P2 valid L/R: {p2v_L}/{p2v_R}
  </div>
</div>'''
        html += '</div></div>'
    return html


def section_full_table(results):
    lstv = [r for r in results if r.get('lstv_detected')]
    lstv.sort(key=lambda r: (
        -{'Type IV':4,'Type III':3,'Type II':2,'Type I':1}.get(r.get('castellvi_type') or '', 0),
        -CONFIDENCE_RANK.get(r.get('confidence','low'), 0),
    ))
    if not lstv:
        return '<h2>LSTV Cases</h2><p style="color:var(--muted)">No LSTV detected.</p>'

    html = f'<h2>All LSTV Cases ({len(lstv)})</h2>'
    html += '''<table class="results">
<thead><tr>
  <th>#</th><th>Study</th><th>Castellvi</th><th>TV</th>
  <th>L ht mm</th><th>R ht mm</th>
  <th>L P1-dist</th><th>R P1-dist</th>
  <th>L class</th><th>R class</th>
  <th>P2-L</th><th>P2-R</th>
  <th>Errors</th>
</tr></thead><tbody>'''

    for i, r in enumerate(lstv, 1):
        sid   = r.get('study_id', '?')
        ct    = r.get('castellvi_type') or 'N/A'
        tv    = r.get('details', {}).get('tv_name', '?')
        ld    = r.get('left',  {}) or {}
        rd    = r.get('right', {}) or {}
        p2l   = ld.get('phase2') or {}
        p2r   = rd.get('phase2') or {}
        errs  = '; '.join(r.get('errors', []))
        ct_c  = CASTELLVI_COLORS.get(ct, '#444')

        def _p2_cell(p2):
            if not p2.get('phase2_attempted'):
                return '<span class="p2-cell">—</span>'
            valid   = p2.get('p2_valid', False)
            cls     = p2.get('classification', '?')
            ax_dist = p2.get('axial_dist_mm')
            color   = P2_VALID_COLOR if valid else P2_FAIL_COLOR
            dist_s  = f' ({ax_dist:.2f}mm)' if _finite(ax_dist) else ''
            return (f'<span class="p2-cell" style="color:{color}">'
                    f'{"✓" if valid else "✗"} {cls}{dist_s}</span>')

        html += f'''<tr>
  <td>{i}</td>
  <td><strong>{sid}</strong></td>
  <td>{_badge(ct, ct_c)}</td>
  <td>{tv}</td>
  <td>{_fmt(ld.get("tp_height_mm"), "")}</td>
  <td>{_fmt(rd.get("tp_height_mm"), "")}</td>
  <td>{_fmt(ld.get("dist_mm"), "")}</td>
  <td>{_fmt(rd.get("dist_mm"), "")}</td>
  <td>{ld.get("classification","—")}</td>
  <td>{rd.get("classification","—")}</td>
  <td>{_p2_cell(p2l)}</td>
  <td>{_p2_cell(p2r)}</td>
  <td style="color:#ff8080;font-size:0.78em">{errs}</td>
</tr>'''
    html += '</tbody></table>'
    return html


# ============================================================================
# BUILD
# ============================================================================

def build_report(results_path: Path, image_dir: Path,
                 output_html: Path, n_reps: int = 3):
    with open(results_path) as f:
        all_results = json.load(f)
    logger.info(f"Loaded {len(all_results)} results")

    stats = compute_stats(all_results)
    reps  = pick_representatives(all_results, image_dir, n_reps)
    ts    = datetime.now().strftime('%Y-%m-%d %H:%M')

    body = (
        section_summary(stats)
        + section_distribution(stats)
        + section_morphometrics(stats)
        + section_representatives(reps, image_dir)
        + section_full_table(all_results)
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LSTV Detection Report — {ts}</title>
  <style>{CSS}</style>
</head>
<body>
  <h1>LSTV Detection Report</h1>
  <p class="subtitle">
    Hybrid Two-Phase Pipeline &nbsp;|&nbsp;
    Phase 1: Sagittal (SPINEPS + VERIDAH + TotalSpineSeg) &nbsp;|&nbsp;
    Phase 2: Axial (registered SPINEPS TP + native TSS sacrum) &nbsp;|&nbsp;
    {ts}
  </p>
  {body}
  <footer>
    LSTV Detection Pipeline — Hybrid Phase1-Sagittal / Phase2-Axial &nbsp;|&nbsp; {ts}
  </footer>
</body>
</html>"""

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding='utf-8')
    size_mb = output_html.stat().st_size / 1e6
    logger.info(f"Report written: {output_html}  ({size_mb:.1f} MB)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Hybrid Pipeline LSTV HTML Report Generator'
    )
    parser.add_argument('--lstv_json',   required=True)
    parser.add_argument('--image_dir',   required=True)
    parser.add_argument('--output_html', required=True)
    parser.add_argument('--n_reps',      type=int, default=3)
    args = parser.parse_args()

    build_report(
        results_path = Path(args.lstv_json),
        image_dir    = Path(args.image_dir),
        output_html  = Path(args.output_html),
        n_reps       = args.n_reps,
    )


if __name__ == '__main__':
    main()
