#!/usr/bin/env python3
"""
06_html_report.py — LSTV Pipeline Summary Report
=================================================
Produces a self-contained HTML report with:
  1. Pipeline summary stats (total studies, LSTV rate, error rate)
  2. Castellvi class distribution (counts + percentages)
  3. Per-class average morphometrics (TP height, TP-sacrum distance,
     P(Type III)) with min/max ranges
  4. Representative overlay images for each Castellvi class
     (up to N_REPS per class, ranked by confidence then dist)
  5. Full ranked table of all LSTV-positive cases

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


# ============================================================================
# STATS HELPERS
# ============================================================================

def _finite(v):
    return v is not None and math.isfinite(float(v))


def _mean(vals):
    v = [x for x in vals if _finite(x)]
    return sum(v) / len(v) if v else None


def _fmt(v, unit='mm', decimals=1):
    if v is None:
        return 'N/A'
    return f'{v:.{decimals}f} {unit}'.strip()


def side_vals(result, key):
    """Pull a metric from both sides of a result."""
    out = []
    for side in ('left', 'right'):
        sd = result.get(side) or {}
        v  = sd.get(key)
        if _finite(v):
            out.append(float(v))
    return out


def compute_stats(results):
    """
    Returns a dict:
      overall   -> {total, lstv_count, error_count, lstv_rate, l6_count}
      by_class  -> {castellvi_type: {count, heights, dists, p3s, tv_names}}
      morpho    -> {castellvi_type: {mean_h, mean_d, mean_p3, min_h, max_h, ...}}
    """
    total       = len(results)
    lstv_count  = sum(1 for r in results if r.get('lstv_detected'))
    error_count = sum(1 for r in results if r.get('errors'))
    l6_count    = sum(1 for r in results
                      if r.get('details', {}).get('has_l6'))

    by_class = defaultdict(lambda: {
        'count': 0, 'heights': [], 'dists': [], 'p3s': [], 'tv_names': []
    })

    for r in results:
        ct = r.get('castellvi_type') or 'None'
        by_class[ct]['count'] += 1
        by_class[ct]['heights'].extend(side_vals(r, 'tp_height_mm'))
        by_class[ct]['dists'].extend(side_vals(r, 'dist_mm'))
        by_class[ct]['p3s'].extend(
            [v for v in side_vals(r, 'p_type_iii') if _finite(v)]
        )
        tv = r.get('details', {}).get('tv_name', '')
        if tv:
            by_class[ct]['tv_names'].append(tv)

    morpho = {}
    for ct, d in by_class.items():
        morpho[ct] = {
            'mean_h':  _mean(d['heights']),
            'mean_d':  _mean(d['dists']),
            'mean_p3': _mean(d['p3s']),
            'min_h':   min(d['heights']) if d['heights'] else None,
            'max_h':   max(d['heights']) if d['heights'] else None,
            'min_d':   min(d['dists'])   if d['dists']   else None,
            'max_d':   max(d['dists'])   if d['dists']   else None,
            'l6_frac': (sum(1 for t in d['tv_names'] if t == 'L6') /
                        len(d['tv_names'])) if d['tv_names'] else None,
        }

    return {
        'overall': {
            'total':       total,
            'lstv_count':  lstv_count,
            'error_count': error_count,
            'lstv_rate':   100 * lstv_count / max(total, 1),
            'l6_count':    l6_count,
        },
        'by_class': dict(by_class),
        'morpho':   morpho,
    }


def pick_representatives(results, image_dir, n_reps=3):
    """
    For each Castellvi class, pick up to n_reps studies that have overlay PNGs,
    ranked by confidence desc then dist asc.
    Returns {castellvi_type: [result, ...]}
    """
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
                r.get('left',  {}).get('dist_mm', 999),
                r.get('right', {}).get('dist_mm', 999),
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
# HTML PIECES
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
  max-width: 1600px; margin: 0 auto;
}
h1 { font-size: 1.9rem; color: var(--accent); margin-bottom: 4px; }
h2 { font-size: 1.25rem; color: var(--accent); margin: 32px 0 12px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }
h3 { font-size: 1.05rem; color: var(--text); margin-bottom: 8px; }
.subtitle { color: var(--muted); margin-bottom: 20px; }

/* Stats bar */
.stats-bar {
  display: flex; gap: 20px; flex-wrap: wrap;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 18px; margin-bottom: 28px;
}
.stat { display: flex; flex-direction: column; min-width: 90px; }
.stat .val { font-size: 1.6rem; font-weight: 700; color: var(--accent); }
.stat .lbl { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }

/* Distribution grid */
.dist-grid { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 24px; }
.dist-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 16px; min-width: 170px; flex: 1;
}
.dist-card .ct-name { font-weight: 700; font-size: 1rem; margin-bottom: 4px; }
.dist-card .ct-count { font-size: 2rem; font-weight: 700; }
.dist-card .ct-pct { color: var(--muted); font-size: 0.85rem; }
.dist-card .ct-bar {
  height: 5px; border-radius: 3px; margin-top: 10px;
}

/* Morpho table */
table.morpho {
  width: 100%; border-collapse: collapse; margin-bottom: 28px;
  font-size: 0.88rem;
}
table.morpho th {
  text-align: left; color: var(--muted); padding: 6px 10px;
  border-bottom: 2px solid var(--border); font-weight: 600;
  white-space: nowrap;
}
table.morpho td {
  padding: 8px 10px; border-bottom: 1px solid #1e1e36;
}
table.morpho tr:last-child td { border-bottom: none; }
table.morpho .ct-badge {
  display: inline-block; padding: 2px 10px; border-radius: 10px;
  font-size: 0.78rem; font-weight: 600; color: #fff;
}

/* Representative images */
.rep-section { margin-bottom: 36px; }
.rep-class-header {
  display: flex; align-items: center; gap: 10px;
  margin-bottom: 14px;
}
.rep-grid { display: flex; gap: 14px; flex-wrap: wrap; }
.rep-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; overflow: hidden; flex: 1; min-width: 320px; max-width: 560px;
}
.rep-card img { width: 100%; height: auto; display: block; }
.rep-card .rep-meta {
  padding: 10px 12px; font-size: 0.82rem; color: var(--muted);
}
.rep-card .rep-meta strong { color: var(--text); }

/* Full results table */
table.results {
  width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-bottom: 32px;
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
footer {
  text-align: center; color: var(--muted); font-size: 0.8rem;
  margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--border);
}
"""


def _badge(text, color):
    return f'<span class="badge" style="background:{color}">{text}</span>'


def section_summary(stats):
    o = stats['overall']
    html = '<div class="stats-bar">'
    items = [
        (o['total'],                          'Total Studies'),
        (o['lstv_count'],                     'LSTV Detected'),
        (f"{o['lstv_rate']:.1f}%",            'LSTV Rate'),
        (o['error_count'],                    'Errors / Incomplete'),
        (o['l6_count'],                       'L6 Present'),
    ]
    for val, lbl in items:
        html += f'<div class="stat"><span class="val">{val}</span><span class="lbl">{lbl}</span></div>'
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
    html = '<h2>Average Morphometrics by Castellvi Class</h2>'
    html += '''<table class="morpho">
<thead><tr>
  <th>Class</th><th>N</th>
  <th>TP Height mean (range) mm</th>
  <th>TP–Sacrum dist mean (range) mm</th>
  <th>P(Type III) mean</th>
  <th>L6 rate</th>
</tr></thead><tbody>'''

    for ct in CASTELLVI_ORDER + ['None']:
        m = morpho.get(ct)
        d = by_class.get(ct)
        if not m or not d:
            continue
        color = CASTELLVI_COLORS.get(ct, '#444466')

        def _rng(mn, mx, unit='mm'):
            if mn is None:
                return 'N/A'
            return f'{mn:.1f}–{mx:.1f} {unit}'

        mean_h_str = (f'{m["mean_h"]:.1f} ({_rng(m["min_h"], m["max_h"])})'
                      if m['mean_h'] is not None else 'N/A')
        mean_d_str = (f'{m["mean_d"]:.1f} ({_rng(m["min_d"], m["max_d"])})'
                      if m['mean_d'] is not None else 'N/A')
        mean_p3    = f'{m["mean_p3"]:.2f}' if m['mean_p3'] is not None else 'N/A'
        l6_frac    = f'{100*m["l6_frac"]:.0f}%' if m['l6_frac'] is not None else 'N/A'

        html += f'''<tr>
  <td><span class="ct-badge" style="background:{color}">{ct}</span></td>
  <td>{d["count"]}</td>
  <td>{mean_h_str}</td>
  <td>{mean_d_str}</td>
  <td>{mean_p3}</td>
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
            dist_L   = r.get('left',  {}).get('dist_mm', None)
            dist_R   = r.get('right', {}).get('dist_mm', None)
            h_L      = r.get('left',  {}).get('tp_height_mm', None)
            h_R      = r.get('right', {}).get('tp_height_mm', None)
            conf_col = CONFIDENCE_COLORS.get(conf, '#888')
            img_path = image_dir / f"{sid}_lstv_overlay.png"
            b64      = img_b64(img_path)
            html += f'''<div class="rep-card">
  <img src="data:image/png;base64,{b64}" alt="overlay {sid}" loading="lazy">
  <div class="rep-meta">
    <strong>{sid}</strong> &nbsp;
    {_badge(ct, color)} {_badge(conf, conf_col)}
    &nbsp; TV: {tv}<br>
    Height L/R: {_fmt(h_L)}/{_fmt(h_R)} &nbsp;|&nbsp;
    TP–Sacrum L/R: {_fmt(dist_L)}/{_fmt(dist_R)}
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
  <th>#</th><th>Study</th><th>Castellvi</th><th>Confidence</th><th>TV</th>
  <th>L ht mm</th><th>R ht mm</th>
  <th>L dist mm</th><th>R dist mm</th>
  <th>L class</th><th>R class</th>
  <th>Errors</th>
</tr></thead><tbody>'''

    for i, r in enumerate(lstv, 1):
        sid   = r.get('study_id', '?')
        ct    = r.get('castellvi_type') or 'N/A'
        conf  = r.get('confidence', 'N/A')
        tv    = r.get('details', {}).get('tv_name', '?')
        ld    = r.get('left',  {})
        rd    = r.get('right', {})
        errs  = '; '.join(r.get('errors', []))
        ct_c  = CASTELLVI_COLORS.get(ct, '#444')
        cf_c  = CONFIDENCE_COLORS.get(conf, '#888')
        html += f'''<tr>
  <td>{i}</td>
  <td><strong>{sid}</strong></td>
  <td>{_badge(ct, ct_c)}</td>
  <td>{_badge(conf, cf_c)}</td>
  <td>{tv}</td>
  <td>{_fmt(ld.get("tp_height_mm"), "")}</td>
  <td>{_fmt(rd.get("tp_height_mm"), "")}</td>
  <td>{_fmt(ld.get("dist_mm"), "")}</td>
  <td>{_fmt(rd.get("dist_mm"), "")}</td>
  <td>{ld.get("classification","—")}</td>
  <td>{rd.get("classification","—")}</td>
  <td style="color:#ff8080;font-size:0.8em">{errs}</td>
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
        + '<h2>Representative Cases by Class</h2>'
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
  <p class="subtitle">SPINEPS + VERIDAH pipeline &nbsp;|&nbsp; Castellvi classification &nbsp;|&nbsp; {ts}</p>
  {body}
  <footer>LSTV Detection Pipeline &nbsp;|&nbsp; SPINEPS + VERIDAH &nbsp;|&nbsp; {ts}</footer>
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
    parser = argparse.ArgumentParser(description='LSTV HTML Report Generator')
    parser.add_argument('--lstv_json',   required=True)
    parser.add_argument('--image_dir',   required=True)
    parser.add_argument('--output_html', required=True)
    parser.add_argument('--n_reps',      type=int, default=3,
                        help='Representative images per Castellvi class (default 3)')
    args = parser.parse_args()

    build_report(
        results_path = Path(args.lstv_json),
        image_dir    = Path(args.image_dir),
        output_html  = Path(args.output_html),
        n_reps       = args.n_reps,
    )


if __name__ == '__main__':
    main()
