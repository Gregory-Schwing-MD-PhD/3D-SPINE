"""
ian_pan_integration.py
======================
Integration guide showing exactly how to wire Ian Pan's full disc-level
confidence sequence into:

  A.  04_detect_lstv.py         — sequence-level H0/H1 alignment tiebreaker
  B.  vertebral_alignment.py    — AlignmentResult fields + tiebreaker logic
  C.  06_visualize_3d.py        — all 5 disc-level peaks on 3D renders
  D.  Diagnostic pandas queries

WHY ALL FIVE DISC LEVELS?
--------------------------
A single L5-S1 distance comparison is fragile because:
  1. Ian Pan's 2D → 3D mapping has ~5–10mm geometric uncertainty.
  2. One outlier disc (e.g. L5-S1 partially hidden by sacrum) corrupts the vote.
  3. The hypotheses are about a WHOLE-SEQUENCE label shift, so the evidence
     is the coherence of ALL disc positions relative to TSS, not just the
     bottommost disc.

With all five levels, the vote becomes:
  H0 score = mean(dist(IP_k, TSS_k))    for k = L1/L2 … L5/S1
  H1 score = mean(dist(IP_k, TSS_{k-1})) for k = L1/L2 … L5/S1

If H0_score < H1_score by > SEQ_VOTE_MIN_MARGIN_MM → H0 wins (TSS labels correct)
If H1_score < H0_score by > SEQ_VOTE_MIN_MARGIN_MM → H1 wins (lumbarization)
Otherwise → neutral (existing Dice result stands)

Only levels where Ian Pan's peak_prob >= SEQ_VOTE_MIN_PROB (0.30) vote.
The paired comparison uses only levels where BOTH H0 and H1 TSS centroids exist
(l1_l2 has no H1 match so it never contributes to the H1 side).

This is precomputed by compute_model_agreement() in ian_pan_disc_coords.py and
stored under model_agreement['_sequence_vote'].  No extra computation needed at
classification time — just read the stored vote.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

DISC_NAMES             = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
SEQ_VOTE_MIN_PROB      = 0.30
SEQ_VOTE_MIN_MARGIN_MM = 8.0
DICE_AMBIGUITY_MAX     = 0.05   # Ian Pan only overrides when Dice margin < this


# ─────────────────────────────────────────────────────────────────────────────
# A.  CHANGES TO 04_detect_lstv.py
# ─────────────────────────────────────────────────────────────────────────────
#
# 1. Add import at top:
#       from ian_pan_disc_coords import load_ian_pan_disc_coords
#
# 2. Add CLI argument in main():
#       parser.add_argument('--ian_pan_coords', default=None,
#           help='results/ian_pan_disc_coords/ian_pan_disc_coords.json')
#
# 3. Load before the study loop in main():
#       IAN_PAN = {}
#       if args.ian_pan_coords:
#           IAN_PAN = load_ian_pan_disc_coords(args.ian_pan_coords)
#           logger.info(f"Ian Pan disc coords: {len(IAN_PAN)} studies loaded")
#
# 4. Pass to classify_study():
#       r, ar = classify_study(
#           sid, spineps_dir, totalspine_dir, registered_dir, nifti_dir,
#           run_morpho=not args.no_morpho,
#           ian_pan_study=IAN_PAN.get(sid),
#       )
#
# 5. In classify_study() body, after the alignment analysis block:
#
#       if ian_pan_study is not None and alignment_result is not None:
#           apply_ian_pan_tiebreaker(alignment_result, ian_pan_study, study_id)
#           out['alignment'] = alignment_result.to_dict()   # refresh serialised copy
#
# 6. Add to 04_lstv_detection.sh singularity exec args:
#       --ian_pan_coords /work/results/ian_pan_disc_coords/ian_pan_disc_coords.json


# ─────────────────────────────────────────────────────────────────────────────
# B.  TIEBREAKER FUNCTION  —  drop into 04_detect_lstv.py
# ─────────────────────────────────────────────────────────────────────────────

def apply_ian_pan_tiebreaker(
    alignment_result,          # vertebral_alignment.AlignmentResult
    ian_pan_study: dict,
    study_id: str,
) -> None:
    """
    Reads the precomputed _sequence_vote from model_agreement and optionally
    overrides alignment_result.preferred_hypothesis.

    Ian Pan overrides ONLY when:
      - Its sequence vote is H0 or H1 (not neutral)
      - Vote confidence is moderate or high
      - At least 3 disc levels contributed
      - The existing Dice margin is ambiguous (< DICE_AMBIGUITY_MAX = 0.05)

    All Ian Pan fields are attached to alignment_result regardless of whether
    the override fires — they appear in lstv_alignment.csv for full auditability.

    New fields on AlignmentResult (set here):
      ip_sequence_vote       H0 | H1 | neutral
      ip_vote_confidence     high | moderate | low | insufficient
      ip_seq_margin_mm       mean_dist_h0 - mean_dist_h1  (mm)
      ip_mean_dist_h0_mm     mean distance under H0 across contributing levels
      ip_mean_dist_h1_mm     mean distance under H1 across contributing levels
      ip_n_levels_voted      n disc levels that contributed to the vote
      ip_tiebreak_applied    True if Ian Pan changed preferred_hypothesis
      ip_per_level           {disc_name: {peak_prob, dist_h0_mm, dist_h1_mm,
                                          closer_hyp}}  for all 5 levels
    """
    if not ian_pan_study:
        return

    agr = ian_pan_study.get("model_agreement", {})
    sv  = agr.get("_sequence_vote", {})

    ip_vote   = sv.get("sequence_vote",  "neutral")
    ip_conf   = sv.get("vote_confidence","insufficient")
    ip_margin = sv.get("margin_mm",       0.0) or 0.0
    ip_h0     = sv.get("mean_dist_h0_mm")
    ip_h1     = sv.get("mean_dist_h1_mm")
    ip_n      = sv.get("n_levels_voted",  0)

    # Build per-level detail dict (all 5 levels, regardless of whether they voted)
    per_level: Dict = {}
    for disc_name in DISC_NAMES:
        d_ag = agr.get(disc_name, {})
        d_lv = ian_pan_study.get("disc_levels", {}).get(disc_name, {})
        d0   = d_ag.get("dist_to_tss_h0_mm")
        d1   = d_ag.get("dist_to_tss_h1_mm")
        per_level[disc_name] = {
            "peak_prob":  d_lv.get("peak_prob"),
            "entropy":    d_lv.get("entropy"),
            "dist_h0_mm": d0,
            "dist_h1_mm": d1,
            "closer_hyp": ("H0" if (d0 is not None and d1 is not None and d0 <= d1)
                           else "H1" if (d0 is not None and d1 is not None)
                           else None),
        }

    # Attach all fields unconditionally
    alignment_result.ip_sequence_vote    = ip_vote
    alignment_result.ip_vote_confidence  = ip_conf
    alignment_result.ip_seq_margin_mm    = round(ip_margin, 2)
    alignment_result.ip_mean_dist_h0_mm  = ip_h0
    alignment_result.ip_mean_dist_h1_mm  = ip_h1
    alignment_result.ip_n_levels_voted   = ip_n
    alignment_result.ip_tiebreak_applied = False
    alignment_result.ip_per_level        = per_level

    logger.info(
        f"  [{study_id}] Ian Pan seq vote: {ip_vote} [{ip_conf}]  "
        f"margin={ip_margin:+.1f}mm  "
        f"H0={ip_h0}mm  H1={ip_h1}mm  "
        f"n={ip_n} levels"
    )

    # ── Log per-level detail ───────────────────────────────────────────────────
    for disc_name in DISC_NAMES:
        pl = per_level[disc_name]
        d0 = pl.get("dist_h0_mm"); d1 = pl.get("dist_h1_mm")
        if d0 is not None and d1 is not None:
            flag = "⚠" if pl["closer_hyp"] == "H1" else "✓"
            logger.info(
                f"    {disc_name}: p={pl['peak_prob']:.2f}  "
                f"H0={d0:.1f}mm  H1={d1:.1f}mm  "
                f"→ {pl['closer_hyp']} {flag}"
            )

    # ── Tiebreaker decision ────────────────────────────────────────────────────
    if ip_vote == "neutral" or ip_conf in ("insufficient", "low"):
        return

    if ip_n < 3:
        logger.info(f"  [{study_id}] Ian Pan tiebreaker skipped: only {ip_n} levels voted")
        return

    # Get existing Dice margin
    dice_h0 = getattr(alignment_result, "mean_dice_h0", None)
    dice_h1 = getattr(alignment_result, "mean_dice_h1", None)
    dice_margin = abs(dice_h0 - dice_h1) if (dice_h0 is not None and dice_h1 is not None) else 0.0

    if dice_margin >= DICE_AMBIGUITY_MAX:
        logger.info(
            f"  [{study_id}] Ian Pan tiebreaker skipped: "
            f"Dice margin={dice_margin:.3f} >= {DICE_AMBIGUITY_MAX} (Dice wins)"
        )
        return

    old_hyp = alignment_result.preferred_hypothesis
    new_hyp = "aligned" if ip_vote == "H0" else "shifted_plus_1"

    if new_hyp == old_hyp:
        logger.info(f"  [{study_id}] Ian Pan agrees with Dice: {old_hyp}")
        return

    alignment_result.preferred_hypothesis = new_hyp
    alignment_result.confidence           = "moderate"
    alignment_result.ip_tiebreak_applied  = True

    logger.info(
        f"  [{study_id}] ⚡ Ian Pan TIEBREAK: {old_hyp} → {new_hyp}  "
        f"(Dice margin={dice_margin:.3f}, IP margin={ip_margin:+.1f}mm [{ip_conf}])"
    )


# ─────────────────────────────────────────────────────────────────────────────
# C.  CHANGES TO vertebral_alignment.py  (AlignmentResult dataclass)
# ─────────────────────────────────────────────────────────────────────────────
#
# Add these optional fields to AlignmentResult:
#
#   @dataclass
#   class AlignmentResult:
#       ...
#       # Ian Pan disc sequence integration (populated by apply_ian_pan_tiebreaker)
#       ip_sequence_vote:    Optional[str]   = None
#       ip_vote_confidence:  Optional[str]   = None
#       ip_seq_margin_mm:    Optional[float] = None
#       ip_mean_dist_h0_mm:  Optional[float] = None
#       ip_mean_dist_h1_mm:  Optional[float] = None
#       ip_n_levels_voted:   Optional[int]   = None
#       ip_tiebreak_applied: bool            = False
#       ip_per_level:        Optional[dict]  = None
#
# Add to AlignmentResult.to_dict():
#   'ip_sequence_vote':    self.ip_sequence_vote,
#   'ip_vote_confidence':  self.ip_vote_confidence,
#   'ip_seq_margin_mm':    self.ip_seq_margin_mm,
#   'ip_mean_dist_h0_mm':  self.ip_mean_dist_h0_mm,
#   'ip_mean_dist_h1_mm':  self.ip_mean_dist_h1_mm,
#   'ip_n_levels_voted':   self.ip_n_levels_voted,
#   'ip_tiebreak_applied': self.ip_tiebreak_applied,
#   'ip_per_level':        self.ip_per_level,
#
# Add to lstv_csv_reporter.py (lstv_alignment.csv columns):
#   'ip_sequence_vote', 'ip_vote_confidence', 'ip_seq_margin_mm',
#   'ip_mean_dist_h0_mm', 'ip_mean_dist_h1_mm',
#   'ip_n_levels_voted', 'ip_tiebreak_applied'


# ─────────────────────────────────────────────────────────────────────────────
# D.  CHANGES TO 06_visualize_3d.py
# ─────────────────────────────────────────────────────────────────────────────

def ian_pan_disc_marker_traces(
    ian_pan_study: dict,
    origin_mm: "np.ndarray",
) -> list:
    """
    Scatter3d markers for ALL five Ian Pan disc peak positions.

    Colour = confidence:  green (high) / yellow (medium) / red (low)
    Label  = disc level + peak_prob + H0 and H1 distances + closer hypothesis

    Integration in build_3d_figure(), after mesh accumulation:

        if ian_pan_lookup:
            ip_study = ian_pan_lookup.get(sid)
            for t in ian_pan_disc_marker_traces(ip_study, origin_mm):
                _add(t, 'focused')

    Pass ian_pan_lookup = load_ian_pan_disc_coords(...) into the visualiser
    via a new --ian_pan_coords CLI arg (same pattern as --lstv_json).
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return []

    if not ian_pan_study:
        return []

    conf_colours = {"high": "#00ff88", "medium": "#ffcc00", "low": "#ff4444"}
    agr          = ian_pan_study.get("model_agreement", {})
    disc_levels  = ian_pan_study.get("disc_levels", {})
    traces       = []

    for disc_name in DISC_NAMES:
        lvl  = disc_levels.get(disc_name, {})
        d_ag = agr.get(disc_name, {})

        ras_mm = lvl.get("world_ras_mm")
        prob   = lvl.get("peak_prob", 0.0)
        conf   = lvl.get("confidence_class", "low")
        colour = conf_colours.get(conf, "#888888")

        if ras_mm is None:
            continue

        d0 = d_ag.get("dist_to_tss_h0_mm")
        d1 = d_ag.get("dist_to_tss_h1_mm")
        if d0 is not None and d1 is not None:
            closer = "H0" if d0 <= d1 else "H1"
            margin = abs(d0 - d1)
            dist_str = f"  d0={d0:.0f} d1={d1:.0f} → {closer} +{margin:.0f}mm"
        else:
            dist_str = ""

        label = f"IP {disc_name.replace('_','/')} p={prob:.2f}{dist_str}"
        disp  = np.array(ras_mm) - origin_mm

        traces.append(go.Scatter3d(
            x=[float(disp[0])], y=[float(disp[1])], z=[float(disp[2])],
            mode="markers+text",
            marker=dict(size=9, color=colour, symbol="diamond", opacity=0.90,
                        line=dict(color="white", width=1)),
            text=[label], textposition="top center",
            textfont=dict(size=9, color=colour),
            name=label, showlegend=True, hoverinfo="text",
        ))

    # Sequence vote summary marker (positioned to the right of the disc column)
    sv        = agr.get("_sequence_vote", {})
    vote      = sv.get("sequence_vote", "")
    vote_conf = sv.get("vote_confidence", "")
    margin    = sv.get("margin_mm", 0.0)
    if vote and vote != "neutral":
        valid_ras = [
            np.array(disc_levels[d]["world_ras_mm"])
            for d in DISC_NAMES
            if disc_levels.get(d, {}).get("world_ras_mm") is not None
        ]
        if valid_ras:
            mid       = np.mean(valid_ras, axis=0) - origin_mm
            vote_col  = "#00ff88" if vote == "H0" else "#ff8800"
            vote_text = f"IP seq: {vote} [{vote_conf}] {margin:+.0f}mm"
            traces.append(go.Scatter3d(
                x=[float(mid[0]) + 10], y=[float(mid[1])], z=[float(mid[2])],
                mode="markers+text",
                marker=dict(size=5, color=vote_col, symbol="square"),
                text=[vote_text], textposition="middle right",
                textfont=dict(size=10, color=vote_col),
                name=vote_text, showlegend=True, hoverinfo="text",
            ))

    return traces


# ─────────────────────────────────────────────────────────────────────────────
# E.  DIAGNOSTIC PANDAS QUERIES
# ─────────────────────────────────────────────────────────────────────────────
#
# Long CSV: one row per study × disc level.  H0 and H1 distances and the
# study-level sequence vote are all present on every row for easy filtering.
#
#   import pandas as pd
#   df   = pd.read_csv('results/ian_pan_disc_coords/ian_pan_disc_per_level.csv')
#   wide = pd.read_csv('results/ian_pan_disc_coords/ian_pan_disc_coords.csv')
#
#   # All disc-level confidences for one study
#   print(df[df.study_id == '12345'][
#       ['disc_level', 'peak_prob', 'confidence_class',
#        'dist_h0_mm', 'dist_h1_mm', 'closer_hyp', 'hyp_margin_mm']
#   ])
#
#   # Studies with Ian Pan H1 vote (likely lumbarization)
#   h1_ids = wide[wide.seq_vote == 'H1']['study_id'].unique()
#   print(f"{len(h1_ids)} studies with Ian Pan H1 sequence vote")
#
#   # At L5-S1 specifically: cases where H1 is clearly closer
#   l5s1 = df[df.disc_level == 'l5_s1']
#   suspicious = l5s1[
#       (l5s1.closer_hyp == 'H1') & (l5s1.hyp_margin_mm > 15)
#   ].sort_values('hyp_margin_mm', ascending=False)
#   print(suspicious[['study_id','peak_prob','dist_h0_mm','dist_h1_mm','hyp_margin_mm']].head(20))
#
#   # Studies where all 5 levels agree on H1 (high-confidence lumbarization signal)
#   per_study_h1 = (
#       df[df.closer_hyp == 'H1'].groupby('study_id').size()
#       .rename('n_h1_levels').reset_index()
#   )
#   all_five_h1 = per_study_h1[per_study_h1.n_h1_levels == 5]
#   print(f"{len(all_five_h1)} studies where all 5 disc levels prefer H1")
#
#   # Cross-reference with LSTV results
#   lstv = pd.read_csv('results/lstv_detection/lstv_alignment.csv')
#   merged = wide.merge(
#       lstv[['study_id', 'lstv_detected', 'preferred_hypothesis',
#             'ip_tiebreak_applied', 'castellvi_type']],
#       on='study_id', how='left'
#   )
#   overrides = merged[merged.ip_tiebreak_applied == True]
#   print(f"\n{len(overrides)} studies where Ian Pan overrode Dice alignment:")
#   print(overrides[['study_id','seq_vote','seq_margin_mm',
#                     'preferred_hypothesis','lstv_detected']].to_string())
#
#   # Studies needing manual review: Ian Pan low-confidence across many levels
#   low_per_study = (
#       df[df.confidence_class == 'low'].groupby('study_id').size()
#       .rename('n_low').reset_index()
#   )
#   flagged = low_per_study[low_per_study.n_low >= 3]
#   print(f"\n{len(flagged)} studies with ≥3 low-confidence disc levels")
