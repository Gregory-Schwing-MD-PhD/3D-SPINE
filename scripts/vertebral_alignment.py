#!/usr/bin/env python3
"""
vertebral_alignment.py — Generalised TSS / VERIDAH Vertebral Alignment Analyser
=================================================================================
v2.0  (Generalised offset model — replaces binary H0/H1)

PURPOSE
-------
Determine the integer level offset between TSS (TotalSpineSeg) and VERIDAH
(SPINEPS seg-vert_msk) lumbar vertebral labelling by finding the offset that
maximises Dice overlap across the entire lumbar sequence simultaneously.

SEQUENCE ALIGNMENT MODEL
-------------------------
TSS labels lumbar vertebrae as sequential integers 41(L1)..45(L5).
VERIDAH labels lumbar vertebrae as sequential integers 20(L1)..25(L6).

Define anatomical index:
    TSS:     i = TSS_label - 41   (i=0 → L1, i=4 → L5)
    VERIDAH: j = VD_label  - 20   (j=0 → L1, j=5 → L6)

An offset k means TSS anatomical index i corresponds to VERIDAH index (i+k):
    TSS label (41+i)  ↔  VD label (20+i+k)

    offset =  0  TSS and VERIDAH agree (normal / aligned)
    offset = -1  VERIDAH one level caudal to TSS
                 TSS L1↔VD L2 ... TSS L5↔VD L6
                 → VERIDAH found an extra caudal segment (lumbarization candidate)
    offset = +1  TSS one level caudal to VERIDAH
                 TSS L2↔VD L1 ... TSS L5↔VD L4
                 → TSS found extra caudal segment / VERIDAH undercounts cranially
                   (sacralization candidate — TSS may subsume one level into sacrum)
    offset = -2  extreme lumbarization (two extra caudal levels in VERIDAH)
    offset = +2  extreme sacralization (two extra caudal levels in TSS)

For each candidate offset the score = mean Dice over all valid pairs (both
masks present and dice >= MIN_PAIR_DICE). A non-zero offset is accepted over
offset=0 only when:
    (a) its score exceeds offset=0 by >= SCORE_MARGIN, OR offset=0 has no
        valid pairs at all (complete sequence displacement)
    (b) >= CONSISTENCY_FRAC of individual pairs favour the winning offset

HANDLES ALL COUNT COMBINATIONS
--------------------------------
    5 TSS × 5 VD  (normal)                 → offset 0 wins
    5 TSS × 6 VD  (lumbarization)          → offset +1 wins
    4 TSS × 5 VD  (TSS overcounts caudally)  → offset -1 wins
    5 TSS × 4 VD  (VERIDAH undercounts)    → offset 0 wins on 4 pairs
    4 TSS × 4 VD  (both short)             → offset 0 wins on 4 pairs
    any other combination                  → naturally handled; insufficient_data
                                             if no offset reaches MIN_PAIRS_REQUIRED
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Label constants ────────────────────────────────────────────────────────────
TSS_LUMBAR_BASE   = 41    # TSS L1=41 ... L5=45
TSS_LUMBAR_MAX    = 45
TSS_SACRUM_LABEL  = 50

VD_LUMBAR_BASE    = 20    # VERIDAH L1=20 ... L6=25
VD_LUMBAR_MAX     = 25
VD_SACRUM_LABEL   = 26

TSS_LUMBAR_LABELS: Dict[int, str] = {41:'L1', 42:'L2', 43:'L3', 44:'L4', 45:'L5'}
VD_LUMBAR_LABELS:  Dict[int, str] = {20:'L1', 21:'L2', 22:'L3', 23:'L4',
                                      24:'L5', 25:'L6'}

# ── Decision thresholds ────────────────────────────────────────────────────────
SCORE_MARGIN        : float = 0.08   # non-zero offset must beat offset=0 by this margin
MIN_PAIRS_REQUIRED  : int   = 3      # min valid pairs to score an offset reliably
MIN_PAIR_DICE       : float = 0.10   # below this is boundary noise, not real overlap
CONSISTENCY_FRAC    : float = 0.60   # fraction of pairs that must favour winning offset
MAX_OFFSET          : int   = 2      # test offsets -2 .. +2

OFFSET_INTERPRETATION: Dict[int, str] = {
    +2: 'VERIDAH 2 levels cranial to TSS — extreme lumbarization candidate '
        '(TSS subsumes 2 extra caudal segments into sacrum)',
    +1: 'VERIDAH 1 level cranial to TSS — lumbarization candidate '
        '(TSS L1↔VD L2 ... TSS L5↔VD L6; VERIDAH has extra caudal level VD L6)',
     0: 'TSS and VERIDAH agree on level identity',
    -1: 'TSS 1 level cranial to VERIDAH — sacralization candidate '
        '(TSS L2↔VD L1 ... TSS L5↔VD L4; TSS has extra caudal segment absent from VERIDAH)',
    -2: 'TSS 2 levels cranial to VERIDAH — extreme sacralization candidate',
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PairDice:
    """Dice overlap between one TSS and one VERIDAH vertebral mask."""
    tss_label:      int
    vd_label:       int
    tss_name:       str
    vd_name:        str
    offset:         int
    dice:           Optional[float]
    tss_voxels:     int
    vd_voxels:      int
    overlap_voxels: int
    valid:          bool  # True when both masks present AND dice >= MIN_PAIR_DICE


@dataclass
class OffsetScore:
    """Summary statistics for one candidate offset."""
    offset:          int
    interpretation:  str
    pairs:           List[PairDice]
    n_valid_pairs:   int
    mean_dice:       Optional[float]
    min_dice:        Optional[float]
    max_dice:        Optional[float]
    std_dice:        Optional[float]
    sufficient_data: bool   # n_valid_pairs >= MIN_PAIRS_REQUIRED


@dataclass
class AlignmentResult:
    """
    Full output of the TSS/VERIDAH alignment analysis for one study.

    best_offset:
        Accepted integer offset.  0 = aligned.
        +1 = VERIDAH one level cranial / extra caudal segment (lumbarization candidate).
        -1 = TSS one level cranial / extra caudal segment (sacralization candidate).
        None = insufficient data.

    preferred_hypothesis (string — for downstream/CSV compatibility):
        'aligned'           offset = 0
        'shifted_plus_1'    offset = +1  (lumbarization — VERIDAH extra caudal level)
        'shifted_minus_1'   offset = -1  (sacralization — TSS extra caudal level)
        'shifted_plus_2'    offset = +2  (extreme lumbarization)
        'shifted_minus_2'   offset = -2  (extreme sacralization)
        'ambiguous'         best offset exists but margin < SCORE_MARGIN
        'insufficient_data'

    ground_truth_label:
        Reserved for prospective annotation via axial nerve morphology,
        iliolumbar ligament, or full-spine CT.
    """
    study_id:              str
    best_offset:           Optional[int]
    preferred_hypothesis:  str
    confidence:            str   # 'high'|'moderate'|'low'|'ambiguous'|'insufficient_data'
    best_score:            Optional[float]
    second_best_score:     Optional[float]
    score_margin:          Optional[float]
    consistency_frac:      Optional[float]
    consistency_beats:     int
    consistency_total:     int

    offset_scores:         Dict[int, OffsetScore]  # keyed by offset int

    tss_count:             int
    vd_count:              int
    tss_labels_present:    List[int]
    vd_labels_present:     List[int]

    # Downstream ensemble — populated externally by 04_detect_lstv.py
    castellvi_at_best:     Optional[str]  = None
    castellvi_at_zero:     Optional[str]  = None
    phenotype_at_best:     Optional[str]  = None
    phenotype_at_zero:     Optional[str]  = None
    lstv_detected_at_best: Optional[bool] = None
    lstv_detected_at_zero: Optional[bool] = None

    # Reserved for prospective ground truth annotation
    ground_truth_label:    Optional[str]  = None
    ground_truth_method:   Optional[str]  = None

    summary:               str = ''

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def _dice(a: np.ndarray, b: np.ndarray) -> Tuple[float, int, int, int]:
    a = a.astype(bool); b = b.astype(bool)
    n_a = int(a.sum()); n_b = int(b.sum())
    n_overlap = int((a & b).sum())
    denom = n_a + n_b
    dice = float(2 * n_overlap / denom) if denom > 0 else 0.0
    return dice, n_a, n_b, n_overlap


def _score_offset(
        sag_tss:            np.ndarray,
        sag_vert:           np.ndarray,
        offset:             int,
        tss_labels_present: List[int],
        vd_labels_present:  List[int],
) -> OffsetScore:
    """
    Score one offset by computing Dice for every pair
    (TSS label 41+i) ↔ (VD label 20+i+offset).
    Iterates over all TSS anatomical indices 0..4 (L1..L5).
    """
    pairs: List[PairDice] = []

    for i in range(TSS_LUMBAR_MAX - TSS_LUMBAR_BASE + 1):
        tss_lbl = TSS_LUMBAR_BASE + i
        vd_lbl  = VD_LUMBAR_BASE  + i + offset

        tss_name = TSS_LUMBAR_LABELS.get(tss_lbl, f'TSS{tss_lbl}')
        vd_name  = VD_LUMBAR_LABELS.get(vd_lbl,  f'VD{vd_lbl}')

        tss_present = tss_lbl in tss_labels_present
        vd_present  = vd_lbl  in vd_labels_present

        if not tss_present or not vd_present:
            pairs.append(PairDice(
                tss_label=tss_lbl, vd_label=vd_lbl,
                tss_name=tss_name, vd_name=vd_name,
                offset=offset, dice=None,
                tss_voxels=0, vd_voxels=0, overlap_voxels=0,
                valid=False))
            continue

        tss_mask = (sag_tss  == tss_lbl)
        vd_mask  = (sag_vert == vd_lbl)

        if not tss_mask.any() or not vd_mask.any():
            pairs.append(PairDice(
                tss_label=tss_lbl, vd_label=vd_lbl,
                tss_name=tss_name, vd_name=vd_name,
                offset=offset, dice=0.0,
                tss_voxels=int(tss_mask.sum()),
                vd_voxels=int(vd_mask.sum()),
                overlap_voxels=0, valid=False))
            continue

        d, n_t, n_v, n_o = _dice(tss_mask, vd_mask)
        valid = (d >= MIN_PAIR_DICE)
        pairs.append(PairDice(
            tss_label=tss_lbl, vd_label=vd_lbl,
            tss_name=tss_name, vd_name=vd_name,
            offset=offset, dice=round(d, 4),
            tss_voxels=n_t, vd_voxels=n_v,
            overlap_voxels=n_o, valid=valid))

    valid_dice = [p.dice for p in pairs if p.valid and p.dice is not None]
    n_valid    = len(valid_dice)
    sufficient = n_valid >= MIN_PAIRS_REQUIRED

    return OffsetScore(
        offset          = offset,
        interpretation  = OFFSET_INTERPRETATION.get(offset, f'offset={offset:+d}'),
        pairs           = pairs,
        n_valid_pairs   = n_valid,
        mean_dice       = round(float(np.mean(valid_dice)), 4) if valid_dice else None,
        min_dice        = round(float(np.min(valid_dice)),  4) if valid_dice else None,
        max_dice        = round(float(np.max(valid_dice)),  4) if valid_dice else None,
        std_dice        = round(float(np.std(valid_dice)),  4) if len(valid_dice) > 1 else None,
        sufficient_data = sufficient,
    )


def _consistency(
        winner:   OffsetScore,
        baseline: OffsetScore,
) -> Tuple[float, int, int]:
    """
    Fraction of individual pairs where winner Dice > baseline Dice.
    Pairs where baseline is invalid and winner is valid count as winner wins.
    Returns (fraction, beats, total).
    """
    beats = total = 0
    for pw, pb in zip(winner.pairs, baseline.pairs):
        if pw.valid and pb.valid:
            total += 1
            if pw.dice > pb.dice:
                beats += 1
        elif pw.valid and not pb.valid and pw.dice is not None and pw.dice >= MIN_PAIR_DICE:
            total += 1
            beats += 1
    frac = float(beats / total) if total > 0 else 0.0
    return frac, beats, total


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_vertebral_alignment(
        study_id: str,
        sag_tss:  np.ndarray,
        sag_vert: np.ndarray,
) -> AlignmentResult:
    """
    Generalised offset-based TSS/VERIDAH alignment analysis.

    Tests offsets -MAX_OFFSET .. +MAX_OFFSET.  The best-scoring offset that
    meets data sufficiency requirements is accepted over offset=0 only when
    margin >= SCORE_MARGIN (or offset=0 has no valid pairs) AND pair-level
    consistency >= CONSISTENCY_FRAC.

    Parameters
    ----------
    study_id  str
    sag_tss   3-D int array — TotalSpineSeg sagittal label volume
    sag_vert  3-D int array — VERIDAH (SPINEPS seg-vert_msk) label volume

    Returns
    -------
    AlignmentResult  (JSON-serialisable via .to_dict())
    """
    tss_labels_present = sorted(
        int(v) for v in np.unique(sag_tss)
        if TSS_LUMBAR_BASE <= v <= TSS_LUMBAR_MAX)
    vd_labels_present = sorted(
        int(v) for v in np.unique(sag_vert)
        if VD_LUMBAR_BASE <= v <= VD_LUMBAR_MAX)

    tss_count = len(tss_labels_present)
    vd_count  = len(vd_labels_present)

    # ── Score every candidate offset ──────────────────────────────────────────
    offset_scores: Dict[int, OffsetScore] = {
        k: _score_offset(sag_tss, sag_vert, k,
                         tss_labels_present, vd_labels_present)
        for k in range(-MAX_OFFSET, MAX_OFFSET + 1)
    }

    # ── Find best and second-best scoreable offsets ───────────────────────────
    scoreable = sorted(
        [(k, s) for k, s in offset_scores.items()
         if s.sufficient_data and s.mean_dice is not None],
        key=lambda t: t[1].mean_dice, reverse=True)

    if not scoreable:
        return _make_insufficient(study_id, offset_scores,
                                  tss_count, vd_count,
                                  tss_labels_present, vd_labels_present)

    best_k, best_s    = scoreable[0]
    second_k, second_s = scoreable[1] if len(scoreable) >= 2 else (None, None)
    score_margin      = (round(float(best_s.mean_dice - second_s.mean_dice), 4)
                         if second_s is not None else None)
    second_best       = second_s.mean_dice if second_s is not None else None

    # ── Decide whether to accept the best offset or fall back to 0 ───────────
    baseline      = offset_scores[0]
    baseline_absent = not baseline.sufficient_data

    if best_k == 0:
        # Aligned — offset=0 wins outright
        accepted_offset = 0
        cons_frac, cons_beats, cons_total = 0.0, 0, 0
    else:
        cons_frac, cons_beats, cons_total = _consistency(best_s, baseline)
        margin_ok = (score_margin is not None and score_margin >= SCORE_MARGIN)

        if (margin_ok or baseline_absent) and cons_frac >= CONSISTENCY_FRAC:
            accepted_offset = best_k
        else:
            # Not convincing enough — fall back to aligned
            accepted_offset = 0
            best_k          = 0
            best_s          = offset_scores[0]

    # ── Confidence ─────────────────────────────────────────────────────────────
    if accepted_offset == 0:
        s0 = offset_scores[0]
        if not s0.sufficient_data:
            confidence = 'insufficient_data'
        elif s0.mean_dice is not None and s0.mean_dice >= 0.75:
            confidence = 'high'
        elif s0.mean_dice is not None and s0.mean_dice >= 0.50:
            confidence = 'moderate'
        else:
            confidence = 'low'
    else:
        if score_margin is None:
            confidence = 'moderate'
        elif score_margin >= 2 * SCORE_MARGIN:
            confidence = 'high'
        elif score_margin >= SCORE_MARGIN:
            confidence = 'moderate'
        else:
            confidence = 'low'

    # ── preferred_hypothesis string ───────────────────────────────────────────
    pref_map = {0: 'aligned', +1: 'shifted_plus_1', -1: 'shifted_minus_1',
                +2: 'shifted_plus_2', -2: 'shifted_minus_2'}
    preferred = pref_map.get(accepted_offset, f'shifted_{accepted_offset:+d}')

    # ── Summary and logging ───────────────────────────────────────────────────
    summary = _build_summary(
        accepted_offset, preferred, confidence,
        best_s, second_s, score_margin,
        cons_frac, cons_beats, cons_total,
        offset_scores, tss_count, vd_count)

    logger.info(f"  [{study_id}] Alignment: {summary}")
    logger.info(f"  [{study_id}] Offset score table:")
    for k in range(-MAX_OFFSET, MAX_OFFSET + 1):
        s  = offset_scores[k]
        sc = (f"{s.mean_dice:.3f} (n={s.n_valid_pairs})"
              if s.mean_dice is not None else "N/A")
        marker = " ← ACCEPTED" if k == accepted_offset else ""
        logger.info(f"    offset {k:+d}  score={sc}  [{s.interpretation[:55]}]{marker}")

    logger.info(f"  [{study_id}] Vertebral Dice matrix:")
    for i in range(TSS_LUMBAR_MAX - TSS_LUMBAR_BASE + 1):
        tss_lbl  = TSS_LUMBAR_BASE + i
        tss_name = TSS_LUMBAR_LABELS.get(tss_lbl, '?')
        parts    = []
        for k in range(-MAX_OFFSET, MAX_OFFSET + 1):
            p     = offset_scores[k].pairs[i]
            d_str = f"{p.dice:.3f}" if p.dice is not None else " --- "
            parts.append(f"VD{p.vd_label}@{k:+d}={d_str}")
        logger.info(f"    TSS {tss_lbl}({tss_name}): {'  '.join(parts)}")

    return AlignmentResult(
        study_id             = study_id,
        best_offset          = accepted_offset,
        preferred_hypothesis = preferred,
        confidence           = confidence,
        best_score           = best_s.mean_dice,
        second_best_score    = second_best,
        score_margin         = score_margin,
        consistency_frac     = round(cons_frac, 4),
        consistency_beats    = cons_beats,
        consistency_total    = cons_total,
        offset_scores        = offset_scores,
        tss_count            = tss_count,
        vd_count             = vd_count,
        tss_labels_present   = tss_labels_present,
        vd_labels_present    = vd_labels_present,
        summary              = summary,
    )


def _make_insufficient(
        study_id: str,
        offset_scores: Dict[int, OffsetScore],
        tss_count: int, vd_count: int,
        tss_labels_present: List[int],
        vd_labels_present:  List[int],
) -> AlignmentResult:
    summary = (
        f"Insufficient valid pairs at all offsets "
        f"(TSS={tss_count} lumbar labels, VERIDAH={vd_count} lumbar labels, "
        f"need ≥{MIN_PAIRS_REQUIRED} pairs). "
        f"Likely registration or segmentation failure.")
    logger.info(f"  Alignment: {summary}")
    return AlignmentResult(
        study_id             = study_id,
        best_offset          = None,
        preferred_hypothesis = 'insufficient_data',
        confidence           = 'insufficient_data',
        best_score           = None,
        second_best_score    = None,
        score_margin         = None,
        consistency_frac     = None,
        consistency_beats    = 0,
        consistency_total    = 0,
        offset_scores        = offset_scores,
        tss_count            = tss_count,
        vd_count             = vd_count,
        tss_labels_present   = tss_labels_present,
        vd_labels_present    = vd_labels_present,
        summary              = summary,
    )


def _build_summary(
        accepted_offset: int,
        preferred:       str,
        confidence:      str,
        best_s:          OffsetScore,
        second_s:        Optional[OffsetScore],
        score_margin:    Optional[float],
        cons_frac:       float,
        cons_beats:      int,
        cons_total:      int,
        offset_scores:   Dict[int, OffsetScore],
        tss_count:       int,
        vd_count:        int,
) -> str:
    best_sc   = f"{best_s.mean_dice:.3f}" if best_s.mean_dice  is not None else "N/A"
    second_sc = (f"{second_s.mean_dice:.3f}"
                 if second_s is not None and second_s.mean_dice is not None else "N/A")
    margin_sc = f"{score_margin:+.3f}" if score_margin is not None else "N/A"

    if accepted_offset == 0:
        s0 = offset_scores[0]
        s0_str = f"{s0.mean_dice:.3f}" if s0.mean_dice is not None else "N/A"
        return (
            f"ALIGNED [offset=0, {confidence}]: "
            f"TSS({tss_count}L) and VERIDAH({vd_count}L) agree. "
            f"offset=0 score={s0_str} (n={s0.n_valid_pairs} pairs). "
            f"Runner-up score={second_sc}. "
            f"{OFFSET_INTERPRETATION[0]}")
    else:
        interp = OFFSET_INTERPRETATION.get(accepted_offset, f'offset={accepted_offset:+d}')
        return (
            f"OFFSET {accepted_offset:+d} DETECTED [{confidence}]: "
            f"TSS({tss_count}L) vs VERIDAH({vd_count}L). "
            f"Best score={best_sc} (n={best_s.n_valid_pairs} pairs), "
            f"margin={margin_sc} over runner-up={second_sc}. "
            f"Pair consistency={cons_frac:.0%} ({cons_beats}/{cons_total} pairs H1>H0). "
            f"{interp}. Ground truth required for confirmation.")


# ═══════════════════════════════════════════════════════════════════════════════
# COHORT STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CohortAlignmentStats:
    """Aggregated alignment statistics across a cohort — for abstract reporting."""
    n_studies:               int
    n_aligned:               int
    n_shifted_plus_1:        int   # lumbarization candidate (offset=+1)
    n_shifted_minus_1:       int   # sacralization candidate (offset=-1)
    n_shifted_other:         int   # |offset| >= 2
    n_ambiguous:             int
    n_insufficient:          int

    tss_vd_count_breakdown:  Dict[str, int]  # e.g. "5v5": 120, "5v6": 8, "4v5": 3

    n_vd_l6_present:         int
    n_l6_confirmed_m1:       int   # VD L6 present AND offset=-1 accepted
    n_l6_rejected:           int   # VD L6 present but offset=0 retained

    mean_best_score:         Optional[float]
    std_best_score:          Optional[float]
    n_classification_differs: int

    n_high:     int
    n_moderate: int
    n_low:      int

    def offset_minus1_rate(self) -> float:
        return self.n_shifted_plus_1 / max(self.n_studies, 1)

    def l6_false_positive_rate(self) -> float:
        return (self.n_l6_rejected / max(self.n_vd_l6_present, 1)
                if self.n_vd_l6_present > 0 else float('nan'))

    def to_dict(self) -> dict:
        d = asdict(self)
        d['offset_minus1_rate_pct'] = round(self.offset_minus1_rate() * 100, 2)
        d['l6_false_positive_rate_pct'] = (
            round(self.l6_false_positive_rate() * 100, 2)
            if self.n_vd_l6_present > 0 else None)
        return d


def compute_cohort_stats(results: List[AlignmentResult]) -> CohortAlignmentStats:
    n = len(results)
    count_bd: Dict[str, int] = {}
    n_aligned = n_m1 = n_p1 = n_other = n_ambig = n_insuf = 0
    n_l6 = n_l6_conf = n_l6_rej = 0
    n_classif_diff = 0
    conf = {'high': 0, 'moderate': 0, 'low': 0}
    best_scores: List[float] = []

    for r in results:
        key = f"{r.tss_count}v{r.vd_count}"
        count_bd[key] = count_bd.get(key, 0) + 1

        pref = r.preferred_hypothesis
        if   pref == 'aligned':           n_aligned += 1
        elif pref == 'shifted_plus_1':    n_m1      += 1
        elif pref == 'shifted_minus_1':   n_p1      += 1
        elif pref == 'insufficient_data': n_insuf   += 1
        elif pref == 'ambiguous':         n_ambig   += 1
        else:                             n_other   += 1

        if VD_LUMBAR_MAX in r.vd_labels_present:
            n_l6 += 1
            if pref == 'shifted_plus_1':  n_l6_conf += 1
            elif pref == 'aligned':       n_l6_rej  += 1

        if r.best_score is not None:
            best_scores.append(r.best_score)

        if r.confidence in conf:
            conf[r.confidence] += 1

        ct_diff = (r.castellvi_at_best != r.castellvi_at_zero
                   and None not in (r.castellvi_at_best, r.castellvi_at_zero))
        ph_diff = (r.phenotype_at_best != r.phenotype_at_zero
                   and None not in (r.phenotype_at_best, r.phenotype_at_zero))
        if ct_diff or ph_diff:
            n_classif_diff += 1

    return CohortAlignmentStats(
        n_studies               = n,
        n_aligned               = n_aligned,
        n_shifted_minus_1       = n_m1,
        n_shifted_plus_1        = n_p1,
        n_shifted_other         = n_other,
        n_ambiguous             = n_ambig,
        n_insufficient          = n_insuf,
        tss_vd_count_breakdown  = count_bd,
        n_vd_l6_present         = n_l6,
        n_l6_confirmed_m1       = n_l6_conf,
        n_l6_rejected           = n_l6_rej,
        mean_best_score         = round(float(np.mean(best_scores)), 4) if best_scores else None,
        std_best_score          = round(float(np.std(best_scores)),  4) if best_scores else None,
        n_classification_differs= n_classif_diff,
        n_high                  = conf['high'],
        n_moderate              = conf['moderate'],
        n_low                   = conf['low'],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    def box(vol, lbl, z, dz=8, x=10, dx=6, y=10, dy=6):
        vol[x:x+dx, y:y+dy, z:z+dz] = lbl

    def run(name, fn, expected_offset):
        tss, vd = fn()
        r = analyse_vertebral_alignment(name, tss, vd)
        status = 'PASS' if r.best_offset == expected_offset else 'FAIL'
        print(f"[{status}] {name}: offset={r.best_offset} "
              f"(expected {expected_offset}) "
              f"conf={r.confidence} score={r.best_score}")
        print(f"       {r.summary[:100]}")
        return status == 'PASS'

    tests_passed = 0

    # 1. Normal aligned 5v5
    def t1():
        t, v = np.zeros((30,30,100),int), np.zeros((30,30,100),int)
        for lbl, z in zip([41,42,43,44,45],[10,20,30,40,50]): box(t,lbl,z)
        for lbl, z in zip([20,21,22,23,24],[10,20,30,40,50]): box(v,lbl,z)
        return t, v
    tests_passed += run("5v5 aligned", t1, 0)

    # 2. Lumbarization offset=-1 (5 TSS, 6 VD, VD shifted one caudal)
    def t2():
        t, v = np.zeros((30,30,100),int), np.zeros((30,30,100),int)
        for lbl, z in zip([41,42,43,44,45,50],[10,20,30,40,50,70]): box(t,lbl,z)
        for lbl, z in zip([20,21,22,23,24,25],[ 0,10,20,30,40,50]): box(v,lbl,z)
        return t, v
    tests_passed += run("5v6 lumbarization (offset=+1)", t2, 1)

    # 3. Sacralization offset=+1 (TSS only has 4 lumbar, VERIDAH has 5)
    #    TSS L2↔VD L1 ... TSS L5↔VD L4 — offset=-1
    def t3():
        t, v = np.zeros((30,30,100),int), np.zeros((30,30,100),int)
        for lbl, z in zip([42,43,44,45],[10,20,30,40]): box(t,lbl,z)  # TSS starts at L2
        for lbl, z in zip([20,21,22,23,24],[10,20,30,40,50]): box(v,lbl,z)  # VD has L1-L5
        # At offset=+1: TSS L2(42)↔VD L1(20), TSS L3(43)↔VD L2(21) etc — positions match
        return t, v
    tests_passed += run("4v5 sacralization (offset=-1)", t3, -1)

    # 4. VERIDAH spurious L6 (VD L6 present but NOT aligned with any TSS)
    def t4():
        t, v = np.zeros((30,30,100),int), np.zeros((30,30,100),int)
        for lbl, z in zip([41,42,43,44,45],[10,20,30,40,50]): box(t,lbl,z)
        for lbl, z in zip([20,21,22,23,24],[10,20,30,40,50]): box(v,lbl,z)
        box(v, 25, 90)  # spurious L6 far from everything
        return t, v
    tests_passed += run("5v6 spurious L6 (should stay aligned)", t4, 0)

    # 5. No overlap at all (registration failure)
    def t5():
        t, v = np.zeros((30,30,100),int), np.zeros((30,30,100),int)
        for lbl, z in zip([41,42,43,44,45],[10,20,30,40,50]): box(t,lbl,z)
        # VD labels in completely different spatial location
        for lbl, z in zip([20,21,22,23,24],[60,68,76,84,92]): box(v,lbl,z)
        return t, v
    tests_passed += run("registration failure (insufficient_data)", t5, None)

    print(f"\n{tests_passed}/5 tests passed")
