"""Compute the official ISIC 2024 competition metric (partial AUC above 80% TPR)
plus standard ROC-AUC from out-of-fold (OOF) prediction CSVs.

The competition's metric is **not** plain AUC. It is the partial area under the ROC
curve restricted to the high-sensitivity region (true-positive rate >= 0.80), then
rescaled. Under this transform the score ranges from 0.02 (random) to 0.12 (perfect) —
so a value near 0.10 is strong. This is a *different scale* from plain AUC; do not
compare the two directly.

This script is the source of the numbers reported in the README. Run it to reproduce
them from the raw OOF files (which are git-ignored; see DATA.md).

Usage:
    python tools/compute_pauc.py --oof-glob "results/dual_hybrid_v2/oof_fold*.csv"
    python tools/compute_pauc.py --oof-glob "results/dual_hybrid_v2/oof_ema_fold*.csv"

OOF CSVs must have columns: target (0/1) and pred (probability).
"""
from __future__ import annotations
import argparse
import csv
import glob
import numpy as np


def load_oof(path: str) -> tuple[np.ndarray, np.ndarray]:
    y, p = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        tcol = "target" if "target" in cols else next(c for c in cols if c.lower() in ("target", "label", "y_true"))
        pcol = "pred" if "pred" in cols else next(c for c in cols if "pred" in c.lower() or c.lower() in ("proba", "prob", "score"))
        for row in reader:
            try:
                y.append(float(row[tcol]))
                p.append(float(row[pcol]))
            except (ValueError, KeyError):
                pass
    return np.asarray(y), np.asarray(p)


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Standard ROC-AUC via the rank (Mann-Whitney) identity."""
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _roc_auc_max_fpr(y: np.ndarray, score: np.ndarray, max_fpr: float) -> float:
    """McClish-corrected partial AUC over FPR in [0, max_fpr] (sklearn-equivalent, range [0.5, 1])."""
    order = np.argsort(-score)
    y = y[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tpr = np.concatenate([[0], tps / tps[-1]])
    fpr = np.concatenate([[0], fps / fps[-1]])
    stop = np.searchsorted(fpr, max_fpr, side="right")
    x, t = fpr[: stop + 1].copy(), tpr[: stop + 1].copy()
    if x[-1] > max_fpr:  # interpolate TPR at the max_fpr boundary
        t[-1] = tpr[stop - 1] + (tpr[stop] - tpr[stop - 1]) * (max_fpr - fpr[stop - 1]) / (fpr[stop] - fpr[stop - 1])
        x[-1] = max_fpr
    partial = np.trapz(t, x)
    min_area, max_area = 0.5 * max_fpr ** 2, max_fpr
    return 0.5 * (1 + (partial - min_area) / (max_area - min_area))


def isic_pauc(y_true: np.ndarray, y_pred: np.ndarray, min_tpr: float = 0.80) -> float:
    """Official ISIC 2024 partial-AUC-above-80%-TPR metric. Range: [0.02 random, 0.12 perfect]."""
    v_gt = np.abs(y_true - 1)
    v_pred = -1.0 * y_pred
    max_fpr = abs(1 - min_tpr)
    pas = _roc_auc_max_fpr(v_gt, v_pred, max_fpr)
    return 0.5 * max_fpr ** 2 + (max_fpr - 0.5 * max_fpr ** 2) / (1.0 - 0.5 * max_fpr) * (pas - 0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-glob", required=True, help='e.g. "results/dual_hybrid_v2/oof_fold*.csv"')
    args = ap.parse_args()
    files = sorted(glob.glob(args.oof_glob))
    if not files:
        raise SystemExit(f"No OOF files matched: {args.oof_glob}")

    aucs, paucs, all_y, all_p = [], [], [], []
    for fp in files:
        y, p = load_oof(fp)
        if len(y) == 0 or y.sum() == 0:
            continue
        aucs.append(roc_auc(y, p))
        paucs.append(isic_pauc(y, p))
        all_y.append(y)
        all_p.append(p)
    aucs, paucs = np.asarray(aucs), np.asarray(paucs)
    Y, P = np.concatenate(all_y), np.concatenate(all_p)

    print(f"Files: {len(files)}")
    print(f"  AUC               (per-fold)  {aucs.mean():.4f} +/- {aucs.std():.4f}")
    print(f"  pAUC@80%TPR       (per-fold)  {paucs.mean():.4f} +/- {paucs.std():.4f}   [range 0.02-0.12]")
    print(f"  AUC               (pooled)    {roc_auc(Y, P):.4f}")
    print(f"  pAUC@80%TPR       (pooled)    {isic_pauc(Y, P):.4f}")
    print(f"  n={len(Y)}  positives={int(Y.sum())}  prevalence={Y.mean()*100:.4f}%")


if __name__ == "__main__":
    main()
