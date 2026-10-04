"""이진 분류 지표. 양성 = label 1 (MCI/AD). numpy만 사용."""
from __future__ import annotations

import numpy as np


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """rank 기반(Mann-Whitney U) AUC. 한 클래스만 있으면 nan."""
    y_true, score = np.asarray(y_true).astype(int), np.asarray(score, dtype=np.float64)
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    i = 0
    while i < len(score):  # tie는 평균 rank
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def confusion_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(prob) >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    def div(a, b):
        return float(a / b) if b else float("nan")

    prec, sens = div(tp, tp + fp), div(tp, tp + fn)
    return dict(
        accuracy=div(tp + tn, len(y_true)), sensitivity=sens, specificity=div(tn, tn + fp),
        precision=prec, f1=div(2 * prec * sens, prec + sens) if tp else (0.0 if len(y_true) else float("nan")),
        tp=float(tp), tn=float(tn), fp=float(fp), fn=float(fn), threshold=float(threshold),
    )


def youden_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    """sens+spec-1 최대 임계값. val에서만 계산해 test에 고정 적용할 것."""
    y_true, prob = np.asarray(y_true).astype(int), np.asarray(prob)
    if len(np.unique(y_true)) < 2:
        return 0.5
    best_t, best_j = 0.5, -np.inf
    for t in np.unique(prob):
        m = confusion_metrics(y_true, prob, float(t))
        j = m["sensitivity"] + m["specificity"] - 1
        if j > best_j:
            best_t, best_j = float(t), j
    return best_t


def summarize(y_true, prob, threshold: float = 0.5) -> dict[str, float]:
    out = confusion_metrics(y_true, prob, threshold)
    out["auc"] = roc_auc(y_true, prob)
    return out
