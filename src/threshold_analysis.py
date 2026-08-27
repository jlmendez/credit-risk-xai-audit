"""Threshold sensitivity and decision-policy trade-offs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from .fairness_metrics import group_metrics, disparity_summary


def threshold_sweep(y_true, y_prob, groups, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.linspace(0.15, 0.75, 25)
    rows = []
    for threshold in thresholds:
        gm = group_metrics(y_true, y_prob, groups, float(threshold))
        gap = disparity_summary(gm)
        approved = np.asarray(y_prob) < threshold
        rows.append({
            "threshold": float(threshold),
            "approval_rate": float(approved.mean()),
            "approved_default_rate": float(np.asarray(y_true)[approved].mean()) if approved.any() else np.nan,
            **gap,
        })
    return pd.DataFrame(rows)


def expected_cost(y_true, y_prob, threshold: float, false_approval_cost=5.0, false_rejection_cost=1.0) -> float:
    y_true = np.asarray(y_true)
    approve = np.asarray(y_prob) < threshold
    false_approval = approve & (y_true == 1)
    false_rejection = (~approve) & (y_true == 0)
    return float((false_approval_cost * false_approval.sum() + false_rejection_cost * false_rejection.sum()) / len(y_true))
