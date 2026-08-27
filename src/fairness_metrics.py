"""Group-level decision and error metrics for binary credit decisions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def group_metrics(y_true, y_prob, groups, threshold: float = 0.5) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    groups = np.asarray(groups)
    approve = y_prob < threshold
    rows = []
    for group in np.unique(groups):
        m = groups == group
        y = y_true[m]
        a = approve[m]
        good = y == 0
        bad = y == 1
        rows.append({
            "group": group,
            "n": int(m.sum()),
            "approval_rate": float(a.mean()),
            "favorable_tpr": float(a[good].mean()) if good.any() else np.nan,
            "bad_approval_rate": float(a[bad].mean()) if bad.any() else np.nan,
            "portfolio_default_rate": float(y[a].mean()) if a.any() else np.nan,
        })
    return pd.DataFrame(rows).set_index("group")


def disparity_summary(metrics: pd.DataFrame) -> dict[str, float]:
    approval = metrics["approval_rate"].dropna()
    opportunity = metrics["favorable_tpr"].dropna()
    return {
        "selection_ratio_min_max": float(approval.min() / approval.max()) if len(approval) > 1 else 1.0,
        "favorable_opportunity_gap": float(opportunity.max() - opportunity.min()) if len(opportunity) > 1 else 0.0,
    }


def bootstrap_selection_ratio(y_true, y_prob, groups, threshold=0.5, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob); groups = np.asarray(groups)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        m = group_metrics(y_true[idx], y_prob[idx], groups[idx], threshold)
        values.append(disparity_summary(m)["selection_ratio_min_max"])
    return np.quantile(values, [0.025, 0.5, 0.975])
