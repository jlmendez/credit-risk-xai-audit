import numpy as np

from src.fairness_metrics import bootstrap_selection_ratio, disparity_summary, group_metrics


def test_identical_group_behavior_has_unit_selection_ratio():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.3, 0.7, 0.2, 0.8, 0.3, 0.7])
    groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    metrics = group_metrics(y_true, y_prob, groups, threshold=0.5)
    summary = disparity_summary(metrics)
    assert summary["selection_ratio_min_max"] == 1.0
    assert summary["favorable_opportunity_gap"] == 0.0


def test_bootstrap_interval_is_ordered():
    y_true = np.array([0, 1] * 50)
    y_prob = np.array([0.2, 0.8] * 50)
    groups = np.array(["A", "A", "B", "B"] * 25)
    q = bootstrap_selection_ratio(y_true, y_prob, groups, n_boot=50, seed=7)
    assert q.shape == (3,)
    assert q[0] <= q[1] <= q[2]
