"""Model-agnostic global importance and compact local surrogate explanations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge


def permutation_table(model, x, y, repeats: int = 15, seed: int = 42) -> pd.DataFrame:
    result = permutation_importance(model, x, y, n_repeats=repeats, random_state=seed, scoring="roc_auc")
    return (pd.DataFrame({"feature": x.columns, "importance": result.importances_mean, "std": result.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def local_surrogate(predict_proba, row: pd.Series, reference: pd.DataFrame, n_samples=1000, kernel_width=1.5, seed=42):
    """LIME-style numerical surrogate for an already numeric feature matrix."""
    rng = np.random.default_rng(seed)
    center = row.to_numpy(dtype=float)
    scale = reference.std(ddof=0).replace(0, 1).to_numpy(dtype=float)
    z = rng.normal(center, scale, size=(n_samples, len(center)))
    distance = np.sqrt(np.square((z - center) / scale).sum(axis=1))
    weights = np.exp(-(distance ** 2) / (kernel_width ** 2))
    y = predict_proba(pd.DataFrame(z, columns=reference.columns))[:, 1]
    surrogate = Ridge(alpha=1.0).fit(z, y, sample_weight=weights)
    fidelity = surrogate.score(z, y, sample_weight=weights)
    coefficients = pd.Series(surrogate.coef_, index=reference.columns).sort_values(key=np.abs, ascending=False)
    return coefficients, float(fidelity)
