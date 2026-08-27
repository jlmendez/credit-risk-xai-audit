"""Compact credit-risk XAI and fairness audit using synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42
FEATURES = [
    "age", "monthly_income", "debt_ratio", "late_payments_12m",
    "savings", "requested_amount", "zone", "channel",
]


def make_data(n: int = 4000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group = rng.choice(["A", "B"], size=n, p=[0.55, 0.45])
    g = (group == "B").astype(int)
    age = np.clip(rng.normal(39 + 2 * g, 10, n), 21, 70).round().astype(int)
    income = np.clip(rng.lognormal(np.log(7800 - 1300 * g), 0.45, n), 1800, 35000)
    debt_ratio = np.clip(rng.beta(2.2 + 0.5 * g, 4.5, n), 0.03, 0.95)
    late = rng.poisson(0.35 + 1.5 * debt_ratio + 0.25 * g)
    savings = np.clip(rng.lognormal(np.log(12000 + 0.9 * income), 0.70, n), 100, 150000)
    amount = np.clip(rng.lognormal(np.log(28000 + 2 * income), 0.50, n), 3000, 250000)
    zone = np.array([
        rng.choice(["Central", "Intermediate", "Peripheral"],
                   p=[0.52, 0.33, 0.15] if flag == 0 else [0.18, 0.40, 0.42])
        for flag in g
    ])
    channel = rng.choice(["Branch", "Web", "Mobile"], size=n, p=[0.38, 0.32, 0.30])
    zone_effect = pd.Series(zone).map({"Central": 0.0, "Intermediate": 0.12, "Peripheral": 0.42}).to_numpy()
    logit = (
        -1.90 + 3.10 * debt_ratio + 0.58 * late - 0.000035 * income
        + 0.000008 * amount - 0.000004 * savings + zone_effect
        + 0.15 * g + rng.normal(0, 0.35, n)
    )
    p_default = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, p_default)
    return pd.DataFrame({
        "audit_group": group, "age": age, "monthly_income": income,
        "debt_ratio": debt_ratio, "late_payments_12m": late, "savings": savings,
        "requested_amount": amount, "zone": zone, "channel": channel,
        "default": default,
    })


def build_model() -> Pipeline:
    categorical = ["zone", "channel"]
    numeric = [c for c in FEATURES if c not in categorical]
    prep = ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([
        ("preprocess", prep),
        ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=8,
                                         random_state=SEED, n_jobs=-1)),
    ])


def group_metrics(y: pd.Series, prob: np.ndarray, groups: pd.Series,
                  threshold: float = 0.50) -> pd.DataFrame:
    rows = []
    approved = prob < threshold
    qualified = y.to_numpy() == 0
    for group in sorted(groups.unique()):
        mask = groups.to_numpy() == group
        rows.append({
            "group": group,
            "n": int(mask.sum()),
            "approval_rate": float(approved[mask].mean()),
            "qualified_approval_rate": float(approved[mask & qualified].mean()),
            "default_rate_among_approved": float(y.to_numpy()[mask & approved].mean())
            if (mask & approved).any() else np.nan,
        })
    return pd.DataFrame(rows).set_index("group")


def threshold_scan(y: pd.Series, prob: np.ndarray, groups: pd.Series) -> pd.DataFrame:
    rows = []
    for threshold in np.arange(0.25, 0.76, 0.05):
        m = group_metrics(y, prob, groups, threshold)
        rates = m["approval_rate"]
        rows.append({
            "threshold": round(float(threshold), 2),
            "overall_approval": float((prob < threshold).mean()),
            "selection_ratio": float(rates.min() / rates.max()),
            "approval_gap": float(rates.max() - rates.min()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    data = make_data()
    train, test = train_test_split(data, test_size=0.25, random_state=SEED,
                                   stratify=data["default"])
    model = build_model()
    model.fit(train[FEATURES], train["default"])
    prob = model.predict_proba(test[FEATURES])[:, 1]

    print(f"ROC-AUC: {roc_auc_score(test['default'], prob):.3f}")
    print("\nGroup metrics at threshold 0.50")
    print(group_metrics(test["default"], prob, test["audit_group"]).round(3))

    importance = permutation_importance(
        model, test[FEATURES], test["default"], scoring="roc_auc",
        n_repeats=10, random_state=SEED, n_jobs=-1,
    )
    imp = pd.Series(importance.importances_mean, index=FEATURES).sort_values(ascending=False)
    print("\nPermutation importance")
    print(imp.round(4))

    print("\nThreshold sensitivity")
    print(threshold_scan(test["default"], prob, test["audit_group"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
