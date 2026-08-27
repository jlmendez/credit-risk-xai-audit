"""Synthetic credit-risk data with a controlled proxy-variable effect."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit


def make_credit_data(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group = rng.choice(["A", "B"], size=n, p=[0.55, 0.45])
    group_b = (group == "B").astype(int)
    age = np.clip(rng.normal(39 + 2 * group_b, 10, n), 21, 70)
    income = np.clip(rng.lognormal(np.log(7800 - 1300 * group_b), 0.45, n), 1800, 35000)
    tenure = np.clip(rng.gamma(2.5, 2.2, n) - 0.7 * group_b, 0, 30)
    history = np.clip(age - 18 - rng.gamma(2.5, 2.0, n), 0.5, 35)
    debt_ratio = np.clip(rng.beta(2.2 + 0.5 * group_b, 4.5, n), 0.03, 0.95)
    arrears = rng.poisson(0.35 + 1.5 * debt_ratio + 0.25 * group_b)
    savings = np.clip(rng.lognormal(np.log(12000 + 0.9 * income), 0.70, n), 100, 150000)
    amount = np.clip(rng.lognormal(np.log(28000 + 2.0 * income), 0.50, n), 3000, 250000)
    zone = np.array([
        rng.choice(["central", "intermediate", "peripheral"], p=[0.52, 0.33, 0.15] if g == "A" else [0.18, 0.40, 0.42])
        for g in group
    ])
    zone_effect = pd.Series(zone).map({"central": 0.0, "intermediate": 0.12, "peripheral": 0.42}).to_numpy()
    logit = (-1.90 + 3.10 * debt_ratio + 0.58 * arrears - 0.000035 * income
             - 0.070 * tenure - 0.055 * history + 0.000008 * amount
             - 0.000004 * savings + zone_effect + 0.15 * group_b + rng.normal(0, 0.35, n))
    default = rng.binomial(1, expit(logit))
    return pd.DataFrame({
        "audit_group": group,
        "age": age.round().astype(int),
        "monthly_income": income.round(2),
        "employment_tenure": tenure.round(2),
        "credit_history_years": history.round(2),
        "debt_ratio": debt_ratio.round(3),
        "arrears_12m": arrears,
        "savings": savings.round(2),
        "requested_amount": amount.round(2),
        "zone": zone,
        "default": default,
    })
