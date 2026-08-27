"""Small helpers for audit summaries and model-card style documentation."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ModelCard:
    name: str
    version: str
    purpose: str
    target: str
    threshold: float
    model_type: str
    data_scope: str
    explainability: str
    known_risks: str
    controls: str

    def to_dict(self):
        return asdict(self)


def deployment_recommendation(selection_ratio: float, opportunity_gap: float, auc: float) -> str:
    if auc < 0.70:
        return "do_not_deploy: predictive performance below minimum review threshold"
    if selection_ratio < 0.80 or opportunity_gap > 0.10:
        return "return_for_mitigation_and_independent_validation"
    return "approve_with_monitoring_and_documented_controls"
