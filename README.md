# Credit Risk XAI & Fairness Audit

An end-to-end audit of a synthetic credit-scoring system covering predictive performance, explainability, fairness, threshold sensitivity, proxy-variable analysis, mitigation, and model governance.

## Why this project matters

Credit models are not only prediction systems: they are decision systems. This project examines how a model behaves, why it makes particular decisions, and whether those decisions remain equitable across groups.

## Highlights

- Reproducible synthetic credit-risk dataset
- Global and local model explanations with permutation importance and SHAP
- Local surrogate explanations inspired by LIME
- Group fairness metrics with bootstrap uncertainty
- Threshold sensitivity and proxy-variable checks
- Mitigation comparison and compact model-card workflow

## Tech stack

Python · pandas · NumPy · scikit-learn · SHAP · Matplotlib

## Repository structure

- `notebooks/credit_risk_xai_fairness_audit.ipynb` — complete analytical workflow
- `requirements.txt` — Python dependencies
- `data/external/` — reserved for optional external datasets

## Reproducibility

The notebook generates its own synthetic data, so no private or external dataset is required.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Notebook outputs are intentionally cleared in the repository so the analysis can be reproduced from a clean state.
