# Credit Risk XAI & Fairness Audit

An end-to-end audit of a synthetic credit-scoring system covering predictive performance, explainability, fairness, threshold sensitivity, proxy-variable analysis, mitigation, and model governance.

## Why this project matters

Credit models are not only prediction systems: they are decision systems. This project examines how a model behaves, why it makes particular decisions, and whether those decisions remain equitable across groups.

## Highlights

- Reproducible synthetic credit-risk data
- Random Forest scoring pipeline
- Permutation-based global explainability
- Group-level approval and error analysis
- Threshold sensitivity and selection-ratio analysis
- Explicit separation between predictive performance and decision fairness

## Tech stack

Python · pandas · NumPy · scikit-learn

## Repository structure

- `src/audit_demo.py` — compact reproducible XAI/fairness audit
- `requirements.txt` — Python dependencies
- `.gitignore` — excludes environments and generated artifacts

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python src/audit_demo.py
```

The dataset is generated synthetically, so no private or external data are required.

## Portfolio context

This repository is a production-style refactoring of a broader analytical notebook on explainable AI, fairness, proxy variables, mitigation, and model governance. The public version emphasizes reusable Python code and reproducibility.
