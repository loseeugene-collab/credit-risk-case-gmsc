# Credit Risk Case — Give Me Some Credit (Logistic Regression)

## Executive summary (what a risk / product lead cares about)
- Built an interpretable **PD (probability‑of‑default)** model (logistic regression baseline) with **train‑only** preprocessing and **probability calibration**.
- **Holdout test (random_state=42):** base rate **6.68%**, ROC‑AUC **0.860**, KS **0.559**.
- **Risk concentration:** the **highest‑risk decile (Decile 10)** contains **52.9%** of all bads with **35.4%** bad rate, while **Decile 1** has **0.40%** bad rate.
- **Decisioning:** approval cut‑off is selected by maximizing **expected profit** using calibrated PDs, with optional **risk appetite constraints** (e.g., bad rate cap).
- **Monitoring:** PSI drift helper included; PSI(score) train→test in this split is **0.0004** (proxy demonstration).

> Note: Profit numbers are in “model dollars” because the public dataset has no true EAD/LGD/cashflows. The notebook exposes **R/L** parameters and supports scenario + constraint analysis.

## Contents
- `Credit_Risk_MiniCase_GMSC_LogReg_Main.ipynb` — full end‑to‑end mini‑case:
  - train/valid/test split (no leakage)
  - **safe preprocessing** fit on train only (median imputation + missingness flags + quantile capping)
  - logistic regression baseline + **probability calibration** (Platt scaling)
  - evaluation: ROC‑AUC, KS; **decile/lift** segmentation
  - decisioning: **expected profit curve**, optimal threshold, **constrained** optimal threshold
  - champion vs challenger comparison
  - monitoring checklist + **PSI drift helper** (score + optional feature PSI)

- `Credit_Risk_MiniCase_GMSC_Summary.ipynb` — slides‑style stakeholder summary:
  - 3 key visuals + policy table + ready executive summary text

## Dataset
Kaggle “Give Me Some Credit”.
Required file: `cs-training.csv` (upload to Colab runtime).
Optional: `Data Dictionary.xls` (feature definitions).

Target: `SeriousDlqin2yrs` (binary distress/default event within 2 years).

## How to run (Google Colab)
1) Open a notebook in Colab  
2) Upload `cs-training.csv` via the **Files** panel  
3) Run all cells  

The notebooks assume:
```python
DATA_PATH = "cs-training.csv"
```

## Decision economics (R/L) and threshold optimization
We use simplified unit economics due to lack of loan cashflows / EAD / LGD in the public dataset.

- `R` = profit from a *good* approved loan (interest + fees − funding − ops)
- `L` = loss from a *bad* approved loan (proxy for LGD×EAD + ops/collections)

Expected profit per approved applicant (using calibrated PD):
`E[profit] = (1 − PD) * R − PD * L`

### Scenario sensitivity (example settings)
Below are example thresholds chosen by maximizing profit on the **test** set (and the best threshold under **bad rate ≤ 2%** when applicable):

| Scenario | R | L | Opt t | Opt appr | Opt bad | Opt $/app | Opt t (bad≤2%) | Appr (bad≤2%) | Bad (bad≤2%) | $/app (bad≤2%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conservative | 15 | 600 | 0.024 | 51.2% | 1.02% | 3.12 | 0.024 | 51.2% | 1.02% | 3.12 |
| base | 15 | 300 | 0.048 | 69.0% | 1.72% | 6.11 | 0.048 | 69.0% | 1.72% | 6.11 |
| aggressive | 20 | 200 | 0.091 | 83.9% | 2.74% | 11.64 | 0.058 | 73.6% | 1.95% | 11.23 |

## Champion vs Challenger (example, base scenario)
Using **R=15, L=300**:
- **Champion (simple business rule):** approve if **PD < 0.080** → approval **81.2%**, bad rate **2.52%**, profit/applicant **5.55**
- **Challenger (profit‑optimal):** approve if **PD < 0.048** → approval **69.0%**, bad rate **1.72%**, profit/applicant **6.11**
- **Profit uplift:** **10.1%** vs champion (in the same R/L units)

## Monitoring (what you would run in production)
- Weekly: approval rate, bad rate, expected profit KPIs
- Model quality: AUC/KS stability (by cohort/vintage once timestamps exist)
- Drift: PSI on **score** and key **features** (baseline vs current period)

- Dataset size: 30,000, base default rate: 6.68%
- Baseline model: Logistic Regression (train-only quantile capping + median imputation + missingness flags + scaling)
- Test performance: AUC 0.860, KS 0.559
- Risk concentration: top decile captures 52.9% of bads with 35.4% bad rate
- Profit-optimal policy (R=15, L=300): threshold t=0.046, approval 68.2%, bad rate 1.7%
- Constrained policy (example: bad rate ≤ 2.0%): threshold t=0.046, approval 68.2%, bad rate 1.7%
- Champion–Challenger uplift: 10.0% profit per applicant (trade-off: approval vs risk)
- Monitoring proxy (train→test): PSI(score) = 0.0004

## Notes / limitations
- Public dataset: no true loan terms, EAD/LGD, or timestamps → economics and monitoring are demonstrated with simplified assumptions.
- In production you would add time splits, vintage tracking, and periodic recalibration.
