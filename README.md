# Homelessness ML Models

This repository evaluates domain-adaptation and machine-learning models for predicting `chronic_homeless` status across Toronto synthetic data, Ottawa synthetic data, and real-world Lanark administrative data. The pipeline uses a standardized feature set, positive-unlabeled (PU) calibration, and a fixed Lanark holdout ($n=114$) to enable paired statistical comparisons across all experimental conditions.

---

## Data and Feature Policy

### Shared Feature Set
- **Numeric/Binary:** `age`, `has_dependents`, `outdoor_sleeping`, `youth`, `no_income`
- **Categorical:** `gender`, `income_type`
- **Availability Indicators:** `has_dependents_available_in_source`, `outdoor_sleeping_available_in_source`
- **Target Variable:** `chronic_homeless`

### Data Preprocessing & Harmonization
- `indigenous_flag` and `years_homeless` are excluded because target-domain administrative records are incomplete.
- `mental_health` and `substance_use` in the Lanark dataset originate from caseworker notes (representing positive instances only). These features are modeled under a Positive-Unlabeled (PU) framework (`pu_learning.py`).
- `income_type` is harmonized into `disability`, `welfare`, `employment`, `informal`, or `other`; missing entries are encoded or imputed per condition protocol.
- Source data processing is managed via `data/merge_data.py` and `advanced_features.py`.

---

## Evaluation Framework & Triage Metrics

Given concept drift, asymmetric missingness, and municipal resource limits, standard AUC metrics are insufficient on their own. The primary performance metric is **Top-$K$ Precision (Precision@$K$)**, simulating caseworker triage capacities:

- **Top 5% ($k=0.05$, $n=6$):** Precision at top 5% risk threshold.
- **Top 10% ($k=0.10$, $n=12$):** Precision at top 10% risk threshold (primary decile benchmark).
- **Top 20% ($k=0.20$, $n=23$):** Precision and recall at top 20% risk threshold.

All conditions are benchmarked against a fixed 114-row Lanark holdout ($46.5\%$ positive base rate) using paired bootstrap confidence intervals (1,000 resamples) and paired permutation tests ($1,000$ iterations).

---

## Experimental Conditions

### Baseline & Transfer Conditions (A–G)
- **Condition A (Target-Only Baseline):** Models (`xgboost_lanark.py`, `logreg_lanark.py`) trained exclusively on Lanark training rows ($n=454$).
- **Condition B (Synthetic Zero-Shot):** Models (`synth_xgboost.py`, `synth_log.py`) trained on combined Toronto and Ottawa synthetic data, evaluated directly on Lanark.
- **Condition C (Pretrain & Fine-Tune):** Pretrained on synthetic pools, then adapted on Lanark training rows (`xgboost_finetune.py`, `logreg_finetune.py`).
- **Condition D / D2 (Bayesian Partial Pooling):** Hierarchical Bernoulli models fit with Bambi/PyMC using source random intercepts (`D`, `partial_pooling.py`) and random slopes (`D2`, `partial_pooling_random_slopes.py`).
- **Condition E (Post-Hoc Recalibration):** Platt scaling (`E_platt`) and Isotonic regression (`E_isotonic`) applied to Condition B predictions (`condition_e_recalibration.py`).
- **Condition F (Instance Reweighting):** Density-ratio reweighting based on source-separability classification (`condition_f_reweighting.py`).
- **Condition G (Serialized BERT Fine-Tuning):** Text-serialized structured representations fine-tuned on BERT encoder layers 10–11 (`condition_g_bert_finetune.py`).

### New Advanced Conditions (H–I)
- **Condition H (Domain-Adversarial Neural Network - DANN):** End-to-end domain adaptation (`condition_h_dann.py`) featuring a feature extractor, label classifier, and domain classifier connected via a Gradient Reversal Layer (GRL). Learns domain-invariant latent representations across Toronto, Ottawa, and Lanark. Supports native Apple Silicon (`mps`) acceleration.
- **Condition I (Positive-Unlabeled Calibration):** Integrates `ElkanNotoPU` (`pu_learning.py`) to estimate true risk probabilities $P(Y=1 \mid X)$ under asymmetric caseworker note logging.

---

## Summary of Results (Lanark Holdout $n=114$, Positives $n=53$)

| Condition | Strategy / Model | Precision @ 5% | Precision @ 10% | Recall @ 20% | Paired Diff vs `H_dann` ($p$-value) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`H_dann`** | Domain-Adversarial Neural Net | **100.0%** | **91.7%** | 26.4% | Reference |
| **`D_bayes`** | Bayesian Partial Pooling (Intercepts) | 83.3% | **91.7%** | **32.1%** | $p = 1.000$ |
| **`D2_bayes`** | Bayesian Partial Pooling (Random Slopes) | 83.3% | **91.7%** | **32.1%** | $p = 1.000$ |
| **`B_logreg`** | Synthetic Zero-Shot Logistic Reg | **100.0%** | **91.7%** | 28.3% | $p = 1.000$ |
| **`A_xgboost`** | Lanark Target-Only XGBoost | **100.0%** | 75.0% | 24.5% | $p = 0.412$ |
| **`C_logreg`** | Pretrain + Fine-Tune LogReg | 83.3% | 66.7% | 28.3% | $p = 0.251$ |
| **`F_xgboost`** | Density Ratio Reweighting | 50.0% | 50.0% | 24.5% | $p = 0.082$ |
| **`E_isotonic`** | Isotonic Recalibration | 16.7% | 33.3% | 13.2% | **$p = 0.008$** |

### Key Findings
1. **Top-Decile Accuracy:** Domain-aligned architectures (`H_dann`, `D_bayes`, `D2_bayes`) achieve **91.7% Precision at $k=10\%$** (11 out of 12 flagged clients are true positive cases).
2. **Head Precision:** `H_dann`, `B_logreg`, `A_logreg`, and `A_xgboost` reach **100% Precision at $k=5\%$** (0 false positives in the top 6 slots).
3. **Recalibration Degradation:** Post-hoc isotonic calibration (`E_isotonic`) disrupts prediction rank order, dropping top-10% precision down to 33.3% ($p = 0.008$). Isotonic recalibration should be avoided for triage-focused systems.

---

## Usage Instructions

### Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt