"""
Condition E: Post-hoc recalibration

Condition B's zero-shot model was trained on a synthetic pool that is ~69%
positive, then evaluated on real Lanark data that's only ~46% positive. That
base-rate mismatch alone can badly hurt a model's predicted probabilities even
if its underlying RANKING of who's more/less at risk is still reasonably good.

This condition tests a much cheaper fix than full fine-tuning: keep Condition
B's model completely frozen (no retraining at all), and fit a small correction
--Platt scaling (a 1-2 parameter logistic regression on the model's own raw
output) or isotonic regression (a flexible monotonic correction) -- using only
Lanark's training rows, to correct the probability CALIBRATION only.

If this closes much of the gap with fine-tuning (Condition C), that's a
notable, efficiency-focused finding: you don't need to retrain the whole
model, just recalibrate its output. If it doesn't, that's evidence the gap is
really about the feature-outcome RELATIONSHIP differing by source, not just
the overall base rate -- strengthening the case for fine-tuning specifically.

Usage:
    python condition_e_recalibration.py \
        --toronto data/synthetic_toronto_sasm.csv \
        --ottawa data/synthetic_ottawa_sasm.csv \
        --lanark data/real_lanark.csv \
        --method both
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier

RANDOM_STATE = 42

NUMERIC_FEATURES = ["age", "has_dependents", "outdoor_sleeping", "youth",
                     "no_income",
                     "has_dependents_available_in_source",
                     "outdoor_sleeping_available_in_source"]
CATEGORICAL_FEATURES = ["gender", "income_type"]
TARGET = "chronic_homeless"


def load_and_tag(path, source_name, data_type):
    df = pd.read_csv(path)
    df["data_source"] = source_name
    df["data_type"] = data_type
    return df


def impute_missing(df, numeric_cols, fill_values=None):
    df = df.copy()
    computed = {}
    for col in numeric_cols:
        if col in df.columns and df[col].isna().any():
            fill_val = fill_values[col] if fill_values and col in fill_values else df[col].median()
            df[col] = df[col].fillna(fill_val)
            computed[col] = fill_val
    return df, computed


def impute_categorical(df, categorical_cols, fill_values=None):
    df = df.copy()
    computed = {}
    for col in categorical_cols:
        if col in df.columns and df[col].isna().any():
            if fill_values is not None and col in fill_values:
                fill_val = fill_values[col]
            else:
                mode = df[col].mode(dropna=True)
                fill_val = mode.iloc[0] if len(mode) > 0 else "unknown"
            df[col] = df[col].fillna(fill_val)
            computed[col] = fill_val
    return df, computed


def build_xy(df, categorical_encoders=None):
    X_numeric = df[NUMERIC_FEATURES].copy()
    X_categorical = pd.get_dummies(df[CATEGORICAL_FEATURES], prefix=CATEGORICAL_FEATURES)
    if categorical_encoders is not None:
        X_categorical = X_categorical.reindex(columns=categorical_encoders, fill_value=0)
    X = pd.concat([X_numeric.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)
    y = df[TARGET].reset_index(drop=True)
    return X, y, list(X_categorical.columns)


def evaluate(y_true, proba, label):
    y_true = np.asarray(y_true)
    preds = (proba >= 0.5).astype(int)
    auc_roc = roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else float("nan")
    auc_pr = average_precision_score(y_true, proba) if len(np.unique(y_true)) > 1 else float("nan")
    brier = brier_score_loss(y_true, proba)

    print(f"\n=== Evaluation: {label} (n={len(y_true)}, positives={int(y_true.sum())}) ===")
    print(f"AUC-ROC:      {auc_roc:.3f}")
    print(f"AUC-PR:       {auc_pr:.3f}")
    print(f"Brier score:  {brier:.3f}")
    print(f"Confusion matrix:\n{confusion_matrix(y_true, preds)}")
    print(classification_report(y_true, preds, zero_division=0))

    return {"label": label, "n": len(y_true), "positives": int(y_true.sum()),
            "auc_roc": auc_roc, "auc_pr": auc_pr, "brier": brier}


def bootstrap_ci(y_true, proba, n_boot=1000, metric_fn=average_precision_score, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], proba[idx]))
    scores = np.array(scores)
    return {"mean": scores.mean(), "ci_lower": np.percentile(scores, 2.5),
            "ci_upper": np.percentile(scores, 97.5), "n_valid_boots": len(scores)}


def save_predictions(condition_name, model_name, lanark_val_index, y_true, proba, out_dir="predictions"):
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"lanark_idx": lanark_val_index, "y_true": y_true, "proba": proba}) \
        .to_csv(os.path.join(out_dir, f"{condition_name}_{model_name}.csv"), index=False)


def main(toronto_path, ottawa_path, lanark_path, val_size=0.2, method="both"):
    # --- Load and tag each source ---
    toronto_df = load_and_tag(toronto_path, "toronto_synth", "synthetic")
    ottawa_df = load_and_tag(ottawa_path, "ottawa_synth", "synthetic")
    lanark_df = load_and_tag(lanark_path, "real_lanark", "observed")

    # --- Same Lanark train/val split as every other condition ---
    lanark_train, lanark_val = train_test_split(
        lanark_df, test_size=val_size, stratify=lanark_df[TARGET],
        random_state=RANDOM_STATE
    )

    # =========================================================
    # STAGE 1: Train the base model on synthetic data (identical to Condition B)
    # =========================================================
    train_pool = pd.concat([toronto_df, ottawa_df], ignore_index=True)
    train_pool, num_fill = impute_missing(train_pool, NUMERIC_FEATURES)
    train_pool, cat_fill = impute_categorical(train_pool, CATEGORICAL_FEATURES)

    X_train, y_train, cat_columns = build_xy(train_pool)

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    base_model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        random_state=RANDOM_STATE,
    )
    base_model.fit(X_train, y_train)
    print("Base model trained on synthetic data only (same as Condition B).")

    # --- Prepare Lanark train/val using the base model's own imputation/encoding ---
    lanark_train_imp, _ = impute_missing(lanark_train, NUMERIC_FEATURES, fill_values=num_fill)
    lanark_train_imp, _ = impute_categorical(lanark_train_imp, CATEGORICAL_FEATURES, fill_values=cat_fill)
    X_lanark_train, y_lanark_train, _ = build_xy(lanark_train_imp, categorical_encoders=cat_columns)

    lanark_val_imp, _ = impute_missing(lanark_val, NUMERIC_FEATURES, fill_values=num_fill)
    lanark_val_imp, _ = impute_categorical(lanark_val_imp, CATEGORICAL_FEATURES, fill_values=cat_fill)
    X_lanark_val, y_lanark_val, _ = build_xy(lanark_val_imp, categorical_encoders=cat_columns)

    # Base model's RAW, uncalibrated probabilities (this is Condition B's result)
    raw_train_proba = base_model.predict_proba(X_lanark_train)[:, 1]
    raw_val_proba = base_model.predict_proba(X_lanark_val)[:, 1]

    evaluate(y_lanark_val, raw_val_proba,
             "Reference - raw, uncalibrated model on Lanark holdout (= Condition B)")

    results = {}

    # =========================================================
    # STAGE 2a: Platt scaling (fit a 1-feature logistic regression
    # mapping raw probability -> corrected probability, using Lanark TRAIN only)
    # =========================================================
    if method in ("platt", "both"):
        platt = LogisticRegression()
        platt.fit(raw_train_proba.reshape(-1, 1), y_lanark_train)
        platt_val_proba = platt.predict_proba(raw_val_proba.reshape(-1, 1))[:, 1]

        platt_results = evaluate(y_lanark_val, platt_val_proba,
                                  "Tier 2 - Platt-scaled model on REAL Lanark holdout (Condition E-Platt)")
        platt_ci = bootstrap_ci(y_lanark_val, platt_val_proba)
        print(f"\nBootstrap AUC-PR (Platt): {platt_ci['mean']:.3f} "
              f"(95% CI: {platt_ci['ci_lower']:.3f}-{platt_ci['ci_upper']:.3f})")
        save_predictions("E", "platt", lanark_val.index, y_lanark_val, platt_val_proba)
        results["platt"] = {"point_estimate": platt_results, "bootstrap": platt_ci}

    # =========================================================
    # STAGE 2b: Isotonic regression (a more flexible, non-parametric
    # monotonic correction -- can capture more complex miscalibration
    # patterns than Platt's single logistic curve, but needs more data
    # to avoid overfitting; Lanark's ~450 training rows is a reasonable
    # amount for this, though still modest)
    # =========================================================
    if method in ("isotonic", "both"):
        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(raw_train_proba, y_lanark_train)
        isotonic_val_proba = isotonic.predict(raw_val_proba)

        isotonic_results = evaluate(y_lanark_val, isotonic_val_proba,
                                     "Tier 2 - Isotonic-calibrated model on REAL Lanark holdout (Condition E-Isotonic)")
        isotonic_ci = bootstrap_ci(y_lanark_val, isotonic_val_proba)
        print(f"\nBootstrap AUC-PR (Isotonic): {isotonic_ci['mean']:.3f} "
              f"(95% CI: {isotonic_ci['ci_lower']:.3f}-{isotonic_ci['ci_upper']:.3f})")
        save_predictions("E", "isotonic", lanark_val.index, y_lanark_val, isotonic_val_proba)
        results["isotonic"] = {"point_estimate": isotonic_results, "bootstrap": isotonic_ci}

    # Also save the raw/reference predictions, so this script alone can
    # contribute the "B" file to compare_conditions.py if you haven't already.
    save_predictions("B", "xgboost", lanark_val.index, y_lanark_val, raw_val_proba)

    print("\nNote: AUC-ROC and AUC-PR are RANK-based metrics, so recalibration "
          "(which only rescales probabilities monotonically) will barely change "
          "them -- the metric to watch here is the BRIER SCORE, which directly "
          "measures whether the predicted probabilities themselves are more "
          "accurate, not just whether the ranking of cases is correct.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Condition E: post-hoc recalibration of the zero-shot model")
    parser.add_argument("--toronto", required=True)
    parser.add_argument("--ottawa", required=True)
    parser.add_argument("--lanark", required=True)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--method", choices=["platt", "isotonic", "both"], default="both")
    args = parser.parse_args()

    main(args.toronto, args.ottawa, args.lanark, args.val_size, args.method)