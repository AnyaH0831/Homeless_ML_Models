"""
Condition A: Target-only baseline

Train on Lanark's real training rows ONLY -- no Toronto/Ottawa synthetic data
involved at all. This is the "floor" condition: how well can you do with just
the small amount of real data you actually have, no transfer learning?

Evaluated on the SAME held-out Lanark validation slice used in Conditions
B/C/D (random_state=42, same val_size), so results are directly comparable.

Runs BOTH XGBoost and logistic regression (via SGDClassifier, consistent with
the Condition C logreg script) versions in one script, so you get the
model-type robustness check for this condition without writing it twice.

Usage:
    python condition_a_target_only.py \
        --lanark data/real_lanark.csv
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier

RANDOM_STATE = 42

# Same shared, leakage-free feature set used in Conditions B/C/D.
NUMERIC_FEATURES = ["age", "has_dependents", "outdoor_sleeping", "youth",
                     "no_income",
                     "has_dependents_available_in_source",
                     "outdoor_sleeping_available_in_source"]
CATEGORICAL_FEATURES = ["gender", "income_type"]
TARGET = "chronic_homeless"


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


def evaluate(y_true, proba, preds, label):
    y_true = np.asarray(y_true)
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
    return {"mean": scores.mean(),
            "ci_lower": np.percentile(scores, 2.5),
            "ci_upper": np.percentile(scores, 97.5),
            "n_valid_boots": len(scores)}


def main(lanark_path, val_size=0.2, xgb_n_estimators=100, xgb_max_depth=3,
         logreg_epochs=100, logreg_alpha=0.01):

    # --- Load Lanark ---
    lanark_df = pd.read_csv(lanark_path)
    lanark_df["data_source"] = "real_lanark"
    lanark_df["data_type"] = "observed"

    # --- Split Lanark into train/val ONCE, stratified, fixed seed ---
    # SAME random_state/val_size as Conditions B/C/D so the held-out rows match.
    lanark_train, lanark_val = train_test_split(
        lanark_df, test_size=val_size, stratify=lanark_df[TARGET],
        random_state=RANDOM_STATE
    )

    # --- Impute using Lanark-TRAIN-only stats (no leakage from val) ---
    lanark_train_imp, num_fill = impute_missing(lanark_train, NUMERIC_FEATURES)
    lanark_train_imp, cat_fill = impute_categorical(lanark_train_imp, CATEGORICAL_FEATURES)

    lanark_val_imp, _ = impute_missing(lanark_val, NUMERIC_FEATURES, fill_values=num_fill)
    lanark_val_imp, _ = impute_categorical(lanark_val_imp, CATEGORICAL_FEATURES, fill_values=cat_fill)

    X_train, y_train, cat_columns = build_xy(lanark_train_imp)
    X_val, y_val, _ = build_xy(lanark_val_imp, categorical_encoders=cat_columns)

    print(f"Lanark training data (target-only, no synthetic data used at all): "
          f"{len(X_train)} rows, {y_train.sum()} positives ({y_train.mean():.1%})")
    print(f"Lanark validation (held out): "
          f"{len(X_val)} rows, {y_val.sum()} positives ({y_val.mean():.1%})")

    all_results = {}

    # =========================================================
    # MODEL 1: XGBoost, trained on Lanark alone
    # =========================================================
    print("\n" + "=" * 60)
    print("MODEL: XGBoost (Condition A, target-only)")
    print("=" * 60)

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_model = XGBClassifier(
        n_estimators=xgb_n_estimators,   # kept modest given only ~450ish training rows
        max_depth=xgb_max_depth,         # shallow trees, guard against overfitting on small n
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,                  # extra L2 regularization given the small sample
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
    )
    xgb_model.fit(X_train, y_train)

    # Tier 1: sanity check on training data itself
    train_proba = xgb_model.predict_proba(X_train)[:, 1]
    train_preds = xgb_model.predict(X_train)
    evaluate(y_train, train_proba, train_preds,
             "Tier 1 - Lanark training rows (sanity check only, XGBoost)")

    # Tier 2: the real result
    val_proba = xgb_model.predict_proba(X_val)[:, 1]
    val_preds = xgb_model.predict(X_val)
    xgb_results = evaluate(y_val, val_proba, val_preds,
                            "Tier 2 - REAL Lanark holdout (Condition A, XGBoost)")
    xgb_ci = bootstrap_ci(y_val, val_proba, n_boot=1000)
    print(f"\nBootstrap AUC-PR (XGBoost): {xgb_ci['mean']:.3f} "
          f"(95% CI: {xgb_ci['ci_lower']:.3f}-{xgb_ci['ci_upper']:.3f}, "
          f"{xgb_ci['n_valid_boots']}/1000 valid resamples)")

    all_results["xgboost"] = {"point_estimate": xgb_results, "bootstrap": xgb_ci, "model": xgb_model}

    # =========================================================
    # MODEL 2: Logistic regression (SGDClassifier), trained on Lanark alone
    # =========================================================
    print("\n" + "=" * 60)
    print("MODEL: Logistic regression (Condition A, target-only)")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    logreg_model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=logreg_alpha,              # heavier regularization given the small sample
        learning_rate="optimal",
        max_iter=logreg_epochs,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    logreg_model.fit(X_train_scaled, y_train)

    train_proba_lr = logreg_model.predict_proba(X_train_scaled)[:, 1]
    train_preds_lr = logreg_model.predict(X_train_scaled)
    evaluate(y_train, train_proba_lr, train_preds_lr,
             "Tier 1 - Lanark training rows (sanity check only, logreg)")

    val_proba_lr = logreg_model.predict_proba(X_val_scaled)[:, 1]
    val_preds_lr = logreg_model.predict(X_val_scaled)
    logreg_results = evaluate(y_val, val_proba_lr, val_preds_lr,
                               "Tier 2 - REAL Lanark holdout (Condition A, logreg)")
    logreg_ci = bootstrap_ci(y_val, val_proba_lr, n_boot=1000)
    print(f"\nBootstrap AUC-PR (logreg): {logreg_ci['mean']:.3f} "
          f"(95% CI: {logreg_ci['ci_lower']:.3f}-{logreg_ci['ci_upper']:.3f}, "
          f"{logreg_ci['n_valid_boots']}/1000 valid resamples)")

    all_results["logreg"] = {"point_estimate": logreg_results, "bootstrap": logreg_ci,
                              "model": logreg_model, "scaler": scaler}

    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n" + "=" * 60)
    print("CONDITION A SUMMARY (target-only, real Lanark holdout)")
    print("=" * 60)
    for name, res in all_results.items():
        pe = res["point_estimate"]
        print(f"{name:10s}  AUC-ROC={pe['auc_roc']:.3f}  AUC-PR={pe['auc_pr']:.3f}  "
              f"Brier={pe['brier']:.3f}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Condition A: target-only baseline (real Lanark data alone)")
    parser.add_argument("--lanark", required=True, help="Path to real Lanark CSV")
    parser.add_argument("--val_size", type=float, default=0.2, help="Lanark holdout fraction")
    parser.add_argument("--xgb_n_estimators", type=int, default=100)
    parser.add_argument("--xgb_max_depth", type=int, default=3)
    parser.add_argument("--logreg_epochs", type=int, default=100)
    parser.add_argument("--logreg_alpha", type=float, default=0.01)
    args = parser.parse_args()

    main(args.lanark, args.val_size, args.xgb_n_estimators, args.xgb_max_depth,
         args.logreg_epochs, args.logreg_alpha)