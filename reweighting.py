"""
Condition F: Instance reweighting via density-ratio estimation

You already built a classifier that predicts data_source from features (the
separability diagnostic). Its predicted P(real_lanark | features) is, in
effect, a density-ratio estimate: rows -- including SYNTHETIC rows -- that
"look like" Lanark get a high score; rows that look distinctly urban/synthetic
get a low score.

This condition uses that score as a per-row SAMPLE WEIGHT when training on the
pooled data: w_i = P(real_lanark | x_i) / P(not real_lanark | x_i). Synthetic
rows that happen to resemble Lanark contribute more to training; synthetic
rows that don't resemble Lanark at all contribute less. This is a standard,
lightweight domain-adaptation technique (importance-weighted training) and
directly reuses infrastructure you've already built, rather than requiring a
new hierarchical/fine-tuning setup.

Usage:
    python condition_f_reweighting.py \
        --toronto data/synthetic_toronto_sasm.csv \
        --ottawa data/synthetic_ottawa_sasm.csv \
        --lanark data/real_lanark.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
GROUP_COL = "data_source"


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


def main(toronto_path, ottawa_path, lanark_path, val_size=0.2, weight_cap=20.0):
    toronto_df = load_and_tag(toronto_path, "toronto_synth", "synthetic")
    ottawa_df = load_and_tag(ottawa_path, "ottawa_synth", "synthetic")
    lanark_df = load_and_tag(lanark_path, "real_lanark", "observed")

    lanark_train, lanark_val = train_test_split(
        lanark_df, test_size=val_size, stratify=lanark_df[TARGET],
        random_state=RANDOM_STATE
    )

    # --- Pooled TRAINING data (Lanark train + full synthetic pools) ---
    train_pool = pd.concat([lanark_train, toronto_df, ottawa_df], ignore_index=True)
    train_pool, num_fill = impute_missing(train_pool, NUMERIC_FEATURES)
    train_pool, cat_fill = impute_categorical(train_pool, CATEGORICAL_FEATURES)

    X_pool, y_pool, cat_columns = build_xy(train_pool)
    is_lanark = (train_pool[GROUP_COL] == "real_lanark").astype(int).reset_index(drop=True)

    print(f"Pooled training data: {len(X_pool)} rows "
          f"({train_pool[GROUP_COL].value_counts().to_dict()})")

    # =========================================================
    # STEP 1: Fit the density-ratio / separability classifier
    # (This is the same diagnostic classifier from earlier in the project,
    # reused here as the actual weighting mechanism rather than just a check.)
    # =========================================================
    separability_model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        eval_metric="logloss", random_state=RANDOM_STATE,
    )
    separability_model.fit(X_pool, is_lanark)
    p_lanark = separability_model.predict_proba(X_pool)[:, 1]

    # Clip to avoid division by ~0 producing enormous/unstable weights
    p_lanark_clipped = np.clip(p_lanark, 1e-3, 1 - 1e-3)
    density_ratio_weights = p_lanark_clipped / (1 - p_lanark_clipped)

    # Cap extreme weights -- without this, a handful of rows that look
    # EXTREMELY Lanark-like can dominate the whole training loss
    density_ratio_weights = np.clip(density_ratio_weights, None, weight_cap)

    print(f"\nDensity-ratio weights: min={density_ratio_weights.min():.3f}, "
          f"median={np.median(density_ratio_weights):.3f}, "
          f"max={density_ratio_weights.max():.3f} (capped at {weight_cap})")
    print(f"Mean weight for real Lanark rows: "
          f"{density_ratio_weights[is_lanark == 1].mean():.3f}")
    print(f"Mean weight for synthetic rows:   "
          f"{density_ratio_weights[is_lanark == 0].mean():.3f}")

    # =========================================================
    # STEP 2: Train the actual chronic_homeless model on the pooled data,
    # using these weights via sample_weight
    # =========================================================
    scale_pos_weight = (y_pool == 0).sum() / max((y_pool == 1).sum(), 1)
    # Combine the class-imbalance weight with the density-ratio weight per row
    class_weights = np.where(y_pool == 1, scale_pos_weight, 1.0)
    combined_weights = density_ratio_weights * class_weights

    reweighted_model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="aucpr", random_state=RANDOM_STATE,
    )
    reweighted_model.fit(X_pool, y_pool, sample_weight=combined_weights)
    print("\nReweighted model trained on pooled data "
          "(synthetic rows downweighted unless they resemble Lanark).")

    # =========================================================
    # EVALUATION on the real Lanark holdout
    # =========================================================
    lanark_val_imp, _ = impute_missing(lanark_val, NUMERIC_FEATURES, fill_values=num_fill)
    lanark_val_imp, _ = impute_categorical(lanark_val_imp, CATEGORICAL_FEATURES, fill_values=cat_fill)
    X_val, y_val, _ = build_xy(lanark_val_imp, categorical_encoders=cat_columns)

    val_proba = reweighted_model.predict_proba(X_val)[:, 1]
    results = evaluate(y_val, val_proba, "Tier 2 - REAL Lanark holdout (Condition F, instance reweighting)")

    ci = bootstrap_ci(y_val, val_proba)
    print(f"\nBootstrap AUC-PR on Lanark holdout: "
          f"{ci['mean']:.3f} (95% CI: {ci['ci_lower']:.3f}-{ci['ci_upper']:.3f}, "
          f"{ci['n_valid_boots']}/1000 valid resamples)")

    save_predictions("F", "xgboost", lanark_val.index, y_val, val_proba)

    return {"point_estimate": results, "bootstrap": ci,
            "separability_model": separability_model, "reweighted_model": reweighted_model,
            "weights": combined_weights}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Condition F: instance reweighting via density-ratio estimation")
    parser.add_argument("--toronto", required=True)
    parser.add_argument("--ottawa", required=True)
    parser.add_argument("--lanark", required=True)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--weight_cap", type=float, default=20.0,
                         help="Maximum allowed density-ratio weight, to prevent a "
                              "small number of rows from dominating the training loss")
    args = parser.parse_args()

    main(args.toronto, args.ottawa, args.lanark, args.val_size, args.weight_cap)