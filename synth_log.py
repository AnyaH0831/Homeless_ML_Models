"""
Condition B: Source-only, zero-shot transfer
Train Logistic Regression on synthetic data (Toronto + Ottawa) only.
Evaluate on a held-out slice of REAL Lanark data that the model never sees during training.

Usage:
    python condition_b_logistic_regression.py --toronto toronto.csv --ottawa ottawa.csv --lanark lanark.csv
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

# The shared, leakage-free feature set agreed on for the 4-condition comparison.
# (year, years_homeless, mental_health, substance_use dropped: leakage / too sparse)
NUMERIC_FEATURES = ["age", "has_dependents", "outdoor_sleeping", "youth",
                     "no_income",
                     "has_dependents_available_in_source",
                     "outdoor_sleeping_available_in_source"]
CATEGORICAL_FEATURES = ["gender", "income_type"]
TARGET = "chronic_homeless"

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_and_tag(path, source_name, data_type):
    df = pd.read_csv(path)
    df["data_source"] = source_name
    df["data_type"] = data_type
    return df


def impute_missing(df, numeric_cols, fill_values=None):
    """
    Fill missing values for columns that carry an '_available_in_source' flag.
    If fill_values is None, compute fill values (mode/median) from this df and return them
    (use this on the TRAINING data only, to avoid leaking target-domain stats).
    If fill_values is provided, apply them (use this on validation/target data).
    """
    df = df.copy()
    computed = {}
    for col in numeric_cols:
        if col in df.columns and df[col].isna().any():
            if fill_values is not None and col in fill_values:
                fill_val = fill_values[col]
            else:
                fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            computed[col] = fill_val
    return df, computed


def build_xy(df, categorical_encoders=None):
    """
    One-hot encode categorical features consistently.
    If categorical_encoders (list of column categories) is None, fit from this df
    and return the category lists (use on TRAINING data).
    If provided, align validation data to the same dummy columns (use on val/target data).
    """
    X_numeric = df[NUMERIC_FEATURES].copy()
    X_categorical = pd.get_dummies(df[CATEGORICAL_FEATURES], prefix=CATEGORICAL_FEATURES)

    if categorical_encoders is not None:
        # Align columns: add any missing dummy columns as 0, drop any extras
        X_categorical = X_categorical.reindex(columns=categorical_encoders, fill_value=0)

    X = pd.concat([X_numeric.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)
    y = df[TARGET].reset_index(drop=True)
    return X, y, list(X_categorical.columns)


def evaluate(model, X, y, label):
    proba = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    auc_roc = roc_auc_score(y, proba) if y.nunique() > 1 else float("nan")
    auc_pr = average_precision_score(y, proba) if y.nunique() > 1 else float("nan")
    brier = brier_score_loss(y, proba)

    print(f"\n=== Evaluation: {label} (n={len(y)}, positives={int(y.sum())}) ===")
    print(f"AUC-ROC:      {auc_roc:.3f}")
    print(f"AUC-PR:       {auc_pr:.3f}")
    print(f"Brier score:  {brier:.3f}")
    print(f"Confusion matrix:\n{confusion_matrix(y, preds)}")
    print(classification_report(y, preds, zero_division=0))

    return {"label": label, "n": len(y), "positives": int(y.sum()),
            "auc_roc": auc_roc, "auc_pr": auc_pr, "brier": brier}


def bootstrap_ci(model, X, y, n_boot=1000, metric_fn=average_precision_score, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y)
    y_arr = y.to_numpy()
    proba_full = model.predict_proba(X)[:, 1]

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_sample = y_arr[idx]
        if len(np.unique(y_sample)) < 2:
            continue  # skip resamples with only one class present
        proba_sample = proba_full[idx]
        scores.append(metric_fn(y_sample, proba_sample))

    scores = np.array(scores)
    return {
        "mean": scores.mean(),
        "ci_lower": np.percentile(scores, 2.5),
        "ci_upper": np.percentile(scores, 97.5),
        "n_valid_boots": len(scores),
    }


def main(toronto_path, ottawa_path, lanark_path, val_size=0.2):
    # --- Load and tag each source ---
    toronto_df = load_and_tag(toronto_path, "toronto_synth", "synthetic")
    ottawa_df = load_and_tag(ottawa_path, "ottawa_synth", "synthetic")
    lanark_df = load_and_tag(lanark_path, "real_lanark", "observed")

    # --- Split Lanark into train/val ONCE, stratified, fixed seed ---
    lanark_train, lanark_val = train_test_split(
        lanark_df, test_size=val_size, stratify=lanark_df[TARGET],
        random_state=RANDOM_STATE
    )

    # --- Condition B training pool: synthetic sources only ---
    train_pool = pd.concat([toronto_df, ottawa_df], ignore_index=True)

    # --- Impute missing values using TRAINING pool stats only (no leakage from Lanark) ---
    train_pool, fill_values = impute_missing(train_pool, NUMERIC_FEATURES)
    lanark_val_imputed, _ = impute_missing(lanark_val, NUMERIC_FEATURES, fill_values=fill_values)

    # --- Build feature matrices, fitting encoders on training pool ---
    X_train, y_train, cat_columns = build_xy(train_pool)
    X_val, y_val, _ = build_xy(lanark_val_imputed, categorical_encoders=cat_columns)

    # --- Scale features using TRAINING pool stats only ---
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

    print(f"Training pool (synthetic only): {len(X_train_scaled)} rows, "
          f"{y_train.sum()} positives ({y_train.mean():.1%})")
    print(f"Lanark validation (real, held out): {len(X_val_scaled)} rows, "
          f"{y_val.sum()} positives ({y_val.mean():.1%})")

    # --- Train Logistic Regression ---
    model = LogisticRegression(
        class_weight="balanced",  # handles class imbalance
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_scaled, y_train)

    # --- Evaluate: Tier 1 sanity check (on synthetic training pool itself) ---
    evaluate(model, X_train_scaled, y_train, "Tier 1 - synthetic training pool (sanity check only)")

    # --- Evaluate: Tier 2 - the real result ---
    results = evaluate(model, X_val_scaled, y_val, "Tier 2 - REAL Lanark holdout (Condition B, zero-shot)")

    # --- Bootstrap CI on the real Lanark holdout ---
    ci = bootstrap_ci(model, X_val_scaled, y_val, n_boot=1000)
    print(f"\nBootstrap AUC-PR on Lanark holdout: "
          f"{ci['mean']:.3f} (95% CI: {ci['ci_lower']:.3f}-{ci['ci_upper']:.3f}, "
          f"{ci['n_valid_boots']}/1000 valid resamples)")

    return {"point_estimate": results, "bootstrap": ci, "model": model,
            "lanark_val_idx": lanark_val.index}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Condition B: train on synthetic, validate on real Lanark")
    parser.add_argument("--toronto", required=True, help="data/synthetic_toronto_sasm.csv")
    parser.add_argument("--ottawa", required=True, help="data/synthetic_ottawa_sasm.csv")
    parser.add_argument("--lanark", required=True, help="data/real_lanark.csv")
    parser.add_argument("--val_size", type=float, default=0.2, help="Lanark holdout fraction")
    args = parser.parse_args()

    main(args.toronto, args.ottawa, args.lanark, args.val_size)