"""
Condition C: Pretrain-then-finetune transfer

Stage 1 (pretrain): Train XGBoost on synthetic data (Toronto + Ottawa) only,
                     same as Condition B.
Stage 2 (finetune):  Continue training ("warm start") that same model on
                     Lanark's real training rows only, using a small number
                     of additional trees, a low learning rate, and heavier
                     regularization -- so the model adapts toward Lanark's
                     real distribution without fully overwriting what it
                     learned from the much larger synthetic pool.

Evaluated on the SAME held-out Lanark validation slice used in Conditions
A/B/D (random_state=42, same val_size), so results are directly comparable.

Usage:
    python condition_c_finetune.py \
        --toronto data/synthetic_toronto_sasm.csv \
        --ottawa data/synthetic_ottawa_sasm.csv \
        --lanark data/real_lanark.csv
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier

RANDOM_STATE = 42

# Same shared, leakage-free feature set used in Conditions A/B/D.
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
    y_arr = y.to_numpy()
    n = len(y)
    proba_full = model.predict_proba(X)[:, 1]

    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_sample = y_arr[idx]
        if len(np.unique(y_sample)) < 2:
            continue
        scores.append(metric_fn(y_sample, proba_full[idx]))

    scores = np.array(scores)
    return {"mean": scores.mean(),
            "ci_lower": np.percentile(scores, 2.5),
            "ci_upper": np.percentile(scores, 97.5),
            "n_valid_boots": len(scores)}


def main(toronto_path, ottawa_path, lanark_path, val_size=0.2,
         finetune_n_estimators=50, finetune_learning_rate=0.01,
         finetune_reg_lambda=5.0):

    # --- Load and tag each source ---
    toronto_df = load_and_tag(toronto_path, "toronto_synth", "synthetic")
    ottawa_df = load_and_tag(ottawa_path, "ottawa_synth", "synthetic")
    lanark_df = load_and_tag(lanark_path, "real_lanark", "observed")

    # --- Split Lanark into train/val ONCE, stratified, fixed seed ---
    # SAME random_state/val_size as Conditions A/B/D so the held-out rows match.
    lanark_train, lanark_val = train_test_split(
        lanark_df, test_size=val_size, stratify=lanark_df[TARGET],
        random_state=RANDOM_STATE
    )

    # =========================================================
    # STAGE 1: PRETRAIN on synthetic data only (same as Condition B)
    # =========================================================
    pretrain_pool = pd.concat([toronto_df, ottawa_df], ignore_index=True)

    pretrain_pool, num_fill_values = impute_missing(pretrain_pool, NUMERIC_FEATURES)
    pretrain_pool, cat_fill_values = impute_categorical(pretrain_pool, CATEGORICAL_FEATURES)

    X_pretrain, y_pretrain, cat_columns = build_xy(pretrain_pool)

    print(f"Stage 1 (pretrain) pool: {len(X_pretrain)} rows, "
          f"{y_pretrain.sum()} positives ({y_pretrain.mean():.1%})")

    scale_pos_weight_pretrain = (y_pretrain == 0).sum() / max((y_pretrain == 1).sum(), 1)
    base_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight_pretrain,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
    )
    base_model.fit(X_pretrain, y_pretrain)
    print("Stage 1 (pretrain) complete.")

    # =========================================================
    # STAGE 2: FINE-TUNE on real Lanark training rows only
    # =========================================================
    # Impute Lanark's own missing values using Lanark-TRAIN-only stats
    # (not the synthetic pool's stats) -- fine-tuning is meant to adapt
    # toward Lanark's real distribution, so its own fill values are used here.
    lanark_train_imp, lanark_num_fill = impute_missing(lanark_train, NUMERIC_FEATURES)
    lanark_train_imp, lanark_cat_fill = impute_categorical(lanark_train_imp, CATEGORICAL_FEATURES)

    # IMPORTANT: align dummy columns to the SAME columns used in Stage 1,
    # so the fine-tuned booster sees the exact same feature layout.
    X_finetune, y_finetune, _ = build_xy(lanark_train_imp, categorical_encoders=cat_columns)

    print(f"\nStage 2 (finetune) pool: {len(X_finetune)} rows, "
          f"{y_finetune.sum()} positives ({y_finetune.mean():.1%})")

    scale_pos_weight_finetune = (y_finetune == 0).sum() / max((y_finetune == 1).sum(), 1)

    # Continue boosting from the pretrained model:
    # - few additional trees (finetune_n_estimators)
    # - low learning rate
    # - stronger L2 regularization (reg_lambda)
    # This lets Lanark nudge the model without overwriting what was learned
    # from the much larger synthetic pool, given how little real data there is.
    finetuned_model = XGBClassifier(
        n_estimators=finetune_n_estimators,
        max_depth=3,                       # shallower than pretrain stage, extra guard against overfitting
        learning_rate=finetune_learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=finetune_reg_lambda,     # heavier L2 penalty during fine-tuning
        scale_pos_weight=scale_pos_weight_finetune,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
    )
    finetuned_model.fit(X_finetune, y_finetune, xgb_model=base_model.get_booster())
    print("Stage 2 (finetune) complete.")

    # =========================================================
    # EVALUATION
    # =========================================================
    # Impute + encode the Lanark validation set using the SAME fill values
    # and dummy columns established in Stage 2 (no leakage from lanark_val).
    lanark_val_imp, _ = impute_missing(lanark_val, NUMERIC_FEATURES, fill_values=lanark_num_fill)
    lanark_val_imp, _ = impute_categorical(lanark_val_imp, CATEGORICAL_FEATURES, fill_values=lanark_cat_fill)
    X_val, y_val, _ = build_xy(lanark_val_imp, categorical_encoders=cat_columns)

    # Tier 0: base (pretrained-only) model on Lanark val, for reference --
    # shows how much fine-tuning actually improved things over Condition B's model.
    evaluate(base_model, X_val, y_val, "Reference - pretrained-only model on Lanark holdout (= Condition B)")

    # Tier 1: sanity check on the fine-tuning data itself
    evaluate(finetuned_model, X_finetune, y_finetune,
              "Tier 1 - Lanark training rows (sanity check only, model has seen these)")

    # Tier 2: the real result
    results = evaluate(finetuned_model, X_val, y_val,
                        "Tier 2 - REAL Lanark holdout (Condition C, fine-tuned)")

    ci = bootstrap_ci(finetuned_model, X_val, y_val, n_boot=1000)
    print(f"\nBootstrap AUC-PR on Lanark holdout: "
          f"{ci['mean']:.3f} (95% CI: {ci['ci_lower']:.3f}-{ci['ci_upper']:.3f}, "
          f"{ci['n_valid_boots']}/1000 valid resamples)")

    return {"point_estimate": results, "bootstrap": ci,
            "base_model": base_model, "finetuned_model": finetuned_model,
            "lanark_val_idx": lanark_val.index}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Condition C: pretrain on synthetic, fine-tune on real Lanark")
    parser.add_argument("--toronto", required=True, help="Path to Toronto synthetic CSV")
    parser.add_argument("--ottawa", required=True, help="Path to Ottawa synthetic CSV")
    parser.add_argument("--lanark", required=True, help="Path to real Lanark CSV")
    parser.add_argument("--val_size", type=float, default=0.2, help="Lanark holdout fraction")
    parser.add_argument("--finetune_n_estimators", type=int, default=50,
                         help="Number of additional trees added during fine-tuning")
    parser.add_argument("--finetune_learning_rate", type=float, default=0.01,
                         help="Learning rate for the fine-tuning stage (kept low on purpose)")
    parser.add_argument("--finetune_reg_lambda", type=float, default=5.0,
                         help="L2 regularization strength during fine-tuning (higher = more conservative)")
    args = parser.parse_args()

    main(args.toronto, args.ottawa, args.lanark, args.val_size,
         args.finetune_n_estimators, args.finetune_learning_rate, args.finetune_reg_lambda)