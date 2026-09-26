"""
Condition C (logistic regression variant): Pretrain-then-finetune transfer

XGBoost supports "continue training" naturally (xgb_model=<previous booster>).
Plain sklearn LogisticRegression does NOT support this the same way: calling
.fit() again just re-solves to the global optimum for whatever data you give
it, discarding the point of "starting from what was learned before."

To get a genuine fine-tuning behavior with logistic regression, this script
uses SGDClassifier(loss="log_loss") instead, which is mathematically the same
model (linear logistic regression) but trained by gradient descent, so it
supports .partial_fit() -- true incremental training that continues from the
current weights rather than re-solving from scratch. This is the standard way
to fine-tune a linear model:
    Stage 1 (pretrain): many epochs of SGD on the synthetic pool.
    Stage 2 (finetune):  a few additional epochs, at a low learning rate, on
                         Lanark's real training rows only -- nudging the
                         weights toward Lanark's distribution without fully
                         overwriting what was learned from the larger
                         synthetic pool.

Evaluated on the SAME held-out Lanark validation slice used in the other
conditions (random_state=42, same val_size).

Usage:
    python condition_c_logreg_finetune.py \
        --toronto data/synthetic_toronto_sasm.csv \
        --ottawa data/synthetic_ottawa_sasm.csv \
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


def evaluate(model, scaler, X, y, label):
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[:, 1]
    preds = model.predict(X_scaled)

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


def bootstrap_ci(model, scaler, X, y, n_boot=1000, metric_fn=average_precision_score, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y_arr = y.to_numpy()
    n = len(y)
    proba_full = model.predict_proba(scaler.transform(X))[:, 1]

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
         pretrain_epochs=50, pretrain_alpha=0.0001,
         finetune_epochs=5, finetune_eta0=0.001, finetune_alpha=0.01):

    # --- Load and tag each source ---
    toronto_df = load_and_tag(toronto_path, "toronto_synth", "synthetic")
    ottawa_df = load_and_tag(ottawa_path, "ottawa_synth", "synthetic")
    lanark_df = load_and_tag(lanark_path, "real_lanark", "observed")

    # --- Split Lanark into train/val ONCE, stratified, fixed seed ---
    lanark_train, lanark_val = train_test_split(
        lanark_df, test_size=val_size, stratify=lanark_df[TARGET],
        random_state=RANDOM_STATE
    )

    # =========================================================
    # STAGE 1: PRETRAIN on synthetic data only
    # =========================================================
    pretrain_pool = pd.concat([toronto_df, ottawa_df], ignore_index=True)
    pretrain_pool, num_fill_values = impute_missing(pretrain_pool, NUMERIC_FEATURES)
    pretrain_pool, cat_fill_values = impute_categorical(pretrain_pool, CATEGORICAL_FEATURES)

    X_pretrain, y_pretrain, cat_columns = build_xy(pretrain_pool)

    print(f"Stage 1 (pretrain) pool: {len(X_pretrain)} rows, "
          f"{y_pretrain.sum()} positives ({y_pretrain.mean():.1%})")

    # Logistic regression via SGD needs standardized inputs to train well.
    # Scaler is fit ONLY on the pretrain pool -- this is part of the
    # "learned representation" being transferred, so it stays fixed
    # afterward rather than re-fitting on Lanark.
    scaler = StandardScaler()
    X_pretrain_scaled = scaler.fit_transform(X_pretrain)

    base_model = SGDClassifier(
        loss="log_loss",              # log_loss = logistic regression
        penalty="l2",
        alpha=pretrain_alpha,          # L2 regularization strength
        learning_rate="optimal",
        max_iter=pretrain_epochs,      # number of passes (epochs) over the pretrain pool
        class_weight="balanced",       # handles class imbalance
        random_state=RANDOM_STATE,
    )
    base_model.fit(X_pretrain_scaled, y_pretrain)
    print(f"Stage 1 (pretrain) complete: {pretrain_epochs} epochs on synthetic pool.")

    # =========================================================
    # STAGE 2: FINE-TUNE on real Lanark training rows only
    # =========================================================
    # Impute Lanark's own missing values using Lanark-TRAIN-only stats,
    # same reasoning as the XGBoost fine-tune script: the point of fine-tuning
    # is adapting toward Lanark's real distribution.
    lanark_train_imp, lanark_num_fill = impute_missing(lanark_train, NUMERIC_FEATURES)
    lanark_train_imp, lanark_cat_fill = impute_categorical(lanark_train_imp, CATEGORICAL_FEATURES)

    X_finetune, y_finetune, _ = build_xy(lanark_train_imp, categorical_encoders=cat_columns)
    # Use the SAME scaler fit during pretraining -- do not refit on Lanark.
    X_finetune_scaled = scaler.transform(X_finetune)

    print(f"\nStage 2 (finetune) pool: {len(X_finetune)} rows, "
          f"{y_finetune.sum()} positives ({y_finetune.mean():.1%})")

    # Copy the pretrained model so the original (Condition B equivalent) stays
    # intact for the reference comparison below, then continue training it.
    import copy
    finetuned_model = copy.deepcopy(base_model)

    # Override the learning rate / regularization for the fine-tuning phase:
    # a low, constant learning rate and stronger regularization keep the
    # updates small and cautious, given how little real data there is.
    finetuned_model.set_params(learning_rate="constant", eta0=finetune_eta0,
                                alpha=finetune_alpha)

    y_finetune_arr = y_finetune.to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)
    for epoch in range(finetune_epochs):
        # shuffle each epoch, standard practice for SGD fine-tuning
        perm = rng.permutation(len(y_finetune_arr))
        finetuned_model.partial_fit(X_finetune_scaled[perm], y_finetune_arr[perm])
    print(f"Stage 2 (finetune) complete: {finetune_epochs} additional epochs on Lanark training rows.")

    # =========================================================
    # EVALUATION
    # =========================================================
    lanark_val_imp, _ = impute_missing(lanark_val, NUMERIC_FEATURES, fill_values=lanark_num_fill)
    lanark_val_imp, _ = impute_categorical(lanark_val_imp, CATEGORICAL_FEATURES, fill_values=lanark_cat_fill)
    X_val, y_val, _ = build_xy(lanark_val_imp, categorical_encoders=cat_columns)

    # Reference: pretrained-only model on Lanark val (~ Condition B with logreg)
    evaluate(base_model, scaler, X_val, y_val,
             "Reference - pretrained-only logreg on Lanark holdout (= Condition B, logreg)")

    # Tier 1: sanity check on the fine-tuning data itself
    evaluate(finetuned_model, scaler, X_finetune, y_finetune,
             "Tier 1 - Lanark training rows (sanity check only, model has seen these)")

    # Tier 2: the real result
    results = evaluate(finetuned_model, scaler, X_val, y_val,
                        "Tier 2 - REAL Lanark holdout (Condition C, logreg fine-tuned)")

    ci = bootstrap_ci(finetuned_model, scaler, X_val, y_val, n_boot=1000)
    print(f"\nBootstrap AUC-PR on Lanark holdout: "
          f"{ci['mean']:.3f} (95% CI: {ci['ci_lower']:.3f}-{ci['ci_upper']:.3f}, "
          f"{ci['n_valid_boots']}/1000 valid resamples)")

    return {"point_estimate": results, "bootstrap": ci,
            "base_model": base_model, "finetuned_model": finetuned_model,
            "scaler": scaler, "lanark_val_idx": lanark_val.index}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Condition C (logreg): pretrain via SGD on synthetic, fine-tune on real Lanark")
    parser.add_argument("--toronto", required=True, help="Path to Toronto synthetic CSV")
    parser.add_argument("--ottawa", required=True, help="Path to Ottawa synthetic CSV")
    parser.add_argument("--lanark", required=True, help="Path to real Lanark CSV")
    parser.add_argument("--val_size", type=float, default=0.2, help="Lanark holdout fraction")
    parser.add_argument("--pretrain_epochs", type=int, default=50)
    parser.add_argument("--pretrain_alpha", type=float, default=0.0001,
                         help="L2 regularization strength during pretraining")
    parser.add_argument("--finetune_epochs", type=int, default=5,
                         help="Number of additional SGD epochs on Lanark (kept small on purpose)")
    parser.add_argument("--finetune_eta0", type=float, default=0.001,
                         help="Learning rate for the fine-tuning stage (kept low on purpose)")
    parser.add_argument("--finetune_alpha", type=float, default=0.01,
                         help="L2 regularization strength during fine-tuning (higher = more conservative)")
    args = parser.parse_args()

    main(args.toronto, args.ottawa, args.lanark, args.val_size,
         args.pretrain_epochs, args.pretrain_alpha,
         args.finetune_epochs, args.finetune_eta0, args.finetune_alpha)