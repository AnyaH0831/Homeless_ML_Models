"""
Condition D: Partial pooling (hierarchical / mixed-effects logistic regression)

Model:  chronic_homeless ~ age + gender + has_dependents + outdoor_sleeping +
                           youth + indigenous_flag + no_income + income_type +
                           has_dependents_available_in_source +
                           outdoor_sleeping_available_in_source +
                           (1 | data_source)
                           [+ (1 | subgroup) if a Family/Single/Youth-type column exists]

Lanark, Toronto, and Ottawa are pooled into one training set. `data_source` is
fit as a random effect: each source gets its own intercept adjustment, but that
adjustment is shrunk toward the shared population estimate in proportion to how
much data that source has. Lanark (small, real) should therefore be pulled
toward the shared estimate; Toronto/Ottawa (large, synthetic) will be trusted
closer to their own values.

Requires: bambi (pip install bambi), which wraps PyMC for lme4-style formulas.

Usage:
    python condition_d_partial_pooling.py \
        --toronto data/synthetic_toronto_sasm.csv \
        --ottawa data/synthetic_ottawa_sasm.csv \
        --lanark data/real_lanark.csv \
        --subgroup_col household_type   # optional, omit if you don't have one
"""

import argparse
import numpy as np
import pandas as pd
import bambi as bmb
import arviz as az
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

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
    """Fill missing numeric values. Pass fill_values=None to compute them from
    this df (use on the pooled TRAINING data); pass a dict to apply precomputed
    values (use on the held-out validation data, to avoid leakage)."""
    df = df.copy()
    computed = {}
    for col in numeric_cols:
        if col in df.columns and df[col].isna().any():
            fill_val = fill_values[col] if fill_values and col in fill_values else df[col].median()
            df[col] = df[col].fillna(fill_val)
            computed[col] = fill_val
    return df, computed


def impute_categorical(df, categorical_cols, fill_values=None):
    """Fill missing categorical values with the mode (or a precomputed fill
    value, to avoid leaking target-domain stats into validation data)."""
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


def build_formula(subgroup_col=None):
    fixed_effects = " + ".join(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    formula = f"{TARGET} ~ {fixed_effects} + (1 | {GROUP_COL})"
    if subgroup_col is not None:
        formula += f" + (1 | {subgroup_col})"
    return formula


def evaluate(y_true, proba, label):
    y_true = np.asarray(y_true)
    auc_roc = roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else float("nan")
    auc_pr = average_precision_score(y_true, proba) if len(np.unique(y_true)) > 1 else float("nan")
    brier = brier_score_loss(y_true, proba)

    print(f"\n=== Evaluation: {label} (n={len(y_true)}, positives={int(y_true.sum())}) ===")
    print(f"AUC-ROC:      {auc_roc:.3f}")
    print(f"AUC-PR:       {auc_pr:.3f}")
    print(f"Brier score:  {brier:.3f}")

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


def main(toronto_path, ottawa_path, lanark_path, val_size=0.2,
         subgroup_col=None, draws=1000, tune=1000, chains=4, cores=1):

    # --- Load and tag each source ---
    toronto_df = load_and_tag(toronto_path, "toronto_synth", "synthetic")
    ottawa_df = load_and_tag(ottawa_path, "ottawa_synth", "synthetic")
    lanark_df = load_and_tag(lanark_path, "real_lanark", "observed")

    # --- Split Lanark into train/val ONCE, stratified, fixed seed ---
    # Use the SAME random_state/val_size as Conditions A/B/C so the held-out
    # rows are identical across all four conditions.
    lanark_train, lanark_val = train_test_split(
        lanark_df, test_size=val_size, stratify=lanark_df[TARGET],
        random_state=RANDOM_STATE
    )

    # --- Condition D training pool: ALL THREE sources pooled together ---
    train_pool = pd.concat([lanark_train, toronto_df, ottawa_df], ignore_index=True)

    # --- Impute missing values using pooled TRAINING data only ---
    train_pool, fill_values = impute_missing(train_pool, NUMERIC_FEATURES)
    lanark_val_imputed, _ = impute_missing(lanark_val, NUMERIC_FEATURES, fill_values=fill_values)

    # --- Also impute missing CATEGORICAL values (gender, income_type) ---
    # This is the piece that was missing before and caused the "incomplete rows" error:
    # any NaN in a categorical fixed-effect column makes the whole row unusable to bambi.
    train_pool, cat_fill_values = impute_categorical(train_pool, CATEGORICAL_FEATURES)
    lanark_val_imputed, _ = impute_categorical(lanark_val_imputed, CATEGORICAL_FEATURES,
                                                fill_values=cat_fill_values)

    remaining_nans = train_pool[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET, GROUP_COL]].isna().sum()
    if remaining_nans.sum() > 0:
        print(f"\nWARNING: still {remaining_nans.sum()} NaNs remaining after imputation:\n"
              f"{remaining_nans[remaining_nans > 0]}\n"
              f"Check these columns exist and are named correctly in all three source files.")

    # Ensure categorical columns are typed as category (bambi/patsy needs this
    # to treat them as fixed-effect factors rather than raw strings)
    for col in CATEGORICAL_FEATURES + [GROUP_COL]:
        train_pool[col] = train_pool[col].astype("category")
    if subgroup_col is not None:
        train_pool[subgroup_col] = train_pool[subgroup_col].astype("category")

    print(f"Pooled training data: {len(train_pool)} rows "
          f"({train_pool[GROUP_COL].value_counts().to_dict()})")
    print(f"Lanark validation (real, held out): {len(lanark_val_imputed)} rows, "
          f"{lanark_val_imputed[TARGET].sum()} positives")

    # --- Fit the hierarchical (partial pooling) model ---
    formula = build_formula(subgroup_col)
    print(f"\nModel formula:\n  {formula}\n")

    model = bmb.Model(formula, data=train_pool, family="bernoulli")
    idata = model.fit(draws=draws, tune=tune, chains=chains, cores=cores,
                       random_seed=RANDOM_STATE, target_accept=0.98)

    # --- Check convergence before trusting anything downstream ---
    summary = az.summary(idata, var_names=["~_log__"], filter_vars="regex")
    max_rhat = summary["r_hat"].max()
    min_ess = summary["ess_bulk"].min()
    print(f"\nConvergence check: max R-hat = {max_rhat:.3f} (want < 1.01), "
          f"min ESS = {min_ess:.0f} (want > ~400)")
    if max_rhat > 1.01:
        print("WARNING: R-hat above 1.01 — chains may not have converged. "
              "Consider increasing `tune`/`draws` or reparameterizing before "
              "trusting these results.")

    # --- Report the group-level (data_source) random effect estimates ---
    # This directly shows how much each source's estimate was shrunk.
    group_effects = az.summary(idata, var_names=[f"1|{GROUP_COL}"], filter_vars="like")
    print(f"\nPer-source random intercept estimates (shrinkage toward pooled mean):\n{group_effects}")

    # --- Predict on the REAL Lanark holdout ---
    # lanark_val_imputed rows belong to the 'real_lanark' group already seen
    # during training, so the model conditions on that group's fitted effect.
    lanark_val_imputed_typed = lanark_val_imputed.copy()
    for col in CATEGORICAL_FEATURES + [GROUP_COL]:
        lanark_val_imputed_typed[col] = pd.Categorical(
            lanark_val_imputed_typed[col], categories=train_pool[col].cat.categories
        )
    if subgroup_col is not None:
        lanark_val_imputed_typed[subgroup_col] = pd.Categorical(
            lanark_val_imputed_typed[subgroup_col],
            categories=train_pool[subgroup_col].cat.categories
        )

    pred_idata = model.predict(
        idata, 
        data=lanark_val_imputed_typed, 
        inplace=False, 
        kind="response"
    )

    # Extract predicted probabilities for the positive outcome (1)
    proba = pred_idata.posterior_predictive["chronic_homeless"].mean(dim=["chain", "draw"]).values

    y_val = lanark_val_imputed[TARGET].to_numpy()

    results = evaluate(y_val, proba, "Tier 2 - REAL Lanark holdout (Condition D, partial pooling)")
    ci = bootstrap_ci(y_val, proba, n_boot=1000)
    print(f"\nBootstrap AUC-PR on Lanark holdout: "
          f"{ci['mean']:.3f} (95% CI: {ci['ci_lower']:.3f}-{ci['ci_upper']:.3f}, "
          f"{ci['n_valid_boots']}/1000 valid resamples)")

    return {"point_estimate": results, "bootstrap": ci, "model": model,
            "idata": idata, "group_effects": group_effects}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Condition D: hierarchical partial-pooling model")
    parser.add_argument("--toronto", required=True, help="Path to Toronto synthetic CSV")
    parser.add_argument("--ottawa", required=True, help="Path to Ottawa synthetic CSV")
    parser.add_argument("--lanark", required=True, help="Path to real Lanark CSV")
    parser.add_argument("--val_size", type=float, default=0.2, help="Lanark holdout fraction")
    parser.add_argument("--subgroup_col", default=None,
                         help="Optional column name for a Family/Single/Youth-type "
                              "subgroup, added as a second random effect (1 | subgroup)")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=1,
                         help="Number of parallel cores for MCMC sampling. "
                              "Default 1 (no multiprocessing) to avoid a known "
                              "macOS/PyMC EOFError with parallel chains. Once "
                              "this runs successfully, you can try --cores 4 "
                              "to speed things up.")
    args = parser.parse_args()

    main(args.toronto, args.ottawa, args.lanark, args.val_size,
         args.subgroup_col, args.draws, args.tune, args.chains, args.cores)