"""
Condition D2: Random-slope partial pooling

Condition D (condition_d_partial_pooling.py) only lets the INTERCEPT vary by
source: (1 | data_source). That means every source shares the exact same
relationship between features and chronic_homeless -- only the baseline rate
shifts per source.

This script tests whether that's too restrictive, given fine-tuning
(Condition C) -- which lets the WHOLE model adapt -- clearly outperformed it.
Here, one or more SLOPES are also allowed to vary by source:

    chronic_homeless ~ [fixed effects] + (1 + no_income | data_source)

This lets e.g. the effect of "no_income" on chronic_homeless differ between
Lanark, Toronto, and Ottawa, not just the overall baseline rate. If this
closes the gap with Condition C's fine-tuning result, it confirms that
letting the FEATURE-OUTCOME relationship (not just the intercept) adapt per
source is what actually matters for this domain-shift problem.

Usage:
    python partial_pooling_random_slopes.py \
        --toronto data/synthetic_toronto_sasm.csv \
        --ottawa data/synthetic_ottawa_sasm.csv \
        --lanark data/real_lanark.csv \
        --random_slope_vars no_income has_dependents
"""

import argparse
import numpy as np
import pandas as pd
import bambi as bmb
import arviz as az
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "age",
    "has_dependents",
    "outdoor_sleeping",
    "youth",
    "no_income",
    "has_dependents_available_in_source",
    "outdoor_sleeping_available_in_source",
]
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
            fill_val = (
                fill_values[col]
                if fill_values and col in fill_values
                else df[col].median()
            )
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


def build_formula(random_slope_vars=None, subgroup_col=None):
    fixed_effects = " + ".join(NUMERIC_FEATURES + CATEGORICAL_FEATURES)

    if random_slope_vars:
        slope_terms = " + ".join(random_slope_vars)
        group_term = f"(1 + {slope_terms} | {GROUP_COL})"
    else:
        group_term = f"(1 | {GROUP_COL})"

    formula = f"{TARGET} ~ {fixed_effects} + {group_term}"
    if subgroup_col is not None:
        formula += f" + (1 | {subgroup_col})"
    return formula


def evaluate(y_true, proba, label):
    y_true = np.asarray(y_true)
    auc_roc = (
        roc_auc_score(y_true, proba)
        if len(np.unique(y_true)) > 1
        else float("nan")
    )
    auc_pr = (
        average_precision_score(y_true, proba)
        if len(np.unique(y_true)) > 1
        else float("nan")
    )
    brier = brier_score_loss(y_true, proba)

    print(
        f"\n=== Evaluation: {label} (n={len(y_true)}, positives={int(y_true.sum())}) ==="
    )
    print(f"AUC-ROC:      {auc_roc:.3f}")
    print(f"AUC-PR:       {auc_pr:.3f}")
    print(f"Brier score:  {brier:.3f}")

    return {
        "label": label,
        "n": len(y_true),
        "positives": int(y_true.sum()),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "brier": brier,
    }


def bootstrap_ci(
    y_true,
    proba,
    n_boot=1000,
    metric_fn=average_precision_score,
    seed=RANDOM_STATE,
):
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
    return {
        "mean": scores.mean(),
        "ci_lower": np.percentile(scores, 2.5),
        "ci_upper": np.percentile(scores, 97.5),
        "n_valid_boots": len(scores),
    }


def main(
    toronto_path,
    ottawa_path,
    lanark_path,
    val_size=0.2,
    random_slope_vars=None,
    subgroup_col=None,
    draws=1000,
    tune=1000,
    chains=4,
    cores=1,
    target_accept=0.99,
):

    # --- Load and tag each source ---
    toronto_df = load_and_tag(toronto_path, "toronto_synth", "synthetic")
    ottawa_df = load_and_tag(ottawa_path, "ottawa_synth", "synthetic")
    lanark_df = load_and_tag(lanark_path, "real_lanark", "observed")

    # --- Split Lanark into train/val ONCE, stratified, fixed seed ---
    lanark_train, lanark_val = train_test_split(
        lanark_df,
        test_size=val_size,
        stratify=lanark_df[TARGET],
        random_state=RANDOM_STATE,
    )

    # --- Pooled training data: all three sources together ---
    train_pool = pd.concat(
        [lanark_train, toronto_df, ottawa_df], ignore_index=True
    )

    # --- Impute numeric + categorical missing values using pooled TRAINING data only ---
    train_pool, num_fill_values = impute_missing(train_pool, NUMERIC_FEATURES)
    train_pool, cat_fill_values = impute_categorical(
        train_pool, CATEGORICAL_FEATURES
    )

    lanark_val_imputed, _ = impute_missing(
        lanark_val, NUMERIC_FEATURES, fill_values=num_fill_values
    )
    lanark_val_imputed, _ = impute_categorical(
        lanark_val_imputed, CATEGORICAL_FEATURES, fill_values=cat_fill_values
    )

    remaining_nans = train_pool[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET, GROUP_COL]
    ].isna().sum()
    if remaining_nans.sum() > 0:
        print(
            f"\nWARNING: still {remaining_nans.sum()} NaNs remaining after imputation:\n"
            f"{remaining_nans[remaining_nans > 0]}"
        )

    for col in CATEGORICAL_FEATURES + [GROUP_COL]:
        train_pool[col] = train_pool[col].astype("category")
    if subgroup_col is not None:
        train_pool[subgroup_col] = train_pool[subgroup_col].astype("category")

    print(
        f"Pooled training data: {len(train_pool)} rows "
        f"({train_pool[GROUP_COL].value_counts().to_dict()})"
    )
    print(
        f"Lanark validation (real, held out): {len(lanark_val_imputed)} rows, "
        f"{lanark_val_imputed[TARGET].sum()} positives"
    )

    # --- Fit the random-slope hierarchical model ---
    formula = build_formula(random_slope_vars, subgroup_col)
    print(f"\nModel formula:\n  {formula}\n")
    if random_slope_vars:
        print(
            f"Random slopes on: {random_slope_vars} (allowed to vary by {GROUP_COL})"
        )
    else:
        print(
            "No random slopes specified -- this is equivalent to Condition D (random intercept only)."
        )

    model = bmb.Model(formula, data=train_pool, family="bernoulli")
    idata = model.fit(
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        random_seed=RANDOM_STATE,
        target_accept=target_accept,
    )

    # --- Convergence check ---
    var_list = [v for v in idata.posterior.data_vars if not v.endswith("__")]
    summary = az.summary(idata, var_names=var_list)
    max_rhat = summary["r_hat"].max()
    min_ess = summary["ess_bulk"].min()
    n_divergences = int(idata.sample_stats["diverging"].sum())
    print(
        f"\nConvergence check: max R-hat = {max_rhat:.3f} (want < 1.01), "
        f"min ESS = {min_ess:.0f} (want > ~400), divergences = {n_divergences}"
    )
    if max_rhat > 1.01 or n_divergences > 0:
        print(
            "WARNING: Divergences detected. Consider raising target_accept further (e.g. 0.99)."
        )

    # --- Report both intercept AND slope random effects ---
    re_var_names = [f"1|{GROUP_COL}"] + [
        f"{v}|{GROUP_COL}" for v in (random_slope_vars or [])
    ]
    for var in re_var_names:
        try:
            re_summary = az.summary(idata, var_names=[var], filter_vars="like")
            print(f"\nRandom effect summary for '{var}':\n{re_summary}")
        except Exception as e:
            print(f"\n(Could not summarize '{var}': {e})")

    # --- Predict on the REAL Lanark holdout ---
    lanark_val_typed = lanark_val_imputed.copy()
    for col in CATEGORICAL_FEATURES + [GROUP_COL]:
        lanark_val_typed[col] = pd.Categorical(
            lanark_val_typed[col], categories=train_pool[col].cat.categories
        )
    if subgroup_col is not None:
        lanark_val_typed[subgroup_col] = pd.Categorical(
            lanark_val_typed[subgroup_col],
            categories=train_pool[subgroup_col].cat.categories,
        )

    # Predict expected probabilities
    pred_idata = model.predict(
        idata, data=lanark_val_typed, inplace=False, kind="response_params"
    )

    # Safely retrieve mean predicted probabilities across posterior keys
    posterior_vars = list(pred_idata.posterior.data_vars)
    if f"{TARGET}_mean" in posterior_vars:
        proba = pred_idata.posterior[f"{TARGET}_mean"].mean(dim=["chain", "draw"]).values
    elif TARGET in posterior_vars:
        proba = pred_idata.posterior[TARGET].mean(dim=["chain", "draw"]).values
    elif f"{TARGET}_obs" in posterior_vars:
        proba = pred_idata.posterior[f"{TARGET}_obs"].mean(dim=["chain", "draw"]).values
    else:
        main_var = posterior_vars[0]
        proba = pred_idata.posterior[main_var].mean(dim=["chain", "draw"]).values

    y_val = lanark_val_imputed[TARGET].to_numpy()

    results = evaluate(
        y_val, proba, "Tier 2 - REAL Lanark holdout (Condition D2, random slopes)"
    )
    ci = bootstrap_ci(y_val, proba, n_boot=1000)
    print(
        f"\nBootstrap AUC-PR on Lanark holdout: "
        f"{ci['mean']:.3f} (95% CI: {ci['ci_lower']:.3f}-{ci['ci_upper']:.3f}, "
        f"{ci['n_valid_boots']}/1000 valid resamples)"
    )

    print(
        "\nCompare this against Condition D (random intercept only) and Condition C "
        "(fine-tuned) to see whether allowing slopes to vary by source closes the gap."
    )

    return {
        "point_estimate": results,
        "bootstrap": ci,
        "model": model,
        "idata": idata,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Condition D2: random-slope partial pooling"
    )
    parser.add_argument(
        "--toronto", required=True, help="Path to Toronto synthetic CSV"
    )
    parser.add_argument(
        "--ottawa", required=True, help="Path to Ottawa synthetic CSV"
    )
    parser.add_argument(
        "--lanark", required=True, help="Path to real Lanark CSV"
    )
    parser.add_argument(
        "--val_size", type=float, default=0.2, help="Lanark holdout fraction"
    )
    parser.add_argument(
        "--random_slope_vars",
        nargs="*",
        default=["no_income"],
        help="Feature(s) allowed to have a random slope by data_source.",
    )
    parser.add_argument(
        "--subgroup_col",
        default=None,
        help="Optional second grouping column (e.g. Family/Single/Youth)",
    )
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Default 1 to avoid multiprocessing issues.",
    )
    parser.add_argument(
        "--target_accept",
        type=float,
        default=0.99,
        help="Set to 0.99 to eliminate sampling divergences.",
    )
    args = parser.parse_args()

    main(
        args.toronto,
        args.ottawa,
        args.lanark,
        args.val_size,
        args.random_slope_vars,
        args.subgroup_col,
        args.draws,
        args.tune,
        args.chains,
        args.cores,
        args.target_accept,
    )