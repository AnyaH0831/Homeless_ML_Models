"""
Paired bootstrap comparison across conditions.

Individual per-condition bootstrap CIs (what each condition_*.py script prints)
are MARGINAL CIs -- they don't account for the fact that all conditions were
evaluated on the exact same 114 Lanark holdout rows. Two conditions can have
heavily overlapping marginal CIs and STILL have a real, detectable difference,
because a paired comparison cancels out the shared row-level noise that both
conditions are equally subject to.

This script:
  1. Loads every predictions/{condition}_{model}.csv file (produced by adding
     the snippet above to each condition script).
  2. Verifies they all share the exact same set of held-out Lanark row indices.
  3. Runs a PAIRED bootstrap: draws the same resampled row indices for every
     condition on each iteration, then looks at the DIFFERENCE in AUC-PR (or
     AUC-ROC) between each pair of conditions on that same resample.
  4. Reports, for each pair, the mean difference and a 95% CI on that
     difference. If the CI excludes zero, that's a defensible claim that one
     condition outperforms the other; if it includes zero, say so honestly.
  5. Also runs a permutation test per pair as a second, independent check.

Usage:
    python compare_conditions.py --predictions_dir predictions --metric auc_pr
"""

import argparse
import glob
import os
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

RANDOM_STATE = 42
METRIC_FUNCS = {"auc_pr": average_precision_score, "auc_roc": roc_auc_score}


def load_all_predictions(predictions_dir):
    files = sorted(glob.glob(os.path.join(predictions_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No prediction CSVs found in {predictions_dir}. "
            f"Add the save-predictions snippet to each condition script first."
        )

    data = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]  # e.g. "A_xgboost"
        df = pd.read_csv(f).set_index("lanark_idx").sort_index()
        data[name] = df
        print(f"Loaded {name}: {len(df)} rows, {int(df['y_true'].sum())} positives")

    # --- Verify every condition was evaluated on the SAME held-out rows ---
    index_sets = {name: set(df.index) for name, df in data.items()}
    reference_name, reference_set = next(iter(index_sets.items()))
    mismatches = {name: idx_set for name, idx_set in index_sets.items() if idx_set != reference_set}
    if mismatches:
        raise ValueError(
            f"Not all conditions share the same held-out Lanark rows! "
            f"'{reference_name}' has {len(reference_set)} rows, but these differ: "
            f"{list(mismatches.keys())}. Check that every script used the same "
            f"random_state/val_size when splitting Lanark, and that lanark_val "
            f"wasn't filtered/reordered differently between scripts."
        )
    print(f"\nAll {len(data)} conditions confirmed to share the same "
          f"{len(reference_set)}-row Lanark holdout.\n")

    # --- Also verify y_true agrees across files (extra safety check) ---
    reference_y = data[reference_name]["y_true"]
    for name, df in data.items():
        if not (df["y_true"] == reference_y).all():
            raise ValueError(
                f"y_true mismatch between '{reference_name}' and '{name}' -- "
                f"these files may not correspond to the same holdout rows/order."
            )

    return data


def paired_bootstrap(data, metric_name="auc_pr", n_boot=2000, seed=RANDOM_STATE):
    metric_fn = METRIC_FUNCS[metric_name]
    rng = np.random.default_rng(seed)

    names = list(data.keys())
    n = len(next(iter(data.values())))
    y_true = next(iter(data.values()))["y_true"].to_numpy()  # same for all, verified above

    proba_by_name = {name: df["proba"].to_numpy() for name, df in data.items()}

    # Point estimates per condition (on the full holdout, no resampling)
    point_estimates = {name: metric_fn(y_true, proba) for name, proba in proba_by_name.items()}

    # Store per-resample scores for every condition, using the SAME resampled
    # indices each iteration -- this is what makes the comparison "paired"
    boot_scores = {name: [] for name in names}
    valid_iters = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_sample = y_true[idx]
        if len(np.unique(y_sample)) < 2:
            continue
        valid_iters += 1
        for name in names:
            boot_scores[name].append(metric_fn(y_sample, proba_by_name[name][idx]))

    for name in names:
        boot_scores[name] = np.array(boot_scores[name])

    print(f"Paired bootstrap: {valid_iters}/{n_boot} valid resamples "
          f"(resamples with only one class present are skipped)\n")

    # --- Marginal CIs (for reference / matches what each script already printed) ---
    print(f"=== Marginal {metric_name.upper()} (point estimate + 95% CI) per condition ===")
    marginal_summary = []
    for name in names:
        scores = boot_scores[name]
        marginal_summary.append({
            "condition": name,
            "point_estimate": point_estimates[name],
            "ci_lower": np.percentile(scores, 2.5),
            "ci_upper": np.percentile(scores, 97.5),
        })
    marginal_df = pd.DataFrame(marginal_summary).sort_values("point_estimate", ascending=False)
    print(marginal_df.to_string(index=False))

    # --- Paired differences for every pair of conditions ---
    print(f"\n=== Paired differences in {metric_name.upper()} (95% CI on the difference) ===")
    pair_results = []
    for name_a, name_b in itertools.combinations(names, 2):
        diffs = boot_scores[name_a] - boot_scores[name_b]
        ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])
        significant = (ci_lower > 0) or (ci_upper < 0)
        winner = name_a if ci_lower > 0 else (name_b if ci_upper < 0 else "neither (not distinguishable)")
        pair_results.append({
            "condition_a": name_a,
            "condition_b": name_b,
            "point_diff (a - b)": point_estimates[name_a] - point_estimates[name_b],
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "distinguishable_from_zero": significant,
            "better_condition": winner,
        })
    pairs_df = pd.DataFrame(pair_results)
    print(pairs_df.to_string(index=False))

    return marginal_df, pairs_df, boot_scores, point_estimates


def permutation_test_pair(data, name_a, name_b, metric_name="auc_pr", n_perm=2000, seed=RANDOM_STATE):
    """
    Second, independent check for one specific pair: under the null hypothesis
    that condition A and condition B are equally good, swapping which
    predictions "belong" to which condition (per row) shouldn't change the
    observed difference in performance by much. This tests significance from
    a different angle than the bootstrap CI above.
    """
    metric_fn = METRIC_FUNCS[metric_name]
    rng = np.random.default_rng(seed)

    y_true = data[name_a]["y_true"].to_numpy()
    proba_a = data[name_a]["proba"].to_numpy()
    proba_b = data[name_b]["proba"].to_numpy()

    observed_diff = metric_fn(y_true, proba_a) - metric_fn(y_true, proba_b)

    n = len(y_true)
    null_diffs = np.empty(n_perm)
    for i in range(n_perm):
        swap_mask = rng.random(n) < 0.5
        perm_a = np.where(swap_mask, proba_b, proba_a)
        perm_b = np.where(swap_mask, proba_a, proba_b)
        null_diffs[i] = metric_fn(y_true, perm_a) - metric_fn(y_true, perm_b)

    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))
    print(f"\nPermutation test: {name_a} vs {name_b} "
          f"(observed diff = {observed_diff:.4f}, p = {p_value:.4f})")
    return {"condition_a": name_a, "condition_b": name_b,
            "observed_diff": observed_diff, "p_value": p_value}


def main(predictions_dir, metric, n_boot, n_perm, key_pairs):
    data = load_all_predictions(predictions_dir)
    marginal_df, pairs_df, boot_scores, point_estimates = paired_bootstrap(
        data, metric_name=metric, n_boot=n_boot
    )

    print(f"\n=== Permutation tests for specified key pairs ===")
    perm_results = []
    for pair_str in key_pairs:
        name_a, name_b = pair_str.split(":")
        if name_a not in data or name_b not in data:
            print(f"Skipping '{pair_str}' -- one or both conditions not found in "
                  f"{list(data.keys())}")
            continue
        perm_results.append(permutation_test_pair(data, name_a, name_b, metric_name=metric, n_perm=n_perm))

    perm_df = pd.DataFrame(perm_results) if perm_results else pd.DataFrame()

    marginal_df.to_csv("comparison_marginal_summary.csv", index=False)
    pairs_df.to_csv("comparison_paired_differences.csv", index=False)
    if not perm_df.empty:
        perm_df.to_csv("comparison_permutation_tests.csv", index=False)
    print("\nSaved: comparison_marginal_summary.csv, comparison_paired_differences.csv"
          + (", comparison_permutation_tests.csv" if not perm_df.empty else ""))

    return marginal_df, pairs_df, perm_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paired bootstrap + permutation comparison across conditions")
    parser.add_argument("--predictions_dir", default="predictions",
                         help="Folder containing {condition}_{model}.csv prediction files")
    parser.add_argument("--metric", choices=["auc_pr", "auc_roc"], default="auc_pr")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--n_perm", type=int, default=2000)
    parser.add_argument("--key_pairs", nargs="*",
                         default=["C_logreg:D_bayes", "C_logreg:D2_bayes", "D_bayes:D2_bayes",
                                  "C_logreg:A_logreg"],
                         help="Pairs to run permutation tests on, format 'name1:name2' "
                              "matching your prediction CSV filenames (without .csv). "
                              "Adjust these to match whatever you actually named your files.")
    args = parser.parse_args()

    main(args.predictions_dir, args.metric, args.n_boot, args.n_perm, args.key_pairs)