"""Command-name wrapper for the paired condition comparison."""

import argparse

from compare import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paired bootstrap and permutation comparison across conditions"
    )
    parser.add_argument("--predictions_dir", default="predictions")
    parser.add_argument("--metric", choices=["auc_pr", "auc_roc"], default="auc_pr")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--n_perm", type=int, default=2000)
    parser.add_argument("--key_pairs", nargs="*", default=[
        "C_logreg:D_bayes",
        "C_logreg:D2_bayes",
        "D_bayes:D2_bayes",
        "C_logreg:A_logreg",
    ])
    args = parser.parse_args()

    main(args.predictions_dir, args.metric, args.n_boot, args.n_perm, args.key_pairs)
