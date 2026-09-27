"""Command-name wrapper for the Condition E recalibration implementation."""

import argparse

from recallibration import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Condition E: post-hoc recalibration of the zero-shot model"
    )
    parser.add_argument("--toronto", required=True)
    parser.add_argument("--ottawa", required=True)
    parser.add_argument("--lanark", required=True)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--method", choices=["platt", "isotonic", "both"], default="both")
    args = parser.parse_args()

    main(args.toronto, args.ottawa, args.lanark, args.val_size, args.method)
