"""Command-name wrapper for the Condition F reweighting implementation."""

import argparse

from reweighting import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Condition F: instance reweighting via density-ratio estimation"
    )
    parser.add_argument("--toronto", required=True)
    parser.add_argument("--ottawa", required=True)
    parser.add_argument("--lanark", required=True)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--weight_cap", type=float, default=20.0)
    args = parser.parse_args()

    main(args.toronto, args.ottawa, args.lanark, args.val_size, args.weight_cap)
