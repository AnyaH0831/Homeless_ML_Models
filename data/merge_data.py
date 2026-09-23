"""
merge_datasets.py

Merges THREE individual-level sources into a single dataset containing
only the columns/concepts they have in common:

    1. toronto_sasm_synthetic_individuals.csv   (SASM schema)
    2. ottawa_sasm_synthetic_individuals.csv    (SASM schema)
    3. lanark_bnl_cleaned.csv                   (output of clean_lanark_bnl.py)

This does a "stack" merge (concatenation), not a row-level join: each
source represents different individuals, so the goal is a combined
individual-level table with a shared, standardized schema plus a
`source_dataset` column (and a `city` column) to track provenance.

The two SASM-schema files share identical columns with each other, so
they're merged the same way; the Lanark file uses a different crosswalk
since its raw columns/concepts don't line up 1:1 (see clean_lanark_bnl.py
and the earlier mapping discussion for details on partial/no matches).

Crosswalk used (standardized column -> SASM column -> Lanark column):
    year                  -> year                 -> (no single "year" field;
                                                        blank unless --lanark-year-col)
    age                   -> age                   -> age
    years_homeless        -> years_homeless        -> years_homeless
    gender                -> gender                -> gender
    has_dependents        -> has_dependents        -> has_dependents
    mental_health         -> mental_health         -> mental_health
    substance_use         -> substance_use         -> substance_use
    outdoor_sleeping      -> outdoor_sleeping       -> outdoor_sleeping
    chronic_homeless      -> chronic_homeless       -> chronic_homeless
    youth                 -> youth                  -> youth
    indigenous_flag       -> indigenous_flag        -> indigenous_flag
    no_income             -> no_income               -> no_income
    income_type           -> income_type             -> income_type

Columns present in only one schema (e.g. race, education, lgbtq,
foster_care_history, housing_loss_income/health from the SASM files, or
client_id, veteran_status, sleeping_arrangement_raw from the Lanark data)
are intentionally left out of this merged file.

Usage:
    python merge_datasets.py \
        --toronto toronto_sasm_synthetic_individuals.csv \
        --ottawa ottawa_sasm_synthetic_individuals.csv \
        --lanark lanark_bnl_cleaned.csv \
        --output merged_common_columns.csv
"""

import argparse
import pandas as pd

# Standardized common schema for the two SASM-schema sources (identical
# column names on both sides, so a single map covers Toronto and Ottawa).
SASM_COMMON_COLUMNS = {
    "year": "year",
    "age": "age",
    "years_homeless": "years_homeless",
    "gender": "gender",
    "has_dependents": "has_dependents",
    "mental_health": "mental_health",
    "substance_use": "substance_use",
    "outdoor_sleeping": "outdoor_sleeping",
    "chronic_homeless": "chronic_homeless",
    "youth": "youth",
    "indigenous_flag": "indigenous_flag",
    "no_income": "no_income",
    "income_type": "income_type",
}

# Standardized common schema for the Lanark (BNL) source -> its cleaned
# column names, as produced by clean_lanark_bnl.py. "year" has no direct
# source column by default (see --lanark-year-col).
LANARK_COMMON_COLUMNS = {
    "year": None,
    "age": "age",
    "years_homeless": "years_homeless",
    "gender": "gender",
    "has_dependents": "has_dependents",
    "mental_health": "mental_health",
    "substance_use": "substance_use",
    "outdoor_sleeping": "outdoor_sleeping",
    "chronic_homeless": "chronic_homeless",
    "youth": "youth",
    "indigenous_flag": "indigenous_flag",
    "no_income": "no_income",
    "income_type": "income_type",
}


def normalize_gender(series):
    """Collapse free-text gender values from all three sources into a
    small shared set of categories so the merged column is comparable."""
    mapping = {
        "m": "male", "male": "male",
        "f": "female", "female": "female",
        "man": "male", "woman": "female",
    }

    def _map(v):
        if pd.isna(v):
            return pd.NA
        s = str(v).strip().lower()
        return mapping.get(s, s if s else pd.NA)

    return series.apply(_map)


def build_subset(df, colmap, source_label, city_label):
    """Given a source dataframe and a {output_col: source_col} mapping,
    return a dataframe with only the output columns (missing source
    columns become all-NA columns) plus source_dataset/city tags."""
    out = pd.DataFrame(index=df.index)
    for out_col, src_col in colmap.items():
        if src_col is not None and src_col in df.columns:
            out[out_col] = df[src_col]
        else:
            out[out_col] = pd.NA
    out["source_dataset"] = source_label
    out["city"] = city_label
    return out


def load_and_subset(path, colmap, source_label, city_label):
    print(f"[info] Loading {source_label}: {path}")
    df = pd.read_csv(path)
    print(f"[info]   {len(df)} rows, {df.shape[1]} columns.")
    return build_subset(df, colmap, source_label, city_label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--toronto",
        default="toronto_sasm_synthetic_individuals.csv",
        help="Path to the Toronto SASM-schema CSV.",
    )
    parser.add_argument(
        "--ottawa",
        default="ottawa_sasm_synthetic_individuals.csv",
        help="Path to the Ottawa SASM-schema CSV.",
    )
    parser.add_argument(
        "--lanark",
        default="lanark_bnl_cleaned.csv",
        help="Path to the cleaned Lanark BNL CSV (output of clean_lanark_bnl.py).",
    )
    parser.add_argument(
        "--output",
        default="merged_common_columns.csv",
        help="Path to write the merged CSV to.",
    )
    parser.add_argument(
        "--lanark-year-col",
        default=None,
        help=(
            "Optional column name in the cleaned Lanark CSV to use as "
            "'year' (e.g. derive one from added_to_bnl_date before "
            "running this script). Left blank/NA if not supplied."
        ),
    )
    args = parser.parse_args()

    lanark_colmap = dict(LANARK_COMMON_COLUMNS)
    if args.lanark_year_col:
        lanark_colmap["year"] = args.lanark_year_col

    toronto_subset = load_and_subset(
        args.toronto, SASM_COMMON_COLUMNS, "sasm_synthetic", "toronto"
    )
    ottawa_subset = load_and_subset(
        args.ottawa, SASM_COMMON_COLUMNS, "sasm_synthetic", "ottawa"
    )
    lanark_subset = load_and_subset(
        args.lanark, lanark_colmap, "lanark_bnl", "lanark"
    )

    merged = pd.concat(
        [toronto_subset, ottawa_subset, lanark_subset],
        ignore_index=True,
        sort=False,
    )

    # Standardize gender categories across all three sources so the
    # merged column is actually usable for cross-source comparison.
    merged["gender"] = normalize_gender(merged["gender"])

    merged.to_csv(args.output, index=False)
    print(
        f"[info] Wrote merged dataset with {len(merged)} rows "
        f"({len(toronto_subset)} Toronto + {len(ottawa_subset)} Ottawa + "
        f"{len(lanark_subset)} Lanark) and {merged.shape[1]} columns "
        f"to: {args.output}"
    )


if __name__ == "__main__":
    main()