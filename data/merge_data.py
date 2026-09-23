"""
merge_datasets.py

Standardizes THREE individual-level sources onto one common schema and
writes them out as:

    - three per-source files, each named after its data_source value
      (real_lanark.csv, synthetic_toronto_sasm.csv,
      synthetic_ottawa_sasm.csv) -- identical column structure, so they
      can be vertically concatenated at any point with a one-line
      pd.concat()
    - one combined file (all three stacked together) for convenience,
      unless --no-combined is passed

Inputs:
    1. toronto_sasm_synthetic_individuals.csv   (SASM schema)
    2. ottawa_sasm_synthetic_individuals.csv    (SASM schema)
    3. lanark_bnl_cleaned.csv                   (output of clean_lanark_bnl.py,
                                                   already trimmed to the
                                                   common columns)

This does a "stack" (concatenation), not a row-level join: each source
represents different individuals, so the goal is a shared, standardized
schema across all three, not a merge keyed by ID.

--- Provenance columns (every row, every file) ---
    data_source: "real_lanark", "synthetic_toronto_sasm", or
                 "synthetic_ottawa_sasm"
    data_type:   "observed" for real_lanark, "synthetic" for the other two

Per TRIPOD-AI / STROBE-style transparent-reporting practice, these let
you (a) report provenance explicitly, and (b) run the diagnostic of
training a classifier to predict data_source from the features -- if
it's near-perfect, the populations are too different to pool naively.

--- Structural-missingness columns (Missing By Design) ---
For every common column that a given source never measures at all --
not "missing data" but "this survey never asked it" -- two things
happen:
    1. The column is added with NaN/NA for that source's rows (never
       silently dropped or imputed).
    2. A companion binary column "{column}_available_in_source" is
       added: 1 for rows from a source that genuinely measures that
       concept, 0 for rows from a source where it's structurally
       absent, only approximate, or a loose proxy.

This distinction matters for downstream modeling: MCAR/MAR imputation
(mean-fill, MICE) assumes the value COULD have been observed. Here it
structurally couldn't for some sources, so it must be flagged rather
than silently imputed.

The Toronto/Ottawa SASM files define every one of the common columns
directly, so they're fully available (1) across the board. The Lanark
availability calls, based on the earlier column-by-column mapping-
quality review:

    year              -> available (1): derived from last_contact_date's
                          year, a genuine (if approximate) proxy
    age               -> available (1): direct field
    years_homeless    -> NOT available (0): Lanark only has "months
                          homeless in the past year", not a true
                          lifetime years-homeless total
    gender            -> available (1): direct field
    has_dependents    -> NOT available (0): derived from head-of-household
                          / number-of-children rather than asked directly
    mental_health     -> NOT available (0): extracted from caseworker
                          notes as Yes/Unknown only -- no explicit "No",
                          so true negatives can't be distinguished from
                          "not mentioned"
    substance_use     -> NOT available (0): same caveat as mental_health
    outdoor_sleeping  -> NOT available (0): recoded from a free-text
                          "current sleeping arrangements" field, not a
                          direct outdoor-sleeping question
    chronic_homeless  -> available (1): direct checkbox field
    youth             -> available (1): direct checkbox field
    indigenous_flag   -> available (1): direct field
    no_income         -> available (1): direct (inverted) field
    income_type       -> available (1): direct field (category vocabulary
                          differs from SASM's, but the concept is
                          genuinely measured)

Adjust the AVAILABILITY dict below if you disagree with any of these
calls -- they're judgment calls about data quality, not hard facts.

Usage:
    python merge_datasets.py \
        --toronto toronto_sasm_synthetic_individuals.csv \
        --ottawa ottawa_sasm_synthetic_individuals.csv \
        --lanark lanark_bnl_cleaned.csv \
        --output-dir ./merged
"""

import argparse
import os
import pandas as pd

# The common schema, in the same order for every source/file.
COMMON_COLUMNS = [
    "year",
    "age",
    "years_homeless",
    "gender",
    "has_dependents",
    "mental_health",
    "substance_use",
    "outdoor_sleeping",
    "chronic_homeless",
    "youth",
    "indigenous_flag",
    "no_income",
    "income_type",
]

# Columns that should stay whole numbers (0/1 flags, counts) rather than
# floats, using pandas' nullable Int64 so missing values don't force the
# whole column to render as e.g. "1.0" instead of "1".
INT_COLUMNS = {
    "year",
    "has_dependents",
    "mental_health",
    "substance_use",
    "outdoor_sleeping",
    "chronic_homeless",
    "youth",
    "indigenous_flag",
    "no_income",
}

DATA_SOURCES = ("real_lanark", "synthetic_toronto_sasm", "synthetic_ottawa_sasm")

DATA_TYPE_BY_SOURCE = {
    "real_lanark": "observed",
    "synthetic_toronto_sasm": "synthetic",
    "synthetic_ottawa_sasm": "synthetic",
}

# {column: {data_source: 0 or 1}}. The two SASM sources define every
# common column directly, so they're fully available (1) everywhere.
# Only real_lanark's per-column calls vary -- see the module docstring
# for the reasoning behind each one.
AVAILABILITY = {
    col: {src: 1 for src in DATA_SOURCES}
    for col in COMMON_COLUMNS
}
for _col in ("years_homeless", "has_dependents", "mental_health",
             "substance_use", "outdoor_sleeping"):
    AVAILABILITY[_col]["real_lanark"] = 0

# Only columns where availability actually differs across sources get a
# companion "_available_in_source" column -- nothing to flag for a
# column that's fully available everywhere.
FLAGGED_COLUMNS = [
    col for col in COMMON_COLUMNS
    if len(set(AVAILABILITY[col].values())) > 1
]

# Final column order used for every output file (per-source AND combined).
ORDERED_COLUMNS = (
    COMMON_COLUMNS
    + ["data_source", "data_type"]
    + [f"{c}_available_in_source" for c in FLAGGED_COLUMNS]
)


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


def build_subset(df, data_source):
    """Return a dataframe in the exact common schema: every common
    column present (NaN/NA where structurally absent for this source),
    the two provenance columns, and one _available_in_source companion
    per flagged column. Nothing else from the source carries over."""
    out = pd.DataFrame(index=df.index)

    for col in COMMON_COLUMNS:
        if col in df.columns:
            out[col] = df[col]
        else:
            # Structurally absent (Missing By Design), not just NaN data.
            out[col] = pd.NA
        if col in INT_COLUMNS:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    out["gender"] = normalize_gender(out["gender"])

    out["data_source"] = data_source
    out["data_type"] = DATA_TYPE_BY_SOURCE[data_source]

    for col in FLAGGED_COLUMNS:
        out[f"{col}_available_in_source"] = pd.array(
            [AVAILABILITY[col][data_source]] * len(df), dtype="Int64"
        )

    return out[ORDERED_COLUMNS]


def load_and_subset(path, data_source):
    print(f"[info] Loading {data_source}: {path}")
    df = pd.read_csv(path)
    print(f"[info]   {len(df)} rows, {df.shape[1]} columns.")
    return build_subset(df, data_source)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
        "--output-dir",
        default=".",
        help="Directory to write the per-source and combined CSVs into.",
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Skip writing the combined (all-three-stacked) CSV; write "
             "only the three per-source files.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    subsets = {
        "real_lanark": load_and_subset(args.lanark, "real_lanark"),
        "synthetic_toronto_sasm": load_and_subset(args.toronto, "synthetic_toronto_sasm"),
        "synthetic_ottawa_sasm": load_and_subset(args.ottawa, "synthetic_ottawa_sasm"),
    }

    # Three per-source files, each named after its data_source value, all
    # in the identical column structure so they're trivially concatenable.
    for data_source, subset_df in subsets.items():
        out_path = os.path.join(args.output_dir, f"{data_source}.csv")
        subset_df.to_csv(out_path, index=False)
        print(f"[info] Wrote {len(subset_df)} rows to: {out_path}")

    if not args.no_combined:
        combined = pd.concat(
            [subsets[src] for src in DATA_SOURCES],
            ignore_index=True,
            sort=False,
        )[ORDERED_COLUMNS]
        combined_path = os.path.join(args.output_dir, "merged_common_columns.csv")
        combined.to_csv(combined_path, index=False)
        print(f"[info] Wrote combined dataset with {len(combined)} rows to: {combined_path}")


if __name__ == "__main__":
    main()