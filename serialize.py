"""
serialize.py

Converts structured rows (Toronto synthetic, Ottawa synthetic, real Lanark)
into natural-language sentences, for use with a BERT-style text classifier
(TabLLM/LIFT-style serialization: Hegselmann et al. 2023).

Uses the SAME shared feature set as every other condition script in this
project (age, gender, has_dependents, outdoor_sleeping, youth, no_income,
income_type, plus the two availability flags), so results stay comparable.

Row order is preserved exactly as in the input CSVs -- this matters for
Lanark specifically, since condition_g_bert_finetune.py re-applies the same
train_test_split(random_state=42) used by every other condition, which only
produces the same split if the row order hasn't changed.

Output: one row per input row, with columns:
    text              -- the serialized sentence
    chronic_homeless  -- the label (kept separate, NOT included in the text)
    data_source       -- 'toronto_synth' / 'ottawa_synth' / 'real_lanark'

Usage:
    python serialize.py \
        --toronto data/synthetic_toronto_sasm.csv \
        --ottawa data/synthetic_ottawa_sasm.csv \
        --lanark data/real_lanark.csv \
        --out_dir data
"""

import argparse
import os
import pandas as pd

TARGET = "chronic_homeless"


def serialize_row(row):
    """Turn one row's fields into a natural-language sentence.

    Fields covered by an '_available_in_source' flag are phrased as
    "unknown" when that flag is 0, rather than asserting a fact the data
    doesn't actually support (e.g. Lanark's imputed fill value for a
    missing field should NOT be stated as if it were observed).
    """
    parts = []

    # --- Age ---
    age = row.get("age")
    if pd.notna(age):
        parts.append(f"{int(round(age))} years old")

    # --- Gender ---
    gender = row.get("gender")
    if pd.notna(gender) and str(gender).lower() not in ("nan", "unknown"):
        parts.append(f"{gender}")

    # --- Dependents (availability-flag aware) ---
    dep_available = row.get("has_dependents_available_in_source", 1)
    if dep_available == 0:
        parts.append("unknown whether they have dependents")
    else:
        parts.append("has dependents" if row.get("has_dependents") == 1 else "has no dependents")

    # --- Outdoor sleeping (availability-flag aware) ---
    outdoor_available = row.get("outdoor_sleeping_available_in_source", 1)
    if outdoor_available == 0:
        parts.append("unknown whether they have slept outdoors recently")
    else:
        parts.append("has slept outdoors recently" if row.get("outdoor_sleeping") == 1
                      else "has not slept outdoors recently")

    # --- Youth flag ---
    youth = row.get("youth")
    if pd.notna(youth):
        parts.append("is classified as a youth" if youth == 1 else "is not classified as a youth")

    # --- Income ---
    no_income = row.get("no_income")
    if pd.notna(no_income):
        parts.append("has no income" if no_income == 1 else "has some form of income")

    income_type = row.get("income_type")
    if pd.notna(income_type) and str(income_type).lower() not in ("nan", "unknown"):
        parts.append(f"income type is {income_type}")

    sentence = "This person is " + ", ".join(parts) + "."
    return sentence


def serialize_dataframe(df, source_name):
    df = df.copy()
    texts = df.apply(serialize_row, axis=1)
    out = pd.DataFrame({
        "text": texts,
        TARGET: df[TARGET].values,
        "data_source": source_name,
    })
    return out


def main(toronto_path, ottawa_path, lanark_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    toronto_df = pd.read_csv(toronto_path)
    ottawa_df = pd.read_csv(ottawa_path)
    lanark_df = pd.read_csv(lanark_path)

    toronto_out = serialize_dataframe(toronto_df, "toronto_synth")
    ottawa_out = serialize_dataframe(ottawa_df, "ottawa_synth")
    lanark_out = serialize_dataframe(lanark_df, "real_lanark")

    toronto_out_path = os.path.join(out_dir, "serialized_toronto_sasm.csv")
    ottawa_out_path = os.path.join(out_dir, "serialized_ottawa_sasm.csv")
    lanark_out_path = os.path.join(out_dir, "serialized_real_lanark.csv")

    toronto_out.to_csv(toronto_out_path, index=False)
    ottawa_out.to_csv(ottawa_out_path, index=False)
    lanark_out.to_csv(lanark_out_path, index=False)

    print(f"Wrote {len(toronto_out)} rows -> {toronto_out_path}")
    print(f"Wrote {len(ottawa_out)} rows -> {ottawa_out_path}")
    print(f"Wrote {len(lanark_out)} rows -> {lanark_out_path}")

    print("\nExample serialized sentences:")
    print("  Toronto:", toronto_out["text"].iloc[0])
    print("  Ottawa: ", ottawa_out["text"].iloc[0])
    print("  Lanark: ", lanark_out["text"].iloc[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serialize structured rows into text for BERT")
    parser.add_argument("--toronto", required=True)
    parser.add_argument("--ottawa", required=True)
    parser.add_argument("--lanark", required=True)
    parser.add_argument("--out_dir", default="data",
                         help="Folder to write serialized CSVs into (default: same 'data' folder as the originals)")
    args = parser.parse_args()

    main(args.toronto, args.ottawa, args.lanark, args.out_dir)