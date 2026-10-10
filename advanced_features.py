"""Leakage-safe, domain-invariant feature construction for tabular models."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


INCOME_TYPE_RISK = {
    "employment": 0.0,
    "informal": 0.5,
    "other": 1.0,
    "welfare": 1.5,
    "disability": 2.0,
}


def fit_region_age_stats(df: pd.DataFrame, source_col: str = "data_source") -> dict[str, tuple[float, float]]:
    """Fit per-source age means and standard deviations on training rows only."""
    stats: dict[str, tuple[float, float]] = {}
    groups = df.groupby(source_col, dropna=False) if source_col in df else [("global", df)]
    for source, group in groups:
        age = pd.to_numeric(group["age"], errors="coerce")
        mean = float(age.mean()) if age.notna().any() else 0.0
        std = float(age.std(ddof=0)) if age.notna().sum() > 1 else 1.0
        stats[str(source)] = (mean, max(std, 1e-8))
    return stats


def build_domain_invariant_features(
    df: pd.DataFrame,
    age_stats: Mapping[str, tuple[float, float]] | None = None,
    source_col: str = "data_source",
) -> pd.DataFrame:
    """Create stable aggregate features without raw age or income_type columns.

    ``systemic_friction_score`` combines income insecurity, youth, and income
    type risk. ``systemic_friction_tier`` is an ordinal 0-4 representation.
    Age is represented as a within-source z-score, with statistics supplied by
    the training split to prevent validation leakage.
    """
    out = pd.DataFrame(index=df.index)
    if age_stats is None:
        age_stats = fit_region_age_stats(df, source_col=source_col)

    if source_col in df:
        source_values = df[source_col].astype(str)
    else:
        source_values = pd.Series("global", index=df.index)

    age = pd.to_numeric(df["age"], errors="coerce")
    age_z = []
    for index, value in age.items():
        mean, std = age_stats.get(str(source_values.loc[index]), age_stats.get("global", (0.0, 1.0)))
        age_z.append((value - mean) / std if pd.notna(value) else 0.0)
    out["age_within_cohort_z"] = np.asarray(age_z, dtype=float)

    no_income = pd.to_numeric(df["no_income"], errors="coerce").fillna(0).clip(0, 1)
    youth = pd.to_numeric(df["youth"], errors="coerce").fillna(0).clip(0, 1)
    income_risk = df["income_type"].astype("string").str.lower().map(INCOME_TYPE_RISK).fillna(1.0)

    out["systemic_friction_score"] = income_risk + no_income * 1.5 + youth * 1.0
    out["systemic_friction_tier"] = pd.cut(
        out["systemic_friction_score"],
        bins=[-np.inf, 0.75, 1.75, 2.75, np.inf],
        labels=[0, 1, 2, 3],
    ).astype(int)

    for column in ("has_dependents", "outdoor_sleeping"):
        out[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(float)
    return out
