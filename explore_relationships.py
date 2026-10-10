"""
explore_relationships.py

Answers one question BEFORE you try any more models: is there learnable
structure in this data at all, and where does it come from?

Parts
-----
1. Feature -> chronic_homeless association, per source (Cramer's V with
   bias correction, chi-square p-values with Benjamini-Hochberg adjustment,
   plus single-feature AUC for numeric/binary features).
2. Feature <-> feature association matrices per source, and how similar the
   dependency structure of each synthetic source is to Lanark's.
3. Learnability ceiling: cross-validated AUC *within* each source (logistic
   regression and gradient boosting) against a label-permutation null, plus a
   source-to-source transfer AUC matrix.
4. Unsupervised structure: PCA and KMeans, and how much of the chronic label
   (and of the data-source label) the clusters explain, against a shuffle null.

Reading the results
-------------------
- Toronto and Ottawa are synthetic rows built from published aggregates, so
  any association structure in them reflects the generator, not real people.
  Only Lanark's associations are real-world relationships (and Lanark is small).
- Unsupervised methods find structure among the features (X). They cannot tell
  you whether X predicts the label; Parts 1 and 3 do that.
- With thousands of rows, tiny effects are "significant". Judge by the effect
  sizes (Cramer's V, AUC), not the p-values alone.

Fields masked as NaN: any value whose '<feature>_available_in_source' flag is 0
(so imputed filler values are never mistaken for observations).

Usage:
    python explore_relationships.py \
        --toronto data/synthetic_toronto_sasm.csv \
        --ottawa data/synthetic_ottawa_sasm.csv \
        --lanark data/real_lanark.csv \
        --out_dir exploration_outputs
"""

import argparse
import itertools
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, entropy, mannwhitneyu, spearmanr
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score, roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

TARGET = "chronic_homeless"
NUMERIC = ["age"]
BINARY = ["has_dependents", "outdoor_sleeping", "youth", "no_income"]
CATEGORICAL = ["gender", "income_type"]
FEATURES = NUMERIC + BINARY + CATEGORICAL
FLAG_FOR = {
    "has_dependents": "has_dependents_available_in_source",
    "outdoor_sleeping": "outdoor_sleeping_available_in_source",
}
SOURCES = ["toronto_synth", "ottawa_synth", "real_lanark"]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_source(path, name):
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"{path} has no '{TARGET}' column.")
    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    for feat, flag in FLAG_FOR.items():
        if flag in df.columns:
            df.loc[df[flag] == 0, feat] = np.nan  # imputed filler, not observed
    for col in CATEGORICAL:
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    out = df[FEATURES + [TARGET]].copy()
    out[TARGET] = out[TARGET].astype(int)
    out["data_source"] = name
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Association helpers
# ----------------------------------------------------------------------------
def to_categorical_frame(df, cols):
    out = {}
    for col in cols:
        s = df[col]
        if col in NUMERIC and s.nunique() > 6:
            out[col] = pd.qcut(s, q=5, duplicates="drop")
        elif col == TARGET:
            out[col] = s.astype(object)
        else:
            out[col] = s.astype(object).where(s.notna(), np.nan)
    return pd.DataFrame(out, index=df.index)


def cramers_v(x, y, min_n=30):
    """Bias-corrected Cramer's V (Bergsma 2013). Returns (V, p_value, n)."""
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(tmp)
    if n < min_n:
        return np.nan, np.nan, n
    ct = pd.crosstab(tmp["x"], tmp["y"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan, np.nan, n
    chi2, p, _, _ = chi2_contingency(ct, correction=False)
    phi2 = chi2 / n
    r, k = ct.shape
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    if denom <= 0:
        return np.nan, p, n
    return float(np.sqrt(phi2c / denom)), p, n


def effect_label(v):
    if pd.isna(v):
        return "n/a"
    if v < 0.1:
        return "negligible"
    if v < 0.3:
        return "small"
    if v < 0.5:
        return "medium"
    return "large"


def bh_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty_like(pv)
    adj[order] = np.minimum(ranked, 1.0)
    out[ok] = adj
    return out


# ----------------------------------------------------------------------------
# Part 1: feature -> label association
# ----------------------------------------------------------------------------
def label_associations(df):
    cat = to_categorical_frame(df, FEATURES)
    y = df[TARGET]
    rows = []
    for f in FEATURES:
        if df[f].notna().sum() == 0:
            rows.append(dict(feature=f, n_used=0, cramers_v=np.nan, p_value=np.nan,
                             oriented_auc=np.nan))
            continue
        v, p, n = cramers_v(cat[f], y)
        auc = np.nan
        if f in NUMERIC + BINARY:
            m = df[f].notna()
            if y[m].nunique() == 2 and df.loc[m, f].nunique() > 1:
                a = roc_auc_score(y[m], df.loc[m, f].astype(float))
                auc = max(a, 1 - a)
        rows.append(dict(feature=f, n_used=n, cramers_v=v, p_value=p, oriented_auc=auc))
    out = pd.DataFrame(rows)
    out["p_adj_BH"] = bh_adjust(out["p_value"])
    out["effect_size"] = out["cramers_v"].map(effect_label)
    return out


def plot_label_heatmap(assoc_by_source, path):
    mat = pd.DataFrame({s: a.set_index("feature")["cramers_v"] for s, a in assoc_by_source.items()})
    fig, ax = plt.subplots(figsize=(1.6 * len(mat.columns) + 3, 0.5 * len(mat) + 2))
    im = ax.imshow(mat.values.astype(float), cmap="viridis", vmin=0, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=20)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i, j]
            ax.text(j, i, "n/a" if pd.isna(val) else f"{val:.2f}", ha="center", va="center",
                    color="black" if (pd.isna(val) or val >= 0.3) else "white", fontsize=9)
    ax.set_title(f"Cramer's V: feature vs {TARGET}\n(<0.1 negligible, 0.1-0.3 small, 0.3-0.5 medium)")
    fig.colorbar(im, ax=ax, label="Cramer's V")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Part 2: feature <-> feature structure
# ----------------------------------------------------------------------------
def association_matrix(df):
    nodes = FEATURES + [TARGET]
    cat = to_categorical_frame(df, nodes)
    arr = np.full((len(nodes), len(nodes)), np.nan)
    for (i, a), (j, b) in itertools.combinations(list(enumerate(nodes)), 2):
        v, _, _ = cramers_v(cat[a], cat[b])
        arr[i, j] = arr[j, i] = v
    np.fill_diagonal(arr, 1.0)  # plain numpy array, so no read-only issue under pandas 3
    return pd.DataFrame(arr, index=nodes, columns=nodes)


def plot_matrices(mats, path):
    fig, axes = plt.subplots(1, len(mats), figsize=(6.2 * len(mats), 5.8))
    axes = np.atleast_1d(axes)
    for ax, (name, mat) in zip(axes, mats.items()):
        im = ax.imshow(mat.values.astype(float), cmap="viridis", vmin=0, vmax=0.5)
        ax.set_xticks(range(len(mat)))
        ax.set_xticklabels(mat.columns, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(mat)))
        ax.set_yticklabels(mat.index, fontsize=8)
        ax.set_title(name)
        for i in range(len(mat)):
            for j in range(len(mat)):
                val = mat.values[i, j]
                if i != j and not pd.isna(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if val < 0.3 else "black")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.7, label="Cramer's V")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def structure_similarity(mats):
    rows = []
    for a, b in itertools.combinations(mats.keys(), 2):
        ma, mb = mats[a].values, mats[b].values
        iu = np.triu_indices_from(ma, k=1)
        va, vb = ma[iu], mb[iu]
        ok = ~np.isnan(va) & ~np.isnan(vb)
        if ok.sum() >= 5:
            rho, p = spearmanr(va[ok], vb[ok])
        else:
            rho, p = np.nan, np.nan
        rows.append(dict(source_a=a, source_b=b, n_pairs=int(ok.sum()),
                         spearman_rho=rho, p_value=p))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Part 3: learnability ceiling and transfer
# ----------------------------------------------------------------------------
def build_preprocessor(shared):
    num = [f for f in shared if f in NUMERIC + BINARY]
    cat = [f for f in shared if f in CATEGORICAL]
    parts = []
    if num:
        parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                       ("sc", StandardScaler())]), num))
    if cat:
        parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("oh", OneHotEncoder(handle_unknown="ignore",
                                                            sparse_output=False))]), cat))
    return ColumnTransformer(parts)


def build_pipeline(model_name, shared):
    if model_name == "logreg":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "hgb":
        model = HistGradientBoostingClassifier(max_iter=60, max_depth=3, learning_rate=0.05,
                                               l2_regularization=1.0, random_state=0)
    else:
        raise ValueError(model_name)
    return Pipeline([("pre", build_preprocessor(shared)), ("model", model)])


def prep_xy(df, shared):
    X = df[shared].copy()
    for c in shared:
        if c in CATEGORICAL:
            X[c] = X[c].astype(object)
    return X, df[TARGET].values


def within_source(df, shared, model_name, n_perm, seed, n_jobs):
    X, y = prep_xy(df, shared)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    score, perm, p = permutation_test_score(
        build_pipeline(model_name, shared), X, y, scoring="roc_auc", cv=cv,
        n_permutations=n_perm, n_jobs=n_jobs, random_state=seed)
    return score, float(np.mean(perm)), float(np.percentile(perm, 95)), p


def cross_source(train_df, test_df, shared, model_name):
    Xtr, ytr = prep_xy(train_df, shared)
    Xte, yte = prep_xy(test_df, shared)
    pipe = build_pipeline(model_name, shared).fit(Xtr, ytr)
    return roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])


# ----------------------------------------------------------------------------
# Part 4: unsupervised structure
# ----------------------------------------------------------------------------
def info_explained(labels, y, n_shuffle, rng):
    """Fraction of the label's entropy that the cluster assignment explains,
    with a shuffle null (MI is biased upward for many clusters, so compare
    the observed value to the null, not to zero)."""
    h = entropy(np.bincount(y) / len(y))
    if h == 0:
        return np.nan, np.nan, np.nan, np.nan
    obs = mutual_info_score(labels, y) / h
    null = np.array([mutual_info_score(labels, rng.permutation(y)) / h
                     for _ in range(n_shuffle)])
    p = (np.sum(null >= obs) + 1) / (n_shuffle + 1)
    return obs, null.mean(), np.percentile(null, 95), p


def cluster_scope(scope, df, shared, ks, n_shuffle, seed, with_source):
    X, y = prep_xy(df, shared)
    Z = build_preprocessor(shared).fit_transform(X)
    rng = np.random.default_rng(seed)
    src = pd.factorize(df["data_source"])[0] if with_source else None
    rows, best = [], None
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        lab = km.labels_
        sil = silhouette_score(Z, lab, sample_size=min(3000, len(Z)), random_state=seed)
        obs, nmean, n95, p = info_explained(lab, y, n_shuffle, rng)
        row = dict(scope=scope, k=k, silhouette=sil, chronic_info_explained=obs,
                   chronic_null_mean=nmean, chronic_null_p95=n95, chronic_p=p)
        if with_source:
            sobs, smean, s95, sp = info_explained(lab, src, n_shuffle, rng)
            row.update(source_info_explained=sobs, source_null_p95=s95, source_p=sp)
        rows.append(row)
        if best is None or sil > best[0]:
            best = (sil, k, lab)
    rates = pd.Series(y).groupby(best[2]).agg(["mean", "size"])
    rates.columns = ["chronic_rate", "n"]
    return pd.DataFrame(rows), best[1], rates, Z


def explained_variance(Z, n=5):
    pca = PCA(n_components=min(n, Z.shape[1])).fit(Z)
    return pca.explained_variance_ratio_


def plot_pca(df, shared, path, per_source=1500, seed=0):
    X, _ = prep_xy(df, shared)
    Z = build_preprocessor(shared).fit_transform(X)
    P = PCA(n_components=2).fit_transform(Z)
    plot_df = pd.DataFrame({"pc1": P[:, 0], "pc2": P[:, 1],
                            "source": df["data_source"].values, TARGET: df[TARGET].values})
    plot_df = pd.concat([g.sample(min(len(g), per_source), random_state=seed)
                         for _, g in plot_df.groupby("source")])
    rng = np.random.default_rng(seed)
    jit = lambda n: rng.normal(0, 0.04, n)  # discrete data: jitter so points are visible
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for s, g in plot_df.groupby("source"):
        axes[0].scatter(g.pc1 + jit(len(g)), g.pc2 + jit(len(g)), s=6, alpha=0.4, label=s)
    axes[0].set_title("PCA coloured by data source")
    axes[0].legend(markerscale=3)
    for lab, g in plot_df.groupby(TARGET):
        axes[1].scatter(g.pc1 + jit(len(g)), g.pc2 + jit(len(g)), s=6, alpha=0.4,
                        label=f"{TARGET}={lab}")
    axes[1].set_title(f"PCA coloured by {TARGET}")
    axes[1].legend(markerscale=3)
    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(toronto, ottawa, lanark, out_dir, n_perm, n_shuffle, seed, n_jobs):
    os.makedirs(out_dir, exist_ok=True)
    dfs = {"toronto_synth": load_source(toronto, "toronto_synth"),
           "ottawa_synth": load_source(ottawa, "ottawa_synth"),
           "real_lanark": load_source(lanark, "real_lanark")}

    for name, df in dfs.items():
        print(f"{name}: {len(df)} rows, chronic rate {df[TARGET].mean():.3f}")
    shared = [f for f in FEATURES if all(df[f].notna().any() for df in dfs.values())]
    dropped = [f for f in FEATURES if f not in shared]
    print(f"\nShared features used for the ML and clustering parts: {shared}")
    if dropped:
        print(f"Not available in every source (excluded there): {dropped}")
    pooled = pd.concat(dfs.values(), ignore_index=True)

    # ---- Part 1 ----
    print("\n" + "=" * 78)
    print("PART 1: association of each feature with chronic_homeless (per source)")
    print("=" * 78)
    assoc = {}
    for name, df in dfs.items():
        a = label_associations(df)
        assoc[name] = a
        a.to_csv(os.path.join(out_dir, f"label_association_{name}.csv"), index=False)
        print(f"\n[{name}]")
        print(a.sort_values("cramers_v", ascending=False).round(4).to_string(index=False))
    plot_label_heatmap(assoc, os.path.join(out_dir, "label_association_heatmap.png"))

    # ---- Part 2 ----
    print("\n" + "=" * 78)
    print("PART 2: feature <-> feature dependency structure")
    print("=" * 78)
    mats = {name: association_matrix(df) for name, df in dfs.items()}
    for name, m in mats.items():
        m.to_csv(os.path.join(out_dir, f"feature_association_{name}.csv"))
    plot_matrices(mats, os.path.join(out_dir, "feature_association_matrices.png"))
    sim = structure_similarity(mats)
    sim.to_csv(os.path.join(out_dir, "structure_similarity.csv"), index=False)
    print("\nSpearman correlation between sources' pairwise association strengths:")
    print(sim.round(4).to_string(index=False))
    print("(High rho: the sources share a dependency structure. Low or negative rho: "
          "the synthetic source's dependencies do not look like Lanark's.)")

    # ---- Part 3 ----
    print("\n" + "=" * 78)
    print(f"PART 3: learnability ceiling (5-fold CV AUC, {n_perm} label permutations)")
    print("=" * 78)
    rows = []
    for name, df in dfs.items():
        for model_name in ["logreg", "hgb"]:
            score, null_mean, null_p95, p = within_source(df, shared, model_name, n_perm, seed, n_jobs)
            rows.append(dict(source=name, model=model_name, cv_auc=score,
                             null_mean=null_mean, null_p95=null_p95, perm_p=p))
    ceiling = pd.DataFrame(rows)
    ceiling.to_csv(os.path.join(out_dir, "learnability_ceiling.csv"), index=False)
    print("\nWithin-source CV AUC vs permutation null:")
    print(ceiling.round(4).to_string(index=False))
    print("(cv_auc near null_mean means nothing is learnable from these features in that source.)")

    print("\nSource -> source transfer AUC (rows = trained on, columns = tested on;"
          " diagonal = within-source CV AUC):")
    for model_name in ["logreg", "hgb"]:
        mat = pd.DataFrame(index=SOURCES, columns=SOURCES, dtype=float)
        for a in SOURCES:
            for b in SOURCES:
                if a == b:
                    mat.loc[a, b] = float(ceiling[(ceiling.source == a) &
                                                  (ceiling.model == model_name)].cv_auc.iloc[0])
                else:
                    mat.loc[a, b] = cross_source(dfs[a], dfs[b], shared, model_name)
        mat.to_csv(os.path.join(out_dir, f"transfer_auc_{model_name}.csv"))
        print(f"\n[{model_name}]")
        print(mat.round(3).to_string())

    # ---- Part 4 ----
    print("\n" + "=" * 78)
    print("PART 4: unsupervised structure (PCA and KMeans)")
    print("=" * 78)
    ks = range(2, 9)
    all_rows = []
    scopes = [("pooled", pooled, True)] + [(n, d, False) for n, d in dfs.items()]
    for scope, df, with_source in scopes:
        res, best_k, rates, Z = cluster_scope(scope, df, shared, ks, n_shuffle, seed, with_source)
        all_rows.append(res)
        ev = explained_variance(Z)
        print(f"\n[{scope}] PCA explained variance, first components: {np.round(ev, 3)}")
        print(f"[{scope}] best silhouette at k={best_k}; chronic rate per cluster:")
        print(rates.round(3).to_string())
        row = res[res.k == best_k].iloc[0]
        print(f"[{scope}] at k={best_k}: clusters explain {row.chronic_info_explained:.4f} of the "
              f"label entropy (shuffle null mean {row.chronic_null_mean:.4f}, "
              f"95th pct {row.chronic_null_p95:.4f}, p={row.chronic_p:.3f})")
        if with_source:
            print(f"[{scope}] clusters explain {row.source_info_explained:.4f} of the data-source "
                  f"entropy (null 95th pct {row.source_null_p95:.4f})")
    pd.concat(all_rows).to_csv(os.path.join(out_dir, "unsupervised_clusters.csv"), index=False)
    plot_pca(pooled, shared, os.path.join(out_dir, "pca_scatter.png"), seed=seed)

    print("\n" + "=" * 78)
    print(f"Done. Tables and figures are in: {out_dir}/")
    print("=" * 78)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Check whether the data contains learnable relationships")
    ap.add_argument("--toronto", required=True)
    ap.add_argument("--ottawa", required=True)
    ap.add_argument("--lanark", required=True)
    ap.add_argument("--out_dir", default="exploration_outputs")
    ap.add_argument("--n_perm", type=int, default=30,
                    help="Label permutations for the learnability null (raise to 200+ for final numbers)")
    ap.add_argument("--n_shuffle", type=int, default=200,
                    help="Label shuffles for the clustering null")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="Parallel jobs for permutation tests (1 avoids multiprocessing issues)")
    args = ap.parse_args()
    main(args.toronto, args.ottawa, args.lanark, args.out_dir, args.n_perm, args.n_shuffle,
         args.seed, args.n_jobs)