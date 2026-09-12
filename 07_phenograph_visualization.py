"""
07_phenograph_visualization.py
================================
STEP 7 of the pipeline. Run standalone with:
    python 07_phenograph_visualization.py
(Requires 04_optuna_weight_learning.py to have been run first -- this step
only needs the domain scores, not the trained MLP/SHAP outputs.)

What this does:
  1. Runs PhenoGraph (Levine et al. 2015 -- graph-based clustering: builds a
     k-nearest-neighbor graph over patients in the 4-domain-score space,
     then finds communities in that graph via Louvain modularity
     optimization) on every patient's [Adiposity, Metabolic, Behavioral,
     Socio-Environmental] score vector. This is a different algorithm from
     a generic k-means -- it finds however many natural clusters the data
     actually supports, rather than you having to pick a cluster count.
  2. Projects that same 4-dimensional score space down to 2 dimensions with
     UMAP, purely so it can be drawn on a page (the clustering itself
     happens in the real 4-D domain-score space, not on the 2-D picture).
  3. Draws the "trend across all CASEIDs" you asked for: one main plot
     colored by which PhenoGraph cluster each patient fell into, plus four
     side-by-side panels -- one per phenotype domain -- colored by that
     domain's own score, so you can see directly that higher score in a
     region of the plot lines up with the patients PhenoGraph grouped
     together there (higher score = higher risk/impact, exactly as you
     described).

Output:
  outputs/phenograph_clusters.xlsx      (CASEID, 4 domain scores, total
                                          score, and PhenoGraph cluster ID)
  outputs/phenograph_cluster_summary.csv (per-cluster size and mean scores)
  outputs/phenograph_umap_clusters.png   (main cluster plot)
  outputs/phenograph_umap_domain_gradients.png (4-panel severity gradient)
"""

import os
import numpy as np
import pandas as pd
import phenograph
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import utils

DOMAIN_SCORE_COLUMNS = [f"{d}_SCORE" for d in config.VARIABLE_RATIONALE]


def main():
    df = utils.load_dataframe("04_final_scored")
    utils.log("07_phenograph", f"Loaded {len(df):,} rows from Step 4.")

    scored = df[[config.CASEID_COLUMN, "TOTAL_PHENOTYPE_SCORE"] + DOMAIN_SCORE_COLUMNS].copy()
    before = len(scored)
    scored = scored.dropna(subset=DOMAIN_SCORE_COLUMNS)
    utils.log("07_phenograph", f"Dropped {before - len(scored):,} patients missing at least one domain "
                                f"score (can't be clustered on a dimension they don't have a value for). "
                                f"Clustering {len(scored):,} patients.")

    if config.SAMPLE_SIZE is not None and len(scored) > config.SAMPLE_SIZE:
        scored = scored.sample(n=config.SAMPLE_SIZE, random_state=config.RANDOM_SEED)
        utils.log("07_phenograph", f"config.SAMPLE_SIZE is set -- clustering a {len(scored):,}-patient subsample.")

    # Small tie-breaking jitter -- see the long comment on
    # config.PHENOGRAPH_JITTER_STD for why this matters on mostly-categorical
    # clinical data. Seeded, so this is reproducible run to run.
    rng = np.random.RandomState(config.RANDOM_SEED)
    X = scored[DOMAIN_SCORE_COLUMNS].values + rng.normal(0, config.PHENOGRAPH_JITTER_STD,
                                                           size=(len(scored), len(DOMAIN_SCORE_COLUMNS)))

    utils.log("07_phenograph", f"Running PhenoGraph (k={config.PHENOGRAPH_K}) -- this typically takes "
                                f"~1-2 minutes on the full cohort...")
    communities, graph, Q = phenograph.cluster(
        X, k=config.PHENOGRAPH_K, min_cluster_size=config.PHENOGRAPH_MIN_CLUSTER_SIZE,
        seed=config.RANDOM_SEED,
    )
    scored["PHENOGRAPH_CLUSTER"] = communities
    n_clusters = len(set(communities)) - (1 if -1 in communities else 0)
    n_outliers = int((communities == -1).sum())
    utils.log("07_phenograph", f"Found {n_clusters} clusters (modularity Q={Q:.3f}); "
                                f"{n_outliers:,} patients did not fit any cluster "
                                f"(PhenoGraph cluster ID = -1, i.e. outliers).")

    # --- save the per-patient cluster assignment table ---
    utils.ensure_output_dirs()
    out_path = os.path.join(config.OUTPUT_DIR, "phenograph_clusters.xlsx")
    scored.rename(columns={config.CASEID_COLUMN: "CASEID"}).to_excel(out_path, index=False, engine="openpyxl")
    utils.log("07_phenograph", f"Saved per-patient cluster assignments -> {out_path}")

    # --- per-cluster summary: size + mean domain scores (this is the "which
    # cluster = which phenotype" interpretability table). DOMINANT_DOMAIN
    # labels each cluster by whichever of the 4 domains scores highest on
    # average for that cluster -- this is what turns "41 numbered clusters"
    # into "clusters for the 4 phenotype domains," as you asked: higher
    # score in a domain = higher risk/impact from that domain for the
    # patients grouped there.
    summary = scored.groupby("PHENOGRAPH_CLUSTER")[DOMAIN_SCORE_COLUMNS + ["TOTAL_PHENOTYPE_SCORE"]].mean()
    summary["n_patients"] = scored.groupby("PHENOGRAPH_CLUSTER").size()
    summary["DOMINANT_DOMAIN"] = summary[DOMAIN_SCORE_COLUMNS].idxmax(axis=1).str.replace("_SCORE", "", regex=False)
    summary = summary.sort_values(["DOMINANT_DOMAIN", "n_patients"], ascending=[True, False]).round(1)
    summary_path = os.path.join(config.OUTPUT_DIR, "phenograph_cluster_summary.csv")
    summary.to_csv(summary_path)
    utils.log("07_phenograph", "Cluster summary (mean domain scores, n patients, dominant domain):\n"
                                + summary.to_string())
    utils.log("07_phenograph", "Patients per dominant domain, across all clusters:\n"
                                + summary.groupby("DOMINANT_DOMAIN")["n_patients"].sum().sort_values(ascending=False).to_string())

    # --- 2-D layout for plotting only (clustering itself already happened
    # above, in the real 4-D domain-score space, on the FULL cohort). UMAP
    # is run on a smaller random subsample purely because (a) a scatterplot
    # of 200k+ overlapping points is unreadable anyway, and (b) UMAP with a
    # fixed random_state runs single-threaded, so at full scale it can take
    # many minutes. config.PHENOGRAPH_PLOT_SAMPLE_SIZE controls this --
    # raise it if you want a denser picture and don't mind waiting longer.
    plot_df = scored.sample(n=min(config.PHENOGRAPH_PLOT_SAMPLE_SIZE, len(scored)),
                             random_state=config.RANDOM_SEED).copy()
    utils.log("07_phenograph", f"Computing a 2-D UMAP layout on {len(plot_df):,} patients for visualization "
                                f"(clustering above already used the full {len(scored):,}-patient cohort)...")
    reducer = umap.UMAP(n_neighbors=config.PHENOGRAPH_K, min_dist=0.3, random_state=config.RANDOM_SEED)
    embedding = reducer.fit_transform(plot_df[DOMAIN_SCORE_COLUMNS].values)
    plot_df["UMAP_1"] = embedding[:, 0]
    plot_df["UMAP_2"] = embedding[:, 1]
    scored = plot_df  # the rest of this script only draws the plots, from here on

    # --- main plot: colored by PhenoGraph cluster ---
    fig, ax = plt.subplots(figsize=(8, 7))
    non_outliers = scored[scored["PHENOGRAPH_CLUSTER"] != -1]
    outliers = scored[scored["PHENOGRAPH_CLUSTER"] == -1]
    scatter = ax.scatter(non_outliers["UMAP_1"], non_outliers["UMAP_2"],
                          c=non_outliers["PHENOGRAPH_CLUSTER"], cmap="tab20", s=4, alpha=0.6)
    if len(outliers):
        ax.scatter(outliers["UMAP_1"], outliers["UMAP_2"], c="lightgray", s=3, alpha=0.4, label="Outliers (-1)")
    ax.set_title(f"PhenoGraph clusters -- {n_clusters} clusters across all patients\n({config.DATASET_LABEL})")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(*scatter.legend_elements(num=min(n_clusters, 10)), title="Cluster", loc="upper right",
              bbox_to_anchor=(1.25, 1.0), fontsize=8)
    fig.tight_layout()
    fig_path = os.path.join(config.OUTPUT_DIR, "phenograph_umap_clusters.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    utils.log("07_phenograph", f"Saved {fig_path}")

    # --- 4-panel plot: same layout, colored by each domain's own score
    # ("higher score = higher risk/impact" gradient, per your request) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, domain_col in zip(axes.flat, DOMAIN_SCORE_COLUMNS):
        sc = ax.scatter(scored["UMAP_1"], scored["UMAP_2"], c=scored[domain_col], cmap="YlOrRd",
                         s=4, alpha=0.7, vmin=0, vmax=100)
        ax.set_title(domain_col.replace("_SCORE", ""))
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        fig.colorbar(sc, ax=ax, label="Domain score (0-100, higher = more severe)")
    fig.suptitle(f"Phenotype domain-score gradients across all patients ({config.DATASET_LABEL})", fontsize=14)
    fig.tight_layout()
    fig_path2 = os.path.join(config.OUTPUT_DIR, "phenograph_umap_domain_gradients.png")
    fig.savefig(fig_path2, dpi=150)
    plt.close(fig)
    utils.log("07_phenograph", f"Saved {fig_path2}")

    return scored, summary


if __name__ == "__main__":
    main()
