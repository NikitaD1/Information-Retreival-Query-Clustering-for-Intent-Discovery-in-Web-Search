"""
visualization.py
----------------
All plotting routines for the assignment.
Every function saves its figure to OUTPUT_DIR and returns the file path.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage as sp_linkage
import warnings
warnings.filterwarnings("ignore")

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        120,
})

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = [
    "#1976D2", "#D32F2F", "#388E3C", "#F57C00", "#7B1FA2",
    "#0097A7", "#795548", "#E91E63", "#546E7A", "#AFB42B",
    "#00796B", "#8D6E63", "#5C6BC0", "#FF7043",
]
LANG_COLORS = {"en": "#1976D2", "hi": "#D32F2F",
               "es": "#388E3C", "fr": "#7B1FA2", "de": "#F57C00"}
LANG_NAMES  = {"en": "English", "hi": "Hindi",
               "es": "Spanish", "fr": "French",  "de": "German"}


# ─────────────────────────────────────────────────────────────────────────────
def _save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _tsne_coords(X, perplexity=None):
    """Compute 2-D t-SNE coordinates."""
    perp = perplexity or min(30, max(5, len(X) // 5))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp,
                max_iter=1000, init="pca")
    return tsne.fit_transform(X)


# ─────────────────────────────────────────────────────────────────────────────
# 1. t-SNE cluster plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_tsne(X, labels, title, filename,
              cluster_names=None, language_labels=None):
    """t-SNE scatter coloured by cluster. Multilingual queries shown as ★."""
    coords = _tsne_coords(X)
    unique = sorted(set(labels))
    is_ml  = (np.array(language_labels) != "en") if language_labels else None

    fig, ax = plt.subplots(figsize=(11, 7))
    for i, cid in enumerate(unique):
        mask  = np.array(labels) == cid
        color = "lightgrey" if cid == -1 else PALETTE[i % len(PALETTE)]
        name  = ("Noise" if cid == -1
                 else (cluster_names.get(cid, f"C{cid}") if cluster_names
                       else f"Cluster {cid}"))
        if is_ml is not None:
            en_mask = mask & ~is_ml
            ml_mask = mask &  is_ml
            ax.scatter(coords[en_mask, 0], coords[en_mask, 1],
                       c=color, s=55, alpha=0.80, label=name)
            ax.scatter(coords[ml_mask, 0], coords[ml_mask, 1],
                       c=color, s=150, alpha=1.0, marker="*",
                       edgecolors="k", linewidths=0.5)
        else:
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=color, s=60, alpha=0.80, label=name)

    if is_ml is not None:
        ax.scatter([], [], marker="*", c="gray", s=150, label="Multilingual ★")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    leg = ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
                    fontsize=8, frameon=True)
    plt.tight_layout()
    return _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dendrogram
# ─────────────────────────────────────────────────────────────────────────────
def plot_dendrogram(X, title, filename, truncate_p=20):
    linked = sp_linkage(X, method="ward")
    fig, ax = plt.subplots(figsize=(16, 6))
    dendrogram(linked, ax=ax, truncate_mode="lastp", p=truncate_p,
               leaf_rotation=90, leaf_font_size=9, show_contracted=True,
               color_threshold=0.7 * max(linked[:, 2]))
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Sample index / cluster size")
    ax.set_ylabel("Ward Distance")
    plt.tight_layout()
    return _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Silhouette analysis plot (per-sample silhouette widths)
# ─────────────────────────────────────────────────────────────────────────────
def plot_silhouette_analysis(X, labels, title, filename):
    arr  = np.array(labels)
    mask = arr != -1
    if mask.sum() < 4 or len(set(arr[mask])) < 2:
        return None
    X_clean = X[mask]; y_clean = arr[mask]
    sil_avg = silhouette_score(X_clean, y_clean)
    sil_vals = silhouette_samples(X_clean, y_clean)
    unique   = sorted(set(y_clean))
    n_cls    = len(unique)
    colors   = cm.nipy_spectral(np.linspace(0.1, 0.9, n_cls))

    fig, ax = plt.subplots(figsize=(8, max(5, n_cls * 0.5)))
    y_lower = 10
    for i, cid in enumerate(unique):
        vals = np.sort(sil_vals[y_clean == cid])
        y_upper = y_lower + len(vals)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                         facecolor=colors[i], edgecolor=colors[i], alpha=0.75)
        ax.text(-0.06, y_lower + 0.5 * len(vals), str(cid), fontsize=8)
        y_lower = y_upper + 8

    ax.axvline(sil_avg, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean silhouette = {sil_avg:.3f}")
    ax.set_xlim([-0.15, 1.0])
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Silhouette coefficient"); ax.set_ylabel("Cluster")
    ax.legend(fontsize=10); plt.tight_layout()
    return _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Silhouette comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_silhouette_comparison(records, filename):
    """
    records : list of {"label": str, "silhouette": float, "algo": str}
    """
    labels = [r["label"]      for r in records]
    scores = [r["silhouette"] for r in records]
    algos  = [r.get("algo", "")  for r in records]

    algo_colors = {"KMeans": "#1976D2", "Hierarchical": "#388E3C",
                   "DBSCAN": "#D32F2F", "": "#78909C"}
    colors = [algo_colors.get(a, "#78909C") for a in algos]

    fig, ax = plt.subplots(figsize=(max(10, len(labels)*0.85), 5))
    bars = ax.bar(range(len(labels)), scores, color=colors,
                  edgecolor="white", linewidth=0.8, width=0.7)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005, f"{s:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Silhouette Score", fontsize=11)
    ax.set_title("Silhouette Score — All Methods & Representations",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, min(1.0, max(scores) + 0.12))
    ax.axhline(np.mean(scores), color="gray", linestyle="--",
               alpha=0.7, label=f"Mean = {np.mean(scores):.3f}")
    # Legend for algo colours
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=c, label=a)
                    for a, c in algo_colors.items() if a]
    legend_elems.append(plt.Line2D([0],[0], color="gray", linestyle="--",
                                   label=f"Mean = {np.mean(scores):.3f}"))
    ax.legend(handles=legend_elems, fontsize=9)
    plt.tight_layout()
    return _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# 5. K sensitivity plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_k_sensitivity(k_results_dict, filename, true_k=14):
    """k_results_dict: {rep_name: [{k:int, silhouette:float}, ...]}"""
    rep_colors = {"TF-IDF": "#1976D2", "N-gram": "#388E3C", "LSA": "#D32F2F"}
    fig, ax = plt.subplots(figsize=(10, 5))
    for rep, results in k_results_dict.items():
        ks     = [r["k"] for r in results]
        scores = [r["silhouette"] for r in results]
        ax.plot(ks, scores, marker="o", linewidth=2, markersize=5,
                color=rep_colors.get(rep, "gray"), label=rep)
    ax.axvline(true_k, color="black", linestyle=":", linewidth=1.5,
               label=f"True intents (k={true_k})")
    ax.set_xlabel("Number of Clusters k", fontsize=11)
    ax.set_ylabel("Silhouette Score", fontsize=11)
    ax.set_title("K-Means Hyperparameter Sensitivity: k vs Silhouette",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); plt.tight_layout()
    return _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# 6. DBSCAN grid heatmaps (silhouette + n_clusters)
# ─────────────────────────────────────────────────────────────────────────────
def plot_dbscan_grid(dbscan_results, filename):
    import pandas as pd
    df = pd.DataFrame(dbscan_results)
    piv_sil = df.pivot_table(index="eps", columns="min_samples",
                              values="silhouette", aggfunc="mean")
    piv_cls = df.pivot_table(index="eps", columns="min_samples",
                              values="n_clusters", aggfunc="mean")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(piv_sil, annot=True, fmt=".3f", cmap="RdYlGn",
                ax=axes[0], vmin=-0.05, vmax=0.5, linewidths=0.5)
    axes[0].set_title("DBSCAN — Silhouette Score\n(eps × min_samples)",
                      fontweight="bold")

    sns.heatmap(piv_cls, annot=True, fmt=".0f", cmap="Blues",
                ax=axes[1], linewidths=0.5)
    axes[1].set_title("DBSCAN — Number of Clusters\n(eps × min_samples)",
                      fontweight="bold")
    plt.tight_layout()
    return _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Cluster size distribution
# ─────────────────────────────────────────────────────────────────────────────
def plot_cluster_sizes(labels, title, filename):
    from collections import Counter
    arr   = np.array(labels)
    sizes = Counter(int(l) for l in arr if l != -1)
    noise = int(np.sum(arr == -1))
    cids  = sorted(sizes)
    vals  = [sizes[c] for c in cids]

    fig, ax = plt.subplots(figsize=(max(8, len(cids)*0.9), 4))
    ax.bar([f"C{c}" for c in cids], vals, color="#1976D2",
           edgecolor="white", alpha=0.85)
    if noise:
        ax.bar(["Noise"], [noise], color="#D32F2F", alpha=0.7)
    for bar, v in zip(ax.patches, vals + ([noise] if noise else [])):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.4, str(v),
                ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Cluster"); ax.set_ylabel("Queries")
    ax.axhline(np.mean(vals), color="orange", linestyle="--",
               alpha=0.8, label=f"Mean = {np.mean(vals):.1f}")
    ax.legend(); plt.tight_layout()
    return _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Cluster stability (ARI across runs)
# ─────────────────────────────────────────────────────────────────────────────
def plot_stability(stability_dict, filename):
    """stability_dict: {label: {mean_ari, std_ari}}"""
    names = list(stability_dict)
    means = [stability_dict[n]["mean_ari"] for n in names]
    stds  = [stability_dict[n]["std_ari"]  for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, means, yerr=stds, capsize=6,
                  color="#4CAF50", edgecolor="white", alpha=0.85,
                  error_kw={"linewidth": 2, "ecolor": "black"})
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.015, f"{m:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color="green", linestyle="--", alpha=0.4, label="Perfect (ARI=1)")
    ax.set_ylabel("Mean Pairwise ARI (10 runs)", fontsize=11)
    ax.set_title("K-Means Stability: Mean ± Std ARI across 10 Runs",
                 fontsize=13, fontweight="bold")
    ax.legend(); plt.tight_layout()
    return _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Top-terms heatmap (text-mining)
# ─────────────────────────────────────────────────────────────────────────────
def plot_top_terms_heatmap(top_terms_dict, title, filename, n_terms=8):
    all_terms = []
    for terms in top_terms_dict.values():
        for t, _ in terms[:n_terms]:
            if t not in all_terms:
                all_terms.append(t)

    cluster_ids = sorted(top_terms_dict)
    mat = np.zeros((len(all_terms), len(cluster_ids)))
    for j, cid in enumerate(cluster_ids):
        td = dict(top_terms_dict[cid])
        for i, term in enumerate(all_terms):
            mat[i, j] = td.get(term, 0.0)

    h = max(8, len(all_terms) * 0.35)
    w = max(8, len(cluster_ids) * 0.9)
    fig, ax = plt.subplots(figsize=(w, h))
    sns.heatmap(mat,
                xticklabels=[f"C{c}" for c in cluster_ids],
                yticklabels=all_terms,
                cmap="YlOrRd", ax=ax, linewidths=0.4,
                annot=False, cbar_kws={"shrink": 0.6})
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Cluster"); ax.set_ylabel("Term")
    plt.tight_layout()
    return _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Cross-lingual dual t-SNE (cluster vs language)
# ─────────────────────────────────────────────────────────────────────────────
def plot_cross_lingual_tsne(X_orig, X_tran, labels_orig, labels_tran,
                             language_labels, filename):
    """Side-by-side t-SNE: original vs translation-normalised corpus."""
    coords_o = _tsne_coords(X_orig)
    coords_t = _tsne_coords(X_tran)
    is_ml    = np.array(language_labels) != "en"

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    for ax, coords, labels, title in [
        (axes[0], coords_o, labels_orig, "Original (Mixed Languages)"),
        (axes[1], coords_t, labels_tran,  "After Translation Normalisation"),
    ]:
        unique = sorted(set(labels))
        for i, cid in enumerate(unique):
            mask  = np.array(labels) == cid
            color = "lightgrey" if cid == -1 else PALETTE[i % len(PALETTE)]
            en_m  = mask & ~is_ml
            ml_m  = mask &  is_ml
            ax.scatter(coords[en_m, 0], coords[en_m, 1],
                       c=color, s=50, alpha=0.75,
                       label=f"C{cid}" if cid != -1 else "Noise")
            ax.scatter(coords[ml_m, 0], coords[ml_m, 1],
                       c=color, s=160, alpha=1.0, marker="*",
                       edgecolors="k", linewidths=0.6)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.suptitle("Cross-Lingual Clustering: Before vs After Translation\n"
                 "(★ = multilingual query)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Language distribution within clusters (stacked bar)
# ─────────────────────────────────────────────────────────────────────────────
def plot_language_in_clusters(queries, labels, language_labels, filename):
    from collections import Counter
    import pandas as pd
    langs_uniq = ["en", "hi", "es", "fr", "de"]
    cluster_ids = sorted(set(labels) - {-1})
    data = {lang: [] for lang in langs_uniq}
    for cid in cluster_ids:
        mask = np.array(labels) == cid
        lang_cnt = Counter(np.array(language_labels)[mask])
        for lang in langs_uniq:
            data[lang].append(lang_cnt.get(lang, 0))

    df = pd.DataFrame(data, index=[f"C{c}" for c in cluster_ids])
    colors_bar = [LANG_COLORS.get(l, "gray") for l in langs_uniq]

    fig, ax = plt.subplots(figsize=(max(10, len(cluster_ids)*0.85), 5))
    df.plot(kind="bar", stacked=True, ax=ax, color=colors_bar,
            edgecolor="white", linewidth=0.5, width=0.75)
    ax.set_title("Language Distribution within Each Cluster",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Cluster"); ax.set_ylabel("Query Count")
    ax.legend([LANG_NAMES[l] for l in langs_uniq], fontsize=9,
              bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# 12. Hierarchical linkage comparison
# ─────────────────────────────────────────────────────────────────────────────
def plot_linkage_comparison(linkage_results_dict, filename):
    """linkage_results_dict: {rep: [{linkage, silhouette}, ...]}"""
    rep_colors = {"TF-IDF": "#1976D2", "N-gram": "#388E3C", "LSA": "#D32F2F"}
    linkages = None
    fig, ax = plt.subplots(figsize=(9, 5))
    x = None
    for rep, results in linkage_results_dict.items():
        lnks   = [r["linkage"]    for r in results]
        scores = [r["silhouette"] for r in results]
        if x is None:
            x = np.arange(len(lnks)); linkages = lnks
        ax.plot(x, scores, marker="o", linewidth=2, markersize=7,
                label=rep, color=rep_colors.get(rep, "gray"))
    if linkages:
        ax.set_xticks(x); ax.set_xticklabels(linkages, fontsize=11)
    ax.set_ylabel("Silhouette Score", fontsize=11)
    ax.set_title("Hierarchical Clustering: Linkage Strategy Comparison",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); plt.tight_layout()
    return _save(fig, filename)
