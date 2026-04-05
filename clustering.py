"""
clustering.py
-------------
Implements three clustering algorithms with hyperparameter experimentation
and cluster-stability analysis via Adjusted Rand Index across multiple seeds.
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score


# ─────────────────────────────────────────────────────────────────────────────
# Individual clustering wrappers
# ─────────────────────────────────────────────────────────────────────────────

def kmeans_cluster(X, k, random_state=42, n_init=20):
    """K-Means clustering. Returns (labels, fitted_model)."""
    model = KMeans(n_clusters=k, random_state=random_state,
                   n_init=n_init, max_iter=500)
    labels = model.fit_predict(X)
    return labels, model


def hierarchical_cluster(X, n_clusters, linkage="ward"):
    """
    Agglomerative hierarchical clustering.
    All representations are L2-normalised, so euclidean distance is used for
    every linkage strategy (for L2-unit vectors, euclidean ~ 2·(1-cosine)).
    """
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        metric="euclidean",
    )
    labels = model.fit_predict(X)
    return labels, model


def dbscan_cluster(X, eps=0.5, min_samples=3):
    """
    DBSCAN clustering with cosine metric.
    Returns (labels, fitted_model).  Noise points are labeled -1.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1)
    labels = model.fit_predict(X)
    return labels, model


# ─────────────────────────────────────────────────────────────────────────────
# Silhouette helper (handles noise and degenerate cases)
# ─────────────────────────────────────────────────────────────────────────────

def _silhouette_safe(X, labels):
    """Return silhouette score, excluding noise points (label == -1)."""
    arr = np.array(labels)
    mask = arr != -1
    unique = set(arr[mask])
    if len(unique) < 2:
        return -1.0
    return float(silhouette_score(X[mask], arr[mask]))


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter experiments
# ─────────────────────────────────────────────────────────────────────────────

def experiment_kmeans_k(X, k_range=range(2, 16)):
    """
    Sweep K-Means over a range of k values.
    Returns list of dicts: {k, silhouette}.
    """
    results = []
    for k in k_range:
        labels, _ = kmeans_cluster(X, k)
        score = _silhouette_safe(X, labels)
        results.append({"k": k, "silhouette": score})
    return results


def experiment_hierarchical_linkage(X, n_clusters, linkages=("ward", "complete", "average")):
    """Try multiple linkage strategies at a fixed n_clusters."""
    results = []
    for lnk in linkages:
        labels, _ = hierarchical_cluster(X, n_clusters, linkage=lnk)
        score = _silhouette_safe(X, labels)
        results.append({"linkage": lnk, "n_clusters": n_clusters, "silhouette": score})
    return results


def experiment_dbscan_grid(X, eps_values, min_samples_values):
    """
    Grid search over DBSCAN (eps × min_samples).
    Returns list of dicts: {eps, min_samples, n_clusters, n_noise, silhouette}.
    """
    results = []
    for eps in eps_values:
        for ms in min_samples_values:
            labels, _ = dbscan_cluster(X, eps=eps, min_samples=ms)
            arr = np.array(labels)
            n_clusters = len(set(arr) - {-1})
            n_noise    = int(np.sum(arr == -1))
            score      = _silhouette_safe(X, labels)
            results.append({
                "eps": eps, "min_samples": ms,
                "n_clusters": n_clusters, "n_noise": n_noise,
                "silhouette": score,
            })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Cluster stability (K-Means, pairwise ARI)
# ─────────────────────────────────────────────────────────────────────────────

def stability_analysis(X, k, n_runs=10):
    """
    Run K-Means n_runs times with different seeds; compute pairwise ARI.
    Returns {mean_ari, std_ari, min_ari, max_ari}.
    """
    all_labels = []
    for seed in range(n_runs):
        labels, _ = kmeans_cluster(X, k, random_state=seed)
        all_labels.append(labels)

    ari_scores = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            ari_scores.append(adjusted_rand_score(all_labels[i], all_labels[j]))

    return {
        "mean_ari": float(np.mean(ari_scores)),
        "std_ari":  float(np.std(ari_scores)),
        "min_ari":  float(np.min(ari_scores)),
        "max_ari":  float(np.max(ari_scores)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cluster statistics helper
# ─────────────────────────────────────────────────────────────────────────────

def cluster_stats(labels):
    """Return basic cluster statistics dict."""
    arr = np.array(labels)
    unique = sorted(set(arr) - {-1})
    sizes = [int(np.sum(arr == c)) for c in unique]
    return {
        "n_clusters":  len(unique),
        "n_noise":     int(np.sum(arr == -1)),
        "sizes":       dict(zip(unique, sizes)),
        "avg_size":    float(np.mean(sizes)) if sizes else 0,
        "std_size":    float(np.std(sizes))  if sizes else 0,
    }
