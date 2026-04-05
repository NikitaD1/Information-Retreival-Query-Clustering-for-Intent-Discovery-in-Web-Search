"""
main.py
=======
Query Clustering for Intent Discovery in Web Search
Assignment #1 — Information Retrieval (S2-25_AIMLZG537)

Pipeline
--------
 1. Load dataset (230 queries; 13 % multilingual)
 2. Build three representations: TF-IDF · N-gram · LSA
 3. K-Means: sweep k=2..15, pick best k, run final clustering
 4. Hierarchical: compare ward/complete/average linkage
 5. DBSCAN: grid search (eps × min_samples)
 6. Text mining: top terms + frequent patterns + cluster labels
 7. Cluster stability analysis (10 seeds, pairwise ARI)
 8. Cross-lingual analysis: before vs after translation normalisation
 9. Save all plots → outputs/
10. Print summary table
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ── Local modules ─────────────────────────────────────────────────────────────
from dataset         import (QUERIES, INTENT_LABELS, LANGUAGE_LABELS,
                              TRANSLATIONS, N_TRUE_INTENTS)
from representations import build_all, build_tfidf, build_ngram, build_lsa, _svd_reduce
from clustering      import (kmeans_cluster, hierarchical_cluster, dbscan_cluster,
                              experiment_kmeans_k, experiment_hierarchical_linkage,
                              experiment_dbscan_grid, stability_analysis, cluster_stats)
from text_mining     import (top_terms_per_cluster, generate_cluster_label,
                              extract_frequent_patterns, queries_by_cluster,
                              print_cluster_summary)
from cross_lingual   import (normalise_corpus, language_distribution,
                              cross_lingual_comparison, multilingual_cluster_membership)
import visualization as viz


# ─────────────────────────────────────────────────────────────────────────────
def section(title):
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def subsection(title):
    print(f"\n  ── {title}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    os.makedirs("outputs", exist_ok=True)
    saved_plots = []
    results_log = []

    # ──────────────────────────────────────────────────────────────────────────
    section("DATASET OVERVIEW")
    # ──────────────────────────────────────────────────────────────────────────
    lang_dist = language_distribution(LANGUAGE_LABELS)
    total_q   = len(QUERIES)
    ml_count  = sum(1 for l in LANGUAGE_LABELS if l != "en")
    print(f"  Total queries   : {total_q}")
    print(f"  Multilingual    : {ml_count}  ({100*ml_count/total_q:.1f} %)")
    print(f"  Intent categories: {N_TRUE_INTENTS}")
    print(f"  Language breakdown:")
    for lang, info in lang_dist.items():
        print(f"    {lang:4s}  {info['count']:4d} queries  ({info['pct']:.1f} %)")

    # ──────────────────────────────────────────────────────────────────────────
    section("QUERY REPRESENTATIONS")
    # ──────────────────────────────────────────────────────────────────────────
    reps = build_all(QUERIES)
    # reps[name] = (X_dense, X_sparse, vectorizer)

    # ──────────────────────────────────────────────────────────────────────────
    section("K-MEANS CLUSTERING — HYPERPARAMETER SWEEP (k = 2 … 15)")
    # ──────────────────────────────────────────────────────────────────────────
    k_results = {}
    best_k    = {}
    for rep_name, (X_dense, _, _) in reps.items():
        results   = experiment_kmeans_k(X_dense, k_range=range(2, 16))
        k_results[rep_name] = results
        best_entry = max(results, key=lambda r: r["silhouette"])
        best_k[rep_name]    = best_entry["k"]
        print(f"  {rep_name:8s}: best k = {best_entry['k']:2d}"
              f"  (silhouette = {best_entry['silhouette']:.4f})")

    p = viz.plot_k_sensitivity(k_results, "01_kmeans_k_sensitivity.png",
                                true_k=N_TRUE_INTENTS)
    saved_plots.append(("K-Means k sensitivity", p))
    print(f"  → Saved: {p}")

    # ──────────────────────────────────────────────────────────────────────────
    section("K-MEANS — FINAL CLUSTERING (best k per representation)")
    # ──────────────────────────────────────────────────────────────────────────
    kmeans_labels  = {}
    for rep_name, (X_dense, X_sparse, vectorizer) in reps.items():
        k          = best_k[rep_name]
        labels, _  = kmeans_cluster(X_dense, k=k)
        kmeans_labels[rep_name] = labels
        stats      = cluster_stats(labels)
        from sklearn.metrics import silhouette_score, adjusted_rand_score
        sil = silhouette_score(X_dense, labels)
        ari = adjusted_rand_score(INTENT_LABELS, labels)
        print(f"\n  {rep_name}  (k={k})")
        print(f"    Silhouette = {sil:.4f}   ARI vs ground-truth = {ari:.4f}")
        print(f"    Cluster sizes — avg: {stats['avg_size']:.1f}  "
              f"std: {stats['std_size']:.1f}")

        results_log.append({
            "Method": "KMeans", "Representation": rep_name,
            "Params": f"k={k}", "Silhouette": round(sil, 4), "ARI": round(ari, 4),
        })

        # t-SNE
        p = viz.plot_tsne(X_dense, labels,
                          f"K-Means (k={k}) on {rep_name}",
                          f"02_tsne_kmeans_{rep_name.lower()}.png",
                          language_labels=LANGUAGE_LABELS)
        saved_plots.append((f"t-SNE K-Means {rep_name}", p))

        # Cluster sizes
        p = viz.plot_cluster_sizes(labels,
                                   f"K-Means Cluster Sizes — {rep_name} (k={k})",
                                   f"03_cluster_sizes_kmeans_{rep_name.lower()}.png")
        saved_plots.append((f"Cluster sizes {rep_name}", p))

        # Silhouette analysis
        p = viz.plot_silhouette_analysis(X_dense, labels,
                                         f"Silhouette Analysis — K-Means {rep_name}",
                                         f"04_silhouette_kmeans_{rep_name.lower()}.png")
        if p:
            saved_plots.append((f"Silhouette {rep_name}", p))

    print(f"\n  → Saved t-SNE, size, and silhouette plots for all representations.")

    # ──────────────────────────────────────────────────────────────────────────
    section("HIERARCHICAL CLUSTERING — LINKAGE COMPARISON")
    # ──────────────────────────────────────────────────────────────────────────
    linkage_results = {}
    hier_labels     = {}
    for rep_name, (X_dense, _, _) in reps.items():
        k = best_k[rep_name]
        results = experiment_hierarchical_linkage(X_dense, n_clusters=k,
                                                  linkages=("ward", "complete", "average"))
        linkage_results[rep_name] = results
        best_lnk = max(results, key=lambda r: r["silhouette"])
        print(f"  {rep_name:8s}  best linkage = {best_lnk['linkage']:8s}"
              f"  silhouette = {best_lnk['silhouette']:.4f}")

        labels, _ = hierarchical_cluster(X_dense, n_clusters=k,
                                          linkage=best_lnk["linkage"])
        hier_labels[rep_name] = labels
        from sklearn.metrics import silhouette_score, adjusted_rand_score
        sil = silhouette_score(X_dense, labels)
        ari = adjusted_rand_score(INTENT_LABELS, labels)
        results_log.append({
            "Method": "Hierarchical", "Representation": rep_name,
            "Params": f"k={k}, {best_lnk['linkage']}",
            "Silhouette": round(sil, 4), "ARI": round(ari, 4),
        })

    p = viz.plot_linkage_comparison(linkage_results, "05_linkage_comparison.png")
    saved_plots.append(("Linkage comparison", p))

    # Dendrogram (best representation = highest silhouette)
    best_rep_name = max(reps, key=lambda n: results_log[0]["Silhouette"]
                        if results_log[0]["Representation"] == n else -1)
    # pick LSA for dendrogram as it's lowest-dim
    X_dend, _, _ = reps["LSA"]
    p = viz.plot_dendrogram(X_dend,
                            "Hierarchical Clustering Dendrogram — LSA Embeddings",
                            "06_dendrogram.png")
    saved_plots.append(("Dendrogram", p))
    print(f"  → Saved: {p}")

    # ──────────────────────────────────────────────────────────────────────────
    section("DBSCAN CLUSTERING — GRID SEARCH (eps × min_samples)")
    # ──────────────────────────────────────────────────────────────────────────
    eps_values  = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    ms_values   = [2, 3, 5, 7]
    dbscan_all  = {}
    dbscan_best = {}

    for rep_name, (X_dense, _, _) in reps.items():
        grid = experiment_dbscan_grid(X_dense, eps_values, ms_values)
        dbscan_all[rep_name] = grid
        # Best config: silhouette > -1 and reasonable cluster count
        valid = [r for r in grid
                 if r["silhouette"] > -0.5
                 and 2 <= r["n_clusters"] <= 20]
        if valid:
            best = max(valid, key=lambda r: r["silhouette"])
        else:
            best = {"eps": 0.4, "min_samples": 3, "silhouette": -1.0,
                    "n_clusters": 0, "n_noise": 0}
        dbscan_best[rep_name] = best
        print(f"  {rep_name:8s}  best (eps={best['eps']:.2f},"
              f" min_s={best['min_samples']})  "
              f"clusters={best['n_clusters']}  "
              f"noise={best['n_noise']}  "
              f"sil={best['silhouette']:.4f}")

        labels, _ = dbscan_cluster(X_dense,
                                   eps=best["eps"],
                                   min_samples=best["min_samples"])
        from sklearn.metrics import silhouette_score, adjusted_rand_score
        mask = np.array(labels) != -1
        sil  = silhouette_score(X_dense[mask], np.array(labels)[mask]) \
               if mask.sum() > 1 and len(set(np.array(labels)[mask])) > 1 else -1.0
        ari  = adjusted_rand_score(np.array(INTENT_LABELS)[mask],
                                   np.array(labels)[mask]) if mask.sum() > 0 else 0.0
        results_log.append({
            "Method": "DBSCAN", "Representation": rep_name,
            "Params": f"eps={best['eps']}, ms={best['min_samples']}",
            "Silhouette": round(sil, 4), "ARI": round(ari, 4),
        })

    # Grid heatmap for LSA (cleanest)
    p = viz.plot_dbscan_grid(dbscan_all["LSA"], "07_dbscan_grid_lsa.png")
    saved_plots.append(("DBSCAN grid LSA", p))
    print(f"  → Saved DBSCAN grid heatmap: {p}")

    # ──────────────────────────────────────────────────────────────────────────
    section("OVERALL SILHOUETTE COMPARISON")
    # ──────────────────────────────────────────────────────────────────────────
    comparison_records = [
        {
            "label": f"{r['Method']} / {r['Representation']}",
            "silhouette": r["Silhouette"],
            "algo": r["Method"],
        }
        for r in results_log
    ]
    p = viz.plot_silhouette_comparison(comparison_records,
                                       "08_silhouette_comparison.png")
    saved_plots.append(("Silhouette comparison", p))
    print(f"  → Saved: {p}")
    df_results = pd.DataFrame(results_log).sort_values("Silhouette", ascending=False)
    print("\n" + df_results.to_string(index=False))

    # ──────────────────────────────────────────────────────────────────────────
    section("TEXT MINING — CLUSTER CHARACTERISATION")
    # ──────────────────────────────────────────────────────────────────────────
    # Use best performing representation for text mining
    best_rep = df_results.iloc[0]["Representation"]
    best_method = df_results.iloc[0]["Method"]
    print(f"  Using: {best_method} / {best_rep}")

    X_dense, X_sparse, vectorizer = reps[best_rep]
    labels_best = kmeans_labels[best_rep]

    # Top TF-IDF terms per cluster
    top_terms  = top_terms_per_cluster(X_sparse, vectorizer, labels_best, n_terms=15)
    cluster_names = {cid: generate_cluster_label(terms)
                     for cid, terms in top_terms.items()}

    # Frequent patterns
    qbc     = queries_by_cluster(QUERIES, labels_best)
    patterns = extract_frequent_patterns(qbc, top_n=8)

    print("\n  Per-cluster summary:")
    print_cluster_summary(top_terms, patterns, qbc, max_queries=3)

    # Heatmap
    p = viz.plot_top_terms_heatmap(top_terms,
                                   f"Top TF-IDF Terms per Cluster "
                                   f"({best_method} / {best_rep})",
                                   "09_top_terms_heatmap.png", n_terms=8)
    saved_plots.append(("Top terms heatmap", p))
    print(f"\n  → Saved: {p}")

    # t-SNE with cluster names
    p = viz.plot_tsne(X_dense, labels_best,
                      f"K-Means Clusters with Intent Labels — {best_rep}",
                      "10_tsne_labeled.png",
                      cluster_names=cluster_names,
                      language_labels=LANGUAGE_LABELS)
    saved_plots.append(("Labeled t-SNE", p))

    # Language distribution in clusters
    p = viz.plot_language_in_clusters(QUERIES, labels_best,
                                      LANGUAGE_LABELS,
                                      "11_language_distribution_clusters.png")
    saved_plots.append(("Language in clusters", p))
    print(f"  → Saved language distribution plot: {p}")

    # ──────────────────────────────────────────────────────────────────────────
    section("CLUSTER STABILITY ANALYSIS (10 random seeds, K-Means)")
    # ──────────────────────────────────────────────────────────────────────────
    stability_results = {}
    for rep_name, (X_dense, _, _) in reps.items():
        k   = best_k[rep_name]
        res = stability_analysis(X_dense, k=k, n_runs=10)
        stability_results[rep_name] = res
        print(f"  {rep_name:8s} (k={k}): "
              f"Mean ARI = {res['mean_ari']:.4f} ± {res['std_ari']:.4f}  "
              f"[{res['min_ari']:.3f} – {res['max_ari']:.3f}]")

    p = viz.plot_stability(stability_results, "12_cluster_stability.png")
    saved_plots.append(("Stability ARI", p))
    print(f"  → Saved: {p}")

    # ──────────────────────────────────────────────────────────────────────────
    section("CROSS-LINGUAL QUERY CLUSTERING")
    # ──────────────────────────────────────────────────────────────────────────
    print("\n  [A] Language distribution in corpus:")
    for lang, info in lang_dist.items():
        print(f"      {LANG_NAMES.get(lang, lang):8s}: {info['count']:3d} "
              f"({info['pct']:.1f} %)")

    # Normalise corpus (translate multilingual queries → English)
    LANG_NAMES_LOCAL = {"en": "English", "hi": "Hindi",
                        "es": "Spanish", "fr": "French", "de": "German"}
    q_normalised = normalise_corpus(QUERIES, TRANSLATIONS)

    print("\n  [B] Re-building representations on normalised corpus …")
    reps_norm = build_all(q_normalised)

    # Cluster original vs normalised (using LSA)
    k_cl = best_k["LSA"]
    X_orig, _, _ = reps["LSA"]
    X_tran, _, _ = reps_norm["LSA"]

    labels_orig, _ = kmeans_cluster(X_orig, k=k_cl)
    labels_tran, _ = kmeans_cluster(X_tran, k=k_cl)

    cl_analysis = cross_lingual_comparison(labels_orig, labels_tran,
                                           INTENT_LABELS, LANGUAGE_LABELS)
    print(f"\n  [C] Cross-lingual clustering results (LSA, k={k_cl}):")
    print(f"      ARI (original  vs ground-truth) = {cl_analysis['ari_original']}")
    print(f"      ARI (translated vs ground-truth) = {cl_analysis['ari_translated']}")
    print(f"      ARI agreement (orig vs tran)     = {cl_analysis['ari_agreement']}")
    print(f"      Multilingual query co-cluster purity = {cl_analysis['ml_purity']:.4f}")
    print(f"      (purity = fraction of non-EN queries that share a cluster")
    print(f"       with EN queries of the same intent after translation)")

    # Which clusters do multilingual queries land in?
    print("\n  [D] Cluster membership by language (translated corpus):")
    membership = multilingual_cluster_membership(q_normalised, labels_tran,
                                                 LANGUAGE_LABELS)
    for lang, cnt in sorted(membership.items()):
        top_clusters = cnt.most_common(3)
        display = {int(k): v for k, v in top_clusters}
        print(f"      {LANG_NAMES_LOCAL.get(lang, lang):8s}: "
              f"{display}")

    # Visualisations
    p = viz.plot_cross_lingual_tsne(X_orig, X_tran, labels_orig, labels_tran,
                                    LANGUAGE_LABELS, "13_cross_lingual_tsne.png")
    saved_plots.append(("Cross-lingual t-SNE", p))
    print(f"\n  → Saved: {p}")

    # N-gram on original (language-agnostic char n-grams)
    X_ng_orig, _, _ = reps["N-gram"]
    labels_ng, _ = kmeans_cluster(X_ng_orig, k=best_k["N-gram"])
    p = viz.plot_tsne(X_ng_orig, labels_ng,
                      "Char N-gram Clustering (Language-Agnostic)",
                      "14_tsne_ngram_crosslingual.png",
                      language_labels=LANGUAGE_LABELS)
    saved_plots.append(("N-gram cross-lingual t-SNE", p))

    # ──────────────────────────────────────────────────────────────────────────
    section("QUERY SUGGESTION IMPROVEMENT DEMO")
    # ──────────────────────────────────────────────────────────────────────────
    print("\n  Clustering improves query suggestion by retrieving semantically")
    print("  related queries within the same cluster.\n")
    demo_queries = [
        "weather in Mumbai today",
        "how to make biryani",
        "IPL 2024 schedule",
        "best laptop for programming",
        "UPSC preparation tips",
    ]
    # Map each demo query to its cluster and show suggestions
    q_idx = {q: i for i, q in enumerate(QUERIES)}
    for demo_q in demo_queries:
        idx   = q_idx.get(demo_q, -1)
        if idx == -1:
            continue
        cid   = kmeans_labels[best_rep][idx]
        peers = qbc.get(cid, [])
        suggs = [q for q in peers if q != demo_q][:5]
        print(f"  Query: '{demo_q}'")
        print(f"  Cluster {cid} suggestions:")
        for s in suggs:
            safe = s.encode("ascii", errors="replace").decode()
            print(f"    • {safe}")
        print()

    # ──────────────────────────────────────────────────────────────────────────
    section("FINAL SUMMARY")
    # ──────────────────────────────────────────────────────────────────────────
    print("\n  Results table (sorted by Silhouette Score):")
    print("\n" + df_results.to_string(index=False))

    # Save CSV
    csv_path = "outputs/results_summary.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n  Results saved → {csv_path}")

    print(f"\n  Saved plots ({len(saved_plots)}):")
    for name, path in saved_plots:
        print(f"    [{name}]  {path}")

    elapsed = time.time() - t_start
    print(f"\n  Total runtime: {elapsed:.1f} s")
    print("\n  Pipeline complete.")


LANG_NAMES = {"en": "English", "hi": "Hindi",
              "es": "Spanish", "fr": "French", "de": "German"}

if __name__ == "__main__":
    main()
