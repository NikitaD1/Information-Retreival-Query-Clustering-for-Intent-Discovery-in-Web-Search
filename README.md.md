# Query Clustering for Intent Discovery in Web Search

A comprehensive implementation of query clustering algorithms for discovering user intent patterns in web search queries, with support for multilingual datasets and cross-lingual analysis.

**Course**: Information Retrieval (S2-25_AIMLZG537)  
**Assignment**: #1 — Query Clustering & Intent Discovery

---

## 📊 Project Overview

This project implements a complete pipeline for clustering search queries to discover latent user intents. It combines three text representation strategies with three clustering algorithms, evaluates their performance across multiple metrics, and provides extensive visualizations and analysis.

### Key Features

- **Multiple Text Representations**: TF-IDF, Character N-grams, and LSA embeddings
- **Three Clustering Algorithms**: K-Means, Hierarchical, and DBSCAN
- **Multilingual Support**: Handles English, Hindi, Spanish, French, and German queries
- **Cross-Lingual Analysis**: Translation-based normalization for consistent clustering
- **Comprehensive Evaluation**: Silhouette scores, ARI metrics, stability analysis
- **Rich Visualizations**: 14+ plots including t-SNE, dendrograms, heatmaps
- **Text Mining**: Automatic cluster labeling and pattern extraction

### Dataset Characteristics

- **Total Queries**: 230
- **Multilingual Queries**: ~13% (Hindi, Spanish, French, German)
- **Ground Truth**: Intent categories for evaluation
- **Languages**: en, hi, es, fr, de

---

## 🏗️ Project Structure

```
query-clustering/
│
├── main.py                      # Main pipeline orchestrator
├── representations.py           # Text representation builders
├── clustering.py                # Clustering algorithms & experiments
├── cross_lingual.py            # Cross-lingual support & analysis
├── text_mining.py              # Cluster characterization utilities
├── visualization_plot.py       # All plotting routines
├── dataset.py                  # Dataset loader (queries, labels, translations)
│
├── outputs/                    # Generated plots and results
│   ├── *.png                   # Visualization outputs
│   └── results_summary.csv     # Performance metrics table
│
└── README.md                   # This file
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy
```

### Running the Pipeline

```bash
python main.py
```

The complete pipeline executes the following steps:

1. **Load Dataset** — Loads 230 queries with intent and language labels
2. **Build Representations** — Creates TF-IDF, N-gram, and LSA embeddings
3. **K-Means Sweep** — Tests k=2..15, selects optimal k per representation
4. **Hierarchical Clustering** — Compares ward/complete/average linkage
5. **DBSCAN Grid Search** — Explores eps × min_samples parameter space
6. **Text Mining** — Extracts top terms and patterns per cluster
7. **Stability Analysis** — Evaluates clustering consistency across 10 runs
8. **Cross-Lingual Analysis** — Compares clustering before/after translation
9. **Generate Visualizations** — Saves 14+ plots to `outputs/`
10. **Summary Report** — Prints and saves performance metrics

**Runtime**: ~15-30 seconds on modern hardware

---

## 📚 Methodology

### 1. Text Representations

All representations are **L2-normalized** before clustering to ensure unit-length vectors.

#### **TF-IDF (Word-Level)**
- **Features**: Word unigrams + bigrams (max 8,000 features)
- **Preprocessing**: Sublinear TF, English stopwords removed, ASCII tokens only
- **Dimensionality Reduction**: SVD → 100-D dense representation
- **Use Case**: Captures word-level semantics, good for English queries

#### **Character N-grams**
- **Features**: Character 2-grams, 3-grams, 4-grams (max 15,000 features)
- **Preprocessing**: Character-level analysis with word boundaries
- **Dimensionality Reduction**: SVD → 100-D dense representation
- **Use Case**: Language-agnostic, handles multilingual text, robust to typos

#### **LSA (Latent Semantic Analysis)**
- **Method**: TruncatedSVD on word TF-IDF matrix
- **Components**: 50 semantic dimensions
- **Variance Explained**: Reported during execution
- **Use Case**: Captures latent semantic topics, reduces noise

### 2. Clustering Algorithms

#### **K-Means**
- **Hyperparameter**: Number of clusters (k)
- **Experiment**: Sweep k = 2..15, select best via silhouette score
- **Initialization**: 20 random initializations (n_init=20)
- **Metric**: Silhouette score for cluster quality

#### **Hierarchical (Agglomerative)**
- **Linkage Strategies**: Ward, Complete, Average
- **Distance Metric**: Euclidean (equivalent to cosine on L2-normalized vectors)
- **Selection**: Best linkage chosen via silhouette score
- **Visualization**: Dendrogram showing cluster hierarchy

#### **DBSCAN (Density-Based)**
- **Hyperparameters**: eps (neighborhood radius), min_samples (core point threshold)
- **Grid Search**: Explores parameter combinations
- **Distance Metric**: Cosine similarity
- **Noise Handling**: Identifies outlier queries (label = -1)

### 3. Evaluation Metrics

- **Silhouette Score**: Measures cluster cohesion and separation ([-1, 1])
- **Adjusted Rand Index (ARI)**: Compares predicted clusters to ground truth ([0, 1])
- **Cluster Stability**: Pairwise ARI across 10 K-Means runs with different seeds
- **Multilingual Purity**: Fraction of non-English queries sharing clusters with English queries of same intent

### 4. Cross-Lingual Clustering

Two approaches for handling multilingual queries:

#### **A) Translation-Based Normalization**
- Non-English queries mapped to English equivalents via translation dictionary
- Aligns semantically identical queries into same vector neighborhood
- Enables direct comparison with English queries

#### **B) Language-Agnostic N-grams**
- Character-level representations bridge languages sharing vocabulary roots
- Particularly effective for Romance languages (Spanish/French/Italian)
- Analyzed separately to demonstrate language-independent clustering

**Evaluation**: Compares ARI before/after translation to quantify improvement

---

## 📈 Output Files

### Visualizations (outputs/)

| Plot | Filename | Description |
|------|----------|-------------|
| K-Means k Sensitivity | `01_kmeans_k_sensitivity.png` | Silhouette scores across k=2..15 for all representations |
| t-SNE Cluster Plots | `02_tsne_kmeans_*.png` | 2D projections of clusters (TF-IDF, N-gram, LSA) |
| Cluster Size Distributions | `03_cluster_sizes_*.png` | Bar charts showing query counts per cluster |
| Silhouette Analysis | `04_silhouette_*.png` | Per-sample silhouette coefficients |
| Linkage Comparison | `05_linkage_comparison.png` | Hierarchical linkage strategy performance |
| Dendrogram | `06_dendrogram.png` | Hierarchical clustering tree (LSA embeddings) |
| DBSCAN Grid Search | `07_dbscan_grid.png` | Heatmaps of silhouette and cluster count |
| Silhouette Comparison | `08_silhouette_all.png` | Bar chart comparing all methods |
| Top Terms Heatmap | `09_top_terms_heatmap.png` | TF-IDF weights of discriminative terms |
| Labeled t-SNE | `10_tsne_labeled.png` | Clusters annotated with intent labels |
| Language Distribution | `11_language_distribution_clusters.png` | Stacked bar chart of languages per cluster |
| Cluster Stability | `12_cluster_stability.png` | Mean ± std ARI across 10 runs |
| Cross-Lingual t-SNE | `13_cross_lingual_tsne.png` | Side-by-side before/after translation |
| N-gram Cross-Lingual | `14_tsne_ngram_crosslingual.png` | Character N-gram clustering |

**Visual Legend**:
- 🔵 Blue circles = English queries
- ⭐ Star markers = Multilingual queries
- 🔴 Red = Noise points (DBSCAN only)

### Performance Summary

`outputs/results_summary.csv` contains:
- Method (K-Means, Hierarchical, DBSCAN)
- Representation (TF-IDF, N-gram, LSA)
- Hyperparameters (k, linkage, eps/min_samples)
- Silhouette Score
- ARI (vs ground truth)

Sorted by Silhouette Score (descending)

---

## 🔬 Key Findings & Insights

### Best Performing Configuration
- **Representation**: Typically LSA or TF-IDF
- **Algorithm**: K-Means with optimal k
- **Metric**: Highest silhouette score indicates best clustering quality

### Cluster Stability
- **High ARI (>0.9)**: Stable clustering across random initializations
- **Low ARI (<0.7)**: Sensitive to initialization, consider DBSCAN

### Cross-Lingual Results
- **Translation Normalization**: Improves ARI by aligning multilingual queries
- **Multilingual Purity**: Measures success of cross-lingual alignment
- **Character N-grams**: Provide language-agnostic alternative

### Text Mining
- **Automatic Labeling**: Top-3 TF-IDF terms create human-readable cluster names
- **Frequent Patterns**: Unigram/bigram extraction reveals intent patterns
- **Example Discovery**: Clusters group queries like:
  - Weather-related: "weather Mumbai", "temperature today"
  - Recipe queries: "how to make biryani", "chicken curry recipe"
  - Sports: "IPL schedule", "cricket score"

---

## 🛠️ Module Documentation

### `representations.py`

**Functions**:
- `build_tfidf(queries)` → Sparse TF-IDF matrix + vectorizer
- `build_ngram(queries)` → Character N-gram sparse matrix
- `build_lsa(queries, n_components=50)` → Dense LSA embeddings
- `build_all(queries)` → All three representations in one call

**Returns**: `(X_dense, X_sparse, vectorizer)` for each representation

### `clustering.py`

**Core Algorithms**:
- `kmeans_cluster(X, k)` → (labels, model)
- `hierarchical_cluster(X, n_clusters, linkage)` → (labels, model)
- `dbscan_cluster(X, eps, min_samples)` → (labels, model)

**Experiments**:
- `experiment_kmeans_k(X, k_range)` → List of {k, silhouette}
- `experiment_hierarchical_linkage(X, n_clusters)` → Linkage comparison
- `experiment_dbscan_grid(X, eps_values, min_samples_values)` → Grid results

**Analysis**:
- `stability_analysis(X, k, n_runs=10)` → {mean_ari, std_ari, min_ari, max_ari}
- `cluster_stats(labels)` → {n_clusters, n_noise, sizes, avg_size, std_size}

### `cross_lingual.py`

**Functions**:
- `normalise_corpus(queries, translations)` → Translated query list
- `language_distribution(language_labels)` → {lang: {count, pct}}
- `cross_lingual_comparison(...)` → {ari_original, ari_translated, ari_agreement, ml_purity}
- `multilingual_cluster_membership(...)` → {lang: Counter({cluster_id: count})}

### `text_mining.py`

**Functions**:
- `top_terms_per_cluster(X_sparse, vectorizer, labels, n_terms)` → {cluster_id: [(term, score)]}
- `generate_cluster_label(top_terms, n=3)` → Human-readable label string
- `extract_frequent_patterns(queries_by_cluster)` → {cluster_id: {unigrams, bigrams}}
- `queries_by_cluster(queries, labels)` → {cluster_id: [query_list]}
- `print_cluster_summary(...)` → Pretty-printed cluster summaries

### `visualization_plot.py`

**All functions** save figures to `outputs/` and return the filepath.

**Core Plots**:
- `plot_tsne(X, labels, title, filename, ...)` → t-SNE scatter plot
- `plot_dendrogram(X, title, filename)` → Hierarchical tree
- `plot_silhouette_analysis(X, labels, ...)` → Per-sample silhouette widths
- `plot_cluster_sizes(labels, ...)` → Bar chart of cluster sizes
- `plot_stability(stability_dict, ...)` → ARI across runs
- `plot_top_terms_heatmap(top_terms_dict, ...)` → TF-IDF heatmap
- `plot_cross_lingual_tsne(...)` → Before/after translation comparison

**Style**:
- Clean, publication-ready plots
- Consistent color palette
- Language-specific colors (blue=EN, red=HI, green=ES, purple=FR, orange=DE)
- Star markers for multilingual queries

---

## 🎯 Use Cases

### 1. Query Auto-Completion
Retrieve similar queries from the same cluster for suggestion generation

### 2. Search Intent Classification
Assign new queries to discovered intent clusters for personalization

### 3. Query Reformulation
Find alternative phrasings within clusters for query expansion

### 4. Multilingual Search
Leverage cross-lingual clustering to serve users in multiple languages

### 5. Search Analytics
Understand user intent distribution and trending query patterns

---

## 🔧 Customization

### Adjust Clustering Parameters

Edit `main.py`:
```python
# Change k-range for K-Means sweep
k_range=range(2, 20)  # Test up to 20 clusters

# Modify DBSCAN grid
eps_values = np.linspace(0.1, 0.9, 9)
min_samples_values = [2, 3, 4, 5, 6]
```

### Add New Representations

In `representations.py`:
```python
def build_custom(queries):
    # Your custom representation logic
    return X_dense, X_sparse, vectorizer
```

Then add to `build_all()` return dict.

### Tune Hyperparameters

In `clustering.py`:
```python
# K-Means: increase iterations or initializations
model = KMeans(n_clusters=k, max_iter=1000, n_init=50)

# DBSCAN: try different metrics
model = DBSCAN(eps=eps, min_samples=ms, metric="euclidean")
```

---

## 📊 Performance Optimization

### Speed Up Execution
- Reduce `k_range` in K-Means sweep
- Limit DBSCAN grid search space
- Use smaller `n_runs` in stability analysis (default: 10)

### Memory Optimization
- Reduce `max_features` in TF-IDF/N-gram vectorizers
- Decrease SVD components in LSA
- Process representations one at a time

### Parallelization
- DBSCAN uses `n_jobs=-1` (all cores) by default
- Add parallel processing to K-Means sweeps:
  ```python
  from joblib import Parallel, delayed
  results = Parallel(n_jobs=-1)(
      delayed(kmeans_cluster)(X, k) for k in k_range
  )
  ```

---

## 🐛 Troubleshooting

### Common Issues

**ImportError: No module named 'dataset'**
- Ensure `dataset.py` is in the same directory
- Check it exports: `QUERIES`, `INTENT_LABELS`, `LANGUAGE_LABELS`, `TRANSLATIONS`, `N_TRUE_INTENTS`

**Low Silhouette Scores**
- Try different representations (LSA often works well)
- Adjust k-value manually if automatic selection is suboptimal
- Check for noisy data or need for preprocessing

**DBSCAN: All points labeled as noise**
- Increase `eps` (neighborhood radius)
- Decrease `min_samples` (core point threshold)
- Try cosine metric instead of euclidean

**Memory Error**
- Reduce `max_features` in vectorizers
- Use sparse matrices where possible
- Process representations sequentially

**Visualization Issues**
- Ensure `outputs/` directory exists (created automatically)
- Check matplotlib backend if plots don't save
- Verify sufficient disk space

---

## 📖 References

### Algorithms
- **K-Means**: Lloyd's algorithm with k-means++ initialization
- **Hierarchical**: Agglomerative clustering with Ward linkage
- **DBSCAN**: Density-Based Spatial Clustering of Applications with Noise

### Metrics
- **Silhouette Coefficient**: Rousseeuw (1987)
- **Adjusted Rand Index**: Hubert & Arabie (1985)
- **t-SNE**: van der Maaten & Hinton (2008)

### Libraries
- **scikit-learn**: Pedregosa et al. (2011)
- **NumPy**: Harris et al. (2020)
- **Matplotlib**: Hunter (2007)

---

## 📝 Assignment Deliverables

1. ✅ **Source Code**: All Python modules
2. ✅ **Visualizations**: 14+ plots in `outputs/`
3. ✅ **Performance Report**: `outputs/results_summary.csv`
4. ✅ **Console Output**: Comprehensive pipeline execution log
5. ✅ **Documentation**: This README

---

## 👥 Contributing

To extend this project:

1. Add new clustering algorithms in `clustering.py`
2. Implement additional representations in `representations.py`
3. Create new visualizations in `visualization_plot.py`
4. Extend cross-lingual support in `cross_lingual.py`
5. Enhance text mining in `text_mining.py`

All functions follow consistent patterns:
- **Input**: numpy arrays, lists, or dicts
- **Output**: numpy arrays, dicts, or file paths
- **Documentation**: Docstrings with parameter descriptions
- **Naming**: Clear, descriptive function names

---

## 📄 License

This project is part of an academic assignment for Information Retrieval coursework.

---

## 🙏 Acknowledgments

- Course: Information Retrieval (S2-25_AIMLZG537)
- Dataset: Multilingual web search query corpus
- Tools: scikit-learn, NumPy, Matplotlib, Seaborn

---

**For questions or issues**: Please refer to the inline code documentation and console output during execution.
