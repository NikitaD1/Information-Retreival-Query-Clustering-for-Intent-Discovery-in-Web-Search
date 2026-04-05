"""
representations.py
------------------
Three query representation strategies:
  1. TF-IDF (word unigrams + bigrams)          → reduced to 100-D via SVD
  2. Character N-gram TF-IDF (2–4 char grams)  → reduced to 100-D via SVD
  3. LSA Semantic Embeddings                   → SVD(50) on word TF-IDF

All dense outputs are L2-normalised before being returned.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer


# ─────────────────────────────────────────────────────────────────────────────
def build_tfidf(queries, max_features=8000):
    """
    Word-level TF-IDF with unigrams and bigrams.
    Returns:
        X_sparse  – raw sparse TF-IDF matrix (for text-mining top-terms)
        vectorizer – fitted TfidfVectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True,
        min_df=1,
        ngram_range=(1, 2),
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",   # ASCII tokens only
    )
    X_sparse = vectorizer.fit_transform(queries)
    return X_sparse, vectorizer


def build_ngram(queries, max_features=15000, ngram_range=(2, 4)):
    """
    Character N-gram TF-IDF (language-agnostic; handles multilingual text).
    Returns sparse matrix + fitted vectorizer.
    """
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        min_df=1,
    )
    X_sparse = vectorizer.fit_transform(queries)
    return X_sparse, vectorizer


def build_lsa(queries, n_components=50):
    """
    Latent Semantic Analysis embedding (TruncatedSVD on word TF-IDF).
    Captures latent semantic topics; serves as our 'embedding' representation.
    Returns dense normalised matrix.
    """
    tfidf_vect = TfidfVectorizer(
        max_features=8000,
        sublinear_tf=True,
        min_df=1,
        ngram_range=(1, 1),
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    X_tfidf = tfidf_vect.fit_transform(queries)

    n_components = min(n_components, X_tfidf.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_lsa = Normalizer(copy=False).fit_transform(svd.fit_transform(X_tfidf))

    explained = svd.explained_variance_ratio_.sum()
    print(f"    LSA: {n_components} components explain "
          f"{explained*100:.1f}% of variance")
    return X_lsa, tfidf_vect, svd


# ─────────────────────────────────────────────────────────────────────────────
def _svd_reduce(X_sparse, n_components=100):
    """Project sparse matrix to dense n_components-D space via SVD + L2-norm."""
    n_components = min(n_components, X_sparse.shape[1] - 1, X_sparse.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_dense = Normalizer(copy=False).fit_transform(svd.fit_transform(X_sparse))
    return X_dense, svd


# ─────────────────────────────────────────────────────────────────────────────
def build_all(queries):
    """
    Build all three representations and return them in a unified dict.

    Returns
    -------
    reps : dict with keys "TF-IDF", "N-gram", "LSA"
        Each value is a tuple:
            (X_dense, X_sparse_or_None, fitted_vectorizer)
        X_dense  – L2-normalised dense matrix used for clustering / t-SNE
        X_sparse – original sparse TF-IDF (used for top-term extraction)
        vectorizer – the underlying TfidfVectorizer
    """
    print("[1/3] Building word TF-IDF representation …")
    X_tfidf_sp, tfidf_vect = build_tfidf(queries)
    X_tfidf_dense, _ = _svd_reduce(X_tfidf_sp, n_components=100)
    print(f"    TF-IDF: shape {X_tfidf_sp.shape} → dense {X_tfidf_dense.shape}")

    print("[2/3] Building character N-gram TF-IDF representation …")
    X_ng_sp, ng_vect = build_ngram(queries)
    X_ng_dense, _ = _svd_reduce(X_ng_sp, n_components=100)
    print(f"    N-gram: shape {X_ng_sp.shape} → dense {X_ng_dense.shape}")

    print("[3/3] Building LSA semantic embeddings …")
    X_lsa, lsa_base_vect, _ = build_lsa(queries, n_components=50)
    # Rebuild sparse TF-IDF with the same vocabulary for text-mining later
    X_lsa_sp, _ = build_tfidf(queries)
    print(f"    LSA: dense shape {X_lsa.shape}")

    return {
        "TF-IDF": (X_tfidf_dense, X_tfidf_sp, tfidf_vect),
        "N-gram": (X_ng_dense,   X_ng_sp,    ng_vect),
        "LSA":    (X_lsa,        X_lsa_sp,   lsa_base_vect),
    }
