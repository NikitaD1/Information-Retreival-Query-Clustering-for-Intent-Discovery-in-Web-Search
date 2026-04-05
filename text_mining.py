"""
text_mining.py
--------------
Text-mining utilities:
  • Top TF-IDF terms per cluster  → cluster characterisation
  • Auto-generated cluster labels (top-N terms)
  • Frequent unigram / bigram intent pattern extraction per cluster
"""

import re
import numpy as np
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
# Top TF-IDF terms per cluster
# ─────────────────────────────────────────────────────────────────────────────

def top_terms_per_cluster(X_sparse, vectorizer, labels, n_terms=12):
    """
    Compute mean TF-IDF weight across all cluster members and return the
    top-n discriminative terms for each cluster.

    Parameters
    ----------
    X_sparse   : sparse TF-IDF matrix (n_samples × n_features)
    vectorizer : fitted TfidfVectorizer (provides feature names)
    labels     : cluster label array (noise = -1 is excluded)
    n_terms    : how many top terms to return per cluster

    Returns
    -------
    dict { cluster_id: [(term, score), ...] }
    """
    feature_names = vectorizer.get_feature_names_out()
    unique = sorted(set(labels) - {-1})
    result = {}
    for cid in unique:
        mask = np.array(labels) == cid
        mean_tfidf = np.asarray(X_sparse[mask].mean(axis=0)).flatten()
        top_idx    = mean_tfidf.argsort()[-n_terms:][::-1]
        result[cid] = [(feature_names[i], float(mean_tfidf[i])) for i in top_idx]
    return result


def generate_cluster_label(top_terms, n=3):
    """Return a human-readable label string from the first n top terms."""
    return " | ".join(t for t, _ in top_terms[:n])


# ─────────────────────────────────────────────────────────────────────────────
# Frequent intent-pattern extraction
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "the", "in", "of", "to", "a", "an", "is", "how", "for", "and", "on",
    "near", "me", "best", "today", "india", "with", "at", "by", "from",
    "my", "your", "new", "now", "get", "free", "online", "latest", "2024",
    "list", "top", "good", "great", "vs", "or", "it",
}


def _tokenise(text):
    """Simple ASCII tokeniser (safe for multilingual corpora)."""
    return [w for w in re.findall(r"[a-zA-Z]{3,}", text.lower())
            if w not in _STOPWORDS]


def extract_frequent_patterns(queries_by_cluster, top_n=10):
    """
    Extract the most frequent unigrams and bigrams within each cluster,
    filtering out stopwords.

    Parameters
    ----------
    queries_by_cluster : dict { cluster_id: [query_string, ...] }
    top_n              : how many patterns to return per cluster

    Returns
    -------
    dict { cluster_id: {"unigrams": [...], "bigrams": [...]} }
    """
    patterns = {}
    for cid, queries in queries_by_cluster.items():
        unigrams, bigrams = [], []
        for q in queries:
            tokens = _tokenise(q)
            unigrams.extend(tokens)
            bigrams.extend(f"{tokens[i]} {tokens[i+1]}"
                           for i in range(len(tokens) - 1))
        patterns[cid] = {
            "unigrams": Counter(unigrams).most_common(top_n),
            "bigrams":  Counter(bigrams).most_common(top_n),
        }
    return patterns


def queries_by_cluster(queries, labels):
    """Group queries by their cluster label (noise excluded)."""
    groups = {}
    for q, lbl in zip(queries, labels):
        if lbl == -1:
            continue
        groups.setdefault(lbl, []).append(q)
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def print_cluster_summary(top_terms_dict, patterns_dict, queries_by_cluster_dict,
                           max_queries=3):
    """Print a readable per-cluster summary to stdout."""
    for cid in sorted(top_terms_dict):
        label  = generate_cluster_label(top_terms_dict[cid])
        terms  = [t for t, _ in top_terms_dict[cid][:8]]
        bgrams = [b for b, _ in patterns_dict[cid]["bigrams"][:5]]
        qs     = queries_by_cluster_dict.get(cid, [])[:max_queries]

        print(f"\n  Cluster {cid:2d}  [{label}]")
        print(f"    Top terms  : {', '.join(terms)}")
        print(f"    Top bigrams: {', '.join(bgrams)}")
        for q in qs:
            # Show only ASCII portion for readability in terminal
            safe_q = q.encode("ascii", errors="replace").decode()
            print(f"    ↳  {safe_q}")
