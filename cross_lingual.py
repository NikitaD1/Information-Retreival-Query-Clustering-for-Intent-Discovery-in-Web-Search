"""
cross_lingual.py
----------------
Cross-lingual query clustering support using translation-based normalisation.

Two approaches are demonstrated:
  A) Translation-based normalisation:
       Non-English queries are mapped to English equivalents before
       representation/clustering.  This aligns semantically identical
       queries across languages into the same vector neighbourhood.

  B) Language-agnostic character N-grams:
       Character-level representations partially bridge languages that share
       vocabulary roots (e.g. Spanish/French/Italian); analysed separately.
"""

import numpy as np
from collections import Counter
from sklearn.metrics import adjusted_rand_score


# ─────────────────────────────────────────────────────────────────────────────
# Translation-based normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalise_corpus(queries, translations):
    """
    Translate non-English queries to English using the provided dictionary.
    Queries not present in the dictionary are returned unchanged.

    Parameters
    ----------
    queries      : list[str]
    translations : dict  { original_query : english_translation }

    Returns
    -------
    list[str]  – normalised (English-only) corpus
    """
    return [translations.get(q, q) for q in queries]


def language_distribution(language_labels):
    """Return a summary dict of language counts and percentages."""
    counts = Counter(language_labels)
    total  = sum(counts.values())
    return {
        lang: {"count": cnt, "pct": round(100 * cnt / total, 1)}
        for lang, cnt in sorted(counts.items())
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-lingual cluster analysis
# ─────────────────────────────────────────────────────────────────────────────

def cross_lingual_comparison(
    labels_original,
    labels_translated,
    intent_labels,
    language_labels,
):
    """
    Compare clustering quality before and after translation normalisation.

    Returns
    -------
    dict with:
      ari_original   – ARI(predicted_original, ground_truth)
      ari_translated – ARI(predicted_translated, ground_truth)
      ari_agreement  – ARI(original_labels, translated_labels) — agreement
      ml_purity      – fraction of multilingual queries that share a cluster
                       with an English query of the same intent after translation
    """
    arr_orig = np.array(labels_original)
    arr_tran = np.array(labels_translated)
    arr_true = np.array(intent_labels)
    is_ml    = np.array(language_labels) != "en"

    # Mask out noise points for ARI
    valid_o = arr_orig != -1
    valid_t = arr_tran != -1

    ari_orig = adjusted_rand_score(arr_true[valid_o], arr_orig[valid_o])
    ari_tran = adjusted_rand_score(arr_true[valid_t], arr_tran[valid_t])
    ari_agr  = adjusted_rand_score(
        arr_orig[valid_o & valid_t],
        arr_tran[valid_o & valid_t],
    )

    # Multilingual co-cluster purity (translated corpus)
    intent_to_en_clusters = {}   # map each intent → set of clusters in EN queries
    for idx, (lbl, intent) in enumerate(zip(arr_tran, intent_labels)):
        if not is_ml[idx] and lbl != -1:
            intent_to_en_clusters.setdefault(intent, set()).add(lbl)

    correct = 0
    total_ml = 0
    for idx, (lbl, intent) in enumerate(zip(arr_tran, intent_labels)):
        if is_ml[idx] and lbl != -1:
            total_ml += 1
            en_clusters = intent_to_en_clusters.get(intent, set())
            if lbl in en_clusters:
                correct += 1

    ml_purity = correct / total_ml if total_ml > 0 else 0.0

    return {
        "ari_original":   round(ari_orig, 4),
        "ari_translated": round(ari_tran, 4),
        "ari_agreement":  round(ari_agr,  4),
        "ml_purity":      round(ml_purity, 4),
        "n_multilingual": int(is_ml.sum()),
    }


def multilingual_cluster_membership(queries, labels, language_labels):
    """
    Show which clusters each language's queries ended up in.

    Returns
    -------
    dict { language: Counter({cluster_id: count}) }
    """
    lang_clusters = {}
    for q, lbl, lang in zip(queries, labels, language_labels):
        if lbl == -1:
            continue
        lang_clusters.setdefault(lang, Counter())[lbl] += 1
    return lang_clusters
