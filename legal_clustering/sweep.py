"""
Data-driven selection of cluster count.

The clusterers were originally tuned with a fixed `dist_threshold`, chosen for
CUAD's 510 contracts. That value does not transfer to an arbitrary upload: on a
small or differently-shaped corpus it over-fragments (many singletons) or
under-separates. `select_k_by_silhouette` removes the hand-tuning by trying a
range of cluster counts and keeping the one with the best silhouette score,
so the number of clusters adapts to whatever was uploaded.

It stays on AgglomerativeClustering (just with n_clusters set instead of a
distance threshold), so the classical-vs-neural comparison elsewhere in the
project is unaffected.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def select_k_by_silhouette(
    X: np.ndarray,
    linkage: str = "ward",
    metric: str = "euclidean",
    k_min: int = 2,
    k_max: int | None = None,
) -> dict:
    """
    Choose the cluster count that maximises the silhouette score.

    For each k in [k_min, k_max], run agglomerative clustering with exactly k
    clusters and score it; return the labels from the best-scoring k.

    Args:
        X: Feature matrix, shape (n_samples, n_features). For these pipelines
            this is the L2-normalised SVD output or the embeddings.
        linkage: Linkage method (matches the clusterer's setting).
        metric: Distance metric (matches the clusterer's setting). With ward
            this must be "euclidean".
        k_min: Smallest cluster count to consider (>= 2; silhouette is
            undefined for a single cluster).
        k_max: Largest cluster count to consider. Defaults to
            min(n_samples - 1, 20) — silhouette needs at most n-1 clusters,
            and very high k is rarely useful.

    Returns:
        dict with:
            labels (np.ndarray): cluster labels from the best k.
            k (int): the chosen cluster count.
            silhouette (float): its silhouette score.
            trace (dict[int, float]): {k: silhouette} for every k tried,
                useful for diagnostics or plotting the selection curve.

    Raises:
        ValueError: If X has fewer than 3 samples (nothing to sweep).
    """
    n = X.shape[0]
    if n < 3:
        raise ValueError(f"need at least 3 samples to select k, got {n}")

    if k_max is None:
        k_max = min(n - 1, 20)
    k_max = min(k_max, n - 1)
    k_min = max(2, min(k_min, k_max))

    best: dict | None = None
    trace: dict[int, float] = {}

    for k in range(k_min, k_max + 1):
        labels = AgglomerativeClustering(
            n_clusters=k, metric=metric, linkage=linkage,
        ).fit_predict(X)

        # Degenerate (shouldn't happen with n_clusters=k, but guard anyway).
        if len(set(labels)) < 2:
            continue

        score = float(silhouette_score(X, labels, metric=metric))
        trace[k] = score
        if best is None or score > best["silhouette"]:
            best = {"labels": labels, "k": k, "silhouette": score}

    if best is None:  # pragma: no cover - only if every k was degenerate
        raise ValueError("could not form more than one cluster at any k")

    best["trace"] = trace
    return best