"""
Tests for select_k_by_silhouette (adaptive cluster-count selection).

Strategy: generate data with a *known* number of well-separated blobs via
make_blobs, and assert the silhouette sweep recovers that number. Well-separated
blobs are the case where silhouette is unambiguous, so this is a stable test of
"does the sweep pick the obviously-correct k", plus the return-shape contract
and the small-corpus guard.
"""
import numpy as np
import pytest
from sklearn.datasets import make_blobs

from legal_clustering.sweep import select_k_by_silhouette


def _blobs(n_centers, n_per=15, seed=0):
    X, _ = make_blobs(
        n_samples=n_centers * n_per,
        centers=n_centers,
        cluster_std=0.40,        # tight, well-separated -> clear silhouette peak
        random_state=seed,
    )
    return X


def test_recovers_three_well_separated_blobs():
    X = _blobs(3)
    result = select_k_by_silhouette(X, k_max=8)
    assert result["k"] == 3


def test_recovers_five_well_separated_blobs():
    X = _blobs(5)
    result = select_k_by_silhouette(X, k_max=10)
    assert result["k"] == 5


def test_return_shape_and_consistency():
    X = _blobs(4)
    result = select_k_by_silhouette(X, k_max=8)

    # keys present
    assert set(result) >= {"labels", "k", "silhouette", "trace"}

    # labels align with the data and contain exactly k groups
    assert len(result["labels"]) == X.shape[0]
    assert len(set(result["labels"])) == result["k"]

    # silhouette is in range and is the best entry in the trace
    assert -1.0 <= result["silhouette"] <= 1.0
    assert result["trace"][result["k"]] == result["silhouette"]
    assert result["silhouette"] == max(result["trace"].values())


def test_respects_k_min_and_k_max_bounds():
    X = _blobs(5)
    result = select_k_by_silhouette(X, k_min=2, k_max=4)
    # every k tried must lie within the requested window
    assert all(2 <= k <= 4 for k in result["trace"])
    assert 2 <= result["k"] <= 4


def test_too_few_samples_raises():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])  # only 2 samples
    with pytest.raises(ValueError):
        select_k_by_silhouette(X)