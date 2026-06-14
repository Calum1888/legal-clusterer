"""
Tests for the silhouette-based adaptive cluster-count selection in sweep.py.

Skipped until sweep.py's public API is confirmed (the plan: generate embeddings
with a known number of blobs via sklearn.datasets.make_blobs and assert the
chosen k matches the blob count, plus an edge case where one blob -> k forced to
its minimum). Remove the skip and fill these in once the signature is pinned.
"""
import pytest

pytestmark = pytest.mark.skip(reason="fill in once sweep.py public API is confirmed")


def test_picks_correct_k_for_well_separated_blobs():
    ...


def test_single_blob_falls_back_to_minimum_k():
    ...
