"""
Contract tests for cluster_documents.

These verify the *shape* of the result rather than any model output: every
document is accounted for, every cluster appears, ordering is largest-first, and
labels/verdicts from the (fake, injected) LLM are wired into the result. No real
models load — fakes go in through the pipeline's dependency-injection seams.
"""
from legal_clustering.pipeline import cluster_documents
from _fakes import FakeClusterer, FakeLLM


def test_every_document_is_accounted_for(corpus):
    result = cluster_documents(
        corpus, doc_type="documents", method="Embeddings",
        label_clusters=False, clusterer=FakeClusterer(),
    )
    assert result.n_documents == len(corpus)
    assert sum(c.size for c in result.clusters) == len(corpus)


def test_two_groups_produced(corpus):
    result = cluster_documents(
        corpus, doc_type="documents", method="Embeddings",
        label_clusters=False, clusterer=FakeClusterer(),
    )
    assert result.n_clusters == 2
    assert len(result.clusters) == 2


def test_clusters_sorted_largest_first(corpus):
    result = cluster_documents(
        corpus, doc_type="documents", method="Embeddings",
        label_clusters=False, clusterer=FakeClusterer(),
    )
    sizes = [c.size for c in result.clusters]
    assert sizes == sorted(sizes, reverse=True)


def test_labels_and_verdicts_wired_in(corpus):
    # verify_labels=True runs the verification pass, so FakeLLM.evaluate_all
    # fires and each cluster gets a verdict wired into `verified`.
    result = cluster_documents(
        corpus, doc_type="documents", method="Embeddings",
        label_clusters=True, verify_labels=True,
        clusterer=FakeClusterer(), llm=FakeLLM(),
    )
    # both fake clusters have 3 members (>= min_cluster_size), so both labelled
    for c in result.clusters:
        assert c.label == f"label-{c.cluster_id}"
        assert c.verified is True


def test_labels_without_verification(corpus):
    # The default path: labels are generated, but verification is skipped, so
    # `verified` is None (the UI shows the label with no verification badge).
    result = cluster_documents(
        corpus, doc_type="documents", method="Embeddings",
        label_clusters=True,  # verify_labels defaults to False
        clusterer=FakeClusterer(), llm=FakeLLM(),
    )
    for c in result.clusters:
        assert c.label == f"label-{c.cluster_id}"
        assert c.verified is None


def test_silhouette_propagated(corpus):
    result = cluster_documents(
        corpus, doc_type="documents", method="Embeddings",
        label_clusters=False, clusterer=FakeClusterer(silhouette=0.5),
    )
    assert result.silhouette == 0.5