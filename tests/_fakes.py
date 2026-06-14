"""
Lightweight fakes that mirror the interfaces the pipeline depends on, so tests
run fast and deterministically without loading any real models.

These exploit the dependency-injection seams cluster_documents already exposes
(`clusterer=` and `llm=`). If your real interfaces differ, adjust here only.
"""


class FakeClusterer:
    """
    Stand-in for EmbeddingClusterer / DocumentClusterer.

    `fit` splits the corpus into two deterministic groups by keyword and sets
    `silhouette_`, mirroring the real `fit() -> {doc_id: cluster_id}` contract.
    """

    def __init__(self, silhouette: float = 0.5):
        self.silhouette_ = silhouette

    def fit(self, documents):
        feline = ("cat", "feline", "kitten")
        mapping = {}
        for doc_id, text in documents.items():
            mapping[doc_id] = 0 if any(w in text for w in feline) else 1
        self.silhouette_ = 0.5
        return mapping


class FakeLLM:
    """
    Stand-in for LLMEvaluation: returns canned labels/verdicts so the labelling
    path can be exercised without TinyLlama. Only labels clusters that meet
    `min_cluster_size`, matching the real behaviour.
    """

    def __init__(self, min_cluster_size: int = 2):
        self.min_cluster_size = min_cluster_size

    def llm_label(self, id_to_cluster, documents, progress=None):
        all_cids = list(id_to_cluster.values())
        sizes = {c: all_cids.count(c) for c in set(all_cids)}
        return {c: f"label-{c}" for c, n in sizes.items() if n >= self.min_cluster_size}

    def evaluate_all(self, labels, id_to_cluster, documents, progress=None):
        return [{"cluster_id": c, "passed": True, "verdict": "ok"} for c in labels]
