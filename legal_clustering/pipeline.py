"""
High-level entry point for the clustering pipeline.

`cluster_documents` is the single function the app, the tests, and any future
API all call. It runs one clusterer (embeddings by default, or TF-IDF), then
optionally labels and verifies each cluster with the LLM, and returns one
structured result object.

Unlike the research scripts, this path takes NO ground-truth labels: there is
no ground truth for an arbitrary upload, so only internal quality (silhouette)
plus the LLM label/verification are reported.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

from .document_clusterer import DocumentClusterer
from .embedding_clusterer import EmbeddingClusterer
from .llm_evaluation import LLMEvaluation
from .validation import (
    CorpusError,
    MIN_DOCUMENTS,
    MAX_DOCUMENTS,
    validate_corpus,
    is_degenerate_clustering_error,
)


@dataclass
class Cluster:
    """One cluster in the result."""
    cluster_id: int
    members: list[str]                 # doc_ids belonging to this cluster
    size: int
    label: Optional[str] = None        # LLM label, or None if not labelled
    verified: Optional[bool] = None    # LLM verification verdict (YES/NO), or None
    verdict: str = ""                  # raw LLM explanation behind `verified`


@dataclass
class ClusteringResult:
    """Everything the app needs to render a run."""
    method: str
    doc_type: str
    n_documents: int
    n_clusters: int
    silhouette: float
    clusters: list[Cluster] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-serialisable form, for an API response or logging."""
        return asdict(self)


def _build_clusterer(method: str, random_state: int):
    """Construct a clusterer with the project's tuned defaults."""
    if method == "Embeddings":
        return EmbeddingClusterer(
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            dist_threshold=None, linkage="ward", metric="euclidean",   # None = adaptive
            max_chars=2000, batch_size=32, random_state=random_state,
        )
    if method == "TF-IDF":
        return DocumentClusterer(
            ngram=(1, 2), n_components=200, n_iter=10,
            dist_threshold=None, linkage="ward", metric="euclidean",   # None = adaptive
            input_type="content", random_state=random_state,
        )
    raise ValueError(f"method must be 'Embeddings' or 'TF-IDF', got {method!r}")


def _fit_or_explain(clusterer, documents: dict) -> dict:
    """
    Run clusterer.fit, translating the single-cluster failure into a
    user-facing CorpusError. The clusterers compute a silhouette score inside
    fit(), which scikit-learn refuses to do when everything lands in one
    cluster; that surfaces here as a clean message instead of a stack trace.
    """
    try:
        return clusterer.fit(documents)
    except ValueError as exc:
        if is_degenerate_clustering_error(exc):
            raise CorpusError(
                "The documents were too similar to separate into distinct "
                "groups — clustering placed them all together. This usually "
                "means the collection is too small or too uniform to cluster. "
                "Try a larger or more varied set of documents."
            ) from exc
        raise


def cluster_documents(
    documents: dict[str, str],
    doc_type: str = "documents",
    method: str = "Embeddings",
    label_clusters: bool = True,
    verify_labels: bool = False,
    *,
    clusterer=None,
    llm=None,
    random_state: int = 42,
    min_documents: int = MIN_DOCUMENTS,
    max_documents: int = MAX_DOCUMENTS,
    progress=None,
) -> ClusteringResult:
    """
    Cluster a collection of documents and (optionally) label each cluster.

    Args:
        documents: Mapping of doc_id -> raw text. This is the whole corpus to
            cluster; clustering is unsupervised and recomputed per call.
        doc_type: Human-readable description of the documents (e.g. "legal
            contracts", "support tickets"). Fed to the LLM prompts so labels
            are domain-appropriate. This is the configurable doc-type hook.
        method: "Embeddings" (sentence-transformer, default) or "TF-IDF".
        label_clusters: If False, skip the LLM entirely and return clusters
            with no labels — fast, and useful for tests that shouldn't download
            a model.
        verify_labels: If True, run a second LLM pass that checks whether each
            cluster's documents fit its generated label. Off by default because
            it doubles the LLM work; when off, every cluster's `verified` is
            None and the UI shows the label with no verification badge.
        clusterer: Optional pre-built clusterer (dependency injection for
            tests). If None, one is built from `method` with tuned defaults.
        llm: Optional pre-built LLMEvaluation (dependency injection for tests).
            If None and label_clusters is True, one is built with the default
            model.
        random_state: Seed passed through to the clusterer.
        min_documents: Reject corpora smaller than this (see validation).
        max_documents: Reject corpora larger than this (see validation).

    Returns:
        ClusteringResult with the silhouette, cluster count, and one Cluster
        per group (every cluster is present; small ones simply have label=None).

    Raises:
        CorpusError: If the corpus is too small, too large, or too uniform to
            separate into more than one cluster. Messages are user-safe.
    """
    if clusterer is None:
        clusterer = _build_clusterer(method, random_state)

    if progress is not None:
        progress(0.35, desc="Embedding & clustering…")

    validate_corpus(documents, min_documents, max_documents)
    id_to_cluster: dict[str, int] = _fit_or_explain(clusterer, documents)

    members_by_cluster: dict[int, list[str]] = defaultdict(list)
    for doc_id, cid in id_to_cluster.items():
        members_by_cluster[int(cid)].append(doc_id)

    labels: dict[int, str] = {}
    verdicts_by_cluster: dict[int, dict] = {}

    if label_clusters:
        if llm is None:
            llm = LLMEvaluation(
                llm_model="Qwen/Qwen2.5-1.5B-Instruct",
                max_tokens=20, token_price=0.0, n_llm_samples=1,
                prompt_type_of_doc=doc_type, seed=random_state,
                batch_size=4, min_cluster_size=2, excerpt_chars=500,
            )
        if progress is not None:
            progress(0.5, desc="Labelling clusters…")
        labels = llm.llm_label(id_to_cluster, documents, progress=progress)

        # Verification is the expensive second pass; only run it when asked.
        if verify_labels:
            if progress is not None:
                progress(0.8, desc="Verifying labels…")
            verdicts = llm.evaluate_all(labels, id_to_cluster, documents, progress=progress)
            verdicts_by_cluster = {v["cluster_id"]: v for v in verdicts}

    clusters: list[Cluster] = []
    for cid in sorted(members_by_cluster):
        members = members_by_cluster[cid]
        verdict = verdicts_by_cluster.get(cid)
        clusters.append(Cluster(
            cluster_id=cid,
            members=members,
            size=len(members),
            label=labels.get(cid),
            verified=verdict["passed"] if verdict else None,
            verdict=verdict["verdict"] if verdict else "",
        ))

    # Largest clusters first — most useful at the top of a results view.
    clusters.sort(key=lambda c: c.size, reverse=True)

    return ClusteringResult(
        method=method,
        doc_type=doc_type,
        n_documents=len(documents),
        n_clusters=len(members_by_cluster),
        silhouette=float(getattr(clusterer, "silhouette_", float("nan"))),
        clusters=clusters,
    )