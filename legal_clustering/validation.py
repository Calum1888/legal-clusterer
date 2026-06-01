"""
Corpus-level guards for the clustering pipeline.

These run before (and around) clustering and translate the ways an arbitrary
upload can break the pipeline into clear, domain-level errors instead of
opaque library stack traces:

  * too few documents to cluster meaningfully,
  * too many for agglomerative clustering's O(n^2) memory footprint,
  * a degenerate result where everything lands in one cluster (which makes
    silhouette undefined and raises deep inside scikit-learn).

`CorpusError` carries messages safe to show a user directly.
"""

from __future__ import annotations

# Defaults. Tunable per call via cluster_documents(min_documents=, max_documents=).
MIN_DOCUMENTS = 5      # below this, clustering can't separate anything useful
MAX_DOCUMENTS = 2000   # agglomerative clustering is O(n^2) in time and memory


class CorpusError(ValueError):
    """A problem with the uploaded corpus. Message is user-safe."""


def validate_corpus(
    documents: dict[str, str],
    min_documents: int = MIN_DOCUMENTS,
    max_documents: int = MAX_DOCUMENTS,
) -> None:
    """
    Check that a corpus is a sensible size to cluster.

    Args:
        documents: Mapping of doc_id -> text (after ingestion).
        min_documents: Reject corpora smaller than this.
        max_documents: Reject corpora larger than this.

    Raises:
        CorpusError: If the corpus is too small or too large. The message is
            phrased for direct display to a user.
    """
    n = len(documents)
    if n < min_documents:
        raise CorpusError(
            f"Only {n} usable document(s) were found, but at least "
            f"{min_documents} are needed to form meaningful clusters. "
            "Add more files (the supported types are .txt, .md, .pdf, .docx) "
            "and try again."
        )
    if n > max_documents:
        raise CorpusError(
            f"{n} documents is more than this tool can cluster in one pass "
            f"(the limit is {max_documents}). The clustering step grows with "
            "the square of the document count, so very large batches exhaust "
            "memory. Split the collection into smaller batches."
        )


def is_degenerate_clustering_error(exc: ValueError) -> bool:
    """
    True if `exc` looks like scikit-learn's complaint that silhouette was
    asked to score a single cluster (n_labels == 1).

    We match on the message because scikit-learn raises a plain ValueError
    here; matching is loose enough to survive minor wording changes.
    """
    msg = str(exc).lower()
    return "number of labels" in msg or "valid values are 2" in msg