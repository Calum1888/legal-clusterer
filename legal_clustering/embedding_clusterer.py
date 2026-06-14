from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from collections import Counter
from .sweep import select_k_by_silhouette


class EmbeddingClusterer:
    """
    Cluster documents using dense sentence embeddings and agglomerative
    clustering.

    The pipeline runs in two stages:
        1. A pre-trained sentence-transformer model encodes each document
           into a dense semantic vector. Documents are truncated to
           max_chars before encoding to stay within the model's context
           window (typically ~512 tokens / ~2000 chars).
        2. Agglomerative (hierarchical) clustering merges documents
           bottom-up until the linkage distance exceeds dist_threshold.

    Embeddings are L2-normalised at encode time. This means Euclidean
    distance on the output is monotonically related to cosine distance,
    so "ward" linkage with "euclidean" metric behaves equivalently to
    cosine-based clustering and is the recommended default.

    This pipeline tends to outperform classical TF-IDF when documents
    use varied vocabulary to express similar concepts, but truncation
    means it effectively clusters on document openings rather than full
    content — important for long documents like legal contracts, where
    titles, parties, and recitals often (but not always) carry the most
    class signal.

    Attributes set by `fit`:
        embeddings_ (np.ndarray): Dense matrix of L2-normalised
            embeddings, shape (n_documents, embedding_dim).
        labels_ (list[int]): Cluster label per document, aligned to doc_ids_.
        doc_ids_ (list): Document identifiers in the order they were fitted.
        silhouette_ (float): Cosine silhouette score over the embeddings.
        _encoder (SentenceTransformer): The loaded encoder, lazily
            initialised on the first call to `embed`.
    """
    def __init__(
        self,
        embedding_model: str,
        dist_threshold: float,
        linkage: str,
        metric: str,
        max_chars: int,
        batch_size: int,
        random_state: int,
        k_min: int = 2,
        k_max: int = None,
        encoder = None
    ):
        """
        Cluster documents using dense sentence embeddings + agglomerative clustering.

        Args:
            embedding_model: HuggingFace model id (e.g. "sentence-transformers/all-mpnet-base-v2").
            dist_threshold: Distance threshold for AgglomerativeClustering.
            linkage: Linkage method ("ward", "average", "complete", "single").
            metric: Distance metric ("euclidean" for ward, "cosine"/"euclidean" otherwise).
            max_chars: Truncate each document to this many characters before encoding.
                Sentence-transformer models have a ~512-token limit (~2000 chars).
            batch_size: Batch size for the encoder. Larger is faster on GPU.
            random_state: Unused by the embedding step but kept for API parity.
        """
        self.embedding_model = embedding_model
        self.dist_threshold = dist_threshold
        self.linkage = linkage
        self.metric = metric
        self.max_chars = max_chars
        self.batch_size = batch_size
        self.random_state = random_state
        self.max_chars = max_chars
        self.k_min = k_min
        self.k_max = k_max

        self.embeddings_ = None
        self.labels_ = None
        self.doc_ids_ = None
        self._encoder = None
        self.selected_k_ = None
        self._encoder = encoder

    def embed(self, documents: dict) -> np.ndarray:
        """
        Encode documents into dense semantic vectors.

        Each document is truncated to self.max_chars before encoding to stay
        within the model's context window. Embeddings are L2-normalised, so
        Euclidean distance on the output is monotonically related to cosine
        distance — ward + euclidean linkage works cleanly.

        Args:
            documents (dict): Mapping of doc_id -> raw document text.

        Returns:
            np.ndarray: Dense array of shape (n_documents, embedding_dim).

        Side effects:
            Sets self._encoder to the loaded SentenceTransformer.
            Sets self.embeddings_ to the encoded matrix.
        """
        if self._encoder is None:
            self._encoder = SentenceTransformer(self.embedding_model)

        texts = [doc[:self.max_chars] for doc in documents.values()]

        self.embeddings_ = self._encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return self.embeddings_

    def clusterer(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster embeddings hierarchically with AgglomerativeClustering.

        Args:
            embeddings (np.ndarray): Dense matrix of shape (n_documents, embedding_dim).

        Returns:
            np.ndarray: Integer cluster labels of shape (n_documents,).

        Note:
            "ward" linkage only supports Euclidean distance. Because embeddings
            are L2-normalised, Euclidean distance here is equivalent to cosine
            up to a monotonic transform, so ward works well.
        """
        if self.dist_threshold is None:
            res = select_k_by_silhouette(
                embeddings, linkage=self.linkage, metric=self.metric,
                k_min=self.k_min, k_max=self.k_max,
            )
            self.selected_k_ = res["k"]
            return res["labels"]
        
        model = AgglomerativeClustering(
            n_clusters=None, metric=self.metric,
            distance_threshold=self.dist_threshold, linkage=self.linkage,
        )
        
        return model.fit_predict(embeddings)

    def fit(self, documents: dict) -> dict:
        """
        Run the full pipeline: embed -> cluster -> print diagnostics.

        Args:
            documents (dict): Mapping of doc_id -> raw text.

        Returns:
            dict: Mapping of doc_id -> cluster label (Python int).
        """
        self.doc_ids_ = list(documents.keys())

        embeddings = self.embed(documents)
        raw_labels = self.clusterer(embeddings)

        # Normalise numpy ints to Python ints once, at the source.
        self.labels_ = [int(l) for l in raw_labels]
        sizes = Counter(self.labels_)

        self.silhouette_ = silhouette_score(embeddings, self.labels_, metric=self.metric)

        # Diagnostics
        print(f"Silhouette Score: {self.silhouette_:.4f}")
        print(f"Number of Cluster Labels: {len(set(self.labels_))}")
        print(f"Top 10 Cluster Sizes: {sizes.most_common(10)}")
        print(f"Number of Singletons: {sum(1 for c in sizes.values() if c == 1)}")
        print(f"Largest Cluster: {max(sizes.values())} documents, "
              f"Smallest Cluster: {min(sizes.values())} documents")

        return dict(zip(self.doc_ids_, self.labels_))