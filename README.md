---
title: Document Clusterer
emoji: 🗂️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Legal Document Clustering: Classical vs Neural Approaches

A comparative study of two unsupervised clustering pipelines for legal contracts, evaluated on the [CUAD](https://www.atticusprojectai.org/cuad) dataset (510 commercial contracts across 25 contract types). The project benchmarks a classical TF-IDF + LSA pipeline against modern sentence-transformer embeddings, evaluates both with internal and external metrics, and uses a small open-source LLM to automatically label and verify the resulting clusters.

The headline result reproduces a well-known pattern in the clustering literature: **geometric quality metrics favour the classical pipeline; ground-truth-aligned metrics favour the neural one**. The README explains why, and what that means for choosing between the two in practice.

---

## Results

Both pipelines were tuned to produce the same number of clusters (36) to make the comparison fair — distance thresholds otherwise live in incomparable geometries.

| Metric            | TF-IDF + LSA | Embeddings | Better      |
|-------------------|-------------:|-----------:|-------------|
| n_clusters        |           36 |         36 | —           |
| Silhouette ↑      |       0.3525 |     0.1062 | TF-IDF      |
| Davies-Bouldin ↓  |       1.6728 |     2.6230 | TF-IDF      |
| ARI ↑             |       0.1927 |     0.2277 | Embeddings  |
| AMI ↑             |       0.3709 |     0.4230 | Embeddings  |
| Homogeneity ↑     |       0.5412 |     0.5969 | Embeddings  |
| Completeness ↑    |       0.5002 |     0.5358 | Embeddings  |
| V-measure ↑       |       0.5199 |     0.5647 | Embeddings  |

**Interpretation.** Internal metrics (Silhouette, Davies-Bouldin) measure only geometric cluster shape — tightness and separation in vector space. TF-IDF + LSA wins these because Truncated SVD produces a low-dimensional space whose dominant axes are precisely the ones used for clustering, so clusters look clean by construction. External metrics (ARI, AMI, V-measure) compare against ground-truth contract types. Embeddings win these consistently, indicating that the embedding clusters — though geometrically messier — better match the categories a domain expert would recognise.

The silhouette gap in particular (0.35 vs 0.11) is largely an artefact of distance concentration in high-dimensional dense spaces and should not be read as the TF-IDF clusters being three times better. This is exactly why both internal and external metrics are reported.

---

## Pipeline

```
┌────────────────────┐
│  CUAD contracts    │  510 docs, 25 contract types
└─────────┬──────────┘
          │
   ┌──────┴──────┐
   │             │
   ▼             ▼
┌──────────┐  ┌─────────────────────┐
│ TF-IDF   │  │ Sentence-Transformer│
│ (1-2)gram│  │ (all-mpnet-base-v2) │
└────┬─────┘  └──────────┬──────────┘
     │                   │
     ▼                   │
┌──────────┐             │
│ Truncated│             │
│   SVD    │             │
└────┬─────┘             │
     │                   │
     └────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │ Agglomerative       │
   │ Clustering          │
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │ LLM labelling +     │
   │ self-verification   │
   └─────────────────────┘
```

### `DocumentClusterer` — classical pipeline
TF-IDF vectorisation (unigrams + bigrams, English stopwords) → Truncated SVD for dimensionality reduction (LSA) → L2 normalisation → hierarchical clustering. Fast, fully reproducible, strong baseline.

### `EmbeddingClusterer` — neural pipeline
Sentence-transformer encoding (`all-mpnet-base-v2`) with documents truncated to ~2000 chars to fit the model's context window → hierarchical clustering directly on the L2-normalised embeddings. Captures semantic similarity that bag-of-words can't see.

### `LLMEvaluation` — automated cluster interpretation
For each cluster, samples a few documents and prompts a small open-source LLM (e.g. TinyLlama) twice:
1. **Label generation** — produce a short descriptive label for the cluster.
2. **Self-verification** — given that label and a fresh sample, judge whether the documents genuinely belong together.

Both passes are batched for GPU efficiency. Verification provides a cheap, automated sanity check that scales to many clusters without manual review.

### `evaluation.py` — metrics
Computes internal (Silhouette, Davies-Bouldin) and external (ARI, AMI, Homogeneity, Completeness, V-measure) metrics, plus cluster-size diagnostics (number of singletons, largest/smallest, distribution).

---

## Methodological choices worth highlighting

A few decisions in this project go beyond the obvious tutorial defaults:

- **L2-normalising the SVD output** so that Euclidean distance in the reduced space is monotonically related to cosine — this means `linkage="ward"` (Euclidean-only) gives equivalent clusterings to cosine-based linkage, and the two clusterers can be compared on the same metric scale.
- **Left-padding the tokenizer** for batched LLM generation. Decoder-only models generate from the right end of the input; right-padding causes them to continue from pad tokens and produce garbage. This is a known footgun and is documented inline.
- **Greedy decoding** (`do_sample=False`) for LLM labelling and verification, so the evaluation is fully reproducible across runs without seed management.
- **Returning only the continuation** (`return_full_text=False`) rather than stripping the prompt from the full output — more robust than regex-based prompt removal.
- **Matched cluster counts** for the TF-IDF/embeddings comparison rather than matched distance thresholds, because the two pipelines live in different geometries.

---

## Known limitations

Included here because being upfront about limitations is itself a signal of methodological maturity.

- **Document truncation.** The embedding pipeline only sees the first ~2000 characters of each contract due to the model's 512-token limit. For CUAD this works reasonably well because contract preambles (title, parties, recitals) usually identify the contract type, but it means the pipeline is effectively clustering on document openings, not full content. A chunk-and-pool strategy would address this.
- **Silhouette is biased against high-dimensional dense embeddings.** Distance concentration makes the metric pessimistic in spaces like 768-dim MPNet output. This is why external metrics are the primary evaluation here.
- **Class imbalance in CUAD.** Service, License, and Distribution dominate; rare types like Joint Filing have only a handful of examples. Global-distance-threshold clustering cannot easily carve out very small clusters without also fragmenting the large ones, capping the achievable V-measure.
- **LLM verification is prompt-sensitive.** The current verification prompt arguably primes a YES response by stating that the algorithm grouped the documents together. A bias-aware version would present the documents and the candidate label independently and ask the model to judge whether the label fits.
- **Hierarchical clustering scales poorly** — O(n² log n) in time and memory. This pipeline is appropriate for low-thousands of documents; larger corpora would need MiniBatch K-Means, HDBSCAN, or HNSW-based approximate methods.

---

## Repository structure

```
.
├── document_clusterer.py      # TF-IDF + SVD + hierarchical clustering
├── embedding_clusterer.py     # Sentence-transformer + hierarchical clustering
├── llm_evaluation.py          # LLM-based labelling and verification
├── full_evaluation.py         # Internal/external clustering metrics + CUAD ground-truth extraction
└── README.md
```

---

## Quickstart

```python
from document_clusterer import DocumentClusterer
from embedding_clusterer import EmbeddingClusterer
from full_evaluation import evaluate_clustering, print_comparison, extract_contract_type
from llm_evaluation import LLMEvaluation

# documents: dict[str, str] mapping doc_id (filename) -> raw text
true_labels = [extract_contract_type(doc_id) for doc_id in documents]

# Classical pipeline
tfidf = DocumentClusterer(
    ngram=(1, 2), n_components=200, n_iter=10,
    dist_threshold=1.2, linkage="ward", metric="euclidean",
    input_type="content", random_state=42,
)
tfidf_labels = tfidf.fit(documents)
tfidf_results = evaluate_clustering(
    "TF-IDF + LSA", tfidf.fdm_, tfidf.labels_, true_labels,
)

# Neural pipeline
emb = EmbeddingClusterer(
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    dist_threshold=0.9, linkage="ward", metric="euclidean",
    max_chars=2000, batch_size=32, random_state=42,
)
emb_labels = emb.fit(documents)
emb_results = evaluate_clustering(
    "Embeddings", emb.embeddings_, emb.labels_, true_labels,
)

print_comparison([tfidf_results, emb_results])

# LLM-based cluster labelling and verification
llm = LLMEvaluation(
    llm_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_tokens=50, token_price=0.0, n_llm_samples=3,
    prompt_type_of_doc="legal contracts", seed=42,
    batch_size=8, min_cluster_size=2, excerpt_chars=500,
)
labels = llm.llm_label(emb_labels, documents)
verdicts = llm.evaluate_all(labels, emb_labels, documents)
```

---

## Stack

Python · scikit-learn · sentence-transformers · Hugging Face Transformers · NumPy · SciPy
