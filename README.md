# Document Clustering: Classical vs Embedding Approaches

![CI](https://github.com/Calum1888/legal-clusterer/actions/workflows/ci.yml/badge.svg)

**🔗 Live demo:** [Document Clusterer on Hugging Face Spaces](https://huggingface.co/spaces/Calum1888/DocumentClusterer) — upload a `.zip` of documents and cluster them with either method, in the browser.

A comparison of two unsupervised clustering pipelines for documents, evaluated on a selection of files. These files include PDFs of UK court rulings on the oil industry, academic papers related to my masters thesis and historical articles. It compares a **classical TF-IDF + LSA** pipeline against a modern **sentence-transformer embeddings** and uses a small open-source LLM to automatically label and verify the resulting clusters. This projected was inspired by work experience at interbational law firm A&O Shearman in their data science department. 

The idea behind the project is to take a large collection of documents and returns clusters where the documents in each cluster are similar or related in some way. This has a wide application to many business cases such as grouping legal contracts in a ligigation case or organising employee files for HR departments.

The project is packaged, tested, and continuously deployed: a `pytest` suite runs in GitHub Actions across Python 3.10–3.12 on every push, and a green push to `main` automatically deploys the Dockerised demo to Hugging Face Spaces.

---

## Demo

---

![Document Clusterer — app interface](docs/screenshot-app.png)

*Upload a zip of documents, pick a method, and the app returns labelled clusters in the browser.*

---

![TF-IDF vs Embeddings on the same upload](docs/screenshot-comparison.png)

*The same upload clustered with TF-IDF (keyword-based) and embeddings (semantic): the neural method separates themes more cleanly.*

**Interpretation.** 

---

## Pipeline

```
            Documents
                 |
        +--------+--------+
        v                 v
   TF-IDF (1-2gram)   Sentence-Transformer
        |             (all-mpnet-base-v2)
        v                 |
   Truncated SVD          |
        +--------+--------+
                 v
        Agglomerative Clustering
                 |
                 v
        LLM labelling + self-verification
```

**`DocumentClusterer` — classical pipeline.** TF-IDF vectorisation (unigrams + bigrams, English stopwords) -> Truncated SVD (LSA) -> L2 normalisation -> hierarchical clustering. Fast, fully reproducible, strong baseline.

**`EmbeddingClusterer` — neural pipeline.** Sentence-transformer encoding (`all-mpnet-base-v2`), documents truncated to ~2000 chars to fit the context window -> hierarchical clustering on the L2-normalised embeddings. Captures semantic similarity that bag-of-words cannot see.

**`LLMEvaluation` — automated cluster interpretation.** For each cluster, samples a few documents and prompts a small open-source LLM (TinyLlama) twice: once to **generate a label**, once to **self-verify** whether the documents genuinely belong together. Both passes are batched for efficiency, giving a cheap automated sanity check that scales without manual review.

**`sweep.py` — adaptive cluster count.** Replaces a hand-tuned distance threshold with a silhouette sweep: tries a range of `k`, keeps the best-scoring one, so the number of clusters adapts to whatever corpus is uploaded rather than being fixed to CUAD's shape.

**`full_evaluation.py` — metrics.** Computes internal (Silhouette, Davies-Bouldin) and external (ARI, AMI, Homogeneity, Completeness, V-measure) metrics, plus cluster-size diagnostics.

---

## Methodological choices worth highlighting

A few decisions go beyond tutorial defaults:

- **L2-normalising the SVD output**, so Euclidean distance in the reduced space is monotonically related to cosine — this makes `linkage="ward"` (Euclidean-only) equivalent to cosine-based linkage, so the two clusterers can be compared on the same metric scale.
- **Left-padding the tokenizer** for batched LLM generation. Decoder-only models generate from the right of the input; right-padding makes them continue from pad tokens and emit garbage. A known footgun, documented inline.
- **Greedy decoding** (`do_sample=False`) for labelling and verification, so the evaluation is fully reproducible across runs without seed management.
- **Returning only the continuation** (`return_full_text=False`) rather than stripping the prompt from the full output — more robust than regex-based removal.
- **Matched cluster counts** for the comparison rather than matched distance thresholds, because the two pipelines live in different geometries.

---

## Testing, CI & deployment

The package is treated as a production artifact, not a notebook:

- **Test suite (`pytest`)** covering the corpus-size guards, zip ingestion, the `cluster_documents` contract (shape, completeness, ordering, label wiring), and the silhouette-based `k`-selection. Dependency-injection seams keep the embedding model and LLM out of the test path, so the suite is fast and deterministic.
- **Continuous integration (GitHub Actions)** runs `ruff` linting and the full test suite on **Python 3.10, 3.11, and 3.12** on every push and pull request.
- **Continuous deployment.** A green push to `main` automatically packages the app and deploys it to the Hugging Face Space, so the live demo always reflects the tested `main` branch — GitHub is the single source of truth.
- **Deployment.** The demo ships as a self-contained, CPU-only Docker image with model weights baked in at build time for a fast, offline cold start. Running CPU-only was a deliberate reliability-over-latency decision after isolating a platform-level GPU-scheduling constraint.

---

## Repository structure

```
legal-clusterer/
├── .github/workflows/ci.yml     # CI: lint + tests (3.10-3.12) + auto-deploy
├── legal_clustering/            # the importable package
│   ├── __init__.py              # public API re-exports
│   ├── document_clusterer.py    # TF-IDF + SVD + hierarchical clustering
│   ├── embedding_clusterer.py   # sentence-transformer + hierarchical clustering
│   ├── llm_evaluation.py        # LLM labelling + self-verification
│   ├── full_evaluation.py       # internal/external metrics + CUAD ground truth
│   ├── sweep.py                 # silhouette-based adaptive k selection
│   ├── ingestion.py             # zip-based multi-format document loading
│   ├── validation.py            # corpus guards (CorpusError)
│   └── pipeline.py              # single entry point: cluster_documents()
├── tests/                       # pytest suite
├── data/                        # example files to cluster
├── app.py                       # Gradio demo (deployed to the Space)
├── Dockerfile                   # CPU-only container for the Space
├── README_SPACE.md              # the Space's README (deployed automatically)
├── running_cluster.py           # CLI entry point for the comparison study
├── requirements.txt             # runtime dependencies
├── requirements-dev.txt         # test / lint tooling
└── setup.py                     # package install (declares dependencies)
```

---

## Quickstart

```bash
git clone https://github.com/Calum1888/legal-clusterer.git
cd legal-clusterer
pip install -e .            # installs the package and its dependencies
```

```python
from legal_clustering import (
    DocumentClusterer,
    EmbeddingClusterer,
    LLMEvaluation,
    evaluate_clustering,
    print_comparison,
    extract_contract_type,
)

# documents: dict[str, str] mapping doc_id (filename) -> raw text
true_labels = [extract_contract_type(doc_id) for doc_id in documents]

# Classical pipeline
tfidf = DocumentClusterer(
    ngram=(1, 2), n_components=200, n_iter=10,
    dist_threshold=1.2, linkage="ward", metric="euclidean",
    input_type="content", random_state=42,
)
tfidf.fit(documents)
tfidf_results = evaluate_clustering("TF-IDF + LSA", tfidf.fdm_, tfidf.labels_, true_labels)

# Neural pipeline
emb = EmbeddingClusterer(
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    dist_threshold=0.9, linkage="ward", metric="euclidean",
    max_chars=2000, batch_size=32, random_state=42,
)
emb.fit(documents)
emb_results = evaluate_clustering("Embeddings", emb.embeddings_, emb.labels_, true_labels)

print_comparison([tfidf_results, emb_results])
```

Run the test suite:

```bash
pip install -r requirements-dev.txt
pytest
```

---

## Known limitations

Listed here because being upfront about limitations is itself a signal of methodological maturity.

- **Document truncation.** The embedding pipeline sees only the first ~2000 characters of each contract (the model's 512-token limit). 
- **LLM verification is prompt-sensitive.** The current verification prompt arguably primes a YES by stating the algorithm grouped the documents together. A bias-aware version would present the documents and candidate label independently and ask whether the label fits.
- **Hierarchical clustering scales poorly** — O(n^2 log n) in time and memory. Appropriate for low-thousands of documents; larger corpora would need MiniBatch K-Means, HDBSCAN, or HNSW-based approximate methods.

---

## Stack

Python · scikit-learn · sentence-transformers · Hugging Face Transformers · Gradio · Docker · GitHub Actions · NumPy · SciPy