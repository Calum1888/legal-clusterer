# TF-IDF + LLM Document Clustering

An unsupervised NLP pipeline that clusters large collections of documents using TF-IDF vectorisation and Latent Semantic Analysis, then uses a locally running LLM to generate human-readable cluster labels — with no cloud API required.

Built and tested on the [CUAD dataset](https://www.atticusprojectai.org/cuad), a benchmark corpus of 500+ legal contracts.

---

## Overview

Grouping large document collections by topic is a common but tedious task. This project automates it end-to-end: raw text goes in, labelled clusters come out. The pipeline combines classical NLP techniques for scalable clustering with a local LLM for interpretable, human-readable output.

The design is intentionally modular — each stage (vectorisation, dimensionality reduction, clustering, labelling) is independently configurable and testable.

---

## Pipeline

```
Raw Documents
      │
      ▼
TF-IDF Vectorisation       ← n-gram range, stopword removal
      │
      ▼
Truncated SVD (LSA)        ← dimensionality reduction to dense semantic space
      │
      ▼
Agglomerative Clustering   ← distance threshold, no fixed k required
      │
      ▼
LLM Cluster Labelling      ← local Ollama model, no cloud API
      │
      ▼
Labelled Clusters + Error Detection
```

---

## Key Features

- **No fixed k** — agglomerative clustering with a distance threshold automatically determines the number of clusters
- **Fully local** — LLM inference runs via [Ollama](https://ollama.com), keeping data private with no external API calls
- **Domain agnostic** — `prompt_type_of_doc` parameter makes the pipeline reusable across any document type
- **Error detection** — LLM-powered coherence check on any specific cluster
- **Tested** — unit, integration, and edge case tests with mocked LLM calls via `pytest`

---

## Tech Stack

| Area | Tools |
|---|---|
| NLP / Vectorisation | `scikit-learn` TF-IDF, n-grams |
| Dimensionality Reduction | Truncated SVD (LSA) |
| Clustering | Agglomerative Clustering |
| LLM Inference | Ollama (`llama3.2:3b`) |
| Testing | `pytest`, `unittest.mock` |
| Data | CUAD v1 (legal contracts) |

---

## Usage

```python
from document_clusterer import DocumentClusterer

clusterer = DocumentClusterer(
    ngram=(1, 3),
    n_components=100,
    n_iter=5,
    dist_threshold=1.5,
    linkage="ward",
    input_type="content",
    random_state=42,
    llm_model="llama3.2:3b",
    n_llm_samples=5,
    prompt_type_of_doc="legal contract titles"
)

# Fit and cluster
results = clusterer.fit(documents)       # {doc_id: cluster_id}

# Generate human-readable labels
labels = clusterer.llm_cluster_label()  # {cluster_id: "Software License Agreements"}

# Check a specific cluster for coherence
verdict = clusterer.error_detection(cluster_id=3, generated_labels=labels)
# {'cluster_id': 3, 'label': 'Software License Agreements', 'verdict': 'YES, ...'}
```

---

## Installation

```bash
git clone https://github.com/Calum1888/TFIDF_LLM_Clustering.git
cd TFIDF_LLM_Clustering
pip install -r requirements.txt
```

Ollama must be running locally with your chosen model pulled:

```bash
ollama pull llama3.2:3b
```

---

## Running Tests

```bash
pytest .
```

Tests cover unit behaviour of each pipeline stage, integration of the full `fit → label → error_detection` flow, and edge cases (single document, identical documents, oversized `n_llm_samples`). All LLM calls are mocked so no local model is needed to run the test suite.

---

## Project Structure

```
TFIDF_LLM_CLUSTERING/
├── document_clusterer/
│   ├── __init__.py
│   └── document_clusterer.py   # Core pipeline class
├── data/
│   └── CUADv1.json             # Legal contracts dataset
├── running_cluster.py          # Example usage script
├── test_document_clusterer.py  # Full test suite
├── requirements.txt
└── setup.py
```

---

## Results

Tested on the CUAD dataset (500+ legal contracts), the pipeline produces clusters that align closely with contract categories such as software licensing, distribution agreements, and employment contracts — with LLM-generated labels that are interpretable without any manual inspection.

---

## Future Work

- `transform()` method to assign unseen documents to existing clusters without re-fitting
- Evaluation metrics (silhouette score, cluster purity) for quantitative assessment
- Support for dense embeddings (e.g. `sentence-transformers`) as an alternative to TF-IDF
