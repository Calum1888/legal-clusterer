---
title: Document Clusterer
emoji: 🗂️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Document Clusterer

Upload a `.zip` of documents (`.txt`, `.md`, `.pdf`, `.docx`) and get them grouped
into labelled clusters — entirely with local models, no cloud API.

- **Embeddings** method: sentence-transformer (`all-mpnet-base-v2`) + agglomerative
  clustering. Semantic, slower.
- **TF-IDF** method: TF-IDF + LSA (Truncated SVD) + agglomerative clustering.
  Keyword-based, faster.
- Each cluster is labelled and verified by a small local LLM
  (`TinyLlama-1.1B-Chat`).

Runs CPU-only in a self-contained Docker image, deployed automatically from the
GitHub repo on every change that passes CI.

The full methodology — a classical-vs-neural comparison on the CUAD contract
corpus, with internal/external metrics and the reasoning behind each design
choice — lives in the GitHub repo:

**→ https://github.com/Calum1888/legal-clusterer**