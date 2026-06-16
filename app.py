"""
Gradio app: upload a zip of documents, get back labelled clusters.

Deployed as a Docker Space on CPU. The models are loaded once at startup and
reused on every request (no GPU, no per-request reload, no scheduling layer).
"""

import torch
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer

from legal_clustering.ingestion import load_documents_from_zip
from legal_clustering.pipeline import cluster_documents
from legal_clustering.validation import CorpusError
from legal_clustering.embedding_clusterer import EmbeddingClusterer
from legal_clustering.llm_evaluation import LLMEvaluation


EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# CPU-only deployment. These guards still resolve correctly if the same file is
# ever run on a GPU box, but on this Space they select CPU.
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_PIPE_DEVICE = 0 if torch.cuda.is_available() else -1

# ---------------------------------------------------------------------------
# Load the heavy models ONCE at startup and reuse them on every request.
# On CPU there's no process fork, so these are plain module-level singletons:
# the first user request pays nothing because loading happened at boot, and the
# Dockerfile bakes the weights into the image for a fast, offline cold start.
# ---------------------------------------------------------------------------
_ENCODER = SentenceTransformer(EMBEDDING_MODEL, device=_DEVICE)

_TOKENIZER = AutoTokenizer.from_pretrained(LLM_MODEL)
# decoder-only models often ship no pad token; reuse EOS so batching works.
if _TOKENIZER.pad_token is None:
    _TOKENIZER.pad_token = _TOKENIZER.eos_token
# left-padding is required for batched generation with decoder-only models:
# right-padding makes them continue from pad tokens and emit garbage.
_TOKENIZER.padding_side = "left"

_LLM_PIPE = pipeline(
    task="text-generation",
    model=LLM_MODEL,
    tokenizer=_TOKENIZER,
    device=_PIPE_DEVICE,
    max_new_tokens=20,
    do_sample=False,   # greedy -> reproducible labels/verdicts
)


def _cluster(documents, doc_type, method, label_clusters, verify_labels):
    """Embedding + clustering + LLM labelling, reusing the module-level models."""
    # Embeddings path reuses the shared encoder; the tfidf path is pure-CPU
    # scikit-learn, so we let the pipeline build it (clusterer=None).
    clusterer = None
    if method == "Embeddings":
        clusterer = EmbeddingClusterer(
            embedding_model=EMBEDDING_MODEL,
            dist_threshold=None, linkage="ward", metric="euclidean",  # None = adaptive
            max_chars=2500, batch_size=32, random_state=42,
            encoder=_ENCODER,                        # inject pre-loaded model
        )

    llm = None
    if label_clusters:
        llm = LLMEvaluation(
            llm_model=LLM_MODEL, max_tokens=20, token_price=0.0, n_llm_samples=1,
            prompt_type_of_doc=doc_type, seed=42, batch_size=4,
            min_cluster_size=2, excerpt_chars=500,
            hf_llm=_LLM_PIPE, tokenizer=_TOKENIZER,   # inject pre-loaded model
        )

    return cluster_documents(
        documents, doc_type=doc_type, method=method,
        label_clusters=label_clusters, verify_labels=verify_labels,
        clusterer=clusterer, llm=llm,
    )


def _render(result) -> str:
    """Turn a ClusteringResult into Markdown for the UI."""
    head = (
        f"### {result.n_documents} documents \u2192 {result.n_clusters} clusters\n"
    )
    blocks = [head]
    for c in result.clusters:
        if c.label is None:
            title = f"**Cluster {c.cluster_id}** \u00b7 {c.size} doc(s)"
        elif c.verified is None:
            # verification was skipped -> show the label with no badge
            title = f"**{c.label}** \u00b7 {c.size} doc(s)"
        else:
            badge = "\u2713 verified" if c.verified else "\u2717 not verified"
            title = f"**{c.label}** \u00b7 {c.size} doc(s) \u00b7 {badge}"
        members = ", ".join(c.members[:8]) + (" \u2026" if c.size > 8 else "")
        blocks.append(f"{title}\n{members}")
    return "\n\n".join(blocks)


def handle(file_path, doc_type, method, label_clusters, verify_labels, progress=gr.Progress()):
    if not file_path:
        return "Please upload a .zip containing your documents."

    progress(0.1, desc="Reading zip\u2026")
    try:
        documents = load_documents_from_zip(file_path)
    except Exception:
        return "That file couldn't be read as a .zip archive."

    progress(0.3, desc="Clustering\u2026")
    try:
        result = _cluster(
            documents, doc_type or "documents", method, label_clusters, verify_labels
        )
    except CorpusError as e:
        return str(e)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Something went wrong while clustering: {e}"

    progress(1.0, desc="Done")
    return _render(result)


with gr.Blocks(title="Document Clusterer") as demo:
    gr.Markdown(
        "# Document Clusterer\n"
        "Upload a **.zip** of documents (`.txt`, `.md`, `.pdf`, `.docx`). They can be clustered with 2 different methods with LLM labelling."
    )
    with gr.Row():
        with gr.Column(scale=1):
            file_in = gr.File(label="Documents (.zip)", file_types=[".zip"], type="filepath")
            doc_type = gr.Textbox(
                label="What kind of documents are these?",
                placeholder="e.g. legal contracts, support tickets, news articles",
                value="",
            )
            method = gr.Radio(
                ["Embeddings", "TF-IDF"], value="Embeddings", label="Method",
                info="Embeddings = semantic (slower); TF-IDF = keyword-based (faster)",
            )
            label = gr.Checkbox(value=True, label="Generate cluster labels (uses the LLM)")
            verify = gr.Checkbox(
                value=False,
                label="Verify labels (slower \u2014 runs a second LLM pass)",
            )
            run = gr.Button("Cluster", variant="primary")
        with gr.Column(scale=2):
            out = gr.Markdown(label="Results")

    run.click(
        handle,
        inputs=[file_in, doc_type, method, label, verify],
        outputs=out,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)