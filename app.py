"""
Gradio app: upload a zip of documents, get back labelled clusters.

Run locally with `python app.py`, in a container via the Dockerfile, or as a
Hugging Face Gradio-SDK Space (which is also where ZeroGPU is available).
"""

import spaces

import gradio as gr


from legal_clustering.ingestion import load_documents_from_zip
from legal_clustering.pipeline import cluster_documents
from legal_clustering.validation import CorpusError

# ZeroGPU: the heavy compute function must be decorated with @spaces.GPU so the
# Space requests a GPU only while it runs. The `spaces` package only exists on
# ZeroGPU, so fall back to a no-op decorator for local / Docker runs.
try:
    import spaces
    gpu = spaces.GPU
except ImportError:
    def gpu(*args, **kwargs):
        # Used either as @gpu or @gpu(duration=...); handle both.
        if args and callable(args[0]):
            return args[0]                 # bare @gpu
        return lambda fn: fn               # @gpu(duration=...)



@gpu(duration=300)
def _cluster(documents, doc_type, method, label_clusters):
    """GPU-bound work: embedding + clustering + LLM labelling."""
    return cluster_documents(
        documents, doc_type=doc_type, method=method, label_clusters=label_clusters,
    )


def _render(result) -> str:
    """Turn a ClusteringResult into Markdown for the UI."""
    head = (
        f"### {result.n_documents} documents \u2192 {result.n_clusters} clusters\n"
        f"*silhouette {result.silhouette:.3f} \u00b7 method: {result.method}*\n"
    )
    blocks = [head]
    for c in result.clusters:
        if c.label is None:
            title = f"**Cluster {c.cluster_id}** \u00b7 {c.size} doc(s) \u00b7 *too small to label*"
        else:
            badge = "\u2713 verified" if c.verified else "\u2717 not verified"
            title = f"**{c.label}** \u00b7 {c.size} doc(s) \u00b7 {badge}"
        members = ", ".join(c.members[:8]) + (" \u2026" if c.size > 8 else "")
        blocks.append(f"{title}\n{members}")
    return "\n\n".join(blocks)


def handle(file_path, doc_type, method, label_clusters, progress=gr.Progress()):
    if not file_path:
        return "Please upload a .zip containing your documents."

    progress(0.1, desc="Reading zip…")
    try:
        documents = load_documents_from_zip(file_path)
    except Exception:
        return "That file couldn't be read as a .zip archive."

    progress(0.3, desc="Clustering…")
    try:
        result = _cluster(documents, doc_type or "documents", method, label_clusters)
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
        "Upload a **.zip** of documents (`.txt`, `.md`, `.pdf`, `.docx`) and get "
        "them grouped into labelled clusters \u2014 no cloud API, all local models."
    )
    with gr.Row():
        with gr.Column(scale=1):
            file_in = gr.File(label="Documents (.zip)", file_types=[".zip"], type="filepath")
            doc_type = gr.Textbox(
                label="What kind of documents are these?",
                placeholder="e.g. legal contracts, support tickets, news articles",
                value="documents",
            )
            method = gr.Radio(
                ["embeddings", "tfidf"], value="embeddings", label="Method",
                info="embeddings = semantic (slower); tfidf = keyword-based (faster)",
            )
            label = gr.Checkbox(value=True, label="Generate cluster labels (uses the LLM)")
            run = gr.Button("Cluster", variant="primary")
        with gr.Column(scale=2):
            out = gr.Markdown(label="Results")

    run.click(handle, inputs=[file_in, doc_type, method, label], outputs=out)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)