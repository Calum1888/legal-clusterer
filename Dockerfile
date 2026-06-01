# Container for the document-clustering Gradio app.
# Runs locally (`docker run -p 7860:7860 ...`), on any cloud, or as a
# Hugging Face Docker-SDK Space. Models are baked in at build time so the
# container starts fast and needs no network at runtime.

FROM python:3.11-slim

# HF Spaces run containers as a non-root user (uid 1000). Match that so the
# image behaves the same locally and on a Space, and so model caches land in
# a writable home directory.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /home/user/app

# Install dependencies first so this layer is cached across code changes.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Bake the models into the image: one slow download at build time instead of
# on every cold start. Comment this out to fetch at runtime (smaller image,
# but the first request is slow and needs network access).
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-mpnet-base-v2'); \
from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0'); \
AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')"

EXPOSE 7860
CMD ["python", "app.py"]
