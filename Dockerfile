FROM python:3.11-slim

# HF Spaces run the container as a non-root user (uid 1000). Set up a writable
# home so the model cache and temp files land somewhere we own.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

# Install the CPU-only build of torch FIRST, from PyTorch's CPU index, so the
# later resolve doesn't drag in the multi-GB CUDA build we don't need here.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the model weights into the image at build time -> fast, offline cold
# starts (no download on first boot). Downloads into HF_HOME, owned by `user`.
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('sentence-transformers/all-mpnet-base-v2'); \
snapshot_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0')"

COPY --chown=user . .

EXPOSE 7860
CMD ["python", "app.py"]