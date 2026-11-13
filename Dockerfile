FROM python:3.12-slim-bookworm

# CPU by default; use https://download.pytorch.org/whl/cu128 for NVIDIA GPUs.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/src:/workspace \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt ./
RUN python -m pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" -r requirements.txt \
    && python -m pip install --no-cache-dir -r requirements-dev.txt

COPY src/ ./src/
COPY examples/ ./examples/
COPY tests/ ./tests/
COPY LICENSE ./LICENSE

# Small synthetic inputs only; deliberately exclude tests that call run_training.
CMD ["python", "-m", "pytest", \
     "tests/test_create_mask.py", \
     "tests/test_positional_encoding.py", \
     "tests/test_ffn.py", \
     "tests/test_mha.py", \
     "tests/test_encoder.py", \
     "tests/test_decoder.py", \
     "tests/test_evaluation.py", \
     "tests/test_model.py::test_transformer_forward_masks_and_backward", \
     "-k", "not device_conversion", "-q", "-p", "no:cacheprovider"]
