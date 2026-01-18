FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/src

# (Wheels exist for these; build tools usually not needed, but harmless to include)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# CPU torch (Linux container)
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

# Inference deps
RUN python -m pip install \
    numpy scipy pandas tqdm pillow \
    scikit-image shapely \
    scikit-learn==1.6.1 \
    joblib \
    segmentation-models-pytorch

WORKDIR /workspace

# Copy code (still mount in compose for live edits)
COPY src/ /workspace/src/
COPY scripts/ /workspace/scripts/
