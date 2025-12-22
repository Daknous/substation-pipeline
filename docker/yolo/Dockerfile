FROM python:3.11-slim
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN python -m pip install --no-cache-dir ultralytics opencv-python-headless pandas numpy tqdm

COPY src/ /workspace/src/
COPY scripts/ /workspace/scripts/