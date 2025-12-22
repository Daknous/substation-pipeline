FROM python:3.11-slim

WORKDIR /workspace

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir pandas numpy

COPY src/ /workspace/src/
COPY scripts/ /workspace/scripts/

ENV PYTHONPATH=/workspace

CMD ["python", "/workspace/scripts/run_substation_scoring.py"]