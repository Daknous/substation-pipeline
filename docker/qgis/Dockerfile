FROM qgis/qgis:3.42.3-noble

USER root
WORKDIR /workspace

RUN mkdir -p /tmp/runtime-root && chmod 700 /tmp/runtime-root
RUN python3 -m pip install --no-cache-dir --break-system-packages overpy shapely pandas tqdm
