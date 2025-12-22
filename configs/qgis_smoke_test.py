import os
import argparse
from pathlib import Path

from qgis.core import (
    QgsApplication, QgsProject, QgsRectangle,
    QgsMapSettings, QgsMapRendererParallelJob
)
from qgis.PyQt.QtCore import QSize

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--expect_layer", default=None)
    ap.add_argument("--test_render_out", required=True)
    return ap.parse_args()

def main():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt/QGIS :contentReference[oaicite:1]{index=1}

    args = parse_args()
    project_path = Path(args.project)
    if not project_path.exists():
        raise SystemExit(f"❌ Project not found: {project_path}")

    QgsApplication.setPrefixPath("/usr", True)
    qgs = QgsApplication([], False)
    qgs.initQgis()

    try:
        prj = QgsProject.instance()
        prj.clear()
        if not prj.read(str(project_path)):
            raise SystemExit(f"❌ Failed to read project: {project_path}")

        layers = list(prj.mapLayers().values())
        print(f"✅ Loaded project: {project_path.name}")
        print(f"✅ Layer count: {len(layers)}")
        print("— Layers —")
        for lyr in layers:
            print(f"  • {lyr.name()} | provider={lyr.providerType()} | crs={lyr.crs().authid()}")
            print(f"    source={lyr.source()}")

        if args.expect_layer:
            if not prj.mapLayersByName(args.expect_layer):
                raise SystemExit(f"❌ Expected layer '{args.expect_layer}' not found.")

        # Render a 256x256 PNG using all visible layers (fallback: all layers)
        root = prj.layerTreeRoot()
        render_layers = [n.layer() for n in root.findLayers() if n.isVisible() and n.layer()]
        if not render_layers:
            render_layers = layers

        ms = QgsMapSettings()
        ms.setLayers(render_layers)
        ms.setOutputSize(QSize(256, 256))

        full = None
        for lyr in render_layers:
            ext = lyr.extent()
            if not ext.isEmpty():
                full = ext if full is None else QgsRectangle(
                    min(full.xMinimum(), ext.xMinimum()),
                    min(full.yMinimum(), ext.yMinimum()),
                    max(full.xMaximum(), ext.xMaximum()),
                    max(full.yMaximum(), ext.yMaximum()),
                )
        if full is None:
            full = QgsRectangle(0, 0, 1, 1)
        ms.setExtent(full)

        job = QgsMapRendererParallelJob(ms)
        job.start(); job.waitForFinished()
        out = Path(args.test_render_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not job.renderedImage().save(str(out), "PNG"):
            raise SystemExit("❌ Render produced an image but could not save it.")
        print(f"✅ Smoke render written: {out}")
        print("✅ SMOKE TEST PASSED")

    finally:
        qgs.exitQgis()

if __name__ == "__main__":
    main()
