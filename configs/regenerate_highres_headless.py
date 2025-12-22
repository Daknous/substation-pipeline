#!/usr/bin/env python3
"""
Headless QGIS snapshot renderer.

Canonical rule:
- OSM id is ALWAYS digits-only (int) via helpers.osm_ids.parse_osm_id_digits
- Filenames are "{osm_id}.png" whenever we can parse an id
- Footprint join uses digits-only keys too

Outputs:
- PNG images (+ .pngw world files)
- image_meters_per_px.csv with columns including osm_id + image + meters-per-px values
"""

import os
import csv
import argparse
from pathlib import Path

from qgis.core import (
    QgsApplication,
    QgsProject,
    QgsVectorLayer,
    QgsCoordinateTransform,
    QgsMapSettings,
    QgsMapRendererSequentialJob,
    QgsRectangle,
    QgsGeometry,
    QgsFeature,
    QgsField,
    QgsVectorDataProvider,
)
from qgis.PyQt.QtCore import QSize, QVariant

from src.helpers.osm_ids import parse_osm_id_digits

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--mapping_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--wms_layer_name", default="Google Satellite")

    ap.add_argument("--buffer_m", type=float, default=300.0)
    ap.add_argument("--img_size_px", type=int, default=1024)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--csv_filename", default="image_meters_per_px.csv")

    ap.add_argument("--footprints_geojson", default=None, help="GeoJSON footprints")
    ap.add_argument("--footprints_style_qml", default=None, help="Optional .qml style for footprints")
    ap.add_argument("--use_footprint_extent", action="store_true",
                    help="Use footprint bbox (if found) instead of point buffer")
    ap.add_argument("--footprint_pad_m", type=float, default=50.0,
                    help="Padding around footprint bbox in destination CRS units")
    ap.add_argument("--footprints_csv", default=None,
                    help="CSV with footprint_wkt + osm_ref (recommended; avoids OGR/GeoJSON)")

    ap.add_argument("--context_pad_m", type=float, default=100.0,
                    help="Extra padding added to the final extent (meters for EPSG:3857).")
    ap.add_argument("--min_extent_m", type=float, default=600.0,
                    help="Minimum width/height of the final extent (meters for EPSG:3857).")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip rendering if output PNG already exists")

    return ap.parse_args()


def pick_id_field(layer: QgsVectorLayer):
    names = [f.name() for f in layer.fields()]
    for c in ("osm_ref_digits", "osm_ref", "Id", "id", "osm_id", "OSM ID", "osmId", "osmID"):
        if c in names:
            return c
    return None


def pick_image_field(layer: QgsVectorLayer):
    names = [f.name() for f in layer.fields()]
    if "image" in names:
        return "image"
    if "Image" in names:
        return "Image"
    return names[0] if names else None


def expand_rect(r: QgsRectangle, pad: float) -> QgsRectangle:
    return QgsRectangle(r.xMinimum() - pad, r.yMinimum() - pad, r.xMaximum() + pad, r.yMaximum() + pad)


def finalize_extent(bbox: QgsRectangle, pad_m: float, min_extent_m: float) -> QgsRectangle:
    """
    1) add padding
    2) enforce min width/height (centered)
    """
    if pad_m and pad_m > 0:
        bbox = expand_rect(bbox, pad_m)

    w = bbox.width()
    h = bbox.height()

    cx = (bbox.xMinimum() + bbox.xMaximum()) / 2.0
    cy = (bbox.yMinimum() + bbox.yMaximum()) / 2.0

    new_w = max(w, min_extent_m) if min_extent_m and min_extent_m > 0 else w
    new_h = max(h, min_extent_m) if min_extent_m and min_extent_m > 0 else h

    return QgsRectangle(cx - new_w / 2.0, cy - new_h / 2.0, cx + new_w / 2.0, cy + new_h / 2.0)


def safe_base_name(value: str) -> str:
    """
    Fallback for filename if no osm id can be parsed.
    Keeps 'way/14037659' -> 'way_14037659'
    """
    s = str(value).strip()
    if "." in s and not s.endswith("."):
        s = s.rsplit(".", 1)[0]
    return s.replace("/", "_").replace("\\", "_")


def main():
    # Strong headless defaults
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_OPENGL", "software")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

    args = parse_args()

    project_path = Path(args.project)
    mapping_csv = Path(args.mapping_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    QgsApplication.setPrefixPath("/usr", True)
    qgs = QgsApplication([], False)
    qgs.initQgis()
    QgsApplication.setMaxThreads(1)  # stability in headless

    try:
        project = QgsProject.instance()
        project.clear()

        print(f"Loading QGIS project: {project_path}", flush=True)
        if not project.read(str(project_path)):
            raise RuntimeError(f"Failed to read QGIS project: {project_path}")

        # Basemap layer
        wms_layers = project.mapLayersByName(args.wms_layer_name)
        if not wms_layers:
            raise RuntimeError(f"WMS/XYZ layer '{args.wms_layer_name}' not found in project.")
        wms = wms_layers[0]
        print(f"Basemap layer: {wms.name()} | provider={wms.providerType()} | crs={wms.crs().authid()}", flush=True)

        # Mapping CSV -> point layer (EPSG:4326)
        uri = f"file:///{mapping_csv}?delimiter=,&xField=Longitude&yField=Latitude&crs=EPSG:4326"
        vlayer = QgsVectorLayer(uri, "mapping_points", "delimitedtext")
        if not vlayer.isValid():
            raise RuntimeError(f"Could not load mapping CSV: {mapping_csv}")

        image_field = pick_image_field(vlayer)
        if not image_field:
            raise RuntimeError("Could not determine image field from mapping CSV.")

        id_field = pick_id_field(vlayer)
        if id_field:
            print(f"Using '{id_field}' as OSM id field", flush=True)
        else:
            print("⚠️ No id_field found in mapping layer; will try parsing id from image_field.", flush=True)

        print(f"Using '{image_field}' as image field", flush=True)

        # --- load footprints (optional) ---
        footprints = None
        footprints_by_id = {}  # key: digits-only string -> bbox (in footprint CRS)
        xf_fp_to_wms = None

        def build_memory_footprints_layer_from_csv(csv_path: Path) -> QgsVectorLayer:
            lyr = QgsVectorLayer("Polygon?crs=EPSG:4326", "footprints", "memory")
            pr: QgsVectorDataProvider = lyr.dataProvider()
            pr.addAttributes([
                QgsField("osm_ref", QVariant.String),
                QgsField("osm_id", QVariant.String),
                QgsField("matched_ref", QVariant.String),
                QgsField("method", QVariant.String),
            ])
            lyr.updateFields()

            feats = []
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Check if 'found' is True (handle both boolean and string)
                    found = str(row.get("found", "")).lower() in ['true', '1', 'yes']
                    if not found:
                        continue
                        
                    wkt = (row.get("footprint_wkt") or "").strip()
                    if not wkt:
                        continue
                    g = QgsGeometry.fromWkt(wkt)
                    if g is None or g.isEmpty():
                        continue

                    ft = QgsFeature(lyr.fields())
                    ft.setGeometry(g)
                    
                    # Support both osm_ref and osm_id columns
                    osm_val = (row.get("osm_ref") or row.get("osm_id") or "").strip()
                    ft["osm_ref"] = osm_val
                    ft["osm_id"] = (row.get("osm_id") or osm_val).strip()
                    ft["matched_ref"] = osm_val
                    ft["method"] = (row.get("method") or "").strip()
                    feats.append(ft)

            pr.addFeatures(feats)
            lyr.updateExtents()
            print(f"✅ Loaded {len(feats)} footprint polygons from CSV")
            return lyr

        # Prefer CSV footprints (recommended)
        if args.footprints_csv:
            fp_csv = Path(args.footprints_csv)
            print(f"Footprints CSV: {fp_csv}", flush=True)
            if fp_csv.exists():
                footprints = build_memory_footprints_layer_from_csv(fp_csv)
                if not footprints.isValid():
                    print(f"⚠️ Footprints memory layer invalid from CSV: {fp_csv}", flush=True)
                    footprints = None
                else:
                    # Index by digits-only
                    fp_id_field = "osm_id" if "osm_id" in [f.name() for f in footprints.fields()] else pick_id_field(footprints)
                    if fp_id_field:
                        for f in footprints.getFeatures():
                            oid = parse_osm_id_digits(f[fp_id_field])
                            if oid is None:
                                continue
                            # Store bbox in footprint CRS (EPSG:4326)
                            footprints_by_id[str(oid)] = f.geometry().boundingBox()
                        print(f"✅ Footprints loaded from CSV: {len(footprints_by_id)} bboxes indexed by digits-only '{fp_id_field}'", flush=True)

                    xf_fp_to_wms = QgsCoordinateTransform(footprints.crs(), wms.crs(), project)
            else:
                print(f"⚠️ Footprints CSV not found: {fp_csv}", flush=True)

        # Fallback: GeoJSON
        elif args.footprints_geojson:
            fp_path = Path(args.footprints_geojson)
            print(f"Footprints GeoJSON: {fp_path}", flush=True)
            if fp_path.exists():
                footprints = QgsVectorLayer(str(fp_path), "footprints", "ogr")
                if not footprints.isValid():
                    print(f"⚠️ Footprints GeoJSON invalid: {fp_path}", flush=True)
                    footprints = None
                else:
                    fp_id_field = pick_id_field(footprints)
                    if fp_id_field:
                        for f in footprints.getFeatures():
                            oid = parse_osm_id_digits(f[fp_id_field])
                            if oid is None:
                                continue
                            footprints_by_id[str(oid)] = f.geometry().boundingBox()
                        print(f"✅ Footprints loaded from GeoJSON: {len(footprints_by_id)} bboxes indexed by digits-only '{fp_id_field}'", flush=True)
                    xf_fp_to_wms = QgsCoordinateTransform(footprints.crs(), wms.crs(), project)
            else:
                print(f"⚠️ Footprints GeoJSON not found: {fp_path}", flush=True)

        # transforms: points -> basemap CRS
        xf_pt_to_wms = QgsCoordinateTransform(vlayer.crs(), wms.crs(), project)

        # map settings
        ms = QgsMapSettings()
        layers_to_render = [wms] + ([footprints] if footprints else [])
        ms.setLayers(layers_to_render)
        ms.setDestinationCrs(wms.crs())
        ms.setOutputSize(QSize(args.img_size_px, args.img_size_px))
        ms.setOutputDpi(args.dpi)

        # CSV metrics
        csv_path = out_dir / args.csv_filename
        fieldnames = [
            "osm_id",          # digits-only
            "image",           # filename
            "ground_width_m",
            "ground_height_m",
            "m_per_px_x",
            "m_per_px_y",
            "meters_per_px",
            "extent_mode",
        ]
        rows = []
        
        skipped_count = 0
        rendered_count = 0

        print("Starting render loop…", flush=True)
        if args.skip_existing:
            print("⏭️  Skip mode enabled: existing images will be reused", flush=True)

        for feat in vlayer.getFeatures():
            # Canonical id (digits-only): prefer id_field, else image_field
            oid = None
            if id_field:
                oid = parse_osm_id_digits(feat[id_field])
            if oid is None:
                oid = parse_osm_id_digits(feat[image_field])

            # Filename: stable if oid exists
            if oid is not None:
                fname = f"{oid}.png"
            else:
                img_val = feat[image_field]
                base = safe_base_name(img_val)
                fname = f"{base}.png"

            out_png = out_dir / fname
            world_path = Path(f"{out_png}w")

            # Check if we should skip rendering
            if args.skip_existing and out_png.exists() and world_path.exists():
                try:
                    # Parse world file to extract metrics
                    with open(world_path, "r") as wf:
                        lines = [line.strip() for line in wf.readlines()]
                        if len(lines) >= 6:
                            px_w = float(lines[0])
                            px_h = float(lines[3])
                            
                            ground_width_m = abs(px_w) * args.img_size_px
                            ground_height_m = abs(px_h) * args.img_size_px
                            meters_per_px = (abs(px_w) + abs(px_h)) / 2
                            
                            rows.append({
                                "osm_id": str(oid) if oid is not None else "",
                                "image": fname,
                                "ground_width_m": ground_width_m,
                                "ground_height_m": ground_height_m,
                                "m_per_px_x": abs(px_w),
                                "m_per_px_y": abs(px_h),
                                "meters_per_px": meters_per_px,
                                "extent_mode": "cached",
                            })
                            
                            skipped_count += 1
                            print(f"⏭ {fname} [cached] {ground_width_m:.1f}m x {ground_height_m:.1f}m", flush=True)
                            continue
                except Exception as e:
                    print(f"⚠️ Could not read world file for {fname}, will re-render: {e}", flush=True)

            # default: point buffer bbox
            geom_pt = feat.geometry()
            geom_pt.transform(xf_pt_to_wms)
            bbox = geom_pt.buffer(args.buffer_m, 50).boundingBox()
            extent_mode = "point_buffer"

            # If enabled + footprint found -> use footprint bbox
            if args.use_footprint_extent and footprints and xf_fp_to_wms and oid is not None:
                key = str(oid)
                if key in footprints_by_id:
                    bb = footprints_by_id[key]  # bbox in EPSG:4326 (or footprints CRS)
                    bb_wms = xf_fp_to_wms.transformBoundingBox(bb)
                    bbox = bb_wms
                    extent_mode = "footprint_bbox"

            # finalize extent (padding + min size)
            bbox = finalize_extent(bbox, args.context_pad_m, args.min_extent_m)
            ground_width_m = bbox.xMaximum() - bbox.xMinimum()
            ground_height_m = bbox.yMaximum() - bbox.yMinimum()

            # render (SEQUENTIAL job – stable in headless)
            ms.setExtent(bbox)
            job = QgsMapRendererSequentialJob(ms)
            job.start()
            job.waitForFinished()
            job.renderedImage().save(str(out_png), "PNG")

            # world file (.pngw)
            px_w = ground_width_m / args.img_size_px
            px_h = -(ground_height_m / args.img_size_px)
            ulx = bbox.xMinimum() + abs(px_w) / 2
            uly = bbox.yMaximum() + px_h / 2

            with open(world_path, "w") as wf:
                wf.write("\n".join([
                    f"{px_w:.10f}",
                    "0.0",
                    "0.0",
                    f"{px_h:.10f}",
                    f"{ulx:.10f}",
                    f"{uly:.10f}",
                ]))

            meters_per_px = (abs(px_w) + abs(px_h)) / 2

            rows.append({
                "osm_id": str(oid) if oid is not None else "",
                "image": fname,
                "ground_width_m": ground_width_m,
                "ground_height_m": ground_height_m,
                "m_per_px_x": abs(px_w),
                "m_per_px_y": abs(px_h),
                "meters_per_px": meters_per_px,
                "extent_mode": extent_mode,
            })

            rendered_count += 1
            print(f"✓ {fname} [{extent_mode}] {ground_width_m:.1f}m x {ground_height_m:.1f}m", flush=True)

        with open(csv_path, "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\n=== Done ===", flush=True)
        print(f"📊 Rendered: {rendered_count} | Skipped: {skipped_count} | Total: {len(rows)}", flush=True)
        print(f"📁 Images & metrics CSV saved to: {csv_path}", flush=True)

    finally:
        qgs.exitQgis()


if __name__ == "__main__":
    main()
    