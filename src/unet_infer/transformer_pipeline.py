import os
import re
import glob
import csv
import json
import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from skimage import filters, measure, feature, draw
from skimage.segmentation import watershed, relabel_sequential, find_boundaries
from skimage.morphology import remove_small_objects, binary_closing, binary_dilation, disk
from scipy.ndimage import distance_transform_edt as edt
from skimage.transform import resize as sk_resize

from src.helpers.osm_ids import parse_osm_id_digits

warnings.filterwarnings("ignore")

try:
    from shapely.geometry import Polygon
    _HAVE_SHAPELY = True
except Exception:
    _HAVE_SHAPELY = False


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("transformer-pipeline")



# =============================================================================
# Postprocessing pipeline
# =============================================================================
class TransformerSegmentationPipeline:
    def __init__(self, constraints=None, debug_bridge=False):
        self.constraints = constraints or {
            "min_transformer_m2": 50.0,
            "typical_transformer_m2": 150.0,
            "max_transformer_m2": 600.0,
            "min_separation_m": 5.0,
            "typical_separation_m": 10.0,
            "fragment_threshold_m2": 50.0,
            "proximity_threshold_m": 3.0,
            "max_merge_distance_m": 15.0,
        }
        self.debug_bridge = bool(debug_bridge)

    def _m2_to_px(self, m2, area_per_px_m2):
        if not np.isfinite(area_per_px_m2) or area_per_px_m2 <= 0:
            return 40
        return max(1, int(round(m2 / area_per_px_m2)))

    def _adaptive_threshold(self, probs):
        from skimage.filters import threshold_otsu
        p = probs[np.isfinite(probs)]
        if p.size < 100:
            return 0.15
        try:
            t = threshold_otsu(p)
            return float(np.clip(t, 0.08, 0.35))
        except Exception:
            return 0.15

    def _create_mask(self, probs, threshold, area_per_px_m2):
        mask = (probs > threshold).astype(np.uint8)
        min_area_m2 = max(0.25 * self.constraints["min_transformer_m2"], 20.0)
        min_area_px = self._m2_to_px(min_area_m2, area_per_px_m2)
        return remove_small_objects(mask.astype(bool), min_size=min_area_px).astype(np.uint8)

    def _generate_seeds(self, probs, mask_lo, area_per_px_m2):
        if not mask_lo.any():
            return np.zeros_like(mask_lo, dtype=np.int32)

        p_masked = probs[mask_lo.astype(bool)]
        p_max = p_masked.max() if p_masked.size > 0 else 0
        if p_max > 0.8:
            sigma, min_dist_factor = 0.5, 0.8
        elif p_max < 0.5:
            sigma, min_dist_factor = 1.5, 1.5
        else:
            sigma, min_dist_factor = 1.0, 1.0

        p_smooth = filters.gaussian(probs, sigma=sigma, preserve_range=True)

        if np.isfinite(area_per_px_m2) and area_per_px_m2 > 0:
            typical_sep_m = self.constraints["typical_separation_m"] * min_dist_factor
            min_dist_px = int(np.clip(typical_sep_m / np.sqrt(area_per_px_m2), 5, 40))
        else:
            min_dist_px = 20

        coords = feature.peak_local_max(
            p_smooth,
            labels=mask_lo.astype(np.uint8),
            min_distance=min_dist_px,
            exclude_border=False,
            threshold_rel=0.3,
        )
        markers = np.zeros_like(mask_lo, dtype=np.int32)
        for i, (r, c) in enumerate(coords, 1):
            markers[r, c] = i
        return markers

    def _watershed_segmentation(self, probs, mask_lo, seeds):
        if seeds.max() > 0:
            gradient = filters.sobel(filters.gaussian(probs, sigma=1.0))
            return watershed(gradient, markers=seeds, mask=mask_lo.astype(bool))
        return measure.label(mask_lo, connectivity=2)

    def _conservative_merge(self, labeled, probs, area_per_px_m2):
        if labeled.max() < 2:
            return labeled
        props = measure.regionprops(labeled, intensity_image=probs)
        px_per_m = 1.0 / np.sqrt(area_per_px_m2) if area_per_px_m2 > 0 else 30
        merge_distance_px = self.constraints["proximity_threshold_m"] * px_per_m
        edges = filters.sobel(probs)

        merged = labeled.copy()
        for i, p1 in enumerate(props):
            for p2 in props[i + 1:]:
                dist = np.hypot(p1.centroid[0] - p2.centroid[0], p1.centroid[1] - p2.centroid[1])
                if dist >= merge_distance_px:
                    continue
                rr, cc = draw.line(int(p1.centroid[0]), int(p1.centroid[1]),
                                   int(p2.centroid[0]), int(p2.centroid[1]))
                valid = (0 <= rr) & (rr < edges.shape[0]) & (0 <= cc) & (cc < edges.shape[1])
                rr, cc = rr[valid], cc[valid]
                if rr.size == 0:
                    continue
                edge_strength = edges[rr, cc].mean()
                prob_continuity = probs[rr, cc].min() / max(p1.mean_intensity, p2.mean_intensity, 1e-6)
                if edge_strength < 0.2 and prob_continuity > 0.7:
                    if p1.area >= p2.area:
                        merged[merged == p2.label] = p1.label
                    else:
                        merged[merged == p1.label] = p2.label
        return relabel_sequential(merged)[0]

    def _aggressive_merge_nearby(self, labeled, area_per_px_m2):
        if labeled.max() < 2:
            return labeled
        px_per_m = 1.0 / np.sqrt(area_per_px_m2) if area_per_px_m2 > 0 else 30
        max_dist = 2.5 * px_per_m

        merged = labeled.copy()
        props = {p.label: p for p in measure.regionprops(merged)}
        groups, used = [], set()

        for l1, p1 in props.items():
            if l1 in used:
                continue
            group = {l1}
            used.add(l1)
            for l2, p2 in props.items():
                if l2 in used or l2 == l1:
                    continue
                d = np.hypot(p1.centroid[0] - p2.centroid[0], p1.centroid[1] - p2.centroid[1])
                if d < max_dist:
                    ang = abs(np.arctan2(p2.centroid[0] - p1.centroid[0], p2.centroid[1] - p1.centroid[1]))
                    if ang < np.pi/6 or ang > 5*np.pi/6 or (np.pi/3 < ang < 2*np.pi/3):
                        group.add(l2); used.add(l2)
            groups.append(group)

        for g in groups:
            if len(g) > 1:
                base = list(g)[0]
                for l in list(g)[1:]:
                    merged[merged == l] = base
        return relabel_sequential(merged)[0]

    def _bridge_merge(self, labeled, probs, area_per_px_m2, base_threshold=0.15):
        if labeled.max() < 2:
            return labeled

        H, W = probs.shape
        lab = labeled.copy()

        px_per_m = 1.0 / np.sqrt(area_per_px_m2) if (np.isfinite(area_per_px_m2) and area_per_px_m2 > 0) else 30.0
        mpp = 1.0 / px_per_m

        near_centroid_m = 6.0
        touch_dilate_rad_m = 0.6
        closing_radius_m = 0.7

        near_centroid_px = near_centroid_m * px_per_m
        touch_dilate_rad = max(1, int(round(touch_dilate_rad_m * px_per_m)))

        sob = filters.sobel(filters.gaussian(probs, sigma=1.0))
        def _dil(mask): return binary_dilation(mask, disk(touch_dilate_rad))

        merged_any = True
        while merged_any:
            merged_any = False
            props = {r.label: r for r in measure.regionprops(lab)}
            labels = list(props.keys())
            dilated = {lbl: _dil(lab == lbl) for lbl in labels}

            for i in range(len(labels)):
                li = labels[i]
                if li not in props:
                    continue
                pi = props[li]

                for j in range(i + 1, len(labels)):
                    lj = labels[j]
                    if lj not in props:
                        continue
                    pj = props[lj]

                    d_cent = np.hypot(pi.centroid[0] - pj.centroid[0], pi.centroid[1] - pj.centroid[1])
                    if d_cent > near_centroid_px:
                        continue
                    if not (dilated[li] & dilated[lj]).any():
                        continue

                    dmap = edt(~(lab == li))
                    try:
                        min_gap_px = float(dmap[lab == lj].min())
                    except ValueError:
                        min_gap_px = float("inf")

                    adapt_close = max(1, int(round(max(closing_radius_m, min_gap_px * mpp * 0.8) * px_per_m)))

                    minr = max(0, min(pi.bbox[0], pj.bbox[0]) - adapt_close - 1)
                    minc = max(0, min(pi.bbox[1], pj.bbox[1]) - adapt_close - 1)
                    maxr = min(H, max(pi.bbox[2], pj.bbox[2]) + adapt_close + 1)
                    maxc = min(W, max(pi.bbox[3], pj.bbox[3]) + adapt_close + 1)

                    roi = lab[minr:maxr, minc:maxc]
                    roi_pairs = ((roi == li) | (roi == lj))
                    roi_closed = binary_closing(roi_pairs, disk(adapt_close))
                    cc_before = measure.label(roi_pairs, connectivity=2).max()
                    cc_after = measure.label(roi_closed, connectivity=2).max()
                    closed_connects = (cc_before >= 2 and cc_after == 1)

                    rr, cc = draw.line(int(pi.centroid[0]), int(pi.centroid[1]),
                                       int(pj.centroid[0]), int(pj.centroid[1]))
                    rr = np.clip(rr, 0, H - 1); cc = np.clip(cc, 0, W - 1)
                    los_min_p = float(probs[rr, cc].min()) if rr.size else 0.0
                    los_edge = float(sob[rr, cc].mean()) if rr.size else 1.0
                    los_pass = (los_min_p >= max(0.18, base_threshold * 0.9) and los_edge <= 0.25)

                    if self.debug_bridge:
                        logger.info(
                            "[bridge] pair(%d,%d) d_cent=%.2f gap_px=%.2f close_r=%d closing=%s los=%s min_p=%.2f edge=%.2f",
                            li, lj, d_cent, min_gap_px, adapt_close, str(closed_connects), str(los_pass),
                            los_min_p, los_edge
                        )

                    if not (closed_connects or los_pass):
                        continue

                    lab[lab == lj] = li
                    merged_any = True
                    break

                if merged_any:
                    break

        return relabel_sequential(lab)[0]

    def _merge_fragments(self, labeled, area_per_px_m2):
        if labeled.max() < 2:
            return labeled

        lab = labeled.copy()
        props = {r.label: r for r in measure.regionprops(lab)}

        if np.isfinite(area_per_px_m2) and area_per_px_m2 > 0:
            px_per_m = 1.0 / np.sqrt(area_per_px_m2)
            max_dist_px = self.constraints["max_merge_distance_m"] * px_per_m
        else:
            max_dist_px = 50

        for lbl, p in props.items():
            area_m2 = p.area * area_per_px_m2 if np.isfinite(area_per_px_m2) else p.area
            if area_m2 >= self.constraints["fragment_threshold_m2"]:
                continue

            min_d, best = float("inf"), None
            for olbl, op in props.items():
                if olbl == lbl:
                    continue
                other_area_m2 = op.area * area_per_px_m2 if np.isfinite(area_per_px_m2) else op.area
                if other_area_m2 < self.constraints["min_transformer_m2"]:
                    continue
                d = np.hypot(p.centroid[0] - op.centroid[0], p.centroid[1] - op.centroid[1])
                if d < min_d and d < max_dist_px:
                    min_d, best = d, olbl

            lab[lab == lbl] = best if best is not None else 0
        return lab

    def _final_cleanup(self, labeled, area_per_px_m2):
        min_area_m2 = max(0.25 * self.constraints["min_transformer_m2"], 20.0)
        min_area_px = self._m2_to_px(min_area_m2, area_per_px_m2)
        labeled = remove_small_objects(labeled, min_size=min_area_px)
        return relabel_sequential(labeled)[0]

    def process(self, probs, area_per_px_m2):
        threshold = self._adaptive_threshold(probs)
        mask_lo = self._create_mask(probs, threshold, area_per_px_m2)
        seeds = self._generate_seeds(probs, mask_lo, area_per_px_m2)
        labeled = self._watershed_segmentation(probs, mask_lo, seeds)

        labeled = self._bridge_merge(labeled, probs, area_per_px_m2, base_threshold=threshold)
        labeled = self._conservative_merge(labeled, probs, area_per_px_m2)
        labeled = self._aggressive_merge_nearby(labeled, area_per_px_m2)
        labeled = self._merge_fragments(labeled, area_per_px_m2)
        labeled = self._final_cleanup(labeled, area_per_px_m2)

        info = {
            "threshold": float(threshold),
            "n_seeds": int(seeds.max()),
            "n_final": int(labeled.max()),
            "success": True,
        }
        return labeled, info


# =============================================================================
# Batch Processor (single-model inference)
# =============================================================================
class BatchTransformerProcessor:
    def __init__(self, output_dir, model, preprocess, device, allow_no_scale=True, debug_bridge=False, skip_existing=False):
        self.output_dir = output_dir
        self.overlay_dir = os.path.join(output_dir, "overlays")
        self.mask_dir = os.path.join(output_dir, "masks")
        self.csv_path = os.path.join(output_dir, "transformer_detections.csv")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.overlay_dir, exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)

        self.model = model
        self.preprocess = preprocess
        self.device = device
        self.allow_no_scale = bool(allow_no_scale)
        self.skip_existing = bool(skip_existing)

        # Two lookup tables:
        # - by osm_id digits (preferred)
        # - by image basename (fallback)
        self.scale_by_osm_id: Dict[str, float] = {}
        self.scale_by_image: Dict[str, float] = {}

        self.pipeline = TransformerSegmentationPipeline(debug_bridge=debug_bridge)
        self.results = []
        
        # Track processed images for skip logic
        self.processed_images: set = set()

    # ---------- world-file helpers ----------
    def find_world_file(self, image_path):
        root, _ = os.path.splitext(image_path)
        for ext in (".pgw", ".pngw", ".wld"):
            p = root + ext
            if os.path.exists(p):
                return p
        return None

    def _deg_to_m_scales(self, lat_deg):
        lat_rad = np.deg2rad(lat_deg)
        return 111_320.0 * np.cos(lat_rad), 110_574.0

    def read_world_file(self, world_file):
        try:
            with open(world_file, "r") as f:
                vals = [float(x.strip()) for x in f.readlines()[:6]]
            if len(vals) < 6:
                return None, None, {}

            A, D, B, E, C, F = vals
            looks_like_deg = (abs(A) < 1e-3 and abs(E) < 1e-3 and -180 <= C <= 180 and -90 <= F <= 90)

            if looks_like_deg:
                mpp_lon, mpp_lat = self._deg_to_m_scales(F)
                area_simple = abs(A) * mpp_lon * abs(E) * mpp_lat
                A_m, E_m = A * mpp_lon, E * mpp_lat
                B_m, D_m = B * mpp_lon, D * mpp_lat
                area_det = abs(A_m * E_m - B_m * D_m)
                world_has_rot = (abs(B) > 0 or abs(D) > 0)
            else:
                area_simple = abs(A) * abs(E)
                area_det = abs(A * E - B * D)
                world_has_rot = (abs(B) > 0 or abs(D) > 0)

            meta = {
                "scale_source": "world_file",
                "looks_like_degrees": bool(looks_like_deg),
                "pixel_area_simple_m2": float(area_simple),
                "pixel_area_det_m2": float(area_det),
                "flag_scale_disagreement": bool(area_simple > 0 and abs(area_det - area_simple) / area_simple > 0.10),
                "world_has_rotation": bool(world_has_rot),
                "A": A, "B": B, "D": D, "E": E, "C": C, "F": F,
            }
            return (A, D, B, E, C, F), float(area_simple), meta
        except Exception as e:
            logger.error(f"Failed to read world file {world_file}: {e}")
            return None, None, {}

    # ---------- ids ----------
    @staticmethod
    def extract_osm_id_digits(x):
        return parse_osm_id_digits(x)

    # ---------- scale manifest ----------
    @staticmethod
    def _float_or_none(v: Any) -> Optional[float]:
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            return float(v)
        except Exception:
            return None

    @classmethod
    def load_scale_manifest(cls, csv_path: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Reads QGIS CSV (image_meters_per_px.csv) and returns:
          (scale_by_osm_id, scale_by_image)
        Both values store: area_per_px_m2 (m_per_px_x * m_per_px_y).

        Supported columns (any subset):
          - osm_ref / osm_id / Id   (we parse digits from any)
          - image
          - m_per_px_x, m_per_px_y
          - meters_per_px  (fallback -> area = mpp^2)
        """
        by_osm: Dict[str, float] = {}
        by_img: Dict[str, float] = {}

        if not csv_path or not os.path.exists(csv_path):
            return by_osm, by_img

        with open(csv_path, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                img = (row.get("image") or "").strip()
                img_base = os.path.basename(img) if img else ""

                # Prefer anisotropic px sizes if present
                mx = cls._float_or_none(row.get("m_per_px_x"))
                my = cls._float_or_none(row.get("m_per_px_y"))
                mpp = cls._float_or_none(row.get("meters_per_px"))

                area_per_px = None
                if mx and my and mx > 0 and my > 0:
                    area_per_px = abs(mx) * abs(my)
                elif mpp and mpp > 0:
                    area_per_px = float(mpp) ** 2

                if not area_per_px or area_per_px <= 0:
                    continue

                # Get osm id (digits-only) from any reasonable field
                oid = (
                    parse_osm_id_digits(row.get("osm_ref"))
                    or parse_osm_id_digits(row.get("osm_id"))
                    or parse_osm_id_digits(row.get("Id"))
                )
                if oid is None and img_base:
                    oid = parse_osm_id_digits(img_base)

                if oid is not None:
                    by_osm[str(int(oid))] = float(area_per_px)

                if img_base:
                    by_img[img_base] = float(area_per_px)

        return by_osm, by_img

    # ---------- model ----------
    def run_inference_probs(self, image_pil):
        img_np = np.array(image_pil)
        H, W = img_np.shape[:2]

        x = self.preprocess(img_np)
        inp = torch.from_numpy(x.astype("float32")).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(inp)

        if isinstance(out, (tuple, list)):
            out = out[0]
        elif isinstance(out, dict):
            out = out.get("out", next(iter(out.values())))

        logits = out
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        if probs.shape != (H, W):
            probs = sk_resize(probs, (H, W), order=1, anti_aliasing=False, preserve_range=True)

        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(probs, 0.0, 1.0)

    def create_overlay(self, image_np, labeled_mask):
        overlay = image_np.copy()
        colors = np.array([
            [228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163],
            [255, 127, 0], [255, 255, 51], [166, 86, 40], [247, 129, 191],
            [153, 153, 153], [31, 120, 180], [178, 223, 138], [51, 160, 44],
            [251, 154, 153], [227, 26, 28], [253, 191, 111], [255, 127, 0],
            [202, 178, 214], [106, 61, 154], [255, 255, 153], [177, 89, 40]
        ])
        for lid in range(1, labeled_mask.max() + 1):
            mask = (labeled_mask == lid)
            overlay[mask] = overlay[mask] * 0.6 + colors[(lid - 1) % len(colors)] * 0.4
        overlay[find_boundaries(labeled_mask, mode="outer")] = [255, 255, 255]
        return overlay.astype(np.uint8)

    # ---------- per image ----------
    def process_single_image(self, image_path, metadata=None):
        metadata = metadata or {}
        filename = os.path.basename(image_path)

        osm_id = self.extract_osm_id_digits(filename)  # digits-only
        osm_key = str(osm_id) if osm_id is not None else None

        img_pil = Image.open(image_path).convert("RGB")
        img_np = np.array(img_pil)

        # scale resolution: pngw -> manifest(osm_id) -> manifest(image) -> none
        world_file = self.find_world_file(image_path)
        area_per_px_m2 = np.nan
        scale_meta: Dict[str, Any] = {"scale_source": None}

        # 1) World file
        if world_file:
            _, a2, meta = self.read_world_file(world_file)
            if a2 and a2 > 0:
                area_per_px_m2 = float(a2)
                scale_meta = meta

        # 2) Manifest by osm id
        if (not np.isfinite(area_per_px_m2) or area_per_px_m2 <= 0) and osm_key and self.scale_by_osm_id:
            a2 = self.scale_by_osm_id.get(osm_key)
            if a2 and a2 > 0:
                area_per_px_m2 = float(a2)
                scale_meta = {
                    "scale_source": "scale_manifest_osm_id",
                    "pixel_area_simple_m2": float(a2),
                    "pixel_area_det_m2": None,
                    "flag_scale_disagreement": None,
                    "world_has_rotation": None,
                    "looks_like_degrees": None,
                }

        # 3) Manifest by image basename
        if (not np.isfinite(area_per_px_m2) or area_per_px_m2 <= 0) and self.scale_by_image:
            a2 = self.scale_by_image.get(filename)
            if a2 and a2 > 0:
                area_per_px_m2 = float(a2)
                scale_meta = {
                    "scale_source": "scale_manifest_image",
                    "pixel_area_simple_m2": float(a2),
                    "pixel_area_det_m2": None,
                    "flag_scale_disagreement": None,
                    "world_has_rotation": None,
                    "looks_like_degrees": None,
                }

        scale_ok = bool(np.isfinite(area_per_px_m2) and area_per_px_m2 > 0)

        if not scale_ok and not self.allow_no_scale:
            logger.warning(f"Skipping {filename}: no scale and allow_no_scale=False")
            return

        probs = self.run_inference_probs(img_pil)
        labeled, info = self.pipeline.process(probs, area_per_px_m2 if scale_ok else np.nan)

        # always save overlay + mask
        base = os.path.splitext(filename)[0]
        overlay_path = os.path.join(self.overlay_dir, f"{base}_overlay.png")
        Image.fromarray(self.create_overlay(img_np, labeled)).save(overlay_path)
        np.save(os.path.join(self.mask_dir, f"{base}_labels.npy"), labeled.astype(np.int32))

        if labeled.max() == 0:
            return

        mpp = float(np.sqrt(area_per_px_m2)) if scale_ok else None

        # per component rows (area-dependent values become None if no scale)
        for r in measure.regionprops(labeled):
            area_px = int(r.area)
            area_m2 = float(area_px * area_per_px_m2) if scale_ok else None

            row = {
                # canonical id fields
                "osm_id": osm_id,
                "osm_id_str": osm_key,

                "image_name": filename,
                "image_path": image_path,
                "component_id": int(r.label),

                "area_px": area_px,
                "area_m2": area_m2,
                "meters_per_px": float(mpp) if mpp else None,

                "threshold": float(info.get("threshold", np.nan)),
                "n_components": int(labeled.max()),
                "scale_source": scale_meta.get("scale_source"),
                "world_file_found": bool(world_file is not None),

                # QA scale fields (kept, you can prune later)
                "pixel_area_simple_m2": scale_meta.get("pixel_area_simple_m2"),
                "pixel_area_det_m2": scale_meta.get("pixel_area_det_m2"),
                "flag_scale_disagreement": scale_meta.get("flag_scale_disagreement"),
                "world_has_rotation": scale_meta.get("world_has_rotation"),
                "world_is_degrees": scale_meta.get("looks_like_degrees"),

                "centroid_x_px": float(r.centroid[1]),
                "centroid_y_px": float(r.centroid[0]),
                "orientation_deg": (np.degrees(r.orientation) + 180.0) % 180.0 if hasattr(r, "orientation") else None,
            }

            if metadata:
                row.update(metadata)

            self.results.append(row)

    # ---------- directory ----------
    def process_directory(self, image_dir, pattern="*.png", metadata_json=None, save_interval=100, scale_manifest_csv=None):
        # scale manifest
        if scale_manifest_csv:
            self.scale_by_osm_id, self.scale_by_image = self.load_scale_manifest(scale_manifest_csv)
            logger.info(
                f"Loaded scale manifest: by_osm_id={len(self.scale_by_osm_id)} | by_image={len(self.scale_by_image)} from {scale_manifest_csv}"
            )
        else:
            self.scale_by_osm_id, self.scale_by_image = {}, {}

        # Load existing results to determine what to skip
        if self.skip_existing and os.path.exists(self.csv_path):
            try:
                existing_df = pd.read_csv(self.csv_path, comment='#')
                existing_images = set(existing_df['image_name'].unique())
                logger.info(f"⏭️  Skip mode: found {len(existing_images)} already-processed images in CSV")
                
                # Also check for existing overlays/masks
                self.processed_images = set()
                for img in existing_images:
                    base = os.path.splitext(img)[0]
                    overlay_path = os.path.join(self.overlay_dir, f"{base}_overlay.png")
                    mask_path = os.path.join(self.mask_dir, f"{base}_labels.npy")
                    
                    # Only skip if both overlay and mask exist
                    if os.path.exists(overlay_path) and os.path.exists(mask_path):
                        self.processed_images.add(img)
                
                logger.info(f"⏭️  Will skip {len(self.processed_images)} images with complete outputs")
            except Exception as e:
                logger.warning(f"Could not load existing CSV for skip logic: {e}")
                self.processed_images = set()
        else:
            self.processed_images = set()

        # metadata indexed by canonical digits-only osm_id
        metadata_dict: Dict[int, Dict[str, Any]] = {}
        if metadata_json and os.path.exists(metadata_json):
            try:
                with open(metadata_json, "r", encoding="utf-8") as f:
                    meta_list = json.load(f)

                for item in meta_list:
                    oid = parse_osm_id_digits(item.get("Id")) or parse_osm_id_digits(item.get("osm_ref")) or parse_osm_id_digits(item.get("osm_id"))
                    if oid is not None:
                        metadata_dict[int(oid)] = item

                logger.info(f"Loaded metadata for {len(metadata_dict)} substations from {metadata_json}.")
            except Exception as e:
                logger.error(f"Failed to load metadata JSON: {e}")

        paths = sorted(glob.glob(os.path.join(image_dir, pattern)))
        logger.info(f"Found {len(paths)} images to process in {image_dir} (pattern={pattern})")

        skipped_count = 0
        processed_count = 0

        for i, p in enumerate(tqdm(paths, desc="Processing images")):
            filename = os.path.basename(p)
            
            # Skip if already processed
            if self.skip_existing and filename in self.processed_images:
                skipped_count += 1
                continue
            
            oid = parse_osm_id_digits(filename)
            meta = metadata_dict.get(int(oid), {}) if oid is not None else {}
            self.process_single_image(p, meta)
            processed_count += 1

            if processed_count > 0 and processed_count % save_interval == 0:
                self.save_results(interim=True)
                logger.info(f"Interim save at {processed_count} processed images (skipped={skipped_count}).")

        self.save_results(interim=False)
        logger.info(f"✅ Complete: Processed={processed_count} | Skipped={skipped_count} | Total={len(paths)}")

    def save_results(self, interim: bool = False):
        final_csv = self.csv_path
        interim_csv = self.csv_path.replace(".csv", "_interim.csv")

        # Nothing new to write (but final may still need to consolidate interim)
        have_new = bool(self.results)
        have_interim = os.path.exists(interim_csv)

        if interim:
            if not have_new:
                logger.info("Interim save: nothing new to write.")
                return

            df_new = pd.DataFrame(self.results)

            # Ensure required columns exist (avoid KeyErrors later)
            required = ["osm_id", "image_name", "component_id"]
            for c in required:
                if c not in df_new.columns:
                    df_new[c] = np.nan

            # Append (create file if missing)
            write_header = not os.path.exists(interim_csv)
            df_new.to_csv(interim_csv, mode="a", index=False, header=write_header)

            logger.info(f"✅ Interim appended: +{len(df_new)} rows -> {interim_csv}")
            self.results = []
            return

        # --- FINAL SAVE ---
        # Load existing final (optional)
        dfs = []

        if os.path.exists(final_csv):
            try:
                dfs.append(pd.read_csv(final_csv, comment="#"))
            except Exception as e:
                logger.warning(f"Could not read existing final CSV '{final_csv}': {e}")

        # Load interim (optional)
        if have_interim:
            try:
                dfs.append(pd.read_csv(interim_csv))
            except Exception as e:
                logger.warning(f"Could not read interim CSV '{interim_csv}': {e}")

        # Add in-memory new rows
        if have_new:
            dfs.append(pd.DataFrame(self.results))

        if not dfs:
            # Nothing exists at all
            logger.warning("Final save: no existing final, no interim, no new results. Nothing to write.")
            return

        combined = pd.concat(dfs, ignore_index=True, sort=False)

        # Ensure stable key columns exist
        for c in ["osm_id", "image_name", "component_id"]:
            if c not in combined.columns:
                combined[c] = np.nan

        # Create a robust dedup key even if osm_id is missing
        # Prefer osm_id; else fallback to image_name.
        combined["_dedup_osm"] = combined["osm_id"].fillna("").astype(str)
        combined["_dedup_img"] = combined["image_name"].fillna("").astype(str)
        combined["_dedup_comp"] = combined["component_id"].fillna(-1).astype(int, errors="ignore")

        combined["_dedup_key"] = (
            combined["_dedup_osm"].where(combined["_dedup_osm"] != "", combined["_dedup_img"])
            + "|" + combined["_dedup_comp"].astype(str)
        )

        # Keep last (newest) occurrence
        combined = combined.drop_duplicates(subset=["_dedup_key"], keep="last").drop(columns=[
            "_dedup_osm", "_dedup_img", "_dedup_comp", "_dedup_key"
        ])

        # Write final with comment header
        header = (
            "# NOTE:\n"
            "# - If no scale was available, area_m2/meters_per_px are empty (pipeline still produces overlays + masks).\n"
        )
        tmp_path = final_csv + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(header)
        combined.to_csv(tmp_path, mode="a", index=False)

        os.replace(tmp_path, final_csv)

        # Clear memory buffer
        self.results = []

        # Optionally remove interim after successful consolidation
        if have_interim:
            try:
                os.remove(interim_csv)
                logger.info(f"🧹 Removed interim file: {interim_csv}")
            except Exception as e:
                logger.warning(f"Could not remove interim file '{interim_csv}': {e}")

        logger.info(f"✅ Final saved: {len(combined)} total rows -> {final_csv}")
