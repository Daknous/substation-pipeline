#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import time
import argparse
import signal
import contextlib
from pathlib import Path
from typing import Optional, Tuple, Any, List, Dict

import pandas as pd
import overpy
from shapely.geometry import Polygon, MultiPolygon, Point, mapping
from shapely.ops import unary_union

from src.helpers.osm_ids import parse_osm_id_digits

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--mapping_csv", required=True, help="CSV with Latitude/Longitude and Id/osm_id")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--radius_m", type=int, default=600)
    ap.add_argument("--rate_limit_sec", type=float, default=1.2)

    ap.add_argument("--cache_json", default=None)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--retry_backoff_sec", type=float, default=2.0)

    ap.add_argument(
        "--overpass_urls",
        default=None,
        help="Comma-separated list of Overpass interpreter URLs. "
             "If omitted, uses $OVERPASS_URLS or $OVERPASS_URL, then defaults.",
    )
    ap.add_argument(
        "--fail_soft",
        action="store_true",
        help="Do not fail the container if Overpass is down/overloaded. "
             "Writes outputs with found=false and continues.",
    )
    ap.add_argument(
        "--request_timeout_sec",
        type=int,
        default=60,
        help="Hard client-side timeout per Overpass HTTP request (seconds).",
    )
    ap.add_argument("--max_rows", type=int, default=None)

    # ✅ New: resume/short-circuit behavior
    ap.add_argument(
        "--resume",
        action="store_true",
        help="If footprints.csv exists, only fetch missing osm_ids and append/merge.",
    )
    ap.add_argument(
        "--require_found",
        action="store_true",
        help="When resuming, treat an osm_id as 'covered' only if found==True and footprint_wkt is non-empty.",
    )

    ap.add_argument("--progress_every", type=int, default=50)

    return ap.parse_args()


# -----------------------------------------------------------------------------
# Hard timeout wrapper (Linux containers)
# -----------------------------------------------------------------------------
class RequestTimeout(Exception):
    pass


@contextlib.contextmanager
def alarm_timeout(seconds: int):
    if not seconds or seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise RequestTimeout(f"Overpass request exceeded {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


# -----------------------------------------------------------------------------
# Overpass endpoint handling (fallback)
# -----------------------------------------------------------------------------
DEFAULT_OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.osm.jp/api/interpreter",
]


def _split_csv_urls(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [u.strip() for u in s.split(",") if u.strip()]


def build_overpass_url_list(args) -> List[str]:
    urls: List[str] = []
    urls += _split_csv_urls(getattr(args, "overpass_urls", None))
    urls += _split_csv_urls(os.environ.get("OVERPASS_URLS"))
    env_single = (os.environ.get("OVERPASS_URL") or "").strip()
    if env_single:
        urls.append(env_single)
    urls += DEFAULT_OVERPASS_URLS

    out: List[str] = []
    seen = set()
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


class OverpassClient:
    def __init__(self, urls: List[str], request_timeout_sec: int):
        self.urls = urls
        self.request_timeout_sec = request_timeout_sec
        self._last_good_url: Optional[str] = None

    def _api(self, url: str) -> overpy.Overpass:
        return overpy.Overpass(url=url)

    def query_with_fallback(self, query: str, retries: int, backoff_sec: float):
        last_err: Optional[Exception] = None

        urls = self.urls[:]
        if self._last_good_url and self._last_good_url in urls:
            urls = [self._last_good_url] + [u for u in urls if u != self._last_good_url]

        for url in urls:
            api = self._api(url)
            for attempt in range(retries):
                try:
                    with alarm_timeout(self.request_timeout_sec):
                        res = api.query(query)
                    self._last_good_url = url
                    return res
                except Exception as e:
                    last_err = e
                    sleep_s = backoff_sec * (2 ** attempt)
                    print(f"⚠️ Overpass failed on {url} (attempt {attempt+1}/{retries}): {type(e).__name__}: {e}")
                    print(f"   sleeping {sleep_s:.1f}s then retrying…")
                    time.sleep(sleep_s)

            print(f"⚠️ Switching Overpass endpoint after failures on: {url}")

        raise last_err if last_err else RuntimeError("All Overpass endpoints failed (unknown error).")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def norm_osm_id(x) -> str:
    oid = parse_osm_id_digits(x)
    return str(oid) if oid is not None else ""


def parse_osm_type(val) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip().lower()
    if not s:
        return None

    for t in ("way", "relation", "node"):
        if s.startswith(f"{t}/") or s.startswith(f"{t}_") or s.startswith(f"{t}-"):
            return t

    if "/" in s:
        head = s.split("/", 1)[0].strip()
        if head in ("way", "relation", "node"):
            return head

    return None


def parse_input_osm(val) -> Tuple[Optional[int], Optional[str], str]:
    raw = "" if val is None else str(val).strip()
    return parse_osm_id_digits(raw), parse_osm_type(raw), raw


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def close_ring(coords):
    if coords and coords[0] != coords[-1]:
        coords = coords + [coords[0]]
    return coords


def way_to_polygon(way):
    coords = [(float(n.lon), float(n.lat)) for n in way.nodes]
    coords = close_ring(coords)
    if len(coords) < 4:
        return None
    poly = Polygon(coords)
    poly = poly if poly.is_valid and not poly.is_empty else poly.buffer(0)
    return poly if (poly is not None and not poly.is_empty) else None


def relation_to_geometry(rel):
    outers, inners = [], []
    for m in rel.members:
        if m.role not in ("outer", "inner"):
            continue
        elem = m.resolve()
        if elem.__class__.__name__ != "Way":
            continue
        poly = way_to_polygon(elem)
        if not poly:
            continue
        (outers if m.role == "outer" else inners).append(poly)

    if not outers:
        return None

    geom = unary_union(outers)
    for hole in inners:
        try:
            geom = geom.difference(hole)
        except Exception:
            pass

    if isinstance(geom, (Polygon, MultiPolygon)) and not geom.is_empty:
        geom = geom.buffer(0)
        return geom if not geom.is_empty else None
    return None


# -----------------------------------------------------------------------------
# Overpass fetching
# -----------------------------------------------------------------------------
def fetch_by_id(client: OverpassClient, osm_id_digits: int, osm_type: Optional[str], retries=3, backoff_sec=2.0):
    types = [osm_type] if osm_type in ("way", "relation") else ["way", "relation"]

    for t in types:
        for tagged in (True, False):
            filt = '["power"="substation"]' if tagged else ""
            q = f"""
            [out:json][timeout:25];
            {t}({osm_id_digits}){filt};
            (._;>;);
            out body;
            """
            res = client.query_with_fallback(q, retries=retries, backoff_sec=backoff_sec)

            if res.ways:
                poly = way_to_polygon(res.ways[0])
                if poly:
                    return poly, f"by_id_{t}_{'tagged' if tagged else 'any'}", f"way/{res.ways[0].id}"

            if res.relations:
                geom = relation_to_geometry(res.relations[0])
                if geom:
                    return geom, f"by_id_{t}_{'tagged' if tagged else 'any'}", f"relation/{res.relations[0].id}"

    return None, None, None


def fetch_around(client: OverpassClient, lat, lon, radius_m, retries=3, backoff_sec=2.0):
    q = f"""
    [out:json][timeout:25];
    (
      way["power"="substation"](around:{radius_m},{lat},{lon});
      relation["power"="substation"](around:{radius_m},{lat},{lon});
    );
    out body;
    >;
    out skel qt;
    """
    res = client.query_with_fallback(q, retries=retries, backoff_sec=backoff_sec)
    pt = Point(float(lon), float(lat))

    for way in res.ways:
        poly = way_to_polygon(way)
        if poly and (poly.contains(pt) or poly.buffer(0).contains(pt)):
            return poly, "around_way", f"way/{way.id}"

    for rel in res.relations:
        geom = relation_to_geometry(rel)
        if geom and (geom.contains(pt) or geom.buffer(0).contains(pt)):
            return geom, "around_relation", f"relation/{rel.id}"

    return None, None, None


# -----------------------------------------------------------------------------
# Existing footprints handling (resume)
# -----------------------------------------------------------------------------
def load_existing_footprints(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"⚠️ Could not read existing footprints.csv ({csv_path}): {e}")
        return pd.DataFrame()


def covered_ids_from_existing(existing: pd.DataFrame, require_found: bool) -> set[str]:
    if existing.empty:
        return set()

    # Accept both schemas:
    # - new: osm_id, footprint_wkt, found, ...
    # - old: osm_ref, footprint_wkt, found, ...
    id_col = "osm_id" if "osm_id" in existing.columns else ("osm_ref" if "osm_ref" in existing.columns else None)
    if not id_col:
        return set()

    tmp = existing.copy()
    tmp["__id"] = tmp[id_col].astype(str).map(norm_osm_id)
    tmp = tmp[tmp["__id"].ne("")]

    if not require_found:
        return set(tmp["__id"].tolist())

    found_col = "found" if "found" in tmp.columns else None
    wkt_col = "footprint_wkt" if "footprint_wkt" in tmp.columns else None
    if not found_col or not wkt_col:
        return set()

    ok = tmp[
        tmp[found_col].astype(str).str.lower().isin(["true", "1"]) &
        tmp[wkt_col].astype(str).str.len().gt(0)
    ]
    return set(ok["__id"].tolist())


def merge_and_write(existing: pd.DataFrame, new_rows: List[Dict[str, Any]], out_csv: Path):
    new_df = pd.DataFrame(new_rows)

    if existing.empty:
        merged = new_df
    else:
        merged = pd.concat([existing, new_df], ignore_index=True)

    # normalize id and de-dup
    id_col = "osm_id" if "osm_id" in merged.columns else ("osm_ref" if "osm_ref" in merged.columns else None)
    if id_col:
        merged["__id"] = merged[id_col].astype(str).map(norm_osm_id)
        merged = merged[merged["__id"].ne("")].copy()
        merged = merged.drop_duplicates(subset=["__id"], keep="last").drop(columns=["__id"])
    else:
        merged = merged.drop_duplicates(keep="last")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return merged


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    urls = build_overpass_url_list(args)
    print("🔎 Overpass endpoints (in order):")
    for u in urls:
        print(f"  - {u}")
    print(
        f"⏱️ request_timeout_sec={args.request_timeout_sec}s "
        f"retries={args.retries} backoff={args.retry_backoff_sec}s rate_limit={args.rate_limit_sec}s "
        f"resume={args.resume} require_found={args.require_found}"
    )

    client = OverpassClient(urls, request_timeout_sec=args.request_timeout_sec)

    df = pd.read_csv(args.mapping_csv)
    if args.max_rows:
        df = df.head(args.max_rows).copy()
        print(f"🧪 max_rows enabled -> processing {len(df)} rows")

    osm_col = "osm_id" if "osm_id" in df.columns else ("Id" if "Id" in df.columns else "id")
    lat_col = "Latitude" if "Latitude" in df.columns else "lat"
    lon_col = "Longitude" if "Longitude" in df.columns else "lon"

    # Normalize ids in mapping
    df["osm_id"] = df[osm_col].astype(str).map(norm_osm_id)
    df = df[df["osm_id"].ne("")].copy()

    # Resume / short-circuit
    footprints_csv = out_dir / "footprints.csv"
    existing = load_existing_footprints(footprints_csv) if args.resume else pd.DataFrame()
    covered = covered_ids_from_existing(existing, require_found=args.require_found) if args.resume else set()

    if args.resume and covered:
        before = len(df)
        df = df[~df["osm_id"].isin(covered)].copy()
        after = len(df)
        print(f"↩️ resume: covered={len(covered)} -> remaining_to_fetch={after} (from {before})")

    if args.resume and df.empty:
        print("✅ footprints.csv already covers all requested substations -> nothing to do.")
        return

    # Cache (still useful to avoid repeated calls inside this run)
    cache_path = Path(args.cache_json) if args.cache_json else (out_dir / "footprint_cache.json")
    cache: dict[str, Any] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
            print(f"📦 loaded cache: {cache_path} (keys={len(cache)})")
        except Exception:
            cache = {}

    new_rows: List[Dict[str, Any]] = []
    features = []
    n_errors = 0

    it = df.iterrows()
    if tqdm is not None:
        it = tqdm(it, total=len(df), desc="OSM footprints", unit="substation")

    for _, r in it:
        osm_id_digits, osm_type, osm_ref_raw = parse_input_osm(r.get(osm_col))
        osm_id_str = r.get("osm_id") or ""

        try:
            lat = float(r.get(lat_col))
            lon = float(r.get(lon_col))
        except Exception:
            lat, lon = float("nan"), float("nan")

        has_coords = (lat == lat) and (lon == lon)

        # cache key: prefer digits id, else coords, else raw
        cache_key = str(osm_id_digits) if osm_id_digits else (f"{lat:.6f},{lon:.6f}" if has_coords else osm_ref_raw)

        if cache_key in cache:
            item = cache[cache_key]
            # Skip if this cache entry has no actual footprint data
            if not item.get("footprint_wkt"):
                continue  # Don't add empty rows that would overwrite good data
            
            # convert cache record to output row shape (ensure osm_id exists)
            out_row = {
                "osm_id": osm_id_str or norm_osm_id(item.get("osm_ref", "")),
                "footprint_wkt": item.get("footprint_wkt", ""),
                "found": bool(item.get("found", False)),
                "method": item.get("method", ""),
                "note": item.get("note", ""),
                "estimated_size_m": item.get("estimated_size_m", ""),
                "Latitude": item.get("Latitude", ""),
                "Longitude": item.get("Longitude", ""),
                "Name": item.get("Name", ""),
                "Voltages": item.get("Voltages", ""),
            }
            new_rows.append(out_row)
            if item.get("found") and item.get("geojson"):
                features.append(item["geojson"])
            continue

        geom, method, matched_ref = (None, None, None)

        try:
            if osm_id_digits:
                geom, method, matched_ref = fetch_by_id(
                    client, osm_id_digits, osm_type, retries=args.retries, backoff_sec=args.retry_backoff_sec
                )

            if geom is None and has_coords:
                geom, method, matched_ref = fetch_around(
                    client, lat, lon, args.radius_m, retries=args.retries, backoff_sec=args.retry_backoff_sec
                )

        except Exception as e:
            n_errors += 1
            geom = None
            matched_ref = None
            method = f"error:{type(e).__name__}"
            print(f"⚠️ OSM fetch failed for {cache_key}: {type(e).__name__}: {e}")
            if not args.fail_soft:
                raise

        found = geom is not None
        wkt = geom.wkt if found else ""
        matched_osm_id = norm_osm_id(matched_ref) if matched_ref else ""

        # output row in your “new footprints.csv” schema
        out_row = {
            "osm_id": osm_id_str,
            "footprint_wkt": wkt,
            "found": bool(found),
            "method": method or "",
            "note": "",               # keep for future (e.g., voltage fallback)
            "estimated_size_m": "",   # keep for future
            "Latitude": lat if has_coords else "",
            "Longitude": lon if has_coords else "",
            "Name": r.get("Name", "") if "Name" in df.columns else "",
            "Voltages": r.get("Voltages", "") if "Voltages" in df.columns else "",
        }
        new_rows.append(out_row)

        if found:
            feat = {
                "type": "Feature",
                "properties": {
                    "osm_id": osm_id_str,
                    "matched_ref": matched_ref or "",
                    "matched_osm_id": matched_osm_id,
                    "found": True,
                    "method": method or "",
                },
                "geometry": mapping(geom),
            }
            features.append(feat)

        # cache record (store both the old-ish keys + your new schema)
        cache[cache_key] = {
            "osm_ref": str(osm_id_digits) if osm_id_digits is not None else "",
            "osm_ref_raw": osm_ref_raw,
            "matched_ref": matched_ref or "",
            "matched_osm_id": matched_osm_id,
            "Latitude": out_row["Latitude"],
            "Longitude": out_row["Longitude"],
            "found": bool(found),
            "method": method or "",
            "footprint_wkt": wkt,
            "note": out_row["note"],
            "estimated_size_m": out_row["estimated_size_m"],
            "Name": out_row["Name"],
            "Voltages": out_row["Voltages"],
            "geojson": feat if found else None,
        }

        if tqdm is None and (len(new_rows) % args.progress_every == 0):
            print(f"… progress {len(new_rows)}/{len(df)} (errors={n_errors})")

        time.sleep(args.rate_limit_sec)

    # Merge with existing + write footprints.csv
    merged = merge_and_write(existing, new_rows, footprints_csv)
    print(f"✅ wrote: {footprints_csv} (rows={len(merged)}, new_rows={len(new_rows)}, errors={n_errors})")

    # Write geojson (only new geometries, safe + small)
    geojson_path = out_dir / "substations.geojson"
    geojson = {"type": "FeatureCollection", "features": features}
    geojson_path.write_text(json.dumps(geojson))
    print(f"✅ wrote: {geojson_path} (features={len(features)})")

    # Write cache
    cache_path.write_text(json.dumps(cache))
    print(f"✅ cache: {cache_path} (keys={len(cache)})")


if __name__ == "__main__":
    main()
    