#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import pandas as pd

# Canonical ID parser (digits-only)
from src.helpers.osm_ids import parse_osm_id_digits


def parse_args():
    ap = argparse.ArgumentParser(description="Convert input JSON into canonical substations manifest CSV.")
    ap.add_argument("--in_json", required=True, help="Input JSON (list of dicts or GeoJSON FeatureCollection)")
    ap.add_argument("--out_csv", required=True, help="Output CSV path (canonical manifest)")
    ap.add_argument("--id_field", default="Id", help="Default ID field name (if present)")
    ap.add_argument("--lat_field", default="Latitude", help="Default latitude field name (if present)")
    ap.add_argument("--lon_field", default="Longitude", help="Default longitude field name (if present)")
    ap.add_argument("--voltage_field", default="Spannung", help="Default voltage string field name (if present)")
    return ap.parse_args()


def _extract_lat_lon(rec: Dict[str, Any], lat_field: str, lon_field: str) -> Tuple[Optional[float], Optional[float]]:
    # 1) direct fields
    for lf in (lat_field, "Latitude", "lat", "Lat", "LAT"):
        if lf in rec:
            try:
                lat = float(rec[lf])
                break
            except Exception:
                lat = None
                break
    else:
        lat = None

    for lf in (lon_field, "Longitude", "lon", "Lon", "LON", "Lng", "lng"):
        if lf in rec:
            try:
                lon = float(rec[lf])
                break
            except Exception:
                lon = None
                break
    else:
        lon = None

    if lat is not None and lon is not None:
        return lat, lon

    # 2) "Coord" fields like "52.5, 13.4" or "52.5 13.4"
    for k in ("Coord", "coord", "coordinates", "Coords", "coords"):
        if k in rec and rec[k] is not None:
            s = str(rec[k])
            nums = re.findall(r"-?\d+(?:\.\d+)?", s)
            if len(nums) >= 2:
                try:
                    lat = float(nums[0])
                    lon = float(nums[1])
                    return lat, lon
                except Exception:
                    pass

    # 3) GeoJSON-like embedded geometry in the record
    geom = rec.get("geometry")
    if isinstance(geom, dict) and "coordinates" in geom:
        coords = geom["coordinates"]
        # Point: [lon, lat]
        if isinstance(coords, (list, tuple)) and len(coords) >= 2 and all(isinstance(x, (int, float)) for x in coords[:2]):
            return float(coords[1]), float(coords[0])

    return None, None


def _voltage_str_from_kv_flags(rec: Dict[str, Any]) -> Optional[str]:
    # Your pattern: KV110=True etc.
    kvs: List[int] = []
    for k, v in rec.items():
        if not (isinstance(k, str) and k.startswith("KV")):
            continue
        if v is True or v == 1 or str(v).lower() == "true":
            try:
                kvs.append(int(k[2:]))
            except Exception:
                pass
    kvs = sorted(set(kvs), reverse=True)
    return "/".join(str(x) for x in kvs) if kvs else None


def _extract_voltage(rec: Dict[str, Any], voltage_field: str) -> Optional[str]:
    # Prefer explicit voltage string if present
    for vf in (voltage_field, "Spannung", "voltage", "Voltage"):
        if vf in rec and rec[vf] is not None:
            s = str(rec[vf]).strip()
            if s:
                return s
    # Otherwise derive from KV flags
    return _voltage_str_from_kv_flags(rec)


def _extract_id(rec: Dict[str, Any], id_field: str) -> Optional[int]:
    # try a few common keys
    for k in (id_field, "Id", "id", "osm_id", "OSM ID", "osmId", "osmID"):
        if k in rec:
            oid = parse_osm_id_digits(rec.get(k))
            if oid is not None:
                return oid
    return None


def _read_any_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    # GeoJSON FeatureCollection
    if isinstance(data, dict) and data.get("type") == "FeatureCollection" and isinstance(data.get("features"), list):
        out = []
        for f in data["features"]:
            if not isinstance(f, dict):
                continue
            props = f.get("properties") if isinstance(f.get("properties"), dict) else {}
            geom = f.get("geometry") if isinstance(f.get("geometry"), dict) else None
            rec = dict(props)
            if geom:
                rec["geometry"] = geom
            out.append(rec)
        return out

    # plain list of dicts
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    # dict containing list under some key
    for key in ("items", "data", "records"):
        if isinstance(data, dict) and isinstance(data.get(key), list):
            return [x for x in data[key] if isinstance(x, dict)]

    raise RuntimeError("Unsupported JSON format. Expected a list of dicts or GeoJSON FeatureCollection.")


def main():
    args = parse_args()
    in_path = Path(args.in_json)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = _read_any_json(in_path)

    rows = []
    for rec in records:
        osm_id = _extract_id(rec, args.id_field)
        lat, lon = _extract_lat_lon(rec, args.lat_field, args.lon_field)
        voltage_str = _extract_voltage(rec, args.voltage_field)

        # Keep only rows we can actually use downstream
        if osm_id is None or lat is None or lon is None:
            continue

        rows.append(
            {
                "osm_id": int(osm_id),            # canonical join key (digits-only)
                "Latitude": float(lat),
                "Longitude": float(lon),
                "voltage_str": voltage_str or "", # optional, but necessary for capacity step
                "osm_ref": str(osm_id),         # preserve original OSM ref
            }
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["osm_id"])
    df.to_csv(out_path, index=False)
    print(f"✅ wrote: {out_path} (rows={len(df)})")


if __name__ == "__main__":
    main()