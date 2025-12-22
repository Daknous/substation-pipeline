#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import joblib

from src.helpers.osm_ids import parse_osm_id_digits
from src.capacity.preprocessing import extract_voltage_features


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--unet_csv", required=True, help="UNet output CSV (transformer_detections.csv)")
    ap.add_argument("--model_joblib", required=True, help="capacity_model.joblib")
    ap.add_argument("--model_config", required=True, help="model_config.json (feature_names order)")
    ap.add_argument("--out_csv", required=True, help="Output CSV (per transformer/component)")
    ap.add_argument("--summary_csv", required=True, help="Output CSV (per substation summary)")

    # Preferred source for voltage (your case)
    ap.add_argument(
        "--manifest_csv",
        required=False,
        default=None,
        help="Substation manifest CSV containing osm_id + voltage (recommended)"
    )
    ap.add_argument("--manifest_id_field", default=None, help="Override manifest id col (default: auto)")
    ap.add_argument("--manifest_voltage_field", default=None, help="Override manifest voltage col (default: auto)")

    # Optional legacy fallback source for voltage
    ap.add_argument(
        "--metadata_json",
        required=False,
        default=None,
        help="Optional substation metadata JSON containing id + voltage field (fallback)"
    )
    ap.add_argument("--id_field", default="Id", help="Field name in metadata JSON for the OSM id/ref")
    ap.add_argument("--voltage_field", default="Spannung", help="Field name in metadata JSON for voltage string")

    ap.add_argument("--class_bins", default="0,40,80,160,999999", help="Comma-separated bin edges in MVA")
    ap.add_argument("--class_labels", default="<=40,40-80,80-160,>160", help="Comma-separated labels")

    return ap.parse_args()


def join_key_digits(x) -> str:
    oid = parse_osm_id_digits(x)
    return str(oid) if oid is not None else ""


def build_feature_row(area_m2: float, voltage_str: str, feature_names: list[str]) -> np.ndarray:
    primary_v, secondary_v, num_levels, v_ratio = extract_voltage_features(voltage_str)

    feats = {
        "Area_m2": float(area_m2) if area_m2 is not None and np.isfinite(area_m2) else 0.0,
        "Primary_Voltage_kV": float(primary_v) if primary_v else 0.0,
        "Secondary_Voltage_kV": float(secondary_v) if secondary_v else 0.0,
        "Voltage_Levels": float(num_levels) if num_levels else 0.0,
        "Voltage_Ratio": float(v_ratio) if v_ratio else 0.0,
        "Has_Secondary": 1.0 if secondary_v else 0.0,
        "Has_Voltage_Ratio": 1.0 if v_ratio else 0.0,
    }
    return np.array([feats.get(f, 0.0) for f in feature_names], dtype=np.float32).reshape(1, -1)


def _guess_manifest_cols(df: pd.DataFrame, id_override: Optional[str], v_override: Optional[str]) -> tuple[str, str]:
    cols = list(df.columns)

    if id_override:
        if id_override not in cols:
            raise RuntimeError(f"--manifest_id_field='{id_override}' not in manifest columns: {cols}")
        id_col = id_override
    else:
        for c in ("osm_id", "Id", "osm_ref", "osm_ref_digits", "id"):
            if c in cols:
                id_col = c
                break
        else:
            raise RuntimeError(f"Could not infer manifest id column. Columns: {cols}")

    if v_override:
        if v_override not in cols:
            raise RuntimeError(f"--manifest_voltage_field='{v_override}' not in manifest columns: {cols}")
        v_col = v_override
    else:
        # IMPORTANT: include your actual column name first
        for c in ("voltage_str", "Spannung", "spannung", "voltage", "Voltage", "netzspannung"):
            if c in cols:
                v_col = c
                break
        else:
            raise RuntimeError(f"Could not infer manifest voltage column. Columns: {cols}")

    return id_col, v_col


def load_voltage_map_from_manifest(
    manifest_csv: str | None,
    id_override: str | None,
    v_override: str | None,
) -> Dict[str, str]:
    if not manifest_csv:
        return {}

    p = Path(manifest_csv)
    if not p.exists():
        raise RuntimeError(f"manifest_csv not found: {p}")

    df = pd.read_csv(p)
    id_col, v_col = _guess_manifest_cols(df, id_override, v_override)

    df["_k"] = df[id_col].apply(join_key_digits)
    df["_v"] = df[v_col].astype(str).fillna("").str.strip()

    mp: Dict[str, str] = {}
    for k, v in zip(df["_k"].tolist(), df["_v"].tolist()):
        if k and v:
            mp[k] = v

    print(f"✅ Loaded voltage map from manifest: {len(mp)} rows (id_col='{id_col}', voltage_col='{v_col}')")
    return mp


def load_voltage_map_from_metadata_json(metadata_json: str | None, id_field: str, voltage_field: str) -> Dict[str, str]:
    if not metadata_json:
        return {}

    p = Path(metadata_json)
    if not p.exists():
        return {}

    data: Any = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for k in ("items", "data", "features"):
            if k in data and isinstance(data[k], list):
                data = data[k]
                break

    if not isinstance(data, list):
        return {}

    mp: Dict[str, str] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        k = join_key_digits(item.get(id_field))
        v = item.get(voltage_field)
        if not k or v is None:
            continue
        v = str(v).strip()
        if v:
            mp[k] = v

    print(f"ℹ️ Loaded voltage map from metadata_json: {len(mp)} rows (id_field='{id_field}', voltage_field='{voltage_field}')")
    return mp


def main():
    args = parse_args()

    model = joblib.load(args.model_joblib)
    cfg = json.loads(Path(args.model_config).read_text(encoding="utf-8"))
    feature_names = cfg.get("feature_names")
    if not feature_names or not isinstance(feature_names, list):
        raise RuntimeError("model_config.json must contain a 'feature_names' list")

    bins = [float(x) for x in args.class_bins.split(",")]
    labels = [x.strip() for x in args.class_labels.split(",")]
    if len(labels) != len(bins) - 1:
        raise RuntimeError(f"class_labels count ({len(labels)}) must equal len(class_bins)-1 ({len(bins)-1}).")

    det = pd.read_csv(args.unet_csv, comment="#")

    # join key: always digits-only
    if "osm_id" in det.columns and det["osm_id"].notna().any():
        det["join_key"] = det["osm_id"].apply(join_key_digits)
    else:
        src_col = "image_name" if "image_name" in det.columns else ("image" if "image" in det.columns else None)
        if not src_col:
            raise RuntimeError("UNet CSV missing osm_id and image_name/image columns; cannot join metadata.")
        det["join_key"] = det[src_col].apply(join_key_digits)

    # voltage map: manifest first (preferred), then JSON fallback (only fill missing)
    meta_map = load_voltage_map_from_manifest(args.manifest_csv, args.manifest_id_field, args.manifest_voltage_field)

    if args.metadata_json:
        fallback = load_voltage_map_from_metadata_json(args.metadata_json, args.id_field, args.voltage_field)
        for k, v in fallback.items():
            if k not in meta_map:
                meta_map[k] = v

    preds: list[float] = []
    reasons: list[str] = []
    used_voltage: list[str] = []

    for _, r in det.iterrows():
        area = r.get("area_m2", None)
        k = (r.get("join_key", "") or "").strip()
        vstr = meta_map.get(k, "")

        if area is None or not np.isfinite(area) or float(area) <= 0:
            preds.append(np.nan)
            reasons.append("missing_or_invalid_area")
            used_voltage.append(vstr)
            continue

        if not str(vstr).strip():
            preds.append(np.nan)
            reasons.append("missing_voltage")
            used_voltage.append(vstr)
            continue

        X = build_feature_row(float(area), str(vstr), feature_names)
        y = float(model.predict(X)[0])
        preds.append(y)
        reasons.append("ok")
        used_voltage.append(vstr)

    det["pred_capacity_mva"] = preds
    det["capacity_reason"] = reasons
    det["voltage_used"] = used_voltage
    det["capacity_class"] = pd.cut(det["pred_capacity_mva"], bins=bins, labels=labels, include_lowest=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    det.to_csv(out_csv, index=False)

    summary = (
        det.groupby("join_key", dropna=False)
        .agg(
            n_components=("pred_capacity_mva", "size"),
            n_pred_ok=("capacity_reason", lambda s: int((s == "ok").sum())),
            capacity_mva_sum=("pred_capacity_mva", "sum"),
            capacity_mva_median=("pred_capacity_mva", "median"),
        )
        .reset_index()
        .rename(columns={"join_key": "osm_id"})
    )

    summary_csv = Path(args.summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)

    print(f"✅ wrote: {out_csv}")
    print(f"✅ wrote: {summary_csv}")
    print("capacity_reason counts:\n", det["capacity_reason"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
    