#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.helpers.osm_ids import parse_osm_id_digits


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--manifest_csv", required=True, help="substations_manifest.csv with osm_id + coords + voltage_str")
    ap.add_argument("--capacity_summary_csv", default=None, help="substations_capacity_summary.csv")
    ap.add_argument("--unet_csv", default=None, help="transformer_detections.csv (optional, for transformer counts/area)")
    ap.add_argument("--yolo_summary_csv", default=None, help="YOLO summary.csv (preferred)")
    ap.add_argument("--yolo_detections_csv", default=None, help="YOLO detections.csv (fallback if no summary)")

    ap.add_argument("--out_csv", required=True, help="Output CSV with extensibility score per substation")

    # Scoring weights (easy to adjust)
    ap.add_argument("--w_capacity", type=float, default=0.55)
    ap.add_argument("--w_area", type=float, default=0.30)
    ap.add_argument("--w_objects", type=float, default=0.15)

    return ap.parse_args()


def key_digits(x) -> str:
    oid = parse_osm_id_digits(x)
    return str(oid) if oid is not None else ""


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def minmax_0_100(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mn = s.min(skipna=True)
    mx = s.max(skipna=True)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return pd.Series(np.full(len(s), 50.0), index=s.index)  # neutral if constant
    return (s - mn) / (mx - mn) * 100.0


def load_capacity_summary(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["osm_id", "capacity_mva_sum", "capacity_mva_median", "n_pred_ok"])
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["osm_id", "capacity_mva_sum", "capacity_mva_median", "n_pred_ok"])

    df = pd.read_csv(p)
    # normalize key column
    key_col = "osm_id" if "osm_id" in df.columns else ("osm_ref" if "osm_ref" in df.columns else df.columns[0])
    df["osm_id"] = df[key_col].apply(key_digits)

    # pick best available capacity column names
    if "capacity_mva_sum" not in df.columns:
        # allow older naming
        for c in ("capacity_sum", "pred_capacity_sum", "capacity_mva_total"):
            if c in df.columns:
                df["capacity_mva_sum"] = df[c]
                break
    if "capacity_mva_sum" not in df.columns:
        df["capacity_mva_sum"] = np.nan

    if "capacity_mva_median" not in df.columns:
        for c in ("capacity_median", "pred_capacity_median"):
            if c in df.columns:
                df["capacity_mva_median"] = df[c]
                break
    if "capacity_mva_median" not in df.columns:
        df["capacity_mva_median"] = np.nan

    if "n_pred_ok" not in df.columns:
        df["n_pred_ok"] = np.nan

    return df[["osm_id", "capacity_mva_sum", "capacity_mva_median", "n_pred_ok"]].drop_duplicates("osm_id")


def load_unet_features(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["osm_id", "n_transformers", "transformer_area_sum_m2"])
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["osm_id", "n_transformers", "transformer_area_sum_m2"])

    det = pd.read_csv(p, comment="#")
    # derive osm_id
    if "osm_id" in det.columns:
        det["osm_id"] = det["osm_id"].apply(key_digits)
    else:
        # fallback to image column
        src = "image_name" if "image_name" in det.columns else ("image" if "image" in det.columns else None)
        det["osm_id"] = det[src].apply(key_digits) if src else ""

    # area column name flexibility
    area_col = "area_m2" if "area_m2" in det.columns else None
    if area_col is None:
        for c in det.columns:
            if "area" in c.lower():
                area_col = c
                break

    det[area_col] = pd.to_numeric(det[area_col], errors="coerce") if area_col else np.nan

    agg = det.groupby("osm_id", dropna=False).agg(
        n_transformers=("osm_id", "size"),
        transformer_area_sum_m2=(area_col, "sum") if area_col else ("osm_id", lambda s: np.nan),
    ).reset_index()

    return agg


def load_yolo_features(summary_csv: str | None, detections_csv: str | None) -> pd.DataFrame:
    # Preferred: summary CSV
    if summary_csv and Path(summary_csv).exists():
        df = pd.read_csv(summary_csv)

        # Find id column
        id_col = None
        for c in ("osm_id", "osm_ref", "id", "Id"):
            if c in df.columns:
                id_col = c
                break
        if id_col is None:
            # maybe the first column is the key
            id_col = df.columns[0]

        df["osm_id"] = df[id_col].apply(key_digits)

        # If it's wide (counts per class), total_objects = sum of numeric cols except key-ish
        numeric_cols = [c for c in df.columns if c != id_col and c != "osm_id"]
        totals = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        out = pd.DataFrame({"osm_id": df["osm_id"], "yolo_total_objects": totals})
        return out.groupby("osm_id", dropna=False).sum(numeric_only=True).reset_index()

    # Fallback: detections CSV (long format)
    if detections_csv and Path(detections_csv).exists():
        det = pd.read_csv(detections_csv)

        # id derivation
        if "osm_id" in det.columns:
            det["osm_id"] = det["osm_id"].apply(key_digits)
        else:
            src = "image" if "image" in det.columns else ("image_name" if "image_name" in det.columns else None)
            det["osm_id"] = det[src].apply(key_digits) if src else ""

        out = det.groupby("osm_id", dropna=False).size().reset_index(name="yolo_total_objects")
        return out

    return pd.DataFrame(columns=["osm_id", "yolo_total_objects"])


def main():
    args = parse_args()

    manifest = pd.read_csv(args.manifest_csv)
    if "osm_id" not in manifest.columns:
        raise RuntimeError("manifest_csv must contain an 'osm_id' column")

    base = manifest.copy()
    base["osm_id"] = base["osm_id"].apply(key_digits)

    cap = load_capacity_summary(args.capacity_summary_csv)
    unet = load_unet_features(args.unet_csv)
    yolo = load_yolo_features(args.yolo_summary_csv, args.yolo_detections_csv)

    df = base.merge(cap, on="osm_id", how="left").merge(unet, on="osm_id", how="left").merge(yolo, on="osm_id", how="left")

    # Fill reasonable defaults
    df["capacity_mva_sum"] = pd.to_numeric(df.get("capacity_mva_sum"), errors="coerce")
    df["transformer_area_sum_m2"] = pd.to_numeric(df.get("transformer_area_sum_m2"), errors="coerce")
    df["yolo_total_objects"] = pd.to_numeric(df.get("yolo_total_objects"), errors="coerce").fillna(0)

    # Score components (z-scored, then weighted)
    z_cap = zscore(df["capacity_mva_sum"].fillna(0))
    z_area = zscore(df["transformer_area_sum_m2"].fillna(0))
    z_obj = zscore(df["yolo_total_objects"].fillna(0))

    score_raw = (args.w_capacity * z_cap) + (args.w_area * z_area) - (args.w_objects * z_obj)
    df["extensibility_score_0_100"] = minmax_0_100(score_raw).round(2)

    # Diagnostics
    df["score_notes"] = ""
    df.loc[df["capacity_mva_sum"].isna(), "score_notes"] += "missing_capacity;"
    df.loc[df["transformer_area_sum_m2"].isna(), "score_notes"] += "missing_unet_area;"
    df.loc[df["yolo_total_objects"].isna(), "score_notes"] += "missing_yolo;"
    df["score_notes"] = df["score_notes"].str.strip(";")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"✅ wrote: {out} (rows={len(df)})")


if __name__ == "__main__":
    main()
    