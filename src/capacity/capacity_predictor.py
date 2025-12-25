from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import joblib

from .preprocessing import prepare_features, FEATURE_NAMES


@dataclass
class CapacityArtifacts:
    model_path: Path
    config_path: Path


class CapacityPredictor:
    def __init__(self, artifacts: CapacityArtifacts):
        self.model = joblib.load(str(artifacts.model_path))

        cfg = json.loads(Path(artifacts.config_path).read_text(encoding="utf-8"))
        self.feature_names = cfg.get("feature_names") or []
        if not self.feature_names:
            raise ValueError(f"model_config.json missing 'feature_names': {artifacts.config_path}")

        # Hard safety: ensure the preprocessing feature order matches training
        if list(self.feature_names) != list(FEATURE_NAMES):
            raise ValueError(
                "Feature order mismatch between model_config.json and preprocessing.FEATURE_NAMES.\n"
                f"config: {self.feature_names}\n"
                f"code:   {FEATURE_NAMES}\n"
                "Fix by aligning preprocessing.FEATURE_NAMES (or implement a config-driven prepare_features)."
            )

    @staticmethod
    def assign_capacity_class(mva: float | None) -> str:
        if mva is None or not np.isfinite(mva):
            return "unknown"
        mva = float(mva)
        if mva <= 40:
            return "≤40 MVA"
        if mva <= 80:
            return "40-80 MVA"
        if mva <= 160:
            return "80-160 MVA"
        if mva <= 250:
            return "160-250 MVA"
        if mva <= 400:
            return "250-400 MVA"
        return ">400 MVA"

    def predict_one(self, area_m2: float | None, voltage_str: str | None):
        # guard: no scale -> no capacity
        if area_m2 is None:
            return None, "unknown"
        try:
            a = float(area_m2)
        except Exception:
            return None, "unknown"
        if not np.isfinite(a) or a <= 0:
            return None, "unknown"

        # guard: no voltage -> no capacity
        vstr = (voltage_str or "").strip()
        if not vstr:
            return None, "unknown"

        X = prepare_features(a, vstr)  # shape (1, n_features) in FEATURE_NAMES order
        pred = self.model.predict(X)
        mva = float(pred[0]) if pred is not None and len(pred) else None
        return mva, self.assign_capacity_class(mva)
    