"""Sklearn ensemble over handcrafted video steganalysis features.

Optional dependency: install with ``pip install -e ".[ml]"``.
When scikit-learn or the bundled model is missing, emits an info-only note.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from vsteg.detect import dct_stats, statistics, video_anomaly
from vsteg.detect.mp4_forensics import inspect_mp4

FEATURE_NAMES = [
    "avg_chi",
    "avg_spa",
    "avg_lsb",
    "avg_rs",
    "dct_near",
    "dct_peakiness",
    "keyframe_zmax",
    "mdat_slack_norm",
]

MODEL_PATH = Path(__file__).resolve().parent / "models" / "video_stego_rf.joblib"
MAX_ML_WEIGHT = 20


def analyze(
    path: str | Path,
    *,
    features: Optional[dict[str, float]] = None,
) -> list[dict]:
    path = Path(path)
    try:
        feats = features if features is not None else extract_features(path)
    except Exception as exc:
        return [
            {
                "signal": "ml_stats",
                "label": f"ML feature extraction failed: {exc}",
                "weight": 0,
            }
        ]

    try:
        import joblib  # noqa: F401
        from sklearn.ensemble import RandomForestClassifier  # noqa: F401
    except Exception:
        return [
            {
                "signal": "ml_stats",
                "label": (
                    "ML ensemble skipped (install optional deps: pip install -e \".[ml]\")"
                ),
                "weight": 0,
                "metrics": feats,
            }
        ]

    if not MODEL_PATH.exists():
        return [
            {
                "signal": "ml_stats",
                "label": (
                    f"ML ensemble skipped (model missing at {MODEL_PATH.name}; "
                    "run scripts/train_ml_detector.py)"
                ),
                "weight": 0,
                "metrics": feats,
            }
        ]

    try:
        import joblib

        model = joblib.load(MODEL_PATH)
        x = np.array([[feats[name] for name in FEATURE_NAMES]], dtype=np.float64)
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(x)[0][1])
        else:
            pred = int(model.predict(x)[0])
            proba = float(pred)
    except Exception as exc:
        return [
            {
                "signal": "ml_stats",
                "label": f"ML inference failed: {exc}",
                "weight": 0,
                "metrics": feats,
            }
        ]

    out: list[dict] = [
        {
            "signal": "ml_stats",
            "label": (
                f"ML ensemble calibration: P(stego)={proba:.3f} "
                f"(RandomForest on {len(FEATURE_NAMES)} features)"
            ),
            "weight": 0,
            "metrics": {**feats, "proba_stego": proba},
        }
    ]

    if proba >= 0.65:
        strength = (proba - 0.65) / max(1e-6, 0.35)
        weight = min(MAX_ML_WEIGHT, 8 + int(strength * 12))
        out.append(
            {
                "signal": "ml_stats",
                "label": (
                    f"ML ensemble likely-stego (P={proba:.3f}) — "
                    "handcrafted video features (chi/SPA/RS/DCT/keyframe)"
                ),
                "weight": weight,
            }
        )
    elif proba >= 0.45:
        out.append(
            {
                "signal": "ml_stats",
                "label": f"ML ensemble suspicious (P={proba:.3f})",
                "weight": 6,
            }
        )
    return out


def extract_features(path: str | Path) -> dict[str, float]:
    """Build the fixed feature vector used by the bundled classifier."""
    path = Path(path)
    lsb = statistics.compute_metrics(path)
    dct = dct_stats.compute_metrics(path)
    kf = video_anomaly.compute_metrics(path)
    slack_norm = 0.0
    try:
        size = max(1, path.stat().st_size)
        report = inspect_mp4(path.read_bytes())
        slack = float(report.get("mdat_slack") or 0)
        slack_norm = min(1.0, max(0.0, slack / size))
    except Exception:
        pass

    return {
        "avg_chi": float(lsb.get("avg_chi", 0.0)),
        "avg_spa": float(lsb.get("avg_spa", 0.0)),
        "avg_lsb": float(lsb.get("avg_lsb", 0.0)),
        "avg_rs": float(lsb.get("avg_rs", 0.0)),
        "dct_near": float(dct.get("near", 0.0)),
        "dct_peakiness": float(dct.get("peakiness", 0.0)),
        "keyframe_zmax": float(kf.get("zmax", 0.0)),
        "mdat_slack_norm": float(slack_norm),
    }


def feature_vector(feats: dict[str, float]) -> np.ndarray:
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float64)
