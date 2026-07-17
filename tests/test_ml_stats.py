"""Tests for StegoForge-style RS, keyframe DCT, and optional ML ensemble."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vsteg.detect import statistics, video_anomaly
from vsteg.detect.report import detect
from vsteg.detect.statistics import _rs_payload_fraction
from vsteg.detect.video_anomaly import frame_score

ROOT = Path(__file__).resolve().parents[1]
CARRIER = ROOT / "carrier.mp4"


def test_rs_fraction_on_random_vs_lsb_like():
    rng = np.random.default_rng(0)
    natural = rng.integers(0, 256, size=50_000, dtype=np.uint8)
    # Force LSB plane toward balanced pairs (embedding-like)
    stego = natural.copy()
    stego[::2] = (stego[::2] & 0xFE) | (stego[1::2] & 1)

    # RS should not crash and return a bounded fraction
    f_nat = _rs_payload_fraction(natural)
    f_stego = _rs_payload_fraction(stego)
    assert 0.0 <= f_nat <= 1.0
    assert 0.0 <= f_stego <= 1.0


def test_frame_score_positive_on_gradient():
    yy, xx = np.mgrid[0:64, 0:64]
    gray = ((xx + yy) % 256).astype(np.uint8)
    score = frame_score(gray)
    assert score >= 0.0


def test_video_anomaly_on_carrier():
    if not CARRIER.exists():
        pytest.skip("carrier.mp4 missing")
    signals = video_anomaly.analyze(CARRIER, max_keyframes=6)
    assert any(s.get("signal") == "video_anomaly" for s in signals)
    # Calibration note always present
    assert any(s.get("weight", 0) == 0 for s in signals)


def test_statistics_includes_rs_metric():
    if not CARRIER.exists():
        pytest.skip("carrier.mp4 missing")
    metrics = statistics.compute_metrics(CARRIER, sample_frames=4)
    assert metrics["frames"] > 0
    assert "avg_rs" in metrics


def test_deep_detect_still_clean_on_carrier():
    if not CARRIER.exists():
        pytest.skip("carrier.mp4 missing")
    report = detect(CARRIER, deep=True)
    assert report.verdict == "clean"
    kinds = {s.get("signal") for s in report.signals}
    assert "video_anomaly" in kinds
    assert "lsb_stats" in kinds


def test_ml_ensemble_graceful_without_model(tmp_path, monkeypatch):
    from vsteg.detect import ml_ensemble

    missing = tmp_path / "nope.joblib"
    monkeypatch.setattr(ml_ensemble, "MODEL_PATH", missing)
    feats = {name: 0.1 for name in ml_ensemble.FEATURE_NAMES}
    signals = ml_ensemble.analyze(CARRIER if CARRIER.exists() else tmp_path, features=feats)
    assert signals
    assert all(s.get("weight", 0) == 0 for s in signals)
    assert "skipped" in signals[0]["label"].lower() or "ML" in signals[0]["label"]
