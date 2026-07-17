"""DCT mid-frequency coefficient histogram analysis (QIM telltales).

Kept low-confidence and high-threshold: robust embedding is stealthy, and clean
H.264 streams can show mild mid-band peakiness from normal quantization.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
from scipy.fft import dctn

MID = [(1, 2), (2, 1), (2, 2), (1, 3), (3, 1)]

NEAR_THRESHOLD = 0.55
PEAKINESS_THRESHOLD = 0.45


def analyze(path: str | Path, sample_frames: int = 4, delta_guess: float = 8.0) -> list[dict]:
    path = Path(path)
    signals: list[dict] = []
    try:
        metrics = compute_metrics(path, sample_frames, delta_guess)
    except Exception as exc:
        return [{"signal": "dct_stats", "label": f"DCT sample failed: {exc}", "weight": 0}]

    if metrics.get("coeff_count", 0) < 100:
        return signals

    near = float(metrics["near"])
    hist_score = float(metrics["peakiness"])

    signals.append(
        {
            "signal": "dct_stats",
            "label": (
                f"DCT calibration: near-fraction={near:.3f}, "
                f"peakiness={hist_score:.3f} (Δ={delta_guess})"
            ),
            "weight": 0,
            "metrics": metrics,
        }
    )

    if near >= NEAR_THRESHOLD:
        strength = (near - NEAR_THRESHOLD) / max(1e-6, 1.0 - NEAR_THRESHOLD)
        weight = min(14, 4 + int(strength * 12))
        signals.append(
            {
                "signal": "dct_stats",
                "label": (
                    f"mid-band coeffs cluster near QIM step Δ={delta_guess} "
                    f"(near-fraction={near:.3f})"
                ),
                "weight": weight,
            }
        )

    if hist_score >= PEAKINESS_THRESHOLD:
        strength = (hist_score - PEAKINESS_THRESHOLD) / max(1e-6, 1.0 - PEAKINESS_THRESHOLD)
        weight = min(12, 4 + int(strength * 10))
        signals.append(
            {
                "signal": "dct_stats",
                "label": f"mid-band histogram peakiness={hist_score:.3f}",
                "weight": weight,
            }
        )

    return signals


def compute_metrics(
    path: str | Path,
    sample_frames: int = 4,
    delta_guess: float = 8.0,
) -> dict:
    """Return mid-band DCT metrics for ML feature extraction."""
    coeffs = _collect_mid_coeffs(Path(path), sample_frames)
    if coeffs.size < 100:
        return {"coeff_count": int(coeffs.size), "near": 0.0, "peakiness": 0.0}
    q = np.round(coeffs / delta_guess)
    residual = np.abs(coeffs - q * delta_guess)
    near = float(np.mean(residual < delta_guess * 0.15))
    hist_score = _peakiness(coeffs, delta_guess)
    return {
        "coeff_count": int(coeffs.size),
        "near": near,
        "peakiness": hist_score,
        "delta": float(delta_guess),
    }


def _collect_mid_coeffs(path: Path, n_frames: int) -> np.ndarray:
    vals: list[float] = []
    container = av.open(str(path))
    count = 0
    for frame in container.decode(video=0):
        gray = frame.to_ndarray(format="gray").astype(np.float64)
        h, w = gray.shape
        h8, w8 = (h // 8) * 8, (w // 8) * 8
        gray = gray[:h8, :w8]
        for by in range(0, h8 // 8, 2):
            for bx in range(0, w8 // 8, 2):
                block = gray[by * 8 : (by + 1) * 8, bx * 8 : (bx + 1) * 8]
                c = dctn(block, type=2, norm="ortho")
                for ri, ci in MID:
                    vals.append(float(c[ri, ci]))
        count += 1
        if count >= n_frames:
            break
    container.close()
    return np.array(vals, dtype=np.float64)


def _peakiness(coeffs: np.ndarray, delta: float) -> float:
    q = np.round(coeffs / delta).astype(np.int32)
    resid = coeffs - q * delta
    hist, _ = np.histogram(resid, bins=21, range=(-delta, delta), density=True)
    center = hist[len(hist) // 2]
    mean = float(np.mean(hist)) + 1e-9
    return float(min(1.0, (center / mean - 1.0) / 5.0))
