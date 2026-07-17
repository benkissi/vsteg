"""StegoForge-style keyframe DCT anomaly scan for video.

Samples I-frames, scores mid-band DCT energy at coefficients (3,4) and (4,3),
then flags unusually large z-scores across the keyframe set.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
from scipy.fft import dctn

# Conservative: natural H.264 keyframes can vary; only extreme outliers score.
ZMAX_THRESHOLD = 4.5
MAX_KEYFRAMES = 12


def analyze(path: str | Path, max_keyframes: int = MAX_KEYFRAMES) -> list[dict]:
    path = Path(path)
    try:
        metrics = compute_metrics(path, max_keyframes)
    except Exception as exc:
        return [
            {
                "signal": "video_anomaly",
                "label": f"keyframe DCT sample failed: {exc}",
                "weight": 0,
            }
        ]

    keyframes = int(metrics.get("keyframes_sampled", 0))
    if keyframes <= 0:
        return [
            {
                "signal": "video_anomaly",
                "label": "No keyframes sampled for DCT anomaly scan",
                "weight": 0,
            }
        ]

    m = float(metrics["score_mean"])
    s = float(metrics["score_std"])
    zmax = float(metrics["zmax"])

    out: list[dict] = [
        {
            "signal": "video_anomaly",
            "label": (
                f"Keyframe DCT calibration: n={keyframes}, "
                f"score_mean={m:.4f}, score_std={s:.4f}, zmax={zmax:.3f}"
            ),
            "weight": 0,
            "metrics": metrics,
        }
    ]

    if zmax >= ZMAX_THRESHOLD:
        strength = (zmax - ZMAX_THRESHOLD) / max(1e-6, 8.0 - ZMAX_THRESHOLD)
        weight = min(18, 6 + int(strength * 14))
        out.append(
            {
                "signal": "video_anomaly",
                "label": (
                    f"I-frame DCT distribution outlier (zmax={zmax:.3f}) — "
                    "StegoForge-style keyframe anomaly"
                ),
                "weight": weight,
            }
        )
    return out


def compute_metrics(path: str | Path, max_keyframes: int = MAX_KEYFRAMES) -> dict:
    """Return keyframe DCT metrics for ML feature extraction."""
    scores, keyframes = _keyframe_scores(Path(path), max_keyframes)
    if not scores:
        return {
            "keyframes_sampled": 0,
            "score_mean": 0.0,
            "score_std": 0.0,
            "zmax": 0.0,
        }
    m = float(np.mean(scores))
    s = float(np.std(scores) + 1e-6)
    zmax = float(max(abs((x - m) / s) for x in scores))
    return {
        "keyframes_sampled": keyframes,
        "score_mean": m,
        "score_std": s,
        "zmax": zmax,
    }


def frame_score(gray: np.ndarray) -> float:
    """Mean |c[3,4]|+|c[4,3]| over 8×8 blocks (StegoForge `_frame_score`)."""
    h, w = gray.shape
    h8 = h - (h % 8)
    w8 = w - (w % 8)
    if h8 == 0 or w8 == 0:
        return 0.0
    g = gray[:h8, :w8].astype(np.float32)
    blocks = g.reshape(h8 // 8, 8, w8 // 8, 8).transpose(0, 2, 1, 3)
    coeffs = dctn(blocks, axes=(2, 3), norm="ortho")
    v1 = np.abs(coeffs[:, :, 3, 4])
    v2 = np.abs(coeffs[:, :, 4, 3])
    return float(np.mean(v1 + v2))


def _keyframe_scores(path: Path, max_keyframes: int) -> tuple[list[float], int]:
    scores: list[float] = []
    keyframes = 0
    container = av.open(str(path))
    try:
        for frame in container.decode(video=0):
            if not getattr(frame, "key_frame", False):
                continue
            keyframes += 1
            gray = frame.to_ndarray(format="gray")
            scores.append(frame_score(gray))
            if keyframes >= max_keyframes:
                break
    finally:
        container.close()
    return scores, keyframes
