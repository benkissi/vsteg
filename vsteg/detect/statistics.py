"""LSB statistical steganalysis: chi-square, sample-pair, and RS analysis.

These checks are intentionally conservative. Natural H.264 content often looks
"balanced" in simplified SPA metrics, so thresholds are set high to limit
false positives on clean delivery video.

RS analysis follows the StegoForge / classical Regular–Singular group method.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import av
import numpy as np

# Tuned against clean H.264 carriers (false-positive prone at lower cutoffs)
CHI_THRESHOLD = 0.78
SPA_THRESHOLD = 0.48
LSB_RATIO_DELTA = 0.12
RS_FRACTION_THRESHOLD = 0.18
RS_GROUP_SIZE = 4
RS_MASK = np.array([0, 1, 1, 0], dtype=np.int32)


def analyze(path: str | Path, sample_frames: int = 8) -> list[dict]:
    path = Path(path)
    try:
        metrics = compute_metrics(path, sample_frames)
    except Exception as exc:
        return [{"signal": "lsb_stats", "label": f"frame sample failed: {exc}", "weight": 0}]

    if metrics.get("frames", 0) <= 0:
        return []

    avg_chi = float(metrics["avg_chi"])
    avg_spa = float(metrics["avg_spa"])
    avg_lsb = float(metrics["avg_lsb"])
    avg_rs = float(metrics["avg_rs"])

    signals: list[dict] = [
        {
            "signal": "lsb_stats",
            "label": (
                f"LSB calibration: chi={avg_chi:.3f}, spa={avg_spa:.3f}, "
                f"lsb_ratio={avg_lsb:.3f}, rs_fraction={avg_rs:.3f}"
            ),
            "weight": 0,
            "metrics": metrics,
        }
    ]

    if avg_chi >= CHI_THRESHOLD:
        strength = (avg_chi - CHI_THRESHOLD) / max(1e-6, 1.0 - CHI_THRESHOLD)
        weight = min(22, 8 + int(strength * 18))
        signals.append(
            {
                "signal": "lsb_stats",
                "label": f"chi-square anomaly score={avg_chi:.3f}",
                "weight": weight,
            }
        )

    if avg_spa >= SPA_THRESHOLD:
        strength = (avg_spa - SPA_THRESHOLD) / max(1e-6, 0.5 - SPA_THRESHOLD)
        weight = min(22, 8 + int(strength * 18))
        signals.append(
            {
                "signal": "lsb_stats",
                "label": f"sample-pair embedding-rate estimate={avg_spa:.3f}",
                "weight": weight,
            }
        )

    if avg_rs >= RS_FRACTION_THRESHOLD:
        strength = (avg_rs - RS_FRACTION_THRESHOLD) / max(1e-6, 0.6 - RS_FRACTION_THRESHOLD)
        weight = min(20, 8 + int(strength * 16))
        signals.append(
            {
                "signal": "lsb_stats",
                "label": (
                    f"RS analysis estimated payload fraction={avg_rs:.3f} "
                    "(Regular–Singular group method)"
                ),
                "weight": weight,
            }
        )

    # Weak alone — only count when also corroborated by chi, spa, or RS
    if abs(avg_lsb - 0.5) > LSB_RATIO_DELTA and (
        avg_chi >= CHI_THRESHOLD
        or avg_spa >= SPA_THRESHOLD
        or avg_rs >= RS_FRACTION_THRESHOLD
    ):
        signals.append(
            {
                "signal": "lsb_stats",
                "label": f"LSB plane ratio={avg_lsb:.3f} (deviates from 0.5)",
                "weight": 6,
            }
        )

    return signals


def compute_metrics(path: str | Path, sample_frames: int = 8) -> dict[str, Any]:
    """Return averaged LSB/RS metrics for ML feature extraction."""
    frames = _sample_frames(Path(path), sample_frames)
    if not frames:
        return {"frames": 0}

    chi_scores = []
    spa_rates = []
    lsb_ratios = []
    rs_fractions = []

    for img in frames:
        flat = img.reshape(-1).astype(np.uint8)
        chi_scores.append(_chi_square(flat))
        spa_rates.append(_sample_pair_rate(flat))
        lsb_ratios.append(float(np.mean(flat & 1)))
        rs_fractions.append(_rs_payload_fraction(flat))

    return {
        "frames": len(frames),
        "avg_chi": float(np.mean(chi_scores)),
        "avg_spa": float(np.mean(spa_rates)),
        "avg_lsb": float(np.mean(lsb_ratios)),
        "avg_rs": float(np.mean(rs_fractions)),
    }


def _sample_frames(path: Path, n: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    total = stream.frames or 0
    frames_list = []
    for i, frame in enumerate(container.decode(video=0)):
        frames_list.append(frame.to_ndarray(format="gray"))
        if total and i >= total:
            break
        if not total and i >= n * 4:
            break
    container.close()
    if not frames_list:
        return out
    idxs = np.linspace(0, len(frames_list) - 1, num=min(n, len(frames_list)), dtype=int)
    for i in idxs:
        out.append(frames_list[int(i)])
    return out


def _chi_square(flat: np.ndarray) -> float:
    """Pairs of values (2k, 2k+1). Uniform pairs suggest LSB embedding."""
    hist = np.bincount(flat, minlength=256).astype(np.float64)
    score = 0.0
    pairs = 0
    for k in range(128):
        a, b = hist[2 * k], hist[2 * k + 1]
        expected = (a + b) / 2.0
        if expected > 0:
            score += ((a - expected) ** 2 + (b - expected) ** 2) / expected
            pairs += 1
    if pairs == 0:
        return 0.0
    avg = score / pairs
    # Higher when average chi is small (more uniform → more suspicious)
    suspicious = 1.0 / (1.0 + avg / 50.0)
    return float(suspicious)


def _sample_pair_rate(flat: np.ndarray) -> float:
    """Simplified sample-pair style estimate in [0, ~0.5]."""
    x = flat[:-1].astype(np.int16)
    y = flat[1:].astype(np.int16)
    d = y - x
    even = np.sum(d % 2 == 0)
    odd = np.sum(d % 2 != 0)
    total = even + odd
    if total == 0:
        return 0.0
    imbalance = abs(even - odd) / total
    return float(max(0.0, 0.5 - imbalance))


def _rs_payload_fraction(flat: np.ndarray) -> float:
    """StegoForge-style RS estimated payload fraction in [0, 1]."""
    arr = flat.astype(np.int32).reshape(-1)
    n_groups = len(arr) // RS_GROUP_SIZE
    if n_groups == 0:
        return 0.0

    groups = arr[: n_groups * RS_GROUP_SIZE].reshape(-1, RS_GROUP_SIZE)
    d_orig = np.sum(np.abs(np.diff(groups, axis=1)), axis=1)

    groups_flip = groups ^ RS_MASK
    d_flip = np.sum(np.abs(np.diff(groups_flip, axis=1)), axis=1)

    f_neg_mask = np.zeros_like(groups)
    masked = RS_MASK == 1
    f_neg_mask[:, masked] = np.where(groups[:, masked] % 2 == 0, -1, 1)
    groups_flip_n = groups + f_neg_mask
    d_flip_n = np.sum(np.abs(np.diff(groups_flip_n, axis=1)), axis=1)

    r_ratio = float(np.sum(d_flip > d_orig) / n_groups)
    s_ratio = float(np.sum(d_flip < d_orig) / n_groups)
    r_n_ratio = float(np.sum(d_flip_n > d_orig) / n_groups)
    s_n_ratio = float(np.sum(d_flip_n < d_orig) / n_groups)

    rs_diff = r_ratio - s_ratio
    neg_diff = r_n_ratio - s_n_ratio
    denominator = neg_diff + rs_diff
    if abs(denominator) < 1e-10:
        return 0.0
    estimated = (neg_diff - rs_diff) / denominator
    return float(min(1.0, max(0.0, estimated)))
