"""LSB statistical steganalysis: chi-square and sample-pair heuristics.

These checks are intentionally conservative. Natural H.264 content often looks
"balanced" in simplified SPA metrics, so thresholds are set high to limit
false positives on clean delivery video.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np

# Tuned against clean H.264 carriers (false-positive prone at lower cutoffs)
CHI_THRESHOLD = 0.78
SPA_THRESHOLD = 0.48
LSB_RATIO_DELTA = 0.12


def analyze(path: str | Path, sample_frames: int = 8) -> list[dict]:
    path = Path(path)
    signals: list[dict] = []
    try:
        frames = _sample_frames(path, sample_frames)
    except Exception as exc:
        return [{"signal": "lsb_stats", "label": f"frame sample failed: {exc}", "weight": 0}]

    if not frames:
        return signals

    chi_scores = []
    spa_rates = []
    lsb_ratios = []

    for img in frames:
        flat = img.reshape(-1).astype(np.uint8)
        chi_scores.append(_chi_square(flat))
        spa_rates.append(_sample_pair_rate(flat))
        lsb_ratios.append(float(np.mean(flat & 1)))

    avg_chi = float(np.mean(chi_scores))
    avg_spa = float(np.mean(spa_rates))
    avg_lsb = float(np.mean(lsb_ratios))

    # Always emit an informational calibration note (no score impact)
    signals.append(
        {
            "signal": "lsb_stats",
            "label": (
                f"LSB calibration: chi={avg_chi:.3f}, spa={avg_spa:.3f}, "
                f"lsb_ratio={avg_lsb:.3f}"
            ),
            "weight": 0,
        }
    )

    if avg_chi >= CHI_THRESHOLD:
        # Stronger only when very close to "too uniform"
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

    # Weak alone — only count when also corroborated by chi or spa
    if abs(avg_lsb - 0.5) > LSB_RATIO_DELTA and (
        avg_chi >= CHI_THRESHOLD or avg_spa >= SPA_THRESHOLD
    ):
        signals.append(
            {
                "signal": "lsb_stats",
                "label": f"LSB plane ratio={avg_lsb:.3f} (deviates from 0.5)",
                "weight": 6,
            }
        )

    return signals


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
    """Simplified sample-pair style estimate in [0, ~0.5].

    Natural compressed video often sits around 0.35–0.45 with this heuristic,
    so only values near 0.5 should be treated as suspicious.
    """
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
