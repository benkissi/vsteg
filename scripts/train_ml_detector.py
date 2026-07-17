#!/usr/bin/env python3
"""Train the optional RandomForest steganalysis model on local samples.

Usage (from repo root, with ml extras installed):

    pip install -e ".[ml,dev]"
    python scripts/train_ml_detector.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vsteg.detect.ml_ensemble import FEATURE_NAMES, MODEL_PATH, extract_features
from vsteg.methods import append, dct, lsb

CARRIER = ROOT / "carrier.mp4"
SECRET = ROOT / "secret.txt"


def _make_tiny(tmp: Path) -> Path:
    out = tmp / "tiny.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=10",
            "-t",
            "3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


def _row(path: Path) -> list[float]:
    feats = extract_features(path)
    return [feats[name] for name in FEATURE_NAMES]


def main() -> int:
    if not CARRIER.exists() or not SECRET.exists():
        print("error: need carrier.mp4 and secret.txt in repo root", file=sys.stderr)
        return 1

    secret = SECRET.read_bytes()
    xs: list[list[float]] = []
    ys: list[int] = []

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tiny = _make_tiny(tmp)

        # Clean samples
        for path in (CARRIER, tiny):
            xs.append(_row(path))
            ys.append(0)

        # Append stego (on carrier)
        stego_a = tmp / "append.mp4"
        append.encode(CARRIER, secret, stego_a, password="train-pw")
        xs.append(_row(stego_a))
        ys.append(1)

        # LSB stego (tiny → mkv)
        stego_l = tmp / "lsb.mkv"
        lsb.encode(tiny, secret[:64], stego_l)
        xs.append(_row(stego_l))
        ys.append(1)

        # DCT stego (tiny)
        stego_d = tmp / "dct.mp4"
        dct.encode(
            tiny,
            b"ml-train-secret",
            stego_d,
            strength=20.0,
            crf=18,
            redundancy=11,
        )
        xs.append(_row(stego_d))
        ys.append(1)

        # Another clean synthetic
        tiny2 = tmp / "tiny2.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "smptebars=size=320x240:rate=10",
                "-t",
                "2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(tiny2),
            ],
            check=True,
            capture_output=True,
        )
        xs.append(_row(tiny2))
        ys.append(0)

    X = np.array(xs, dtype=np.float64)
    y = np.array(ys, dtype=np.int32)

    clf = RandomForestClassifier(
        n_estimators=64,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)

    print(f"trained on {len(y)} samples → {MODEL_PATH}")
    print(f"  labels: clean={int(np.sum(y == 0))} stego={int(np.sum(y == 1))}")
    print(f"  train accuracy: {clf.score(X, y):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
