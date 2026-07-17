#!/usr/bin/env python3
"""Train the optional RandomForest steganalysis model from ``data/``.

Expected layout
---------------
data/
  carrier-sample.mp4          # optional — used for vsteg synthetic stego
  payload-sample.txt          # optional — secret for vsteg synthetic stego
  tools/
    <tool-name>/              # e.g. openpuff, vsteg, …
      original-1.mp4          # clean / cover  (also accepts original1.mp4)
      stego-1.mp4             # stego twin
      target-1.txt            # optional payload (metadata; not required for train)
      original-2.mp4
      stego-2.mp4
      target-2.txt
      …

Outputs
-------
models/runs/<timestamp>.joblib   # archived run
models/latest-model              # always overwritten with the newest run

Usage
-----
    pip install -e ".[ml,dev]"
    python scripts/train_ml_detector.py
    python scripts/train_ml_detector.py --no-synthetic   # tools pairs only
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vsteg.detect.ml_ensemble import FEATURE_NAMES, MODEL_PATH, extract_features
from vsteg.methods import append, dct, lsb

DATA_DIR = ROOT / "data"
TOOLS_DIR = DATA_DIR / "tools"
CARRIER = DATA_DIR / "carrier-sample.mp4"
PAYLOAD = DATA_DIR / "payload-sample.txt"
MODELS_DIR = ROOT / "models"
RUNS_DIR = MODELS_DIR / "runs"

# original-1.mp4 | original1.mp4 | original_1.mp4
PAIR_RE = re.compile(
    r"^(?P<kind>original|stego|target)[-_]?(?P<idx>\d+)\.(?P<ext>.+)$",
    re.IGNORECASE,
)
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi"}


@dataclass
class Sample:
    path: Path
    label: int  # 0=clean, 1=stego
    source: str  # e.g. "tools/openpuff#1/original"


def _row(path: Path) -> list[float]:
    feats = extract_features(path)
    return [feats[name] for name in FEATURE_NAMES]


def discover_tool_pairs(tools_dir: Path) -> list[Sample]:
    """Scan data/tools/<tool>/ for original-N / stego-N video pairs."""
    samples: list[Sample] = []
    if not tools_dir.is_dir():
        return samples

    for tool_dir in sorted(p for p in tools_dir.iterdir() if p.is_dir()):
        buckets: dict[str, dict[str, Path]] = {}
        for path in sorted(tool_dir.iterdir()):
            if not path.is_file():
                continue
            m = PAIR_RE.match(path.name)
            if not m:
                continue
            kind = m.group("kind").lower()
            idx = m.group("idx")
            buckets.setdefault(idx, {})[kind] = path

        for idx in sorted(buckets, key=lambda x: int(x)):
            pair = buckets[idx]
            orig = pair.get("original")
            stego = pair.get("stego")
            if orig is not None and orig.suffix.lower() in VIDEO_EXTS:
                samples.append(
                    Sample(
                        path=orig,
                        label=0,
                        source=f"tools/{tool_dir.name}#{idx}/original",
                    )
                )
            if stego is not None and stego.suffix.lower() in VIDEO_EXTS:
                samples.append(
                    Sample(
                        path=stego,
                        label=1,
                        source=f"tools/{tool_dir.name}#{idx}/stego",
                    )
                )
            if orig is None and stego is None:
                continue
            if orig is None or stego is None:
                print(
                    f"warning: incomplete pair in {tool_dir.name} #{idx} "
                    f"(have: {sorted(pair)})",
                    file=sys.stderr,
                )
    return samples


def build_synthetic_vsteg(tmp: Path) -> list[Sample]:
    """Generate a few vsteg clean/stego samples from data/carrier-sample + payload."""
    samples: list[Sample] = []
    if not CARRIER.exists():
        print(
            f"warning: {CARRIER.relative_to(ROOT)} missing — skipping vsteg synthetics",
            file=sys.stderr,
        )
        return samples

    secret = (
        PAYLOAD.read_bytes()
        if PAYLOAD.exists()
        else b"vsteg-ml-train-secret"
    )

    tiny = tmp / "tiny.mp4"
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
            str(tiny),
        ],
        check=True,
        capture_output=True,
    )

    samples.append(Sample(CARRIER, 0, "synthetic/carrier-sample"))
    samples.append(Sample(tiny, 0, "synthetic/testsrc"))

    stego_a = tmp / "append.mp4"
    append.encode(CARRIER, secret, stego_a, password="train-pw")
    samples.append(Sample(stego_a, 1, "synthetic/vsteg-append"))

    stego_l = tmp / "lsb.mkv"
    lsb.encode(tiny, secret[:64], stego_l)
    samples.append(Sample(stego_l, 1, "synthetic/vsteg-lsb"))

    stego_d = tmp / "dct.mp4"
    dct.encode(
        tiny,
        b"ml-train-secret",
        stego_d,
        strength=20.0,
        crf=18,
        redundancy=11,
    )
    samples.append(Sample(stego_d, 1, "synthetic/vsteg-dct"))
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Only train on data/tools pairs (skip vsteg synthetic samples)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Data root (default: {DATA_DIR})",
    )
    args = parser.parse_args(argv)

    tools_dir = args.data_dir / "tools"
    samples = discover_tool_pairs(tools_dir)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        if not args.no_synthetic:
            samples.extend(build_synthetic_vsteg(tmp))

        if not samples:
            print(
                "error: no training samples found.\n"
                f"  Add pairs under {tools_dir}/<tool>/original-N.mp4 + stego-N.mp4\n"
                "  or provide data/carrier-sample.mp4 for synthetic vsteg samples.",
                file=sys.stderr,
            )
            return 1

        print(f"training on {len(samples)} samples:")
        for s in samples:
            print(f"  [{s.label}] {s.source} ← {s.path}")

        xs: list[list[float]] = []
        ys: list[int] = []
        for s in samples:
            try:
                xs.append(_row(s.path))
                ys.append(s.label)
            except Exception as exc:
                print(f"warning: skip {s.path}: {exc}", file=sys.stderr)

        if len(xs) < 2 or len(set(ys)) < 2:
            print(
                "error: need at least one clean and one stego sample after feature extract",
                file=sys.stderr,
            )
            return 1

        X = np.array(xs, dtype=np.float64)
        y = np.array(ys, dtype=np.int32)

        clf = RandomForestClassifier(
            n_estimators=64,
            max_depth=6,
            random_state=42,
            class_weight="balanced",
        )
        clf.fit(X, y)

        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_path = RUNS_DIR / f"{stamp}.joblib"
        joblib.dump(clf, run_path)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        # Atomic-ish replace of latest-model
        tmp_latest = MODELS_DIR / f".latest-model.tmp-{stamp}"
        shutil.copy2(run_path, tmp_latest)
        tmp_latest.replace(MODEL_PATH)

        print(f"\nrun saved → {run_path.relative_to(ROOT)}")
        print(f"latest   → {MODEL_PATH.relative_to(ROOT)}")
        print(f"  labels: clean={int(np.sum(y == 0))} stego={int(np.sum(y == 1))}")
        print(f"  train accuracy: {clf.score(X, y):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
