# vsteg — Video Steganography Toolkit

Encode, decode, and detect hidden data in video files.

License: [MIT](LICENSE)

## Why this exists

- **StegoForge** embeds in pixel LSBs then re-encodes with **lossy** H.264, which destroys the payload.
- **OpenPuff** can encode/decode but **cannot detect**.

`vsteg` fixes both: reliable round-trips, and a detection pipeline.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,web]"
# optional ML steganalysis ensemble (sklearn RandomForest)
pip install -e ".[ml]"
python scripts/train_ml_detector.py   # once, to build the small model
# needs ffmpeg/ffprobe on PATH
```

## Web UI (beginner-friendly)

```bash
python -m vsteg web
# open http://127.0.0.1:5000
```

Four tabs: **Hide**, **Reveal**, **Check**, and **Compare** (side-by-side attributes + graphs).

## Usage (CLI)

```bash
# Method A — append (default, 100% reliable, keeps .mp4)
python -m vsteg encode -i carrier.mp4 -s secret.txt -o out.mp4
python -m vsteg encode -i carrier.mp4 -s secret.txt -o out.mp4 -p mypassword

# Method B — lossless LSB (true pixel stego → .mkv / FFV1)
python -m vsteg encode -i carrier.mp4 -s secret.txt -o out.mkv -m lsb
python -m vsteg encode -i carrier.mp4 -s secret.txt -o out.mkv -m lsb --bits 1 -p pw

# Method C — DCT/QIM robust (survives lossy re-encode; small payloads)
python -m vsteg encode -i carrier.mp4 -s secret.txt -o out.mp4 -m dct --strength 12 --crf 18

# Decode (auto-tries methods)
python -m vsteg decode -i out.mp4 -o recovered.bin
python -m vsteg decode -i out.mkv -o recovered.bin -p pw

# Detect
python -m vsteg detect -i suspect.mp4
python -m vsteg detect -i suspect.mp4 --json

# Compare two videos (text + optional HTML graphs)
python -m vsteg compare -a carrier.mp4 -b out.mp4
python -m vsteg compare -a carrier.mp4 -b out.mp4 --html output/compare.html
```

## Detection notes (vs StegoForge)

Deep **Check** includes StegoForge-inspired stats: chi-square, **RS analysis**, keyframe DCT anomaly, plus an optional sklearn ensemble on handcrafted video features. We do **not** ship StegoForge’s image ONNX CNN (trained on still images / BOSSbase) — that model is the wrong domain for H.264 video.

## Methods at a glance

| Method | Output | Survives re-encode? | Capacity |
|--------|--------|---------------------|----------|
| `append` | same container | No (trailer stripped) | Filesystem-limited |
| `lsb` | lossless `.mkv` (FFV1) | No (needs lossless) | ~frames×W×H×3×bits/8 |
| `dct` | `.mp4` H.264 | **Yes** (within limits) | KB–~1 MiB default |

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Usage / unexpected error |
| 2 | Decode / auth failure |
| 3 | Detect: suspicious or likely-stego |
| 4 | Detect: clean |

## Tests

```bash
pytest -q
```
