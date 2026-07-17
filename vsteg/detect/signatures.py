"""Signature scanning for known steganography markers."""

from __future__ import annotations

from pathlib import Path

from vsteg import MAGIC

# Known / heuristic markers
SIGNATURES = [
    (MAGIC, "vsteg VSTG magic"),
    # Full "OpenPuff" string only — short "OPF" is a common false positive in media
    (b"OpenPuff", "OpenPuff marker"),
    (b"StegoForge", "StegoForge marker"),
]


def scan(path: str | Path, max_scan: int = 64 * 1024 * 1024) -> list[dict]:
    path = Path(path)
    size = path.stat().st_size
    hits: list[dict] = []

    # Scan head and tail (append payloads live at the end)
    regions = []
    with open(path, "rb") as f:
        head = f.read(min(size, max_scan // 2))
        regions.append(("head", 0, head))
        if size > len(head):
            tail_size = min(size, max_scan // 2)
            f.seek(size - tail_size)
            tail = f.read(tail_size)
            regions.append(("tail", size - tail_size, tail))

    for region_name, base, data in regions:
        for sig, label in SIGNATURES:
            start = 0
            while True:
                idx = data.find(sig, start)
                if idx < 0:
                    break
                hits.append(
                    {
                        "signal": "signature",
                        "label": label,
                        "offset": base + idx,
                        "region": region_name,
                        "weight": 40 if sig == MAGIC else 25,
                    }
                )
                start = idx + 1
                if len(hits) > 20:
                    return hits
    return hits
