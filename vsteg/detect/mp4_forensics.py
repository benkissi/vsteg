"""MP4 container forensics — OpenPuff-style mdat slack and atom consistency.

OpenPuff (and similar tools) can grow the `mdat` box without updating `stsz`
sample totals, leaving high-entropy unreferenced bytes the decoder never reads.
Decoded frames stay identical while the file gets larger.
"""

from __future__ import annotations

import struct
from pathlib import Path


def analyze(path: str | Path) -> list[dict]:
    path = Path(path)
    try:
        data = path.read_bytes()
    except Exception as exc:
        return [{"signal": "mp4_forensics", "label": f"read failed: {exc}", "weight": 0}]

    if len(data) < 16 or data[4:8] not in (b"ftyp", b"wide", b"mdat", b"moov"):
        # Not an ISO-BMFF / MP4-like file
        return []

    signals: list[dict] = []
    try:
        report = inspect_mp4(data)
    except Exception as exc:
        return [
            {
                "signal": "mp4_forensics",
                "label": f"mp4 parse failed: {exc}",
                "weight": 0,
            }
        ]

    signals.append(
        {
            "signal": "mp4_forensics",
            "label": (
                f"MP4 forensics: mdat_payload={report['mdat_payload']}, "
                f"stsz_total={report['stsz_total']}, "
                f"mdat_slack={report['mdat_slack']}"
            ),
            "weight": 0,
        }
    )

    slack = report["mdat_slack"]
    if slack >= 64:
        # Strong OpenPuff-like signal observed on real OpenPuff output (~1.8KB slack)
        weight = 50 if slack >= 256 else 35
        signals.append(
            {
                "signal": "mp4_forensics",
                "label": (
                    f"unreferenced mdat slack of {slack} bytes "
                    f"(mdat payload exceeds sum of stsz sample sizes) — "
                    "common in OpenPuff-style video steganography"
                ),
                "weight": weight,
            }
        )
        # Entropy of trailing slack region
        ent = report.get("slack_entropy")
        if ent is not None and ent >= 7.5 and slack >= 128:
            signals.append(
                {
                    "signal": "mp4_forensics",
                    "label": (
                        f"mdat slack is high-entropy ({ent:.2f} bits/byte) — "
                        "consistent with encrypted/whitened hidden data"
                    ),
                    "weight": 15,
                }
            )
    elif slack < 0:
        signals.append(
            {
                "signal": "mp4_forensics",
                "label": (
                    f"stsz sample total exceeds mdat payload by {-slack} bytes — "
                    "inconsistent sample table"
                ),
                "weight": 12,
            }
        )

    trailing = report.get("trailing_after_atoms", 0)
    if trailing >= 16:
        signals.append(
            {
                "signal": "mp4_forensics",
                "label": f"{trailing} bytes after the last top-level MP4 atom",
                "weight": 40 if trailing >= 64 else 25,
            }
        )

    return signals


def inspect_mp4(data: bytes) -> dict:
    atoms = list_top_level_atoms(data)
    mdat_payload = 0
    mdat_end = 0
    for off, typ, size in atoms:
        if typ == b"mdat":
            # payload excludes the 8-byte atom header (or 16 for largesize)
            hdr = 16 if size > 0 and data[off : off + 4] == b"\x00\x00\x00\x01" else 8
            payload = size - hdr
            mdat_payload += max(0, payload)
            mdat_end = max(mdat_end, off + size)

    stsz_total = sum_stsz_totals(data)
    slack = mdat_payload - stsz_total if stsz_total > 0 and mdat_payload > 0 else 0

    slack_entropy = None
    if slack >= 32 and mdat_end > 0:
        # Approximate: last `slack` bytes of the final mdat payload
        # Find last mdat
        for off, typ, size in reversed(atoms):
            if typ != b"mdat":
                continue
            hdr = 16 if data[off : off + 4] == b"\x00\x00\x00\x01" else 8
            payload = data[off + hdr : off + size]
            region = payload[-slack:] if len(payload) >= slack else payload
            slack_entropy = round(_entropy(region), 3)
            break

    last_atom_end = atoms[-1][0] + atoms[-1][2] if atoms else 0
    trailing = max(0, len(data) - last_atom_end)

    return {
        "atoms": [(off, typ.decode("latin1", "replace"), size) for off, typ, size in atoms],
        "mdat_payload": mdat_payload,
        "stsz_total": stsz_total,
        "mdat_slack": slack,
        "slack_entropy": slack_entropy,
        "trailing_after_atoms": trailing,
    }


def list_top_level_atoms(data: bytes) -> list[tuple[int, bytes, int]]:
    out: list[tuple[int, bytes, int]] = []
    i = 0
    n = len(data)
    while i + 8 <= n:
        size = struct.unpack(">I", data[i : i + 4])[0]
        typ = data[i + 4 : i + 8]
        if size == 1 and i + 16 <= n:
            size = struct.unpack(">Q", data[i + 8 : i + 16])[0]
        elif size == 0:
            size = n - i
        if size < 8 or i + size > n + 0:
            # tolerate minor overhang
            if size < 8:
                break
        out.append((i, typ, size))
        i += size
        if i >= n:
            break
    return out


def sum_stsz_totals(data: bytes) -> int:
    total = 0
    i = 0
    while True:
        j = data.find(b"stsz", i)
        if j < 0:
            break
        atom = j - 4
        i = j + 4
        if atom < 0 or atom + 20 > len(data):
            continue
        try:
            size = struct.unpack(">I", data[atom : atom + 4])[0]
            if size < 20 or atom + size > len(data):
                continue
            if data[atom + 4 : atom + 8] != b"stsz":
                continue
            body = data[atom + 8 : atom + size]
            sample_size = struct.unpack(">I", body[4:8])[0]
            count = struct.unpack(">I", body[8:12])[0]
            if count < 0 or count > 50_000_000:
                continue
            if sample_size != 0:
                total += sample_size * count
            else:
                need = 12 + 4 * count
                if need > len(body):
                    continue
                entries = struct.unpack(">" + "I" * count, body[12 : 12 + 4 * count])
                total += sum(entries)
        except Exception:
            continue
    return total


def _entropy(buf: bytes) -> float:
    if not buf:
        return 0.0
    counts = [0] * 256
    for b in buf:
        counts[b] += 1
    n = len(buf)
    ent = 0.0
    for c in counts:
        if c:
            p = c / n
            ent -= p * (p and __import__("math").log2(p))
    return ent
