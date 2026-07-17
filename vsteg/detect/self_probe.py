"""Active probes for vsteg's own embedding methods.

Generic statistical tests are noisy. These probes look for vsteg-specific
fingerprints and only score highly when a payload can be at least partially
verified (trailer decode, LSB header parse, or DCT sync + VSTG header).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vsteg import HEADER_SIZE, MAGIC
from vsteg import ecc
from vsteg.container import parse_header
from vsteg.methods import append
from vsteg.methods.dct import (
    DEFAULT_REDUNDANCY,
    DEFAULT_STRENGTH,
    SYNC_PATTERN,
    _extract_bits_from_y,
    _get_y_plane,
    _seed,
)
from vsteg.methods.lsb import _extract_from_planes, _seed as lsb_seed
from vsteg.probe import is_lossless_codec, probe


def analyze(path: str | Path) -> list[dict]:
    path = Path(path)
    signals: list[dict] = []
    signals.extend(_probe_append(path))
    signals.extend(_probe_lsb_header(path))
    signals.extend(_probe_dct_sync(path))
    return signals


def _probe_append(path: Path) -> list[dict]:
    out: list[dict] = []
    try:
        if not append.has_appended_payload(path):
            return out
        try:
            append.decode(path)
            out.append(
                {
                    "signal": "self_probe",
                    "label": "vsteg append payload verified (VSTG trailer decoded)",
                    "weight": 55,
                }
            )
        except Exception as exc:
            msg = str(exc).lower()
            if "password" in msg:
                out.append(
                    {
                        "signal": "self_probe",
                        "label": "vsteg append trailer present (encrypted; password required to fully verify)",
                        "weight": 48,
                    }
                )
            else:
                out.append(
                    {
                        "signal": "self_probe",
                        "label": f"vsteg-like trailer found but decode incomplete: {exc}",
                        "weight": 35,
                    }
                )
    except Exception as exc:
        out.append(
            {
                "signal": "self_probe",
                "label": f"append probe failed: {exc}",
                "weight": 0,
            }
        )
    return out


def _probe_lsb_header(path: Path) -> list[dict]:
    out: list[dict] = []
    try:
        info = probe(path)
    except Exception:
        return out

    if not is_lossless_codec(info.codec):
        return out

    try:
        import av
    except Exception:
        return out

    try:
        rng = np.random.default_rng(lsb_seed(None))
        extracted: list[int] = []
        need = HEADER_SIZE * 8
        container = av.open(str(path))
        for frame in container.decode(video=0):
            planes = frame.reformat(format="yuv444p").to_ndarray()
            extracted.extend(
                _extract_from_planes(planes, need - len(extracted), rng, bits=1)
            )
            if len(extracted) >= need:
                break
        container.close()
        if len(extracted) < need:
            return out
        raw = np.packbits(np.array(extracted[:need], dtype=np.uint8)).tobytes()
        if not raw.startswith(MAGIC):
            return out
        try:
            hdr = parse_header(raw)
            out.append(
                {
                    "signal": "self_probe",
                    "label": (
                        f"vsteg LSB header detected in lossless frames "
                        f"(method={hdr.method}, payload_len={hdr.length})"
                    ),
                    "weight": 55,
                }
            )
        except Exception:
            out.append(
                {
                    "signal": "self_probe",
                    "label": "vsteg LSB magic detected in lossless frame LSBs",
                    "weight": 45,
                }
            )
    except Exception as exc:
        out.append(
            {
                "signal": "self_probe",
                "label": f"LSB probe failed: {exc}",
                "weight": 0,
            }
        )
    return out


def _probe_dct_sync(path: Path) -> list[dict]:
    """Require sync match AND a parsable VSTG header after ECC decode."""
    out: list[dict] = []
    try:
        import av  # noqa: F401
    except Exception:
        return out

    candidates = [
        (DEFAULT_STRENGTH, DEFAULT_REDUNDANCY),
        (16.0, 9),
        (16.0, 11),
        (20.0, 11),
        (12.0, 7),
        (8.0, 5),
    ]
    seen: set[tuple[float, int]] = set()
    best_info = {
        "sync": -1,
        "verified": False,
        "strength": None,
        "redundancy": None,
    }

    for strength, redundancy in candidates:
        key = (strength, redundancy)
        if key in seen:
            continue
        seen.add(key)
        try:
            sync_score, verified = _dct_probe_once(
                path, strength=strength, redundancy=redundancy
            )
        except Exception:
            continue
        if sync_score > best_info["sync"] or (
            sync_score == best_info["sync"] and verified and not best_info["verified"]
        ):
            best_info = {
                "sync": sync_score,
                "verified": verified,
                "strength": strength,
                "redundancy": redundancy,
            }
        if verified:
            break

    sync_len = len(SYNC_PATTERN)
    sync_score = int(best_info["sync"])
    if sync_score < 0:
        return out

    out.append(
        {
            "signal": "self_probe",
            "label": (
                f"vsteg DCT sync probe best match={sync_score}/{sync_len} "
                f"(Δ={best_info['strength']}, redundancy={best_info['redundancy']}, "
                f"verified={best_info['verified']})"
            ),
            "weight": 0,
        }
    )

    # Only score when the payload header actually verifies — sync alone is too
    # collision-prone across thousands of bit offsets on clean video.
    if best_info["verified"]:
        out.append(
            {
                "signal": "self_probe",
                "label": (
                    f"vsteg DCT payload verified (sync {sync_score}/{sync_len}, "
                    f"VSTG header recovered)"
                ),
                "weight": 55,
            }
        )

    return out


def _dct_probe_once(
    path: Path,
    strength: float,
    redundancy: int,
    max_bits: int = 12000,
) -> tuple[int, bool]:
    """Return (best_sync_matches, verified_vstg_header)."""
    import av

    rng = np.random.default_rng(_seed(None))
    delta = float(strength)
    target = min(max_bits * redundancy, 80000)
    extracted: list[int] = []

    container = av.open(str(path))
    for frame in container.decode(video=0):
        if len(extracted) >= target:
            break
        y, _ = _get_y_plane(frame)
        need = target - len(extracted)
        extracted.extend(_extract_bits_from_y(y, need, rng, delta))
    container.close()

    n_full = (len(extracted) // redundancy) * redundancy
    if n_full < len(SYNC_PATTERN) * redundancy:
        return 0, False

    trimmed = np.array(extracted[:n_full], dtype=np.int8)
    groups = trimmed.reshape(-1, redundancy)
    bits = (groups.sum(axis=1) >= (redundancy / 2.0)).astype(np.uint8)

    sync = SYNC_PATTERN
    best = 0
    verified = False
    limit = min(len(bits) - len(sync), 4000)

    for i in range(max(0, limit)):
        score = int(np.sum(bits[i : i + len(sync)] == sync))
        if score > best:
            best = score
        # Only attempt expensive verify on near-perfect sync
        if score >= len(sync) - 1:
            payload_bits = bits[i + len(sync) :]
            if _bits_look_like_vstg(payload_bits):
                verified = True
                best = score
                break
    return best, verified


def _bits_look_like_vstg(payload_bits: np.ndarray) -> bool:
    """Pack bits → ECC decode → require a valid VSTG header."""
    n_bytes = len(payload_bits) // 8
    if n_bytes < 64:
        return False
    raw = np.packbits(payload_bits[: n_bytes * 8]).tobytes()
    # Try a few truncated ECC lengths near the start
    # ecc.decode already tries decreasing codeword counts
    try:
        container = ecc.decode(raw)
    except Exception:
        # Try shorter prefixes in case of trailing noise
        for cut in (0.75, 0.5, 0.35):
            end = int(len(raw) * cut)
            end -= end % 255  # RS_N
            if end < 255:
                continue
            try:
                container = ecc.decode(raw[:end])
                break
            except Exception:
                container = None
        if not container:
            return False

    if not container.startswith(MAGIC):
        return False
    try:
        parse_header(container[:HEADER_SIZE])
        return True
    except Exception:
        return False
