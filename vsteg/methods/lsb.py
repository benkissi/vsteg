"""Method B — lossless LSB embed in YUV planes, re-encode as FFV1/MKV."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import av
import numpy as np

from vsteg import HEADER_SIZE, METHOD_LSB
from vsteg.container import (
    ContainerError,
    pack_payloads,
    parse_header,
    unpack_matching,
)
from vsteg.probe import probe

CAPACITY_MARGIN = 0.90


class LSBError(Exception):
    pass


def capacity_bytes(
    width: int, height: int, frames: int, bits: int = 1, channels: int = 3
) -> int:
    raw = (frames * width * height * channels * bits) // 8
    return max(0, raw - HEADER_SIZE)


def _seed(password: Optional[str]) -> int:
    material = (password or "vsteg-lsb-default").encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _embed_into_planes(
    planes: np.ndarray,
    bit_array: np.ndarray,
    bit_idx: int,
    rng: np.random.Generator,
    bits: int,
) -> tuple[np.ndarray, int]:
    """planes: (3, H, W) uint8. Return (planes, new_bit_idx)."""
    n_bits = len(bit_array)
    if bit_idx >= n_bits:
        return planes, bit_idx

    flat = planes.reshape(-1).copy()
    remaining = n_bits - bit_idx
    mask = (0xFF << bits) & 0xFF

    if bits == 1:
        n = min(remaining, flat.size)
        positions = rng.permutation(flat.size)[:n]
        vals = flat[positions].astype(np.uint8)
        flat[positions] = (vals & mask) | bit_array[bit_idx : bit_idx + n]
        bit_idx += n
    else:
        n_samples = min((remaining + bits - 1) // bits, flat.size)
        positions = rng.permutation(flat.size)[:n_samples]
        vals = flat[positions].astype(np.uint8)
        out_vals = np.empty(n_samples, dtype=np.uint8)
        for i in range(n_samples):
            v = 0
            for b in range(bits):
                src = bit_idx + i * bits + b
                if src < n_bits:
                    v |= int(bit_array[src]) << b
            out_vals[i] = (vals[i] & mask) | v
        flat[positions] = out_vals
        bit_idx = min(n_bits, bit_idx + n_samples * bits)

    return flat.reshape(planes.shape), bit_idx


def _extract_from_planes(
    planes: np.ndarray,
    need: int,
    rng: np.random.Generator,
    bits: int,
) -> list[int]:
    flat = planes.reshape(-1)
    out: list[int] = []
    if bits == 1:
        n = min(need, flat.size)
        positions = rng.permutation(flat.size)[:n]
        vals = flat[positions].astype(np.uint8)
        out.extend((vals & 1).tolist())
    else:
        n_samples = min((need + bits - 1) // bits, flat.size)
        positions = rng.permutation(flat.size)[:n_samples]
        vals = flat[positions].astype(np.uint8)
        for v in vals:
            for b in range(bits):
                if len(out) >= need:
                    break
                out.append((int(v) >> b) & 1)
    return out


def encode(
    carrier: str | Path,
    secret: bytes,
    output: str | Path,
    password: Optional[str] = None,
    bits: int = 1,
    *,
    decoy: Optional[bytes] = None,
    decoy_password: Optional[str] = None,
) -> Path:
    if bits < 1 or bits > 3:
        raise LSBError("--bits must be 1..3")

    carrier = Path(carrier)
    output = Path(output)
    info = probe(carrier)
    if info.frames <= 0 or info.width <= 0:
        raise LSBError("could not determine video dimensions/frames")

    cap = capacity_bytes(info.width, info.height, info.frames, bits=bits)
    try:
        container_bytes = pack_payloads(
            secret,
            METHOD_LSB,
            password=password,
            decoy=decoy,
            decoy_password=decoy_password,
            compress=True,
        )
    except ContainerError as exc:
        raise LSBError(str(exc)) from exc
    if len(container_bytes) > int(cap * CAPACITY_MARGIN):
        raise LSBError(
            f"payload too large: need {len(container_bytes)} bytes, "
            f"usable ~{int(cap * CAPACITY_MARGIN)} (raw {cap}, bits={bits})"
        )

    # Decoy mode uses the default seed so either password can locate the bits.
    position_password = None if decoy is not None else password
    bit_array = np.unpackbits(np.frombuffer(container_bytes, dtype=np.uint8))
    rng = np.random.default_rng(_seed(position_password))
    bit_idx = 0

    in_c = av.open(str(carrier))
    in_stream = in_c.streams.video[0]
    in_stream.thread_type = "AUTO"
    rate = in_stream.average_rate or 30

    out_c = av.open(str(output), mode="w")
    out_stream = out_c.add_stream("ffv1", rate=rate)
    out_stream.width = in_stream.width
    out_stream.height = in_stream.height
    out_stream.pix_fmt = "yuv444p"

    for frame in in_c.decode(video=0):
        planes = frame.reformat(format="yuv444p").to_ndarray().copy()
        planes, bit_idx = _embed_into_planes(planes, bit_array, bit_idx, rng, bits)
        new_frame = av.VideoFrame.from_ndarray(planes, format="yuv444p")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        for pkt in out_stream.encode(new_frame):
            out_c.mux(pkt)

    for pkt in out_stream.encode():
        out_c.mux(pkt)

    in_c.close()
    out_c.close()

    if bit_idx < len(bit_array):
        raise LSBError(f"capacity exhausted ({bit_idx}/{len(bit_array)} bits)")
    return output


def decode(
    path: str | Path,
    password: Optional[str] = None,
    bits: int = 1,
    max_bytes: int = 64 * 1024 * 1024,
) -> bytes:
    if bits < 1 or bits > 3:
        raise LSBError("--bits must be 1..3")

    path = Path(path)
    # Try password-seeded positions first (legacy), then default seed (decoy mode).
    seed_candidates: list[Optional[str]] = []
    if password is not None:
        seed_candidates.append(password)
    seed_candidates.append(None)

    errors: list[str] = []
    for seed_pw in seed_candidates:
        try:
            return _decode_with_seed(path, password, bits, max_bytes, seed_pw)
        except Exception as exc:
            errors.append(str(exc))
    raise LSBError(
        "failed to unpack LSB payload: " + (errors[-1] if errors else "unknown")
    )


def _decode_with_seed(
    path: Path,
    password: Optional[str],
    bits: int,
    max_bytes: int,
    seed_password: Optional[str],
) -> bytes:
    info = probe(path)
    rng = np.random.default_rng(_seed(seed_password))

    header_bits = HEADER_SIZE * 8
    cap_bits = capacity_bytes(info.width, info.height, max(info.frames, 1), bits) * 8
    max_bits = min(max_bytes * 8, cap_bits + header_bits)

    extracted: list[int] = []
    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    for frame in container.decode(video=0):
        need = max_bits - len(extracted)
        if need <= 0:
            break
        planes = frame.reformat(format="yuv444p").to_ndarray()
        extracted.extend(_extract_from_planes(planes, need, rng, bits))
        max_bits = min(max_bits, _target_bits_from_extracted(extracted, max_bits))

    container.close()

    if len(extracted) < header_bits:
        raise LSBError("not enough bits for header")

    n_bytes = len(extracted) // 8
    raw = np.packbits(np.array(extracted[: n_bytes * 8], dtype=np.uint8)).tobytes()
    try:
        return unpack_matching(raw, password)
    except ContainerError as exc:
        raise LSBError(f"failed to unpack LSB payload: {exc}") from exc


def _target_bits_from_extracted(extracted: list[int], current_max: int) -> int:
    """Grow/shrink extract target once container header(s) are readable."""
    header_bits = HEADER_SIZE * 8
    if len(extracted) < header_bits:
        return current_max
    hdr_bytes = np.packbits(
        np.array(extracted[:header_bits], dtype=np.uint8)
    ).tobytes()
    try:
        hdr = parse_header(hdr_bytes)
    except ContainerError:
        return current_max

    first_end = HEADER_SIZE + hdr.length
    # Allow a second (decoy) container after the real one.
    dual_bits = (first_end + HEADER_SIZE) * 8
    if len(extracted) >= dual_bits:
        second = np.packbits(
            np.array(extracted[first_end * 8 : dual_bits], dtype=np.uint8)
        ).tobytes()
        try:
            hdr2 = parse_header(second)
            return min(current_max, (first_end + HEADER_SIZE + hdr2.length) * 8)
        except ContainerError:
            return min(current_max, first_end * 8)
    return min(current_max, max(first_end * 8, dual_bits))
