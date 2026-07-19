"""Method C — DCT-domain QIM embed with Reed-Solomon ECC (compression-robust)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import av
import numpy as np
from scipy.fft import dctn, idctn

from vsteg import HEADER_SIZE, MAGIC, METHOD_DCT
from vsteg import ecc
from vsteg.container import (
    ContainerError,
    pack_payloads,
    parse_header,
    unpack_matching,
)
from vsteg.probe import probe

MID_COEFFS = [
    (1, 2),
    (2, 1),
    (2, 2),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
]

DEFAULT_STRENGTH = 16.0
DEFAULT_REDUNDANCY = 9
DEFAULT_CRF = 18
DEFAULT_MAX_ROBUST = 1 * 1024 * 1024
SYNC_PATTERN = np.array(
    [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0], dtype=np.uint8
)


class DCTError(Exception):
    pass


def _seed(password: Optional[str]) -> int:
    material = (password or "vsteg-dct-default").encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _block_grid(h: int, w: int) -> tuple[int, int]:
    return h // 8, w // 8


def capacity_bits(
    width: int,
    height: int,
    frames: int,
    redundancy: int = DEFAULT_REDUNDANCY,
    coeffs_per_block: int = len(MID_COEFFS),
) -> int:
    bh, bw = _block_grid(height, width)
    usable = bh * bw * frames * coeffs_per_block
    usable -= len(SYNC_PATTERN) * redundancy
    return max(0, usable // redundancy)


def capacity_bytes(
    width: int,
    height: int,
    frames: int,
    redundancy: int = DEFAULT_REDUNDANCY,
) -> int:
    bits = capacity_bits(width, height, frames, redundancy=redundancy)
    ecc_bytes = bits // 8
    from vsteg.ecc import RS_K, RS_N

    approx = int(ecc_bytes * RS_K / RS_N) - 4
    return max(0, approx - HEADER_SIZE)


def _qim_embed(coeff: float, bit: int, delta: float) -> float:
    q = np.round(coeff / delta)
    if int(q) % 2 == bit:
        return float(q * delta)
    down, up = q - 1, q + 1
    if abs(coeff - down * delta) <= abs(coeff - up * delta):
        return float(down * delta)
    return float(up * delta)


def _qim_extract(coeff: float, delta: float) -> int:
    return int(np.round(coeff / delta)) % 2


def _iter_block_positions(h: int, w: int, rng: np.random.Generator):
    bh, bw = _block_grid(h, w)
    positions = [(by, bx) for by in range(bh) for bx in range(bw)]
    order = rng.permutation(len(positions))
    for i in order:
        yield positions[i]


def _get_y_plane(frame: av.VideoFrame) -> np.ndarray:
    """Return a writable (H, W) copy of the luma plane."""
    yuv = frame.reformat(format="yuv420p")
    # to_ndarray for yuv420p returns a single buffer; use planes
    y = np.frombuffer(yuv.planes[0], dtype=np.uint8).reshape(
        yuv.planes[0].height, yuv.planes[0].line_size
    )[:, : yuv.width].copy()
    return y, yuv


def _copy_plane(src, dst) -> None:
    """Copy a video plane accounting for linesize padding (PyAV/FFmpeg stride).

    ``bytes(src)`` can be larger than ``dst`` when the decoder pads rows
    (common on Linux/WSL). Copy only the active ``width × height`` region.
    """
    src_view = np.frombuffer(src, dtype=np.uint8).reshape(src.height, src.line_size)
    dst_view = np.frombuffer(dst, dtype=np.uint8).reshape(dst.height, dst.line_size)
    w = min(src.width, dst.width)
    h = min(src.height, dst.height)
    dst_view[:h, :w] = src_view[:h, :w]
    if dst.line_size > w:
        dst_view[:h, w:] = 0


def _frame_from_yuv420(y: np.ndarray, template: av.VideoFrame) -> av.VideoFrame:
    """Build a yuv420p frame with modified Y, copying UV from template."""
    base = template.reformat(format="yuv420p")
    new = av.VideoFrame(width=base.width, height=base.height, format="yuv420p")

    # Y plane — respect linesize padding
    y_plane = new.planes[0]
    y_buf = np.frombuffer(y_plane, dtype=np.uint8).reshape(
        y_plane.height, y_plane.line_size
    )
    y_buf[:, : base.width] = np.clip(y, 0, 255).astype(np.uint8)
    if y_plane.line_size > base.width:
        y_buf[:, base.width :] = 0

    # Copy U and V from template (stride-safe; never bytes(src) → update)
    for i in (1, 2):
        _copy_plane(base.planes[i], new.planes[i])

    new.pts = template.pts
    new.time_base = template.time_base
    return new


def _embed_bits_in_y(
    y: np.ndarray,
    robust_bits: np.ndarray,
    bit_idx: int,
    rng: np.random.Generator,
    delta: float,
) -> tuple[np.ndarray, int]:
    h, w = y.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    y = y[:h8, :w8].astype(np.float64)
    n_bits = len(robust_bits)

    for by, bx in _iter_block_positions(h8, w8, rng):
        if bit_idx >= n_bits:
            break
        block = y[by * 8 : (by + 1) * 8, bx * 8 : (bx + 1) * 8]
        coeffs = dctn(block, type=2, norm="ortho")
        for ri, ci in MID_COEFFS:
            if bit_idx >= n_bits:
                break
            coeffs[ri, ci] = _qim_embed(coeffs[ri, ci], int(robust_bits[bit_idx]), delta)
            bit_idx += 1
        recon = idctn(coeffs, type=2, norm="ortho")
        y[by * 8 : (by + 1) * 8, bx * 8 : (bx + 1) * 8] = recon

    # Write back into full-size array if cropped
    out = np.zeros((h, w), dtype=np.float64)
    out[:h8, :w8] = y
    if h > h8:
        out[h8:, :] = 0
    if w > w8:
        out[:, w8:] = 0
    # Preserve any leftover rows/cols from original by returning clipped y only
    # Callers pass already-sized Y matching frame — restore unused border from input
    return np.clip(y, 0, 255), bit_idx


def _extract_bits_from_y(
    y: np.ndarray,
    need: int,
    rng: np.random.Generator,
    delta: float,
) -> list[int]:
    h, w = y.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    y = y[:h8, :w8].astype(np.float64)
    out: list[int] = []
    for by, bx in _iter_block_positions(h8, w8, rng):
        if len(out) >= need:
            break
        block = y[by * 8 : (by + 1) * 8, bx * 8 : (bx + 1) * 8]
        coeffs = dctn(block, type=2, norm="ortho")
        for ri, ci in MID_COEFFS:
            if len(out) >= need:
                break
            out.append(_qim_extract(coeffs[ri, ci], delta))
    return out


def encode(
    carrier: str | Path,
    secret: bytes,
    output: str | Path,
    password: Optional[str] = None,
    strength: float = DEFAULT_STRENGTH,
    crf: int = DEFAULT_CRF,
    redundancy: int = DEFAULT_REDUNDANCY,
    max_robust: int = DEFAULT_MAX_ROBUST,
    *,
    decoy: Optional[bytes] = None,
    decoy_password: Optional[str] = None,
) -> Path:
    carrier = Path(carrier)
    output = Path(output)
    info = probe(carrier)
    if info.frames <= 0:
        raise DCTError("could not determine frame count")

    payload_len = len(secret) + (len(decoy) if decoy is not None else 0)
    if payload_len > max_robust:
        raise DCTError(
            f"payload {payload_len} bytes exceeds --max-robust {max_robust}"
        )

    cap = capacity_bytes(info.width, info.height, info.frames, redundancy=redundancy)
    if payload_len > cap:
        raise DCTError(
            f"payload too large for robust capacity: need {payload_len}, have ~{cap}"
        )

    try:
        container = pack_payloads(
            secret,
            METHOD_DCT,
            password=password,
            decoy=decoy,
            decoy_password=decoy_password,
            compress=True,
            ecc=True,
        )
    except ContainerError as exc:
        raise DCTError(str(exc)) from exc
    ecc_payload = ecc.encode(container)
    bit_array = np.unpackbits(np.frombuffer(ecc_payload, dtype=np.uint8))
    stream_bits = np.concatenate([SYNC_PATTERN, bit_array])
    robust_bits = np.repeat(stream_bits, redundancy)

    # Decoy mode uses the default seed so either password can locate the bits.
    position_password = None if decoy is not None else password
    rng = np.random.default_rng(_seed(position_password))
    delta = float(strength)

    in_c = av.open(str(carrier))
    in_stream = in_c.streams.video[0]
    in_stream.thread_type = "AUTO"
    rate = in_stream.average_rate or 30

    out_c = av.open(str(output), mode="w")
    out_stream = out_c.add_stream("libx264", rate=rate)
    out_stream.width = in_stream.width
    out_stream.height = in_stream.height
    out_stream.pix_fmt = "yuv420p"
    out_stream.options = {"crf": str(crf), "preset": "medium", "tune": "film"}

    bit_idx = 0
    n_bits = len(robust_bits)

    for frame in in_c.decode(video=0):
        y, yuv = _get_y_plane(frame)
        if bit_idx < n_bits:
            # Keep full Y; embed only in 8-aligned region
            h, w = y.shape
            h8, w8 = (h // 8) * 8, (w // 8) * 8
            region = y[:h8, :w8].copy()
            region, bit_idx = _embed_bits_in_y(region, robust_bits, bit_idx, rng, delta)
            y = y.copy()
            y[:h8, :w8] = region
        new_frame = _frame_from_yuv420(y, frame)
        for pkt in out_stream.encode(new_frame):
            out_c.mux(pkt)

    for pkt in out_stream.encode():
        out_c.mux(pkt)

    in_c.close()
    out_c.close()

    if bit_idx < n_bits:
        raise DCTError(f"ran out of capacity ({bit_idx}/{n_bits} robust bits)")
    return output


def decode(
    path: str | Path,
    password: Optional[str] = None,
    strength: float = DEFAULT_STRENGTH,
    redundancy: int = DEFAULT_REDUNDANCY,
    max_bytes: int = DEFAULT_MAX_ROBUST,
) -> bytes:
    path = Path(path)
    seed_candidates: list[Optional[str]] = []
    if password is not None:
        seed_candidates.append(password)
    seed_candidates.append(None)

    errors: list[str] = []
    for seed_pw in seed_candidates:
        try:
            return _decode_with_seed(
                path, password, strength, redundancy, max_bytes, seed_pw
            )
        except Exception as exc:
            errors.append(str(exc))
    raise DCTError(errors[-1] if errors else "DCT decode failed")


def _decode_with_seed(
    path: Path,
    password: Optional[str],
    strength: float,
    redundancy: int,
    max_bytes: int,
    seed_password: Optional[str],
) -> bytes:
    info = probe(path)
    rng = np.random.default_rng(_seed(seed_password))
    delta = float(strength)

    max_robust_bits = capacity_bits(
        info.width, info.height, max(info.frames, 1), redundancy
    )
    # Extra room for an optional decoy container after the real one.
    target = min(
        (len(SYNC_PATTERN) + (HEADER_SIZE + max_bytes + 2048) * 8 * 2) * redundancy,
        (max_robust_bits + len(SYNC_PATTERN)) * redundancy + redundancy * 128,
    )

    extracted: list[int] = []
    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    for frame in container.decode(video=0):
        if len(extracted) >= target:
            break
        y, _ = _get_y_plane(frame)
        need = target - len(extracted)
        extracted.extend(_extract_bits_from_y(y, need, rng, delta))

    container.close()

    n_full = (len(extracted) // redundancy) * redundancy
    if n_full == 0:
        raise DCTError("no bits extracted")
    trimmed = np.array(extracted[:n_full], dtype=np.int8)
    groups = trimmed.reshape(-1, redundancy)
    bits = (groups.sum(axis=1) >= (redundancy / 2.0)).astype(np.uint8)

    sync = SYNC_PATTERN
    sync_pos = -1
    search_limit = min(len(bits) - len(sync), 20000)
    best_i, best_score = 0, -1
    for i in range(max(0, search_limit)):
        score = int(np.sum(bits[i : i + len(sync)] == sync))
        if score > best_score:
            best_score = score
            best_i = i
        if score == len(sync):
            sync_pos = i
            break
    if sync_pos < 0:
        if best_score < len(sync) - 3:
            raise DCTError(
                f"sync pattern not found (best match {best_score}/{len(sync)})"
            )
        sync_pos = best_i

    payload_bits = bits[sync_pos + len(sync) :]
    n_bytes = len(payload_bits) // 8
    raw_ecc = np.packbits(payload_bits[: n_bytes * 8]).tobytes()

    try:
        container_bytes = ecc.decode(raw_ecc)
    except ecc.ECCError as exc:
        raise DCTError(f"ECC decode failed: {exc}") from exc

    try:
        return unpack_matching(container_bytes, password)
    except ContainerError as exc:
        raise DCTError(f"container unpack failed: {exc}") from exc


def looks_like_payload(
    path: str | Path,
    password: Optional[str] = None,
    strength: float = DEFAULT_STRENGTH,
    redundancy: int = DEFAULT_REDUNDANCY,
    max_bits: int = 12000,
) -> bool:
    """Cheap preflight: true only if DCT sync + VSTG header verify quickly.

    Used by auto-reveal to skip multi-minute frame scans on foreign encodings.
    """
    path = Path(path)
    try:
        rng = np.random.default_rng(_seed(password))
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
            return False

        trimmed = np.array(extracted[:n_full], dtype=np.int8)
        groups = trimmed.reshape(-1, redundancy)
        bits = (groups.sum(axis=1) >= (redundancy / 2.0)).astype(np.uint8)
        sync = SYNC_PATTERN
        limit = min(len(bits) - len(sync), 4000)

        for i in range(max(0, limit)):
            score = int(np.sum(bits[i : i + len(sync)] == sync))
            if score < len(sync) - 1:
                continue
            if _bits_look_like_vstg(bits[i + len(sync) :]):
                return True
        return False
    except Exception:
        return False


def _bits_look_like_vstg(payload_bits: np.ndarray) -> bool:
    n_bytes = len(payload_bits) // 8
    if n_bytes < 64:
        return False
    raw = np.packbits(payload_bits[: n_bytes * 8]).tobytes()
    try:
        container_bytes = ecc.decode(raw)
    except Exception:
        container_bytes = None
        for cut in (0.75, 0.5, 0.35):
            end = int(len(raw) * cut)
            end -= end % 255
            if end < 255:
                continue
            try:
                container_bytes = ecc.decode(raw[:end])
                break
            except Exception:
                continue
        if not container_bytes:
            return False

    if not container_bytes.startswith(MAGIC):
        return False
    try:
        parse_header(container_bytes[:HEADER_SIZE])
        return True
    except Exception:
        return False
