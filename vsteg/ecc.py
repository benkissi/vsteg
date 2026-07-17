"""Reed-Solomon ECC + block interleaving for Method C."""

from __future__ import annotations

from reedsolo import RSCodec

# RS(n, k): n = 255, k = 223 → 32 parity bytes (~12.5% overhead)
# We use a higher overhead for robustness: RS(255, 191) → 64 parity (~25%)
RS_N = 255
RS_K = 191
RS_NSYM = RS_N - RS_K  # 64


class ECCError(Exception):
    """ECC encode/decode failure."""


def _codec() -> RSCodec:
    return RSCodec(RS_NSYM)


def encode(data: bytes) -> bytes:
    """Apply Reed-Solomon framing then interleave.

    Data is split into RS_K-byte chunks, each encoded to RS_N bytes,
    then columns are interleaved so burst errors from compression
    don't wipe an entire codeword.
    """
    if not data:
        return b""
    # Length prefix (4 bytes) so decode knows original size
    framed = len(data).to_bytes(4, "big") + data
    pad = (-len(framed)) % RS_K
    if pad:
        framed += b"\x00" * pad

    rs = _codec()
    codewords: list[bytes] = []
    for i in range(0, len(framed), RS_K):
        codewords.append(bytes(rs.encode(framed[i : i + RS_K])))

    return _interleave(codewords)


def decode(encoded: bytes) -> bytes:
    """De-interleave and Reed-Solomon decode; return original payload.

    Extra trailing bytes (common when extracting a bit-budget larger than the
    true payload) would corrupt column-interleaved codewords, so we try
    decreasing codeword counts until RS+length-prefix validates.
    """
    if not encoded:
        raise ECCError("empty ECC payload")
    if len(encoded) % RS_N != 0:
        encoded = encoded[: len(encoded) - (len(encoded) % RS_N)]
    if not encoded:
        raise ECCError("no complete codewords")

    max_words = len(encoded) // RS_N
    last_err: Exception | None = None
    # Prefer longer candidates first (correct length is usually near the top
    # when extraction overshoots only modestly).
    for n_words in range(max_words, 0, -1):
        chunk = encoded[: n_words * RS_N]
        try:
            return _decode_exact(chunk, n_words)
        except ECCError as exc:
            last_err = exc
            continue
    raise ECCError(f"Reed-Solomon decode failed: {last_err}")


def _decode_exact(encoded: bytes, n_words: int) -> bytes:
    codewords = _deinterleave(encoded, n_words)
    rs = _codec()
    recovered = bytearray()
    for cw in codewords:
        try:
            decoded, _, _ = rs.decode(cw)
            recovered.extend(decoded)
        except Exception as exc:
            raise ECCError(f"Reed-Solomon decode failed: {exc}") from exc

    if len(recovered) < 4:
        raise ECCError("recovered data too short")
    length = int.from_bytes(recovered[:4], "big")
    if length < 0 or length > len(recovered) - 4:
        raise ECCError(f"invalid length prefix: {length}")
    payload = bytes(recovered[4 : 4 + length])
    if len(payload) != length:
        raise ECCError(f"length mismatch: expected {length}, got {len(payload)}")
    # Reject absurd lengths that only "fit" due to padding
    if length == 0:
        raise ECCError("empty payload")
    return payload


def overhead_ratio() -> float:
    """Approximate expansion factor (encoded/plaintext) ignoring pad."""
    return RS_N / RS_K


def encoded_size(plaintext_len: int) -> int:
    """Bytes after length-prefix + pad + RS."""
    framed = 4 + plaintext_len
    pad = (-framed) % RS_K
    n_words = (framed + pad) // RS_K
    return n_words * RS_N


def _interleave(codewords: list[bytes]) -> bytes:
    """Column-major interleave of equal-length codewords."""
    if not codewords:
        return b""
    n = len(codewords)
    out = bytearray(n * RS_N)
    for col in range(RS_N):
        for row, cw in enumerate(codewords):
            out[col * n + row] = cw[col]
    return bytes(out)


def _deinterleave(data: bytes, n_words: int) -> list[bytes]:
    words = [bytearray(RS_N) for _ in range(n_words)]
    for col in range(RS_N):
        for row in range(n_words):
            words[row][col] = data[col * n_words + row]
    return [bytes(w) for w in words]
