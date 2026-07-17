"""Authenticated payload container: header + optional AES-GCM + CRC32."""

from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass
from typing import BinaryIO, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from vsteg import (
    FLAG_COMPRESSED,
    FLAG_ECC,
    FLAG_ENCRYPTED,
    HEADER_SIZE,
    MAGIC,
)

SCRYPT_N = 2**15
SCRYPT_R = 8
SCRYPT_P = 1
KEY_LEN = 32
SALT_LEN = 16
NONCE_LEN = 12
CHUNK = 1024 * 1024  # 1 MiB streaming chunks


class ContainerError(Exception):
    """Raised when packing/unpacking a container fails."""


@dataclass
class Header:
    version: int
    flags: int
    method: int
    salt: bytes
    nonce: bytes
    length: int
    crc32: int

    @property
    def encrypted(self) -> bool:
        return bool(self.flags & FLAG_ENCRYPTED)

    @property
    def compressed(self) -> bool:
        return bool(self.flags & FLAG_COMPRESSED)

    @property
    def ecc(self) -> bool:
        return bool(self.flags & FLAG_ECC)


def _derive_key(password: str, salt: bytes) -> bytes:
    kdf = Scrypt(salt=salt, length=KEY_LEN, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P)
    return kdf.derive(password.encode("utf-8"))


def pack(
    plaintext: bytes,
    method: int,
    password: Optional[str] = None,
    compress: bool = True,
    ecc: bool = False,
) -> bytes:
    """Build a complete container (header + payload bytes)."""
    flags = 0
    body = plaintext
    if compress:
        body = zlib.compress(body, level=6)
        flags |= FLAG_COMPRESSED
    if ecc:
        flags |= FLAG_ECC

    salt = b"\x00" * SALT_LEN
    nonce = b"\x00" * NONCE_LEN

    if password:
        salt = os.urandom(SALT_LEN)
        nonce = os.urandom(NONCE_LEN)
        key = _derive_key(password, salt)
        body = AESGCM(key).encrypt(nonce, body, None)
        flags |= FLAG_ENCRYPTED

    crc = zlib.crc32(body) & 0xFFFFFFFF
    header = struct.pack(
        ">4sBBB16s12sQI",
        MAGIC,
        1,  # version
        flags,
        method & 0xFF,
        salt,
        nonce,
        len(body),
        crc,
    )
    assert len(header) == HEADER_SIZE
    return header + body


def unpack(data: bytes, password: Optional[str] = None) -> tuple[Header, bytes]:
    """Parse container bytes and return (header, plaintext)."""
    if len(data) < HEADER_SIZE:
        raise ContainerError("data too short for header")
    header = parse_header(data[:HEADER_SIZE])
    body = data[HEADER_SIZE : HEADER_SIZE + header.length]
    if len(body) != header.length:
        raise ContainerError(
            f"truncated payload: expected {header.length} bytes, got {len(body)}"
        )
    return _finalize(header, body, password)


def parse_header(raw: bytes) -> Header:
    if len(raw) < HEADER_SIZE:
        raise ContainerError("header too short")
    magic, version, flags, method, salt, nonce, length, crc = struct.unpack(
        ">4sBBB16s12sQI", raw[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise ContainerError(f"bad magic: {magic!r}")
    if version != 1:
        raise ContainerError(f"unsupported version: {version}")
    return Header(
        version=version,
        flags=flags,
        method=method,
        salt=salt,
        nonce=nonce,
        length=length,
        crc32=crc,
    )


def find_magic(data: bytes, start: int = 0) -> int:
    """Return index of MAGIC in data, or -1."""
    return data.find(MAGIC, start)


def validate_decoy(
    decoy: Optional[bytes],
    password: Optional[str],
    decoy_password: Optional[str],
) -> None:
    """Enforce OpenPuff-style decoy rules when a decoy payload is supplied."""
    if decoy is None:
        if decoy_password:
            raise ContainerError("decoy password set but no decoy payload was provided")
        return
    if not decoy:
        raise ContainerError("decoy payload is empty")
    if not password:
        raise ContainerError("real password is required when using a decoy")
    if not decoy_password:
        raise ContainerError("decoy password is required when using a decoy")
    if password == decoy_password:
        raise ContainerError("decoy password must differ from the real password")


def pack_payloads(
    secret: bytes,
    method: int,
    password: Optional[str] = None,
    *,
    decoy: Optional[bytes] = None,
    decoy_password: Optional[str] = None,
    compress: bool = True,
    ecc: bool = False,
) -> bytes:
    """Pack secret, optionally followed by an encrypted decoy container.

    Layout when decoy is set: ``real_container || decoy_container``.
    Reveal tries each container with the supplied password and returns the match.
    """
    validate_decoy(decoy, password, decoy_password)
    real = pack(secret, method, password=password, compress=compress, ecc=ecc)
    if decoy is None:
        return real
    decoy_c = pack(
        decoy, method, password=decoy_password, compress=compress, ecc=ecc
    )
    return real + decoy_c


def unpack_matching(data: bytes, password: Optional[str] = None) -> bytes:
    """Scan for VSTG containers and return the first that unlocks with password."""
    if len(data) < HEADER_SIZE:
        raise ContainerError("data too short for header")

    idx = find_magic(data, 0)
    last_err: Exception | None = None
    seen = 0
    while idx >= 0:
        try:
            header = parse_header(data[idx : idx + HEADER_SIZE])
            end = idx + HEADER_SIZE + header.length
            if end > len(data):
                idx = find_magic(data, idx + 1)
                continue
            _, plaintext = unpack(data[idx:end], password)
            return plaintext
        except ContainerError as exc:
            last_err = exc
            seen += 1
            idx = find_magic(data, idx + 1)

    if seen == 0:
        raise ContainerError("no VSTG container found")
    if last_err:
        raise last_err
    raise ContainerError("no container matched the supplied password")


def _finalize(
    header: Header, body: bytes, password: Optional[str]
) -> tuple[Header, bytes]:
    actual_crc = zlib.crc32(body) & 0xFFFFFFFF
    if actual_crc != header.crc32:
        raise ContainerError(
            f"CRC mismatch: expected {header.crc32:#010x}, got {actual_crc:#010x}"
        )

    if header.encrypted:
        if not password:
            raise ContainerError("password required (payload is encrypted)")
        key = _derive_key(password, header.salt)
        try:
            body = AESGCM(key).decrypt(header.nonce, body, None)
        except Exception as exc:
            raise ContainerError(
                "decryption failed (wrong password or tampered data)"
            ) from exc

    if header.compressed:
        try:
            body = zlib.decompress(body)
        except zlib.error as exc:
            raise ContainerError("decompression failed") from exc

    return header, body


def pack_stream(
    src: BinaryIO,
    method: int,
    password: Optional[str] = None,
    compress: bool = True,
    ecc: bool = False,
) -> bytes:
    """Read src via chunks then pack into a container."""
    chunks: list[bytes] = []
    while True:
        block = src.read(CHUNK)
        if not block:
            break
        chunks.append(block)
    return pack(b"".join(chunks), method, password=password, compress=compress, ecc=ecc)
