"""Unit tests for the payload container."""

from __future__ import annotations

import pytest

from vsteg import METHOD_APPEND
from vsteg.container import (
    ContainerError,
    pack,
    pack_payloads,
    unpack,
    unpack_matching,
    validate_decoy,
)


def test_roundtrip_no_password():
    raw = b"hello secret world"
    blob = pack(raw, METHOD_APPEND)
    hdr, out = unpack(blob)
    assert out == raw
    assert not hdr.encrypted
    assert hdr.compressed


def test_roundtrip_with_password():
    raw = b"top secret" * 100
    blob = pack(raw, METHOD_APPEND, password="hunter2")
    hdr, out = unpack(blob, password="hunter2")
    assert out == raw
    assert hdr.encrypted


def test_wrong_password():
    blob = pack(b"data", METHOD_APPEND, password="right")
    with pytest.raises(ContainerError, match="decryption failed|password"):
        unpack(blob, password="wrong")


def test_missing_password():
    blob = pack(b"data", METHOD_APPEND, password="secret")
    with pytest.raises(ContainerError, match="password required"):
        unpack(blob)


def test_tamper_detected():
    blob = bytearray(pack(b"data", METHOD_APPEND))
    blob[-1] ^= 0xFF
    with pytest.raises(ContainerError, match="CRC"):
        unpack(bytes(blob))


def test_bad_magic():
    blob = bytearray(pack(b"data", METHOD_APPEND))
    blob[0:4] = b"XXXX"
    with pytest.raises(ContainerError, match="magic"):
        unpack(bytes(blob))


def test_decoy_pair_unlocks_by_password():
    blob = pack_payloads(
        b"real-secret",
        METHOD_APPEND,
        password="real-pw",
        decoy=b"harmless-note",
        decoy_password="decoy-pw",
    )
    assert unpack_matching(blob, "real-pw") == b"real-secret"
    assert unpack_matching(blob, "decoy-pw") == b"harmless-note"
    with pytest.raises(ContainerError):
        unpack_matching(blob, "wrong")


def test_decoy_requires_distinct_passwords():
    with pytest.raises(ContainerError, match="differ"):
        validate_decoy(b"decoy", "same", "same")
    with pytest.raises(ContainerError, match="real password"):
        validate_decoy(b"decoy", None, "decoy-pw")
