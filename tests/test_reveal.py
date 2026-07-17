"""Reveal fail-fast and round-trip via shared decode_payload."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from vsteg.methods import append
from vsteg.reveal import RevealError, decode_payload

ROOT = Path(__file__).resolve().parents[1]
CARRIER = ROOT / "carrier.mp4"
SECRET = ROOT / "secret.txt"
OPENPUFF_STEGO = ROOT / "openpuff" / "stego.mp4"


def test_reveal_openpuff_fails_fast():
    if not OPENPUFF_STEGO.exists():
        pytest.skip("openpuff sample not present")

    start = time.perf_counter()
    with pytest.raises(RevealError, match="OpenPuff-like|does not appear to contain a vsteg"):
        decode_payload(OPENPUFF_STEGO, method="auto")
    elapsed = time.perf_counter() - start
    # Must not fall into multi-minute LSB/DCT scans
    assert elapsed < 8.0


def test_reveal_append_roundtrip(tmp_path):
    assert CARRIER.exists() and SECRET.exists()
    stego = tmp_path / "stego.mp4"
    secret = SECRET.read_bytes()
    append.encode(CARRIER, secret, stego)
    assert decode_payload(stego, method="auto") == secret
