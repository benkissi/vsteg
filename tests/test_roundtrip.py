"""Round-trip encode/decode tests for all methods."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from vsteg.methods import append, dct, lsb

ROOT = Path(__file__).resolve().parents[1]
CARRIER = ROOT / "carrier.mp4"
SECRET = ROOT / "secret.txt"


@pytest.fixture(scope="module")
def secret_bytes() -> bytes:
    return SECRET.read_bytes()


@pytest.fixture(scope="module")
def tiny_carrier(tmp_path_factory) -> Path:
    """Generate a short synthetic carrier for LSB/DCT tests (fast)."""
    out = tmp_path_factory.mktemp("vid") / "tiny.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=10",
            "-t",
            "3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


def test_append_roundtrip(tmp_path, secret_bytes):
    assert CARRIER.exists()
    stego = tmp_path / "stego_append.mp4"
    append.encode(CARRIER, secret_bytes, stego)
    recovered = append.decode(stego)
    assert recovered == secret_bytes


def test_append_roundtrip_password(tmp_path, secret_bytes):
    stego = tmp_path / "stego_append_pw.mp4"
    append.encode(CARRIER, secret_bytes, stego, password="testpass")
    recovered = append.decode(stego, password="testpass")
    assert recovered == secret_bytes


def test_append_wrong_password(tmp_path, secret_bytes):
    stego = tmp_path / "stego_append_badpw.mp4"
    append.encode(CARRIER, secret_bytes, stego, password="right")
    with pytest.raises(Exception):
        append.decode(stego, password="wrong")


def test_append_decoy_roundtrip(tmp_path, secret_bytes):
    stego = tmp_path / "stego_decoy.mp4"
    decoy = b"just a shopping list"
    append.encode(
        CARRIER,
        secret_bytes,
        stego,
        password="real-pw",
        decoy=decoy,
        decoy_password="decoy-pw",
    )
    assert append.decode(stego, password="real-pw") == secret_bytes
    assert append.decode(stego, password="decoy-pw") == decoy


def test_lsb_roundtrip(tmp_path, tiny_carrier, secret_bytes):
    stego = tmp_path / "stego_lsb.mkv"
    # Use a small secret if needed — secret.txt should be tiny
    lsb.encode(tiny_carrier, secret_bytes, stego)
    recovered = lsb.decode(stego)
    assert recovered == secret_bytes


def test_lsb_roundtrip_password(tmp_path, tiny_carrier, secret_bytes):
    stego = tmp_path / "stego_lsb_pw.mkv"
    lsb.encode(tiny_carrier, secret_bytes, stego, password="pw")
    recovered = lsb.decode(stego, password="pw")
    assert recovered == secret_bytes


def test_dct_roundtrip(tmp_path, tiny_carrier):
    # Small payload for robust method
    secret = b"robust-secret-ok"
    stego = tmp_path / "stego_dct.mp4"
    dct.encode(tiny_carrier, secret, stego, strength=20.0, crf=18, redundancy=11)
    recovered = dct.decode(stego, strength=20.0, redundancy=11)
    assert recovered == secret


def test_dct_survives_reencode(tmp_path, tiny_carrier):
    """Acceptance: payload survives an extra lossy H.264 pass."""
    secret = b"survive-me"
    stego = tmp_path / "stego_dct.mp4"
    dct.encode(tiny_carrier, secret, stego, strength=20.0, crf=18, redundancy=11)

    recompressed = tmp_path / "recompressed.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(stego),
            "-c:v",
            "libx264",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(recompressed),
        ],
        check=True,
        capture_output=True,
    )
    recovered = dct.decode(recompressed, strength=20.0, redundancy=11)
    assert recovered == secret
