"""Detection pipeline tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from vsteg.detect.report import detect
from vsteg.methods import append

ROOT = Path(__file__).resolve().parents[1]
CARRIER = ROOT / "carrier.mp4"
SECRET = ROOT / "secret.txt"


def test_clean_carrier_low_score():
    report = detect(CARRIER, deep=False)
    # Clean carrier should not be likely-stego on structural/signature alone
    assert report.verdict in {"clean", "suspicious"}
    assert report.score < 60


def test_detects_append_output(tmp_path):
    stego = tmp_path / "stego.mp4"
    append.encode(CARRIER, SECRET.read_bytes(), stego)
    report = detect(stego, deep=False)
    assert report.verdict in {"suspicious", "likely-stego"}
    assert report.score >= 25
    labels = " ".join(s.get("label", "") for s in report.signals)
    assert "VSTG" in labels or "trailer" in labels.lower() or "appended" in labels.lower()


def test_ffprobe_fields_in_report():
    report = detect(CARRIER, deep=False)
    assert report.media.get("probe_source") in {"ffprobe", "pyav"}
    assert "codec" in report.media
    assert "audio_count" in report.media
    assert any(c["id"] == "ffmpeg" for c in report.categories)
    # ffprobe layer should at least emit informational notes
    assert any(s.get("signal") == "ffmpeg" for s in report.signals)


def test_clean_carrier_deep_not_suspicious():
    report = detect(CARRIER, deep=True)
    assert report.verdict == "clean"
    assert report.score < 25


def test_detects_openpuff_mdat_slack():
    stego = ROOT / "openpuff" / "stego.mp4"
    original = ROOT / "openpuff" / "original.mp4"
    if not stego.exists() or not original.exists():
        pytest.skip("openpuff sample pair not present")
    clean = detect(original, deep=False)
    assert clean.verdict == "clean"
    report = detect(stego, deep=False)
    assert report.verdict in {"suspicious", "likely-stego"}
    assert any(
        s.get("signal") == "mp4_forensics" and s.get("weight", 0) >= 35
        for s in report.signals
    )


def test_detects_dct_via_self_probe(tmp_path):
    import subprocess

    from vsteg.methods import dct

    tiny = tmp_path / "tiny.mp4"
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
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(tiny),
        ],
        check=True,
        capture_output=True,
    )
    stego = tmp_path / "stego_dct.mp4"
    dct.encode(tiny, b"hide-me", stego, strength=20.0, crf=18, redundancy=11)
    report = detect(stego, deep=False)
    assert report.verdict in {"suspicious", "likely-stego"}
    assert any(s.get("signal") == "self_probe" and s.get("weight", 0) >= 35 for s in report.signals)
