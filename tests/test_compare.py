from pathlib import Path

from vsteg.compare import compare, write_html
from vsteg.methods import append

ROOT = Path(__file__).resolve().parents[1]
CARRIER = ROOT / "carrier.mp4"
SECRET = ROOT / "secret.txt"


def test_compare_same_file_similar(tmp_path):
    # Copy identical content via append of empty-ish? Just compare file to itself
    report = compare(CARRIER, CARRIER, sample_frames=4, deep=True)
    assert report.similarity in {"identical", "similar"}
    assert report.changed_count == 0
    assert "scalar" in report.charts
    assert report.media_a.get("codec")


def test_compare_append_differs(tmp_path):
    stego = tmp_path / "stego.mp4"
    append.encode(CARRIER, SECRET.read_bytes(), stego)
    report = compare(CARRIER, stego, sample_frames=4, deep=False)
    assert report.changed_count >= 1
    assert any(r["key"] == "size_bytes" and r["changed"] for r in report.attributes)


def test_compare_html(tmp_path):
    out = tmp_path / "report.html"
    report = compare(CARRIER, CARRIER, deep=False)
    write_html(report, out)
    text = out.read_text(encoding="utf-8")
    assert "chart.js" in text.lower()
    assert "vsteg compare" in text


def test_compare_openpuff_pair():
    orig = ROOT / "openpuff" / "original.mp4"
    stego = ROOT / "openpuff" / "stego.mp4"
    if not orig.exists() or not stego.exists():
        import pytest

        pytest.skip("openpuff sample pair not present")
    report = compare(orig, stego, sample_frames=4, deep=True)
    assert report.container_analysis.get("openpuff_like") is True
    assert report.container_analysis.get("mdat_slack_b", 0) >= 64
    assert report.frame_analysis.get("mean_abs_diff", 1) < 0.5
