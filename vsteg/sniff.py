"""Guess a file extension from payload magic bytes / content."""

from __future__ import annotations

import json
import re


def guess_extension(data: bytes) -> str:
    """Return a file extension (with leading dot) for *data*."""
    if not data:
        return ".bin"

    # Image
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    if data.startswith(b"BM"):
        return ".bmp"

    # Documents / archives
    if data.startswith(b"%PDF"):
        return ".pdf"
    if data.startswith(b"PK\x03\x04"):
        # ZIP family — try to distinguish Office
        lower = data[:4096].lower()
        if b"word/" in lower:
            return ".docx"
        if b"xl/" in lower:
            return ".xlsx"
        if b"ppt/" in lower:
            return ".pptx"
        return ".zip"
    if data.startswith(b"\x1f\x8b"):
        return ".gz"
    if data.startswith(b"Rar!\x1a\x07"):
        return ".rar"
    if data.startswith(b"7z\xbc\xaf\x27\x1c"):
        return ".7z"

    # Media
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"qt  ", b"M4V ", b"M4A "):
            return ".mov" if brand == b"qt  " else ".mp4"
        return ".mp4"
    if data.startswith(b"\x1a\x45\xdf\xa3"):
        return ".mkv"
    if data.startswith(b"ID3") or (
        len(data) > 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0
    ):
        return ".mp3"
    if data.startswith(b"OggS"):
        return ".ogg"
    if data.startswith(b"fLaC"):
        return ".flac"
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        return ".wav"

    # Text / structured text
    if _looks_like_json(data):
        return ".json"
    if _looks_like_html(data):
        return ".html"
    if _looks_like_xml(data):
        return ".xml"
    if _looks_like_text(data):
        return ".txt"

    return ".bin"


def suggested_filename(data: bytes, stem: str = "recovered") -> str:
    return f"{stem}{guess_extension(data)}"


def _looks_like_json(data: bytes) -> bool:
    sample = data[:8192].lstrip()
    if not sample or sample[0:1] not in (b"{", b"["):
        return False
    try:
        json.loads(data.decode("utf-8"))
        return True
    except Exception:
        return False


def _looks_like_html(data: bytes) -> bool:
    head = data[:1024].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html")


def _looks_like_xml(data: bytes) -> bool:
    head = data[:256].lstrip()
    return head.startswith(b"<?xml")


def _looks_like_text(data: bytes) -> bool:
    sample = data[:8192]
    # Reject if too many NUL / control bytes (except tab/lf/cr)
    if b"\x00" in sample:
        return False
    try:
        text = sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    if not text.strip():
        return False
    # Mostly printable
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
    return printable / max(len(text), 1) >= 0.95 and bool(
        re.search(r"[\w]", text)
    )
