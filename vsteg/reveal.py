"""Shared reveal/decode orchestration with fail-fast foreign-format detection."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from vsteg.detect.mp4_forensics import inspect_mp4
from vsteg.methods import append, dct, lsb
from vsteg.probe import is_lossless_codec, probe


class RevealError(Exception):
    """Raised when a payload cannot be revealed (with a user-facing message)."""


def decode_payload(
    path: str | Path,
    password: Optional[str] = None,
    method: str = "auto",
) -> bytes:
    """Extract a vsteg payload, skipping methods that cannot apply.

    Auto mode avoids long frame scans on foreign encodings (e.g. OpenPuff).
    """
    path = Path(path)
    method = (method or "auto").strip().lower()
    if method not in {"auto", "append", "lsb", "dct"}:
        raise RevealError("method must be auto, append, lsb, or dct")

    pwd = password.strip() if password else None
    errors: list[str] = []

    if method == "append":
        return _try_append(path, pwd)
    if method == "lsb":
        return _try_lsb(path, pwd)
    if method == "dct":
        return _try_dct(path, pwd)

    # --- auto ---
    has_append = append.has_appended_payload(path)
    lossless = False
    try:
        lossless = is_lossless_codec(probe(path).codec)
    except Exception:
        pass

    # Foreign tools (e.g. OpenPuff): no vsteg trailer / lossless LSB carrier.
    # Skip expensive DCT frame scans and fail immediately with a clear hint.
    if not has_append and not lossless and _looks_foreign(path):
        raise RevealError(_foreign_message(path, pwd))

    candidates: list[str] = []
    if has_append:
        candidates.append("append")
    if lossless:
        candidates.append("lsb")
    # DCT only if a quick sync+header probe confirms a vsteg payload.
    if dct.looks_like_payload(path, password=pwd):
        candidates.append("dct")

    if not candidates:
        raise RevealError(_foreign_message(path, pwd))

    for name in candidates:
        try:
            if name == "append":
                return _try_append(path, pwd)
            if name == "lsb":
                return _try_lsb(path, pwd)
            return _try_dct(path, pwd)
        except RevealError as exc:
            errors.append(str(exc))
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    # Passworded vsteg append/LSB/DCT may fail auth while still being ours.
    if pwd and any("password" in e.lower() for e in errors):
        raise RevealError(
            "Found a vsteg-like payload but could not decrypt it. "
            "Check the password and try again."
        )

    detail = "; ".join(errors[:2]) if errors else ""
    raise RevealError(
        _foreign_message(path, pwd)
        + (f" ({detail})" if detail else "")
    )


def _looks_foreign(path: Path) -> bool:
    """True when container forensics suggest a non-vsteg embedding."""
    return bool(_foreign_hints(path))


def _try_append(path: Path, pwd: Optional[str]) -> bytes:
    try:
        return append.decode(path, password=pwd)
    except Exception as exc:
        raise RevealError(f"append: {exc}") from exc


def _try_lsb(path: Path, pwd: Optional[str]) -> bytes:
    try:
        info = probe(path)
    except Exception as exc:
        raise RevealError(f"lsb: cannot probe video ({exc})") from exc
    if not is_lossless_codec(info.codec):
        raise RevealError(
            "lsb: carrier is not a lossless codec (vsteg LSB uses FFV1/MKV)"
        )
    try:
        return lsb.decode(path, password=pwd)
    except Exception as exc:
        raise RevealError(f"lsb: {exc}") from exc


def _try_dct(path: Path, pwd: Optional[str]) -> bytes:
    try:
        return dct.decode(path, password=pwd)
    except Exception as exc:
        raise RevealError(f"dct: {exc}") from exc


def _foreign_message(path: Path, pwd: Optional[str]) -> str:
    hints = _foreign_hints(path)
    base = (
        "This video does not appear to contain a vsteg payload. "
        "Reveal only extracts secrets hidden with vsteg "
        "(append / LSB / DCT) — not other tools."
    )
    if hints:
        base += " " + " ".join(hints)
    if pwd:
        base += " If this was hidden with vsteg and a password, double-check the password."
    return base


def _foreign_hints(path: Path) -> list[str]:
    hints: list[str] = []
    try:
        info = inspect_mp4(path.read_bytes())
    except Exception:
        return hints
    slack = int(info.get("mdat_slack") or 0)
    if slack >= 64:
        entropy = info.get("slack_entropy")
        ent_note = (
            f", ~{entropy:.2f} bits/byte entropy" if isinstance(entropy, float) else ""
        )
        hints.append(
            f"Container looks OpenPuff-like (unreferenced mdat slack of {slack} bytes"
            f"{ent_note}). Use OpenPuff to extract that secret."
        )
    return hints
