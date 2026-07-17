"""Container structure checks focused on appended / trailer payloads."""

from __future__ import annotations

from pathlib import Path

from vsteg.methods.append import has_appended_payload


def analyze(path: str | Path) -> list[dict]:
    path = Path(path)
    signals: list[dict] = []

    # Trailing / appended payload (vsteg Method A and similar tools)
    if has_appended_payload(path):
        signals.append(
            {
                "signal": "structure",
                "label": "VSTG-like trailer / appended data after media",
                "weight": 45,
            }
        )

    return signals
