"""Detection report aggregation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vsteg.detect import (
    dct_stats,
    ffmpeg_consistency,
    mp4_forensics,
    self_probe,
    signatures,
    statistics,
    structure,
)
from vsteg.probe import probe

SIGNAL_META = {
    "signature": {
        "title": "Signature scan",
        "summary": "Looks for known steganography markers in the file bytes (for example the vsteg VSTG header).",
    },
    "structure": {
        "title": "Container structure",
        "summary": "Checks whether the file has trailing/appended data after the media payload.",
    },
    "self_probe": {
        "title": "vsteg active probe",
        "summary": (
            "Actively probes for vsteg’s own methods: verified append trailers, "
            "LSB headers in lossless frames, and DCT sync preambles used by Method C."
        ),
    },
    "mp4_forensics": {
        "title": "MP4 container forensics",
        "summary": (
            "Parses MP4 atoms and sample tables. Flags unreferenced mdat slack "
            "(mdat larger than stsz totals) — a strong OpenPuff-style indicator."
        ),
    },
    "ffmpeg": {
        "title": "FFmpeg / ffprobe consistency",
        "summary": (
            "Uses ffprobe (or PyAV) to inspect codec, resolution, bitrate, frame rate, "
            "duration, audio/subtitle streams, and metadata. These checks do not prove "
            "steganography; they flag unexpected remux/encode traits."
        ),
    },
    "lsb_stats": {
        "title": "LSB statistics",
        "summary": "Samples frames and runs chi-square / sample-pair style tests for least-significant-bit embedding.",
    },
    "dct_stats": {
        "title": "DCT mid-band analysis",
        "summary": "Looks for quantization patterns in mid-frequency DCT coefficients that can appear after robust embedding.",
    },
}

CHECK_PIPELINE = [
    {
        "id": "signature",
        "title": "1. Signature scan",
        "body": "Search the start and end of the file for known tool markers (vsteg, OpenPuff, StegoForge).",
    },
    {
        "id": "structure",
        "title": "2. Container structure",
        "body": "Detect appended trailers after the media — common for append-style stego.",
    },
    {
        "id": "self_probe",
        "title": "3. vsteg active probe",
        "body": (
            "Try to confirm vsteg payloads directly: decode append trailers, read LSB "
            "headers in lossless video, and search for the Method C DCT sync pattern."
        ),
    },
    {
        "id": "mp4_forensics",
        "title": "4. MP4 container forensics",
        "body": (
            "Compare mdat size to stsz sample totals. Extra unreferenced slack is a "
            "hallmark of OpenPuff-style embedding (decoded frames unchanged, file grows)."
        ),
    },
    {
        "id": "ffmpeg",
        "title": "5. FFmpeg / ffprobe consistency",
        "body": (
            "Probe codec, pix_fmt, bitrate, fps, duration, audio/subs/data streams, "
            "and metadata tags for unusual or inconsistent encoder fingerprints."
        ),
    },
    {
        "id": "lsb_stats",
        "title": "6. LSB statistical tests",
        "body": "Sample frames and measure LSB-plane anomalies with chi-square and sample-pair heuristics.",
    },
    {
        "id": "dct_stats",
        "title": "7. DCT-domain checks",
        "body": "Inspect mid-frequency coefficients for QIM-like clustering (best-effort; robust stego is stealthier).",
    },
]


@dataclass
class DetectionReport:
    path: str
    score: int
    verdict: str
    summary: str
    confidence_label: str
    signals: list[dict] = field(default_factory=list)
    categories: list[dict] = field(default_factory=list)
    media: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    pipeline: list[dict] = field(default_factory=list)
    deep: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect(path: str | Path, deep: bool = True) -> DetectionReport:
    path = Path(path)
    info = None
    try:
        info = probe(path)
    except Exception:
        info = None

    signals: list[dict] = []
    signals.extend(signatures.scan(path))
    signals.extend(structure.analyze(path))
    signals.extend(self_probe.analyze(path))
    signals.extend(mp4_forensics.analyze(path))
    signals.extend(ffmpeg_consistency.analyze(path, info=info))
    if deep:
        signals.extend(statistics.analyze(path))
        signals.extend(dct_stats.analyze(path))

    enriched: list[dict] = []
    for s in signals:
        kind = s.get("signal", "other")
        meta = SIGNAL_META.get(kind, {"title": kind, "summary": ""})
        item = dict(s)
        item["title"] = meta["title"]
        item["explanation"] = meta["summary"]
        item["severity"] = _severity(int(s.get("weight", 0)))
        enriched.append(item)
    # Highest-weight findings first in the flat list too
    enriched.sort(key=lambda i: int(i.get("weight", 0)), reverse=True)

    # Informational (weight 0) findings do not affect the score
    raw_score = min(100, sum(int(s.get("weight", 0)) for s in enriched))
    scored_kinds = {s.get("signal") for s in enriched if int(s.get("weight", 0)) > 0}
    has_hard_evidence = bool(
        scored_kinds
        & {"signature", "structure", "ffmpeg", "self_probe", "mp4_forensics"}
    )
    # Statistical-only hits are easy false positives on clean H.264 — require a
    # higher bar before calling the file suspicious.
    if not has_hard_evidence and raw_score < 45:
        score = min(raw_score, 24)
    else:
        score = raw_score

    if score >= 60:
        verdict = "likely-stego"
        confidence_label = "High"
        summary = (
            "Several independent checks point to hidden data. "
            "This video is likely carrying a steganographic payload."
        )
    elif score >= 25:
        verdict = "suspicious"
        confidence_label = "Medium"
        summary = (
            "Some unusual signals were found, but they are not conclusive. "
            "The file may contain steganography, or the signals may be false positives "
            "from remux/encode quirks that ffprobe also surfaces."
        )
    else:
        verdict = "clean"
        confidence_label = "Low risk"
        if raw_score > score:
            summary = (
                "No strong steganography indicators were found. "
                "Weak statistical hints were present but are common in normal compressed "
                "video, so they were not enough to mark this file suspicious."
            )
        else:
            summary = (
                "No strong steganography indicators were found. "
                "FFmpeg/ffprobe consistency checks also look normal enough under our thresholds. "
                "A clean result is not a mathematical proof — stealthy methods can still hide."
            )

    categories = _category_breakdown(enriched, deep=deep)
    media = (
        ffmpeg_consistency.media_snapshot(info)
        if info is not None
        else {"filename": path.name, "error": "probe unavailable"}
    )

    return DetectionReport(
        path=str(path),
        score=score,
        verdict=verdict,
        summary=summary,
        confidence_label=confidence_label,
        signals=enriched,
        categories=categories,
        media=media,
        thresholds={
            "clean_max": 24,
            "suspicious_min": 25,
            "likely_stego_min": 60,
        },
        pipeline=CHECK_PIPELINE,
        deep=deep,
    )


def _severity(weight: int) -> str:
    if weight >= 35:
        return "high"
    if weight >= 15:
        return "medium"
    if weight > 0:
        return "low"
    return "info"


def _category_breakdown(signals: list[dict], deep: bool) -> list[dict]:
    order = [
        "signature",
        "structure",
        "self_probe",
        "mp4_forensics",
        "ffmpeg",
        "lsb_stats",
        "dct_stats",
    ]
    by_kind: dict[str, list[dict]] = {k: [] for k in order}
    for s in signals:
        kind = s.get("signal", "other")
        by_kind.setdefault(kind, []).append(s)

    out: list[dict] = []
    for kind in order:
        meta = SIGNAL_META.get(kind, {"title": kind, "summary": ""})
        items = by_kind.get(kind, [])
        skipped = kind in {"lsb_stats", "dct_stats"} and not deep
        weight = sum(int(i.get("weight", 0)) for i in items)
        if skipped:
            status = "skipped"
            detail = "Not run (fast mode)."
        scored = [i for i in items if int(i.get("weight", 0)) > 0]
        info_only = [i for i in items if int(i.get("weight", 0)) <= 0]
        if weight > 0:
            status = "triggered"
            detail = (
                f"{len(scored)} scored finding(s) (+{weight}); "
                f"{len(info_only)} info note(s)."
            )
        elif items:
            status = "noted"
            detail = f"{len(items)} informational note(s), no score impact."
        else:
            status = "clear"
            detail = "No anomalies in this check."
        # Show scored findings first so +0 info notes don't look like the trigger
        ordered = sorted(items, key=lambda i: int(i.get("weight", 0)), reverse=True)
        out.append(
            {
                "id": kind,
                "title": meta["title"],
                "summary": meta["summary"],
                "status": status,
                "score_contribution": weight,
                "detail": detail,
                "findings": ordered,
            }
        )
    return out


def format_text(report: DetectionReport) -> str:
    lines = [
        f"file:       {report.path}",
        f"verdict:    {report.verdict}",
        f"score:      {report.score}/100 ({report.confidence_label})",
        f"summary:    {report.summary}",
        "",
        "media:",
    ]
    media = report.media or {}
    for key in (
        "probe_source",
        "filename",
        "size_human",
        "codec",
        "pix_fmt",
        "profile",
        "bitrate_human",
        "fps",
        "duration_sec",
        "width",
        "height",
        "audio_count",
        "subtitle_count",
        "data_count",
        "encoder",
        "container",
    ):
        if key in media and media[key] not in (None, "", [], {}):
            lines.append(f"  {key}: {media[key]}")
    if media.get("tags"):
        lines.append(f"  tags: {media['tags']}")
    if media.get("audio_streams"):
        lines.append(f"  audio_streams: {media['audio_streams']}")
    if media.get("subtitle_streams"):
        lines.append(f"  subtitle_streams: {media['subtitle_streams']}")

    lines.append("")
    lines.append("categories:")
    for cat in report.categories:
        lines.append(
            f"  - {cat['title']}: {cat['status']} "
            f"(+{cat['score_contribution']}) — {cat['detail']}"
        )
    lines.append("")
    lines.append("signals:")
    if not report.signals:
        lines.append("  (none)")
    else:
        for s in report.signals:
            lines.append(
                f"  - [{s.get('signal')}|{s.get('severity')}] "
                f"+{s.get('weight', 0)}: {s.get('label')}"
            )
    return "\n".join(lines)


def format_json(report: DetectionReport) -> str:
    return json.dumps(report.to_dict(), indent=2)
