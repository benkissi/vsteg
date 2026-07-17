"""Detection report aggregation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vsteg.detect import (
    dct_stats,
    ffmpeg_consistency,
    ml_ensemble,
    mp4_forensics,
    self_probe,
    signatures,
    statistics,
    structure,
    video_anomaly,
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
        "title": "LSB / RS statistics",
        "summary": (
            "Samples frames and runs chi-square, sample-pair, and Regular–Singular (RS) "
            "tests for least-significant-bit embedding (StegoForge-style RS)."
        ),
    },
    "dct_stats": {
        "title": "DCT mid-band analysis",
        "summary": "Looks for quantization patterns in mid-frequency DCT coefficients that can appear after robust embedding.",
    },
    "video_anomaly": {
        "title": "Keyframe DCT anomaly",
        "summary": (
            "StegoForge-style I-frame scan: mid-band DCT energy at coefficients "
            "(3,4)/(4,3), flagged when keyframe z-scores are extreme."
        ),
    },
    "ml_stats": {
        "title": "ML ensemble",
        "summary": (
            "Optional scikit-learn RandomForest over handcrafted video features "
            "(chi/SPA/RS/DCT/keyframe/mdat slack). Install with pip install -e \".[ml]\"."
        ),
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
        "title": "6. LSB / RS statistical tests",
        "body": (
            "Sample frames for chi-square, sample-pair, and Regular–Singular (RS) "
            "LSB embedding estimates."
        ),
    },
    {
        "id": "dct_stats",
        "title": "7. DCT mid-band checks",
        "body": "Inspect mid-frequency coefficients for QIM-like clustering (best-effort; robust stego is stealthier).",
    },
    {
        "id": "video_anomaly",
        "title": "8. Keyframe DCT anomaly",
        "body": (
            "StegoForge-style keyframe scan of mid-band DCT energy; outliers across "
            "I-frames raise suspicion."
        ),
    },
    {
        "id": "ml_stats",
        "title": "9. ML ensemble",
        "body": (
            "Optional RandomForest over handcrafted features. Soft signal only — "
            "cannot alone force a likely-stego verdict under dampening."
        ),
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
    # Handcrafted / ML feature breakdown for the Check UI
    stat_features: list[dict] = field(default_factory=list)

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
    feats: dict[str, float] = {}
    mdat_slack_bytes = 0
    ml_proba: float | None = None
    if deep:
        lsb_signals = statistics.analyze(path)
        dct_signals = dct_stats.analyze(path)
        va_signals = video_anomaly.analyze(path)
        signals.extend(lsb_signals)
        signals.extend(dct_signals)
        signals.extend(va_signals)
        feats, mdat_slack_bytes = _features_from_signals(
            lsb_signals, dct_signals, va_signals, path
        )
        ml_signals = ml_ensemble.analyze(path, features=feats)
        signals.extend(ml_signals)
        for s in ml_signals:
            m = s.get("metrics") or {}
            if "proba_stego" in m:
                ml_proba = float(m["proba_stego"])
                break

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
    soft_kinds = {"lsb_stats", "dct_stats", "video_anomaly", "ml_stats"}
    has_hard_evidence = bool(
        scored_kinds
        & {"signature", "structure", "ffmpeg", "self_probe", "mp4_forensics"}
    )
    # Statistical / ML-only hits are easy false positives on clean H.264 — require
    # a higher bar before calling the file suspicious.
    only_soft = scored_kinds and scored_kinds.issubset(soft_kinds)
    if (not has_hard_evidence or only_soft) and raw_score < 45 and not has_hard_evidence:
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
    stat_features = (
        _stat_feature_panel(feats, mdat_slack_bytes, ml_proba)
        if deep and feats
        else []
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
        stat_features=stat_features,
    )


def _features_from_signals(
    lsb_signals: list[dict],
    dct_signals: list[dict],
    va_signals: list[dict],
    path: Path,
) -> tuple[dict[str, float], int]:
    """Assemble ML features from analyzer metrics without a second full decode."""
    feats = {name: 0.0 for name in ml_ensemble.FEATURE_NAMES}
    for s in lsb_signals:
        m = s.get("metrics") or {}
        if "avg_chi" in m:
            feats["avg_chi"] = float(m["avg_chi"])
            feats["avg_spa"] = float(m["avg_spa"])
            feats["avg_lsb"] = float(m["avg_lsb"])
            feats["avg_rs"] = float(m["avg_rs"])
            break
    for s in dct_signals:
        m = s.get("metrics") or {}
        if "near" in m:
            feats["dct_near"] = float(m["near"])
            feats["dct_peakiness"] = float(m["peakiness"])
            break
    for s in va_signals:
        m = s.get("metrics") or {}
        if "zmax" in m:
            feats["keyframe_zmax"] = float(m["zmax"])
            break
    slack_bytes = 0
    try:
        size = max(1, path.stat().st_size)
        report = mp4_forensics.inspect_mp4(path.read_bytes())
        slack_bytes = int(report.get("mdat_slack") or 0)
        feats["mdat_slack_norm"] = min(1.0, max(0.0, slack_bytes / size))
    except Exception:
        pass
    return feats, slack_bytes


def _stat_feature_panel(
    feats: dict[str, float],
    mdat_slack_bytes: int,
    ml_proba: float | None,
) -> list[dict]:
    """Human-readable breakdown of chi/SPA/RS/DCT/keyframe/mdat (+ ML)."""
    rows: list[dict] = []

    def add(
        key: str,
        name: str,
        value: float,
        display: str,
        status: str,
        note: str,
    ) -> None:
        rows.append(
            {
                "key": key,
                "name": name,
                "value": value,
                "display": display,
                "status": status,  # ok | watch | flag | info
                "note": note,
            }
        )

    chi = float(feats.get("avg_chi", 0.0))
    add(
        "chi",
        "Chi-square (LSB)",
        chi,
        f"{chi:.3f}",
        "flag" if chi >= statistics.CHI_THRESHOLD else "ok",
        f"Threshold ≥ {statistics.CHI_THRESHOLD:.2f} (pair uniformity)",
    )
    spa = float(feats.get("avg_spa", 0.0))
    add(
        "spa",
        "Sample-pair (SPA)",
        spa,
        f"{spa:.3f}",
        "flag" if spa >= statistics.SPA_THRESHOLD else "ok",
        f"Threshold ≥ {statistics.SPA_THRESHOLD:.2f} (embedding-rate estimate)",
    )
    rs = float(feats.get("avg_rs", 0.0))
    add(
        "rs",
        "RS analysis",
        rs,
        f"{rs:.3f}",
        "flag" if rs >= statistics.RS_FRACTION_THRESHOLD else "ok",
        f"Threshold ≥ {statistics.RS_FRACTION_THRESHOLD:.2f} (payload fraction)",
    )
    lsb = float(feats.get("avg_lsb", 0.0))
    lsb_delta = abs(lsb - 0.5)
    add(
        "lsb_ratio",
        "LSB plane ratio",
        lsb,
        f"{lsb:.3f}",
        "watch" if lsb_delta > statistics.LSB_RATIO_DELTA else "ok",
        f"Natural ≈ 0.5 · watch if |Δ| > {statistics.LSB_RATIO_DELTA:.2f}",
    )
    near = float(feats.get("dct_near", 0.0))
    add(
        "dct_near",
        "DCT near-QIM",
        near,
        f"{near:.3f}",
        "flag" if near >= dct_stats.NEAR_THRESHOLD else "ok",
        f"Threshold ≥ {dct_stats.NEAR_THRESHOLD:.2f} (mid-band clustering)",
    )
    peak = float(feats.get("dct_peakiness", 0.0))
    add(
        "dct_peakiness",
        "DCT peakiness",
        peak,
        f"{peak:.3f}",
        "flag" if peak >= dct_stats.PEAKINESS_THRESHOLD else "ok",
        f"Threshold ≥ {dct_stats.PEAKINESS_THRESHOLD:.2f} (histogram peaks)",
    )
    zmax = float(feats.get("keyframe_zmax", 0.0))
    add(
        "keyframe",
        "Keyframe DCT z-max",
        zmax,
        f"{zmax:.3f}",
        "flag" if zmax >= video_anomaly.ZMAX_THRESHOLD else "ok",
        f"Threshold ≥ {video_anomaly.ZMAX_THRESHOLD:.1f} (I-frame outlier)",
    )
    slack_norm = float(feats.get("mdat_slack_norm", 0.0))
    if mdat_slack_bytes >= 64:
        slack_status = "flag"
    elif mdat_slack_bytes > 0:
        slack_status = "watch"
    else:
        slack_status = "ok"
    add(
        "mdat_slack",
        "MP4 mdat slack",
        float(mdat_slack_bytes),
        f"{mdat_slack_bytes} B (norm {slack_norm:.4f})",
        slack_status,
        "OpenPuff-like when unreferenced slack ≥ 64 bytes",
    )
    if ml_proba is not None:
        if ml_proba >= 0.65:
            ml_status = "flag"
        elif ml_proba >= 0.45:
            ml_status = "watch"
        else:
            ml_status = "ok"
        add(
            "ml_proba",
            "ML ensemble P(stego)",
            ml_proba,
            f"{ml_proba:.3f}",
            ml_status,
            "RandomForest over the features above · soft signal only",
        )
    else:
        add(
            "ml_proba",
            "ML ensemble P(stego)",
            0.0,
            "n/a",
            "info",
            'Install optional ML: pip install -e ".[ml]" && train model',
        )
    return rows


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
        "video_anomaly",
        "ml_stats",
    ]
    by_kind: dict[str, list[dict]] = {k: [] for k in order}
    for s in signals:
        kind = s.get("signal", "other")
        by_kind.setdefault(kind, []).append(s)

    out: list[dict] = []
    soft = {"lsb_stats", "dct_stats", "video_anomaly", "ml_stats"}
    for kind in order:
        meta = SIGNAL_META.get(kind, {"title": kind, "summary": ""})
        items = by_kind.get(kind, [])
        skipped = kind in soft and not deep
        weight = sum(int(i.get("weight", 0)) for i in items)
        scored = [i for i in items if int(i.get("weight", 0)) > 0]
        info_only = [i for i in items if int(i.get("weight", 0)) <= 0]
        if skipped:
            status = "skipped"
            detail = "Not run (fast mode)."
        elif weight > 0:
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

    if report.stat_features:
        lines.append("")
        lines.append("statistical / ML features:")
        for row in report.stat_features:
            lines.append(
                f"  - [{row.get('status')}] {row.get('name')}: {row.get('display')} "
                f"— {row.get('note')}"
            )

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
