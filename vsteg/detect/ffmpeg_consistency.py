"""FFmpeg/ffprobe consistency checks — not a stego oracle, but useful anomalies."""

from __future__ import annotations

from pathlib import Path

from vsteg.probe import VideoInfo, is_lossless_codec, is_unusual_pix_fmt, probe


def analyze(path: str | Path, info: VideoInfo | None = None) -> list[dict]:
    """Return lightweight signals about unexpected encoder / container traits."""
    path = Path(path)
    signals: list[dict] = []
    try:
        info = info or probe(path)
    except Exception as exc:
        return [
            {
                "signal": "ffmpeg",
                "label": f"ffprobe/media probe failed: {exc}",
                "weight": 0,
            }
        ]

    # Codec / pixel format anomalies
    if is_lossless_codec(info.codec):
        signals.append(
            {
                "signal": "ffmpeg",
                "label": (
                    f"video codec '{info.codec}' is lossless — unusual for normal "
                    "delivery video; often used so hidden pixel bits survive"
                ),
                "weight": 22,
            }
        )

    if is_unusual_pix_fmt(info.pix_fmt):
        signals.append(
            {
                "signal": "ffmpeg",
                "label": (
                    f"pixel format '{info.pix_fmt}' is uncommon for everyday H.264/H.265 "
                    "uploads (more common in lossless / analysis pipelines)"
                ),
                "weight": 12,
            }
        )

    # Bitrate / duration / size consistency
    if info.bitrate > 0 and info.duration > 0:
        expected = int(info.bitrate * info.duration / 8)
        surplus = info.size_bytes - expected
        if info.size_bytes > expected * 2 + 1024 * 1024 and surplus > 4096:
            signals.append(
                {
                    "signal": "ffmpeg",
                    "label": (
                        f"container size ({info.size_bytes} bytes) is much larger than "
                        f"bitrate×duration suggests (~{expected} bytes); "
                        "possible appended data or remux artifact"
                    ),
                    "weight": 18,
                }
            )
        elif expected > 0 and info.size_bytes < expected * 0.35 and info.duration > 1:
            signals.append(
                {
                    "signal": "ffmpeg",
                    "label": (
                        "file is much smaller than bitrate×duration implies — "
                        "container/stream metadata may be inconsistent"
                    ),
                    "weight": 8,
                }
            )

    # Duration / frame-rate sanity
    if info.fps <= 0 and info.duration > 0:
        signals.append(
            {
                "signal": "ffmpeg",
                "label": "missing or zero frame rate in stream metadata",
                "weight": 6,
            }
        )
    if info.duration <= 0:
        signals.append(
            {
                "signal": "ffmpeg",
                "label": "missing duration metadata",
                "weight": 4,
            }
        )
    if info.width <= 0 or info.height <= 0:
        signals.append(
            {
                "signal": "ffmpeg",
                "label": "missing video resolution metadata",
                "weight": 6,
            }
        )

    # Audio / subtitle / data stream inventory notes
    if info.audio_count == 0:
        signals.append(
            {
                "signal": "ffmpeg",
                "label": "no audio stream present (can be normal; also common after stego re-encodes that drop audio)",
                "weight": 3,
            }
        )
    elif info.audio_count > 2:
        signals.append(
            {
                "signal": "ffmpeg",
                "label": f"unusual audio stream count ({info.audio_count})",
                "weight": 4,
            }
        )

    if info.subtitle_count > 0:
        signals.append(
            {
                "signal": "ffmpeg",
                "label": (
                    f"{info.subtitle_count} subtitle stream(s) present — "
                    "not stego by itself; noted for completeness"
                ),
                "weight": 0,
            }
        )

    if info.data_count > 0:
        signals.append(
            {
                "signal": "ffmpeg",
                "label": (
                    f"{info.data_count} data/attachment stream(s) — "
                    "worth inspecting; some tools stash payloads in side streams"
                ),
                "weight": 10,
            }
        )

    # Metadata / encoder tags
    tags = info.tags or {}
    interesting = {
        k: v
        for k, v in tags.items()
        if k.lower()
        in {
            "encoder",
            "encoding_tool",
            "handler_name",
            "compatible_brands",
            "major_brand",
            "comment",
            "description",
            "title",
            "artist",
            "software",
        }
    }
    if interesting:
        preview = ", ".join(f"{k}={v}" for k, v in list(interesting.items())[:6])
        signals.append(
            {
                "signal": "ffmpeg",
                "label": f"container metadata tags: {preview}",
                "weight": 0,
            }
        )
    else:
        signals.append(
            {
                "signal": "ffmpeg",
                "label": "no common encoder/title metadata tags found",
                "weight": 2,
            }
        )

    encoder = (info.encoder or "").lower()
    # Heuristic: some research tools leave fingerprints
    for needle, label, weight in (
        ("stego", "encoder/metadata mentions 'stego'", 20),
        ("stegoforge", "encoder/metadata mentions StegoForge", 30),
        ("openpuff", "encoder/metadata mentions OpenPuff", 30),
        ("vsteg", "encoder/metadata mentions vsteg", 25),
    ):
        blob = " ".join([encoder, *[str(v).lower() for v in tags.values()]])
        if needle in blob:
            signals.append(
                {
                    "signal": "ffmpeg",
                    "label": label,
                    "weight": weight,
                }
            )

    # Low ffprobe score can mean damaged/odd containers
    score = int((info.metadata or {}).get("probe_score") or 0)
    if score and score < 50:
        signals.append(
            {
                "signal": "ffmpeg",
                "label": f"ffprobe confidence score is low ({score}/100) — container may be unusual or damaged",
                "weight": 8,
            }
        )

    return signals


def media_snapshot(info: VideoInfo) -> dict:
    """Structured media block for the detection report UI/CLI."""
    audio = [s for s in info.streams if s.type == "audio"]
    subs = [s for s in info.streams if s.type == "subtitle"]
    return {
        "filename": info.path.name,
        "probe_source": info.probe_source,
        "size_bytes": info.size_bytes,
        "size_human": _human_bytes(info.size_bytes),
        "width": info.width,
        "height": info.height,
        "frames": info.frames,
        "fps": round(info.fps, 3) if info.fps else 0,
        "duration_sec": round(info.duration, 3) if info.duration else 0,
        "codec": info.codec,
        "profile": info.profile,
        "pix_fmt": info.pix_fmt,
        "container": info.container,
        "format_long_name": info.format_long_name,
        "bitrate": info.bitrate,
        "bitrate_human": _human_bitrate(info.bitrate),
        "encoder": info.encoder,
        "audio_count": info.audio_count,
        "subtitle_count": info.subtitle_count,
        "data_count": info.data_count,
        "audio_streams": [
            {
                "index": s.index,
                "codec": s.codec,
                "sample_rate": s.sample_rate,
                "channels": s.channels,
                "language": s.language,
            }
            for s in audio
        ],
        "subtitle_streams": [
            {
                "index": s.index,
                "codec": s.codec,
                "language": s.language,
                "title": s.title,
            }
            for s in subs
        ],
        "tags": info.tags,
        "metadata": info.metadata,
    }


def _human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


def _human_bitrate(bps: int) -> str:
    if not bps:
        return "unknown"
    if bps >= 1_000_000:
        return f"{bps / 1_000_000:.2f} Mbps"
    if bps >= 1_000:
        return f"{bps / 1_000:.1f} kbps"
    return f"{bps} bps"
