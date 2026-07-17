"""Video probing helpers via ffprobe (preferred) / PyAV fallback."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class StreamInfo:
    index: int
    type: str  # video / audio / subtitle / data / unknown
    codec: str
    codec_long: str = ""
    profile: str = ""
    width: int = 0
    height: int = 0
    pix_fmt: str = ""
    fps: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    channel_layout: str = ""
    bit_rate: int = 0
    language: str = ""
    title: str = ""
    disposition: dict = field(default_factory=dict)


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    frames: int
    fps: float
    duration: float
    codec: str
    bitrate: int
    container: str
    size_bytes: int
    # Rich fields (optional / defaulted for backward compatibility)
    pix_fmt: str = ""
    profile: str = ""
    encoder: str = ""
    format_name: str = ""
    format_long_name: str = ""
    probe_source: str = ""  # ffprobe | pyav
    audio_count: int = 0
    subtitle_count: int = 0
    data_count: int = 0
    streams: list[StreamInfo] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    tags: dict = field(default_factory=dict)  # format tags
    raw_format: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["path"] = str(self.path)
        return d


class ProbeError(Exception):
    pass


def probe(path: str | Path) -> VideoInfo:
    """Return rich media info; prefer ffprobe, fall back to PyAV."""
    path = Path(path)
    if not path.exists():
        raise ProbeError(f"file not found: {path}")

    if shutil.which("ffprobe"):
        try:
            return _ffprobe_rich(path)
        except Exception:
            pass

    return _pyav_probe(path)


def free_disk_bytes(path: str | Path) -> int:
    return shutil.disk_usage(Path(path).resolve().parent).free


def is_lossless_codec(codec: str) -> bool:
    return codec.lower() in {
        "ffv1",
        "huffyuv",
        "ffvhuff",
        "utvideo",
        "rawvideo",
        "png",
        "qtrle",
        "prores_ks",
        "alac",
        "flac",
        "pcm_s16le",
        "pcm_s24le",
    }


def is_unusual_pix_fmt(pix_fmt: str) -> bool:
    """Pixel formats uncommon for everyday delivery video."""
    p = (pix_fmt or "").lower()
    if not p:
        return False
    unusual = {
        "yuv444p",
        "yuv444p10le",
        "gbrp",
        "gbrp10le",
        "rgb24",
        "bgr24",
        "bgr0",
        "rgba",
        "gray",
        "gray16le",
    }
    return p in unusual


def _parse_fps(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(value)
    except Exception:
        return 0.0


def _ffprobe_rich(path: Path) -> VideoInfo:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-show_chapters",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ProbeError(f"ffprobe failed: {result.stderr.strip() or 'unknown error'}")

    data = json.loads(result.stdout)
    fmt = data.get("format", {}) or {}
    tags = fmt.get("tags") or {}
    streams_raw = data.get("streams", []) or []

    streams: list[StreamInfo] = []
    video: Optional[dict] = None
    audio_count = subtitle_count = data_count = 0

    for s in streams_raw:
        stype = s.get("codec_type") or "unknown"
        tags_s = s.get("tags") or {}
        disp = s.get("disposition") or {}
        info = StreamInfo(
            index=int(s.get("index", 0)),
            type=stype,
            codec=s.get("codec_name") or "unknown",
            codec_long=s.get("codec_long_name") or "",
            profile=s.get("profile") or "",
            width=int(s.get("width") or 0),
            height=int(s.get("height") or 0),
            pix_fmt=s.get("pix_fmt") or "",
            fps=_parse_fps(s.get("avg_frame_rate") or s.get("r_frame_rate")),
            sample_rate=int(s.get("sample_rate") or 0),
            channels=int(s.get("channels") or 0),
            channel_layout=s.get("channel_layout") or "",
            bit_rate=int(s.get("bit_rate") or 0),
            language=tags_s.get("language") or "",
            title=tags_s.get("title") or "",
            disposition={k: v for k, v in disp.items() if v not in (0, "0", None, "")},
        )
        streams.append(info)
        if stype == "video" and video is None:
            video = s
        elif stype == "audio":
            audio_count += 1
        elif stype == "subtitle":
            subtitle_count += 1
        elif stype in {"data", "attachment"}:
            data_count += 1

    if not video:
        raise ProbeError("no video stream")

    fps = _parse_fps(video.get("avg_frame_rate") or video.get("r_frame_rate"))
    duration = float(fmt.get("duration") or video.get("duration") or 0)
    frames = int(video.get("nb_frames") or 0)
    if frames <= 0 and duration > 0 and fps > 0:
        frames = int(duration * fps)

    encoder = (
        tags.get("encoder")
        or tags.get("Encoder")
        or tags.get("encoding_tool")
        or ""
    )

    return VideoInfo(
        path=path,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        frames=frames,
        fps=fps,
        duration=duration,
        codec=video.get("codec_name") or "unknown",
        bitrate=int(fmt.get("bit_rate") or video.get("bit_rate") or 0),
        container=fmt.get("format_name") or path.suffix.lstrip("."),
        size_bytes=int(fmt.get("size") or path.stat().st_size),
        pix_fmt=video.get("pix_fmt") or "",
        profile=str(video.get("profile") or ""),
        encoder=encoder,
        format_name=fmt.get("format_name") or "",
        format_long_name=fmt.get("format_long_name") or "",
        probe_source="ffprobe",
        audio_count=audio_count,
        subtitle_count=subtitle_count,
        data_count=data_count,
        streams=streams,
        metadata={
            "chapters": len(data.get("chapters") or []),
            "nb_streams": int(fmt.get("nb_streams") or len(streams)),
            "probe_score": int(fmt.get("probe_score") or 0),
            "start_time": fmt.get("start_time"),
        },
        tags={str(k): str(v) for k, v in tags.items()},
        raw_format={
            k: fmt.get(k)
            for k in ("bit_rate", "duration", "size", "format_name", "format_long_name")
            if k in fmt
        },
    )


def _pyav_probe(path: Path) -> VideoInfo:
    try:
        import av
    except Exception as exc:
        raise ProbeError("neither ffprobe nor PyAV available") from exc

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ProbeError("no video stream")
        stream = container.streams.video[0]
        width = stream.width
        height = stream.height
        codec = stream.codec_context.name or "unknown"
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        frames = stream.frames or 0
        duration = (
            float(stream.duration * stream.time_base) if stream.duration else 0.0
        )
        if frames <= 0 and duration > 0 and fps > 0:
            frames = int(duration * fps)
        bitrate = int(stream.bit_rate or container.bit_rate or 0)
        fmt = container.format.name if container.format else path.suffix.lstrip(".")
        pix_fmt = getattr(stream.codec_context, "pix_fmt", None) or ""
        if hasattr(pix_fmt, "name"):
            pix_fmt = pix_fmt.name

        streams: list[StreamInfo] = []
        audio_count = subtitle_count = data_count = 0
        for s in container.streams:
            stype = s.type or "unknown"
            si = StreamInfo(
                index=s.index,
                type=stype,
                codec=(s.codec_context.name if s.codec_context else "unknown") or "unknown",
                width=getattr(s, "width", 0) or 0,
                height=getattr(s, "height", 0) or 0,
                fps=float(s.average_rate) if getattr(s, "average_rate", None) else 0.0,
                sample_rate=int(getattr(s.codec_context, "sample_rate", 0) or 0)
                if s.codec_context
                else 0,
                channels=int(getattr(s.codec_context, "channels", 0) or 0)
                if s.codec_context
                else 0,
                bit_rate=int(s.bit_rate or 0),
            )
            streams.append(si)
            if stype == "audio":
                audio_count += 1
            elif stype == "subtitle":
                subtitle_count += 1
            elif stype in {"data", "attachment"}:
                data_count += 1

        tags = {str(k): str(v) for k, v in (container.metadata or {}).items()}
        return VideoInfo(
            path=path,
            width=width,
            height=height,
            frames=frames,
            fps=fps,
            duration=duration,
            codec=codec,
            bitrate=bitrate,
            container=fmt,
            size_bytes=path.stat().st_size,
            pix_fmt=str(pix_fmt or ""),
            encoder=tags.get("encoder") or tags.get("Encoder") or "",
            format_name=fmt,
            probe_source="pyav",
            audio_count=audio_count,
            subtitle_count=subtitle_count,
            data_count=data_count,
            streams=streams,
            metadata={"nb_streams": len(streams)},
            tags=tags,
        )
