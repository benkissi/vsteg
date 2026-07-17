"""Compare two videos: attributes, stream inventory, optional frame stats."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vsteg.detect.ffmpeg_consistency import media_snapshot
from vsteg.detect.mp4_forensics import inspect_mp4
from vsteg.probe import VideoInfo, probe


@dataclass
class AttributeRow:
    key: str
    label: str
    a: Any
    b: Any
    delta: Any = None
    changed: bool = False
    unit: str = ""


@dataclass
class CompareReport:
    a_path: str
    b_path: str
    summary: str
    similarity: str  # identical / similar / different
    changed_count: int
    attributes: list[dict] = field(default_factory=list)
    media_a: dict = field(default_factory=dict)
    media_b: dict = field(default_factory=dict)
    charts: dict = field(default_factory=dict)
    frame_analysis: dict = field(default_factory=dict)
    container_analysis: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare(
    path_a: str | Path,
    path_b: str | Path,
    sample_frames: int = 12,
    deep: bool = True,
) -> CompareReport:
    path_a = Path(path_a)
    path_b = Path(path_b)
    info_a = probe(path_a)
    info_b = probe(path_b)

    rows = _attribute_rows(info_a, info_b)
    changed = [r for r in rows if r.changed]
    notes: list[str] = []

    if info_a.codec != info_b.codec:
        notes.append(f"Video codec differs: {info_a.codec} vs {info_b.codec}.")
    if info_a.pix_fmt and info_b.pix_fmt and info_a.pix_fmt != info_b.pix_fmt:
        notes.append(f"Pixel format differs: {info_a.pix_fmt} vs {info_b.pix_fmt}.")
    if info_a.audio_count != info_b.audio_count:
        notes.append(
            f"Audio stream count differs: {info_a.audio_count} vs {info_b.audio_count}."
        )
    if abs(info_a.size_bytes - info_b.size_bytes) > 1024:
        notes.append(
            f"File size differs by {_human_bytes(abs(info_a.size_bytes - info_b.size_bytes))}."
        )
    if (info_a.width, info_a.height) != (info_b.width, info_b.height):
        notes.append("Resolution differs — frame-difference analysis may be limited.")

    container_analysis = _container_analysis(path_a, path_b)
    if container_analysis.get("mdat_slack_a") is not None:
        notes.append(
            f"MP4 mdat slack — A: {container_analysis['mdat_slack_a']} B: "
            f"{container_analysis['mdat_slack_b']} (Δ {container_analysis['mdat_slack_delta']})."
        )
    if container_analysis.get("openpuff_like"):
        notes.append(
            "OpenPuff-like pattern: decoded video frames match (or nearly match) but the "
            "container grew with unreferenced mdat slack / silent bitstream changes."
        )

    frame_analysis: dict[str, Any] = {"enabled": False}
    if deep:
        frame_analysis = _frame_analysis(path_a, path_b, info_a, info_b, sample_frames)
        if frame_analysis.get("mean_abs_diff") is not None:
            mad = frame_analysis["mean_abs_diff"]
            if mad < 0.5:
                notes.append(
                    f"Sampled frames are nearly identical (mean abs diff ≈ {mad:.3f})."
                )
            elif mad < 5:
                notes.append(
                    f"Sampled frames are close (mean abs diff ≈ {mad:.3f}) — small encode drift possible."
                )
            else:
                notes.append(
                    f"Sampled frames differ substantially (mean abs diff ≈ {mad:.3f})."
                )

    mad = frame_analysis.get("mean_abs_diff")
    frames_ok = mad is not None
    openpuff_like = bool(container_analysis.get("openpuff_like"))

    if openpuff_like:
        similarity = "similar"
        summary = (
            "Decoded frames match, but the container changed in an OpenPuff-like way "
            "(unreferenced mdat growth / silent bitstream mutation)."
        )
    elif not changed and (not frames_ok or mad < 0.5):
        similarity = "identical"
        summary = (
            "The two videos look effectively the same under probed attributes"
            + (" and sampled frames." if frames_ok else ".")
        )
    elif len(changed) <= 3 and (not frames_ok or mad < 8):
        similarity = "similar"
        summary = (
            f"{len(changed)} attribute(s) differ, but the videos are still broadly similar."
        )
    else:
        similarity = "different"
        summary = (
            f"{len(changed)} attribute(s) differ; the videos are meaningfully different "
            "in container and/or decoded frame content."
        )

    charts = _chart_payload(info_a, info_b, frame_analysis, container_analysis)

    return CompareReport(
        a_path=str(path_a),
        b_path=str(path_b),
        summary=summary,
        similarity=similarity,
        changed_count=len(changed),
        attributes=[asdict(r) for r in rows],
        media_a=media_snapshot(info_a),
        media_b=media_snapshot(info_b),
        charts=charts,
        frame_analysis=frame_analysis,
        container_analysis=container_analysis,
        notes=notes,
    )


def format_text(report: CompareReport) -> str:
    lines = [
        f"A:          {report.a_path}",
        f"B:          {report.b_path}",
        f"similarity: {report.similarity}",
        f"changed:    {report.changed_count} attributes",
        f"summary:    {report.summary}",
        "",
        "attributes:",
    ]
    for row in report.attributes:
        mark = "*" if row["changed"] else " "
        delta = f"  (Δ {row['delta']})" if row.get("delta") not in (None, "") else ""
        lines.append(
            f"  {mark} {row['label']}: {row['a']}  |  {row['b']}{delta}"
        )
    if report.notes:
        lines.append("")
        lines.append("notes:")
        for n in report.notes:
            lines.append(f"  - {n}")
    fa = report.frame_analysis or {}
    if fa.get("enabled"):
        lines.append("")
        lines.append("frame analysis:")
        for k in (
            "frames_compared",
            "mean_abs_diff",
            "max_abs_diff",
            "comparable",
            "message",
        ):
            if k in fa:
                lines.append(f"  {k}: {fa[k]}")
    ca = report.container_analysis or {}
    if ca:
        lines.append("")
        lines.append("container analysis:")
        for k in (
            "size_delta",
            "mdat_payload_a",
            "mdat_payload_b",
            "mdat_slack_a",
            "mdat_slack_b",
            "mdat_slack_delta",
            "lcp_bytes",
            "openpuff_like",
            "message",
        ):
            if k in ca and ca[k] is not None:
                lines.append(f"  {k}: {ca[k]}")
    return "\n".join(lines)


def format_json(report: CompareReport) -> str:
    return json.dumps(report.to_dict(), indent=2)


def write_html(report: CompareReport, output: str | Path) -> Path:
    """Write a self-contained HTML report with Chart.js graphs."""
    output = Path(output)
    payload = json.dumps(report.to_dict())
    html = _HTML_TEMPLATE.replace("__REPORT_JSON__", payload)
    output.write_text(html, encoding="utf-8")
    return output


def _attribute_rows(a: VideoInfo, b: VideoInfo) -> list[AttributeRow]:
    specs = [
        ("filename", "Filename", a.path.name, b.path.name),
        ("size_bytes", "File size (bytes)", a.size_bytes, b.size_bytes, "bytes"),
        ("size_human", "File size", _human_bytes(a.size_bytes), _human_bytes(b.size_bytes)),
        ("container", "Container", a.container, b.container),
        ("codec", "Video codec", a.codec, b.codec),
        ("profile", "Profile", a.profile or "—", b.profile or "—"),
        ("pix_fmt", "Pixel format", a.pix_fmt or "—", b.pix_fmt or "—"),
        ("width", "Width", a.width, b.width, "px"),
        ("height", "Height", a.height, b.height, "px"),
        ("fps", "Frame rate", round(a.fps, 3), round(b.fps, 3), "fps"),
        ("duration", "Duration", round(a.duration, 3), round(b.duration, 3), "s"),
        ("frames", "Frames", a.frames, b.frames),
        ("bitrate", "Bitrate", a.bitrate, b.bitrate, "bps"),
        (
            "bitrate_human",
            "Bitrate (human)",
            _human_bitrate(a.bitrate),
            _human_bitrate(b.bitrate),
        ),
        ("encoder", "Encoder tag", a.encoder or "—", b.encoder or "—"),
        ("audio_count", "Audio streams", a.audio_count, b.audio_count),
        ("subtitle_count", "Subtitle streams", a.subtitle_count, b.subtitle_count),
        ("data_count", "Data streams", a.data_count, b.data_count),
        ("probe_source", "Probe source", a.probe_source, b.probe_source),
    ]
    rows: list[AttributeRow] = []
    for item in specs:
        key, label, va, vb = item[0], item[1], item[2], item[3]
        unit = item[4] if len(item) > 4 else ""
        changed = va != vb
        delta: Any = None
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = round(vb - va, 3)
            # ignore tiny float noise
            if isinstance(va, float) or isinstance(vb, float):
                changed = abs(float(va) - float(vb)) > 1e-3
        rows.append(
            AttributeRow(
                key=key,
                label=label,
                a=va,
                b=vb,
                delta=delta,
                changed=changed,
                unit=unit,
            )
        )
    return rows


def _frame_analysis(
    path_a: Path,
    path_b: Path,
    info_a: VideoInfo,
    info_b: VideoInfo,
    sample_frames: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "enabled": True,
        "comparable": False,
        "frames_compared": 0,
        "mean_abs_diff": None,
        "max_abs_diff": None,
        "per_frame_mad": [],
        "hist_a": [],
        "hist_b": [],
        "message": "",
    }
    try:
        import av
    except Exception as exc:
        result["message"] = f"PyAV unavailable for frame analysis: {exc}"
        return result

    if info_a.width != info_b.width or info_a.height != info_b.height:
        # Still compute separate histograms, skip pixel MAD
        ha = _luma_histogram(path_a, sample_frames)
        hb = _luma_histogram(path_b, sample_frames)
        result["hist_a"] = ha
        result["hist_b"] = hb
        result["message"] = "Different resolutions — compared luma histograms only."
        return result

    frames_a = _sample_gray_frames(path_a, sample_frames)
    frames_b = _sample_gray_frames(path_b, sample_frames)
    n = min(len(frames_a), len(frames_b))
    if n == 0:
        result["message"] = "Could not decode sample frames."
        return result

    mads = []
    hist_a = np.zeros(32, dtype=np.float64)
    hist_b = np.zeros(32, dtype=np.float64)
    for i in range(n):
        fa = frames_a[i].astype(np.float64)
        fb = frames_b[i].astype(np.float64)
        # resize mismatch guard
        if fa.shape != fb.shape:
            continue
        mad = float(np.mean(np.abs(fa - fb)))
        mads.append(mad)
        hist_a += np.histogram(fa, bins=32, range=(0, 255))[0]
        hist_b += np.histogram(fb, bins=32, range=(0, 255))[0]

    if not mads:
        result["message"] = "No comparable frame pairs."
        return result

    hist_a = (hist_a / hist_a.sum()).tolist() if hist_a.sum() else []
    hist_b = (hist_b / hist_b.sum()).tolist() if hist_b.sum() else []

    result.update(
        {
            "comparable": True,
            "frames_compared": len(mads),
            "mean_abs_diff": round(float(np.mean(mads)), 4),
            "max_abs_diff": round(float(np.max(mads)), 4),
            "per_frame_mad": [round(x, 4) for x in mads],
            "hist_a": hist_a,
            "hist_b": hist_b,
            "message": f"Compared {len(mads)} sampled grayscale frames.",
        }
    )
    return result


def _sample_gray_frames(path: Path, n: int) -> list[np.ndarray]:
    import av

    frames: list[np.ndarray] = []
    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    decoded: list[np.ndarray] = []
    for frame in container.decode(video=0):
        decoded.append(frame.to_ndarray(format="gray"))
        if len(decoded) >= max(n * 4, n):
            break
    container.close()
    if not decoded:
        return frames
    idxs = np.linspace(0, len(decoded) - 1, num=min(n, len(decoded)), dtype=int)
    return [decoded[int(i)] for i in idxs]


def _luma_histogram(path: Path, n: int) -> list[float]:
    frames = _sample_gray_frames(path, n)
    if not frames:
        return []
    hist = np.zeros(32, dtype=np.float64)
    for f in frames:
        hist += np.histogram(f, bins=32, range=(0, 255))[0]
    if hist.sum() == 0:
        return []
    return (hist / hist.sum()).tolist()


def _container_analysis(path_a: Path, path_b: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "size_delta": path_b.stat().st_size - path_a.stat().st_size,
        "mdat_payload_a": None,
        "mdat_payload_b": None,
        "mdat_slack_a": None,
        "mdat_slack_b": None,
        "mdat_slack_delta": None,
        "lcp_bytes": None,
        "openpuff_like": False,
        "message": "",
    }
    try:
        data_a = path_a.read_bytes()
        data_b = path_b.read_bytes()
    except Exception as exc:
        out["message"] = f"could not read files: {exc}"
        return out

    n = min(len(data_a), len(data_b))
    lcp = 0
    for i in range(n):
        if data_a[i] != data_b[i]:
            break
        lcp += 1
    out["lcp_bytes"] = lcp

    try:
        fa = inspect_mp4(data_a)
        fb = inspect_mp4(data_b)
        out["mdat_payload_a"] = fa["mdat_payload"]
        out["mdat_payload_b"] = fb["mdat_payload"]
        out["mdat_slack_a"] = fa["mdat_slack"]
        out["mdat_slack_b"] = fb["mdat_slack"]
        out["mdat_slack_delta"] = fb["mdat_slack"] - fa["mdat_slack"]
        # OpenPuff-like: B has slack, A has little/none, size grew
        if (
            fa["mdat_slack"] <= 16
            and fb["mdat_slack"] >= 64
            and out["size_delta"] >= 64
        ):
            out["openpuff_like"] = True
            out["message"] = (
                "B gained unreferenced mdat slack while sample tables stayed consistent "
                "with A's media — typical of OpenPuff."
            )
        elif abs(out["size_delta"]) >= 64 and lcp > 1024:
            out["message"] = (
                f"Files share a {lcp}-byte identical prefix then diverge; "
                f"size Δ={out['size_delta']} bytes."
            )
    except Exception as exc:
        out["message"] = f"mp4 forensics unavailable: {exc}"
    return out


def _chart_payload(
    a: VideoInfo,
    b: VideoInfo,
    frame_analysis: dict,
    container_analysis: dict | None = None,
) -> dict[str, Any]:
    ca = container_analysis or {}
    return {
        "scalar": {
            "labels": ["Size (MB)", "Duration (s)", "FPS", "Bitrate (Mbps)", "Frames/1000"],
            "a": [
                round(a.size_bytes / (1024 * 1024), 3),
                round(a.duration, 3),
                round(a.fps, 3),
                round(a.bitrate / 1_000_000, 3) if a.bitrate else 0,
                round(a.frames / 1000, 3) if a.frames else 0,
            ],
            "b": [
                round(b.size_bytes / (1024 * 1024), 3),
                round(b.duration, 3),
                round(b.fps, 3),
                round(b.bitrate / 1_000_000, 3) if b.bitrate else 0,
                round(b.frames / 1000, 3) if b.frames else 0,
            ],
        },
        "streams": {
            "labels": ["Audio", "Subtitles", "Data"],
            "a": [a.audio_count, a.subtitle_count, a.data_count],
            "b": [b.audio_count, b.subtitle_count, b.data_count],
        },
        "container": {
            "labels": ["mdat payload (KB)", "mdat slack (B)", "size Δ (B)"],
            "a": [
                round((ca.get("mdat_payload_a") or 0) / 1024, 3),
                ca.get("mdat_slack_a") or 0,
                0,
            ],
            "b": [
                round((ca.get("mdat_payload_b") or 0) / 1024, 3),
                ca.get("mdat_slack_b") or 0,
                ca.get("size_delta") or 0,
            ],
        },
        "frame_mad": {
            "labels": [f"f{i+1}" for i in range(len(frame_analysis.get("per_frame_mad") or []))],
            "values": frame_analysis.get("per_frame_mad") or [],
        },
        "histogram": {
            "labels": [str(i) for i in range(32)],
            "a": frame_analysis.get("hist_a") or [],
            "b": frame_analysis.get("hist_b") or [],
        },
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


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>vsteg compare report</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root { --ink:#0c1f1c; --mist:#e8f2ee; --lime:#c8f560; --fog:rgba(232,242,238,.75); --line:rgba(232,242,238,.16); }
    body { margin:0; font-family: DM Sans, system-ui, sans-serif; background:#071412; color:var(--mist); }
    main { width:min(960px, calc(100% - 2rem)); margin:0 auto; padding:2rem 0 3rem; }
    h1,h2 { font-family: Syne, system-ui, sans-serif; letter-spacing:-.03em; }
    .card { border:1px solid var(--line); border-radius:16px; padding:1rem 1.1rem; margin:1rem 0; background:rgba(8,22,20,.7); }
    table { width:100%; border-collapse:collapse; font-size:.92rem; }
    th,td { text-align:left; padding:.45rem .35rem; border-bottom:1px solid rgba(232,242,238,.08); vertical-align:top; }
    tr.changed td { background:rgba(200,245,96,.06); }
    .badge { display:inline-block; padding:.15rem .5rem; border-radius:999px; background:var(--lime); color:var(--ink); font-weight:700; text-transform:uppercase; font-size:.75rem; }
    .charts { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
    canvas { max-width:100%; }
    @media (max-width:720px){ .charts{grid-template-columns:1fr;} }
  </style>
</head>
<body>
<main>
  <h1>vsteg compare</h1>
  <div id="summary" class="card"></div>
  <div class="card"><h2>Attributes</h2><div id="table"></div></div>
  <div class="charts">
    <div class="card"><h2>Scalars</h2><canvas id="cScalar"></canvas></div>
    <div class="card"><h2>Streams</h2><canvas id="cStreams"></canvas></div>
    <div class="card"><h2>Frame MAD</h2><canvas id="cMad"></canvas></div>
    <div class="card"><h2>Luma histogram</h2><canvas id="cHist"></canvas></div>
  </div>
  <div class="card"><h2>Notes</h2><ul id="notes"></ul></div>
</main>
<script>
const report = __REPORT_JSON__;
const nameA = report.media_a.filename || 'A';
const nameB = report.media_b.filename || 'B';
document.getElementById('summary').innerHTML = `
  <p><span class="badge">${report.similarity}</span></p>
  <p>${report.summary}</p>
  <p><strong>A:</strong> ${report.a_path}<br/><strong>B:</strong> ${report.b_path}</p>
`;
const rows = report.attributes.map(r => `
  <tr class="${r.changed ? 'changed' : ''}">
    <td>${r.label}</td><td>${r.a}</td><td>${r.b}</td><td>${r.delta ?? ''}</td>
  </tr>`).join('');
document.getElementById('table').innerHTML = `
  <table><thead><tr><th>Attribute</th><th>${nameA}</th><th>${nameB}</th><th>Δ</th></tr></thead>
  <tbody>${rows}</tbody></table>`;
document.getElementById('notes').innerHTML = (report.notes||[]).map(n=>`<li>${n}</li>`).join('') || '<li>None</li>';

const charts = report.charts || {};
const common = { responsive:true, plugins:{ legend:{ labels:{ color:'#e8f2ee' } } },
  scales:{ x:{ ticks:{ color:'#e8f2ee' }, grid:{ color:'rgba(232,242,238,.08)' } },
           y:{ ticks:{ color:'#e8f2ee' }, grid:{ color:'rgba(232,242,238,.08)' } } } };
new Chart(document.getElementById('cScalar'), { type:'bar', data:{
  labels: charts.scalar.labels,
  datasets:[
    { label:nameA, data:charts.scalar.a, backgroundColor:'#c8f560' },
    { label:nameB, data:charts.scalar.b, backgroundColor:'#ff6b4a' }
  ]}, options:common });
new Chart(document.getElementById('cStreams'), { type:'bar', data:{
  labels: charts.streams.labels,
  datasets:[
    { label:nameA, data:charts.streams.a, backgroundColor:'#c8f560' },
    { label:nameB, data:charts.streams.b, backgroundColor:'#ff6b4a' }
  ]}, options:common });
if ((charts.frame_mad.values||[]).length) {
  new Chart(document.getElementById('cMad'), { type:'line', data:{
    labels: charts.frame_mad.labels,
    datasets:[{ label:'Mean abs diff', data:charts.frame_mad.values, borderColor:'#c8f560', tension:.25 }]
  }, options:common });
}
if ((charts.histogram.a||[]).length) {
  new Chart(document.getElementById('cHist'), { type:'line', data:{
    labels: charts.histogram.labels,
    datasets:[
      { label:nameA+' luma', data:charts.histogram.a, borderColor:'#c8f560', tension:.2 },
      { label:nameB+' luma', data:charts.histogram.b, borderColor:'#ff6b4a', tension:.2 }
    ]
  }, options:common });
}
</script>
</body>
</html>
"""
