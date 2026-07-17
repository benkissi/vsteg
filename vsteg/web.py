"""Beginner-friendly web UI for vsteg."""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from vsteg.compare import compare as run_compare
from vsteg.detect.report import detect as run_detect
from vsteg.methods import append, dct, lsb
from vsteg.reveal import RevealError, decode_payload
from vsteg.sniff import suggested_filename

STATIC_DIR = Path(__file__).parent / "web_static"
WORK_ROOT = Path(tempfile.gettempdir()) / "vsteg-web"
WORK_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="vsteg", description="Hide, reveal, and check video steganography")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _job_dir() -> Path:
    d = WORK_ROOT / uuid.uuid4().hex
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_upload(upload: UploadFile, dest: Path) -> Path:
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


def _cleanup(path: Path) -> None:
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink(missing_ok=True)
    except Exception:
        pass


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.post("/api/encode")
async def api_encode(
    carrier: UploadFile = File(...),
    secret: UploadFile = File(...),
    method: str = Form("append"),
    password: Optional[str] = Form(None),
    decoy: Optional[UploadFile] = File(None),
    decoy_password: Optional[str] = Form(None),
):
    method = (method or "append").strip().lower()
    if method not in {"append", "lsb", "dct"}:
        raise HTTPException(400, "method must be append, lsb, or dct")

    pwd = password.strip() if password else None
    decoy_pwd = decoy_password.strip() if decoy_password else None
    job = _job_dir()
    try:
        carrier_path = _save_upload(carrier, job / (carrier.filename or "carrier.mp4"))
        secret_path = _save_upload(secret, job / (secret.filename or "secret.bin"))
        secret_bytes = secret_path.read_bytes()

        decoy_bytes = None
        if decoy is not None and decoy.filename:
            decoy_path = _save_upload(decoy, job / (decoy.filename or "decoy.bin"))
            decoy_bytes = decoy_path.read_bytes()

        kwargs = {"password": pwd, "decoy": decoy_bytes, "decoy_password": decoy_pwd}
        if method == "append":
            out = job / "stego.mp4"
            append.encode(carrier_path, secret_bytes, out, **kwargs)
        elif method == "lsb":
            out = job / "stego.mkv"
            lsb.encode(carrier_path, secret_bytes, out, **kwargs)
        else:
            out = job / "stego.mp4"
            dct.encode(carrier_path, secret_bytes, out, **kwargs)

        return FileResponse(
            path=str(out),
            filename=out.name,
            media_type="application/octet-stream",
            background=BackgroundTask(_cleanup, job),
        )
    except Exception as exc:
        _cleanup(job)
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/decode")
async def api_decode(
    stego: UploadFile = File(...),
    password: Optional[str] = Form(None),
    method: str = Form("auto"),
):
    method = (method or "auto").strip().lower()
    if method not in {"auto", "append", "lsb", "dct"}:
        raise HTTPException(400, "method must be auto, append, lsb, or dct")

    pwd = password.strip() if password else None
    job = _job_dir()
    try:
        stego_path = _save_upload(stego, job / (stego.filename or "stego.mp4"))
        try:
            plaintext = decode_payload(stego_path, password=pwd, method=method)
        except RevealError as exc:
            raise HTTPException(400, str(exc)) from exc

        name = suggested_filename(plaintext)
        out = job / name
        out.write_bytes(plaintext)
        return FileResponse(
            path=str(out),
            filename=name,
            media_type="application/octet-stream",
            background=BackgroundTask(_cleanup, job),
        )
    except HTTPException:
        _cleanup(job)
        raise
    except Exception as exc:
        _cleanup(job)
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/detect")
async def api_detect(
    video: UploadFile = File(...),
    deep: bool = Form(True),
):
    job = _job_dir()
    try:
        path = _save_upload(video, job / (video.filename or "suspect.mp4"))
        report = run_detect(path, deep=deep)
        return JSONResponse(report.to_dict())
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        _cleanup(job)


@app.post("/api/compare")
async def api_compare(
    video_a: UploadFile = File(...),
    video_b: UploadFile = File(...),
    deep: bool = Form(True),
    samples: int = Form(12),
):
    job = _job_dir()
    try:
        path_a = _save_upload(video_a, job / ("a_" + (video_a.filename or "a.mp4")))
        path_b = _save_upload(video_b, job / ("b_" + (video_b.filename or "b.mp4")))
        report = run_compare(
            path_a,
            path_b,
            sample_frames=max(2, min(int(samples), 48)),
            deep=deep,
        )
        return JSONResponse(report.to_dict())
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        _cleanup(job)


def main(host: str = "127.0.0.1", port: int = 5000) -> None:
    import uvicorn

    print(f"vsteg web UI → http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
