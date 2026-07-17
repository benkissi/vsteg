"""Method A — append container after the last MP4 atom / as file trailer."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from vsteg import HEADER_SIZE, METHOD_APPEND
from vsteg.container import (
    ContainerError,
    find_magic,
    pack_payloads,
    unpack_matching,
)
from vsteg.probe import free_disk_bytes

SOFT_WARN_BYTES = 32 * 1024**3  # 32 GiB


class AppendError(Exception):
    pass


def encode(
    carrier: str | Path,
    secret: bytes,
    output: str | Path,
    password: Optional[str] = None,
    *,
    decoy: Optional[bytes] = None,
    decoy_password: Optional[str] = None,
) -> Path:
    carrier = Path(carrier)
    output = Path(output)
    try:
        container = pack_payloads(
            secret,
            METHOD_APPEND,
            password=password,
            decoy=decoy,
            decoy_password=decoy_password,
            compress=True,
        )
    except ContainerError as exc:
        raise AppendError(str(exc)) from exc

    needed = carrier.stat().st_size + len(container)
    free = free_disk_bytes(output)
    if needed > free:
        raise AppendError(
            f"not enough disk space: need {needed} bytes, have {free}"
        )
    if len(container) > SOFT_WARN_BYTES:
        # Soft warning only — caller/CLI may print it
        pass

    shutil.copyfile(carrier, output)
    with open(output, "ab") as f:
        f.write(container)
    return output


def decode(path: str | Path, password: Optional[str] = None) -> bytes:
    path = Path(path)
    data = path.read_bytes()
    scan_from = max(0, len(data) - 64 * 1024 * 1024)
    region = data[scan_from:]
    try:
        return unpack_matching(region, password)
    except ContainerError as exc:
        raise AppendError(f"no valid VSTG container found in file trailer: {exc}") from exc


def has_appended_payload(path: str | Path) -> bool:
    """Cheap check: VSTG magic present in trailing region."""
    path = Path(path)
    size = path.stat().st_size
    if size < HEADER_SIZE:
        return False
    read_size = min(size, 64 * 1024 * 1024)
    with open(path, "rb") as f:
        f.seek(size - read_size)
        tail = f.read(read_size)
    return find_magic(tail) >= 0
