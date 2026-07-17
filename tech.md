# vsteg — Technical Documentation

**Course / project notes:** how this toolkit works under the hood.

| | |
|---|---|
| Package | `vsteg` v0.1.0 |
| Language | Python ≥ 3.10 |
| Purpose | Encode, decode, detect, and compare video steganography |
| Interfaces | CLI (`python -m vsteg`) and web UI (`vsteg web`) |

This document explains **what** the system does, **why** design choices were made, and **how** each layer is implemented. It is meant to be read alongside the source under `vsteg/`.

---

## 1. Problem statement and design goals

### 1.1 Why existing tools failed

| Tool | What went wrong | Lesson for vsteg |
|------|-----------------|------------------|
| **StegoForge** | Hides data in pixel LSBs, then re-encodes with **lossy** H.264. Quantization destroys LSBs → encode “works”, decode fails. | Never carry payload bits through a lossy codec unless the embed method is designed for that. |
| **OpenPuff** | Can encode/decode video, but has **no detection**. | Ship steganalysis, not only embed/extract. |

### 1.2 Design principles

1. **Reliability first** — every method must round-trip: `decode(encode(x)) == x` (covered by tests).
2. **Authenticated containers** — wrong password / tampering fails loudly; never return silent garbage.
3. **Method-appropriate media path** — append leaves video untouched; LSB uses lossless FFV1; DCT uses QIM + ECC so H.264 is survivable.
4. **Layered detection** — cheap structural checks first; statistical tests second; OpenPuff-style container forensics included.
5. **Optional decoy** — OpenPuff-style second payload for plausible deniability.

---

## 2. System architecture

```
┌─────────────────────────────────────────────────────────────┐
│  CLI (cli.py)          Web UI (web.py + web_static/)        │
└─────────────┬───────────────────────────┬───────────────────┘
              │                           │
              ▼                           ▼
┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐
│ encode / decode  │  │ detect pipeline  │  │ compare         │
│ methods/*        │  │ detect/*         │  │ compare.py      │
│ reveal.py        │  │ report.py        │  │                 │
└────────┬─────────┘  └────────┬─────────┘  └────────┬────────┘
         │                     │                     │
         ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐
│ container.py     │  │ probe.py (ffprobe│  │ mp4_forensics   │
│ ecc.py           │  │ / PyAV)          │  │ frame sampling  │
└──────────────────┘  └──────────────────┘  └─────────────────┘
```

### 2.1 Package map

| Module | Responsibility |
|--------|----------------|
| `container.py` | `VSTG` binary format, zlib, scrypt, AES-GCM, CRC32, decoy packing |
| `ecc.py` | Reed–Solomon + interleaving (Method C) |
| `methods/append.py` | Trailer append (Method A) |
| `methods/lsb.py` | Pixel LSB → FFV1/MKV (Method B) |
| `methods/dct.py` | DCT-domain QIM → H.264 (Method C) |
| `reveal.py` | Auto-decode with fail-fast for foreign tools |
| `detect/*` | Steganalysis pipeline |
| `compare.py` | Side-by-side forensic compare |
| `probe.py` | Media metadata (ffprobe preferred, PyAV fallback) |
| `sniff.py` | Guess recovered filename from magic bytes |
| `cli.py` / `web.py` | User interfaces |

### 2.2 Technology stack

| Layer | Technology | Role |
|-------|------------|------|
| Crypto | `cryptography` — Scrypt + AES-GCM | Password → key; authenticated encryption |
| Compression | `zlib` (level 6) | Shrink payload before encrypt/embed |
| Video I/O | **PyAV** (`av`) wrapping FFmpeg libs | Frame-accurate decode/encode |
| Probe | **ffprobe** (CLI) / PyAV | Codec, fps, frames, bitrate, tags |
| Numerics | `numpy`, `scipy.fft` | LSB planes, 2-D DCT/IDCT |
| ECC | `reedsolo` | RS(255,191) for Method C |
| Web | FastAPI + uvicorn + multipart | Upload/encode/decode/detect/compare UI |
| Charts | Chart.js (CDN) | Compare graphs in browser / HTML export |

**System dependency:** `ffmpeg` / `ffprobe` on `PATH`.

---

## 3. The `VSTG` payload container

Every secret (and every decoy) is wrapped in the same binary format before embedding. Embedding methods differ; the container does not.

### 3.1 Header layout (47 bytes, big-endian)

Packed with:

```text
struct ">4sBBB16s12sQI"
```

| Offset | Size | Field | Meaning |
|--------|------|-------|---------|
| 0 | 4 | `magic` | ASCII `VSTG` |
| 4 | 1 | `version` | Must be `1` |
| 5 | 1 | `flags` | Bitmask (see below) |
| 6 | 1 | `method` | `0`=append, `1`=LSB, `2`=DCT |
| 7 | 16 | `salt` | Scrypt salt (all zeros if not encrypted) |
| 23 | 12 | `nonce` | AES-GCM nonce (zeros if not encrypted) |
| 35 | 8 | `length` | `uint64` body length in bytes |
| 43 | 4 | `crc32` | CRC-32 of the **body** (after compression/encryption) |

Wire format:

```text
[ 47-byte header ][ body of `length` bytes ]
```

Constants (`vsteg/__init__.py`):

```text
MAGIC       = b"VSTG"
HEADER_SIZE = 47
METHOD_APPEND = 0
METHOD_LSB    = 1
METHOD_DCT    = 2
FLAG_ENCRYPTED  = 0x01
FLAG_COMPRESSED = 0x02
FLAG_ECC        = 0x04
```

### 3.2 Pack pipeline (encode-time)

```text
plaintext
    │
    ▼  (optional) zlib.compress(level=6)     → set FLAG_COMPRESSED
    │
    ▼  (optional) AES-256-GCM encrypt         → set FLAG_ENCRYPTED
    │         key = Scrypt(password, salt)
    │         salt, nonce = os.urandom(...)
    │
    ▼  crc32(body)
    │
    ▼  emit header || body
```

**Scrypt parameters** (`container.py`):

| Param | Value | Notes |
|-------|-------|-------|
| `n` | \(2^{15}\) = 32768 | CPU/memory cost |
| `r` | 8 | Block size factor |
| `p` | 1 | Parallelization |
| output | 32 bytes | AES-256 key |

AES-GCM uses a 12-byte nonce; associated data is unused (`None`). GCM’s auth tag is part of the ciphertext returned by `AESGCM.encrypt`.

**Why CRC and GCM?**  
- No password: CRC catches truncation/corruption.  
- With password: GCM already authenticates; CRC still guards the stored ciphertext blob for quick rejection before decrypt.

### 3.3 Unpack pipeline (decode-time)

```text
parse header (magic, version)
    → read body[length]
    → verify CRC32
    → if FLAG_ENCRYPTED: Scrypt + AES-GCM decrypt  (fails ⇒ wrong password / tamper)
    → if FLAG_COMPRESSED: zlib.decompress
    → plaintext
```

Wrong password never yields plausible garbage: GCM decrypt raises → `ContainerError`.

### 3.4 Decoy (plausible deniability)

Optional second payload, OpenPuff-style.

**Rules** (`validate_decoy`):

- Decoy file requires a **real password** and a **decoy password**.
- Passwords must **differ**.
- Empty decoy is rejected.

**Layout:**

```text
real_VSTG_container  ||  decoy_VSTG_container
```

**Reveal:** `unpack_matching` scans for every `VSTG` magic, tries the supplied password on each container, returns the first that decrypts/validates.

| Password used | Result |
|---------------|--------|
| Real password | Real secret |
| Decoy password | Decoy file |
| Wrong password | Error |

**Implication for class discussion:** a coerced reveal can produce the decoy; presence of *two* `VSTG` headers is still a forensic signal (detectability vs deniability trade-off).

---

## 4. Embedding methods

### 4.1 Method A — Trailer append (default)

**File:** `methods/append.py`  
**Idea:** Copy the carrier file, then append the `VSTG` blob after the media. Players ignore trailing bytes after the last MP4 atom / MKV segment.

```text
[ original media bytes unchanged ][ VSTG ... ][ optional decoy VSTG ]
```

| Property | Value |
|----------|-------|
| Output | Same container type (e.g. `.mp4`) |
| Video stream | Untouched |
| Capacity | Effectively disk / filesystem limited |
| Survives re-encode / remux? | **No** — trailer is stripped |
| Reliability | Highest (byte-identical round-trip) |

**Decode:** read file, scan the last **64 MiB** for `VSTG`, `unpack_matching`.

**Cheap probe:** `has_appended_payload()` — seek near EOF, search for magic (used by detect + reveal fail-fast).

**Disk check:** refuse encode if `carrier_size + container > free_disk`. Soft threshold constant `32 GiB` exists for oversized payloads.

---

### 4.2 Method B — Lossless LSB (true pixel steganography)

**File:** `methods/lsb.py`  
**Idea:** Put container bits into least-significant bits of pixels, then re-encode with a **lossless** codec so bits survive.

| Property | Value |
|----------|-------|
| Pixel format | `yuv444p` (3 planes) |
| Codec | **FFV1** |
| Container | typically `.mkv` |
| Bits/sample | `--bits` ∈ {1, 2, 3} |
| Survives lossy H.264? | **No** |

#### Capacity

\[
\text{raw} = \left\lfloor \frac{\text{frames} \times W \times H \times 3 \times \text{bits}}{8} \right\rfloor
\]

\[
\text{capacity\_bytes} = \max(0,\ \text{raw} - 47)
\]

Encode uses only **90%** of that (`CAPACITY_MARGIN = 0.90`) as a safety margin.

#### Position scrambling

Bit locations are not sequential. For each frame’s flattened plane samples:

1. Seed RNG:  
   `seed = SHA256(password or "vsteg-lsb-default")[:8]` interpreted as big-endian `uint64`.
2. `rng.permutation(num_samples)` picks which samples get LSBs replaced.

**Decoy special case:** when a decoy is present, **position seed uses the default** (`password=None` → `"vsteg-lsb-default"`), so *either* password can walk the same bit order; encryption alone selects which container unlocks.

#### Encode / decode flow

**Encode:** pack container → unpack to bits → walk frames → overwrite LSBs → write FFV1.

**Decode:** try seeds `[user_password, None]` (legacy files used password-seeded positions; decoy files use default). Expand extract length once the first header’s `length` is known; allow room for a second (decoy) header. Then `unpack_matching`.

#### Why StegoForge failed and this does not

StegoForge used lossy re-encode after LSB embed. vsteg **changes the codec to lossless FFV1**, so LSBs are preserved. Trade-off: larger files; output is MKV, not MP4.

---

### 4.3 Method C — DCT / QIM (compression-robust)

**File:** `methods/dct.py`  
**Idea:** Hide bits in **mid-frequency DCT coefficients** of 8×8 luma blocks using **Quantization Index Modulation (QIM)**, protect with **Reed–Solomon + interleaving**, and output H.264 at high quality.

| Property | Value |
|----------|-------|
| Domain | 8×8 DCT on **Y** (luma) only |
| Embed rule | QIM with step \(\Delta\) (`--strength`, default `16.0`) |
| Redundancy | Default **9** (each logical bit repeated 9×; majority vote on decode) |
| Output | H.264 `libx264`, `yuv420p`, CRF default **18** |
| Default max payload | **1 MiB** (`--max-robust`) |
| Survives mild re-encode? | **Yes**, within capacity/strength limits |

#### Why mid-frequency coefficients?

- **Low frequency (DC / near-DC):** highly visible if changed; carefully coded by the codec.
- **High frequency:** first to be zeroed by quantization → payload dies.
- **Mid-band:** compromise between visibility and survival.

Coefficients used (row, col) inside each 8×8 block:

```text
(1,2) (2,1) (2,2) (1,3) (3,1) (2,3) (3,2)
```

#### QIM mathematics

Let \(\Delta\) = strength, \(c\) = coefficient, \(b \in \{0,1\}\).

**Embed:**

1. \(q = \mathrm{round}(c / \Delta)\)
2. If \(q \bmod 2 = b\), keep \(q\); else move to nearest \(q\pm 1\) with correct parity.
3. Write \(c' = q' \cdot \Delta\), inverse DCT back to pixels.

**Extract:** \(b = \mathrm{round}(c / \Delta) \bmod 2\).

DCT/IDCT: `scipy.fft.dctn` / `idctn`, type 2, `norm="ortho"`.

#### Bitstream before embedding

```text
VSTG container(+decoy)
        │
        ▼  ecc.encode  (Reed–Solomon + interleave)
        │
        ▼  bits = unpackbits(...)
        │
        ▼  [ SYNC_PATTERN (16 bits) ] || bits
        │
        ▼  repeat each bit `redundancy` times
        │
        ▼  QIM into random-ordered 8×8 blocks (seeded RNG)
```

**Sync pattern** (helps locate the payload after compression noise):

```text
1 0 1 0  1 1 0 0  1 1 1 0  0 0 1 0
```

Decode searches for this pattern (exact match preferred; accepts best match if score \(\ge 13/16\)).

#### Reed–Solomon ECC (`ecc.py`)

| Param | Value |
|-------|-------|
| Code | RS(255, 191) |
| Parity symbols | 64 per codeword (~25% overhead) |
| Framing | 4-byte big-endian length + payload + pad to multiple of 191 |

**Interleaving:** codewords are written **column-major**. A burst of bit errors from a damaged GOP hits many codewords lightly instead of destroying one codeword completely.

Decode tries decreasing numbers of codewords until the length prefix validates (handles over-extraction).

#### Capacity (approximate)

\[
\text{usable slots} = (H/8)\times(W/8)\times\text{frames}\times 7
\]

\[
\text{capacity\_bits} = \max\!\left(0,\ \left\lfloor\frac{\text{usable} - 16\cdot R}{R}\right\rfloor\right)
\]

where \(R\) = redundancy. Then account for RS expansion and subtract header overhead. Realistically: **kilobytes to low megabytes**, not gigabytes.

#### Decoy + seeding

Same rule as LSB: with decoy, **position RNG uses default seed** so both passwords share bit locations.

---

## 5. Reveal (decode orchestration)

**File:** `reveal.py`  
Used by CLI auto-decode and `POST /api/decode`.

### 5.1 Forced method

`method ∈ {append, lsb, dct}` → call that method only.  
LSB refuses non-lossless codecs (vsteg LSB always ships FFV1).

### 5.2 Auto mode (fail-fast)

1. Check append magic in trailer (`has_appended_payload`).
2. Probe codec; note if lossless.
3. **Foreign short-circuit:** if no append trailer, not lossless, **and** MP4 forensics looks foreign (e.g. OpenPuff `mdat` slack ≥ 64 bytes) → **stop immediately** with a clear error. This avoids multi-minute DCT frame scans on OpenPuff files.
4. Build candidate list:
   - append if trailer present
   - LSB if lossless
   - DCT only if `dct.looks_like_payload()` (quick sync + ECC + `VSTG` header verify)
5. Try candidates; return first success.
6. If nothing matches → explain that Reveal only unlocks **vsteg** payloads; mention OpenPuff-like slack when present.

**Important classroom point:** detection ≠ extraction. vsteg can **flag** OpenPuff-like video but cannot decrypt OpenPuff’s proprietary format.

---

## 6. Detection (steganalysis)

**Orchestrator:** `detect/report.py`  
**CLI:** `vsteg detect`  
**Web:** Check tab → `POST /api/detect`

### 6.1 Pipeline (in order)

| Step | Module | What it looks for |
|------|--------|-------------------|
| 1 | `signatures` | Known byte markers (`VSTG`, strings related to OpenPuff/StegoForge) in head/tail of file |
| 2 | `structure` | Appended trailer via `has_appended_payload` |
| 3 | `self_probe` | **Active** confirmation: decode append; parse LSB header on lossless; DCT sync + ECC + header |
| 4 | `mp4_forensics` | ISO-BMFF atoms; `mdat` vs `stsz` slack; trailing bytes; slack entropy |
| 5 | `ffmpeg_consistency` | Unusual codec/pix_fmt/size vs bitrate×duration/metadata |
| 6 | `statistics` *(deep)* | Chi-square, sample-pair, and **RS (Regular–Singular)** LSB tests on sampled frames |
| 7 | `dct_stats` *(deep)* | Mid-band coefficient clustering near a \(\Delta\) grid |
| 8 | `video_anomaly` *(deep)* | StegoForge-style **keyframe** DCT energy at coeffs `(3,4)/(4,3)`; z-score outliers |
| 9 | `ml_ensemble` *(deep, optional)* | sklearn RandomForest over handcrafted features (`pip install -e ".[ml]"` + train script) |

`--fast` / `deep=False` skips steps 6–9.

**StegoForge note:** their ONNX CNN is image/BOSSbase-trained; we intentionally do **not** ship it for video. Our ML path is a small RF on video-native features.

### 6.2 Scoring and verdicts

Each signal contributes a **weight** (0 = informational only).

\[
\text{raw\_score} = \min(100,\ \sum \text{weights})
\]

**Dampening:** if there is no hard evidence (`signature` / `structure` / `ffmpeg` / `self_probe` / `mp4_forensics`) and `raw_score < 45`, cap score at **24** (soft stats + ML alone cannot mark clean H.264 “suspicious”).

| Score | Verdict |
|-------|---------|
| 0–24 | `clean` |
| 25–59 | `suspicious` |
| 60–100 | `likely-stego` |

Example strong weights (approximate, from source):

- Verified vsteg append / LSB / DCT: **~55**
- Append structure alone: **~45**
- `mdat` slack ≥ 256 bytes: **~50**; ≥ 64: **~35**
- High-entropy slack (≥ 7.5 bits/byte): **+15**

### 6.3 OpenPuff MP4 forensics (deep dive)

OpenPuff-style video stego often:

1. Leaves **decoded frames identical** to the original (MAD ≈ 0).
2. **Grows the file** by embedding into the MP4 `mdat` region without updating sample-size tables (`stsz`).

vsteg computes:

\[
\text{mdat\_slack} = \text{mdat\_payload} - \text{stsz\_total}
\]

(when both totals are positive). Slack ≥ 64 bytes is suspicious; high Shannon entropy on the slack region suggests encrypted/whitened hidden data.

**Calibration:** real pair `openpuff/original.mp4` vs `openpuff/stego.mp4` showed **+1819** bytes slack, identical frames — used to tune weights and compare’s `openpuff_like` flag.

---

## 7. Compare

**File:** `compare.py`

Compares two videos (e.g. original vs stego):

1. **Metadata attributes** — size, codec, profile, pix_fmt, resolution, fps, duration, frames, bitrate, streams, encoder tags (via `probe`).
2. **Optional deep frame sample** — decode N grayscale frames; mean/max absolute difference; histogram charts.
3. **Container analysis** — `mdat` payload/slack on both sides; longest common prefix of file bytes; `openpuff_like` when B gains slack while pictures (frames) stay matched.

Similarity labels: `identical` / `similar` / `different`, plus a human summary string.

Outputs: CLI text, JSON, HTML (Chart.js), or web Compare tab.

---

## 8. Interfaces

### 8.1 CLI

```bash
vsteg encode  -i carrier -s secret -o out [-p PW] [-m append|lsb|dct] \
              [--decoy file --decoy-password PW2] ...
vsteg decode  -i stego -o recovered [-p PW] [-m auto|append|lsb|dct]
vsteg detect  -i suspect [--json] [--fast]
vsteg compare -a A -b B [--html report.html] [--json] [--fast]
vsteg web     [--host 127.0.0.1] [--port 5000]
```

| Exit code | Meaning |
|-----------|---------|
| 0 | Success |
| 1 | Usage / unexpected error |
| 2 | Decode / auth failure |
| 3 | Detect: suspicious or likely-stego |
| 4 | Detect: clean |

### 8.2 Web API

| Route | Role |
|-------|------|
| `GET /` | SPA-like static UI |
| `POST /api/encode` | Multipart hide → download stego |
| `POST /api/decode` | Multipart reveal → download secret |
| `POST /api/detect` | JSON detection report |
| `POST /api/compare` | JSON compare report |

Uploads land under `$TMPDIR/vsteg-web/<uuid>/` and are cleaned after response.

UI shows a busy overlay during long operations; Reveal fail-fast returns foreign/OpenPuff errors quickly.

---

## 9. End-to-end data paths (summary)

### Append

```text
secret → VSTG → append after carrier copy → stego.mp4
stego.mp4 → scan trailer → unpack_matching(password) → secret
```

### LSB

```text
secret → VSTG → bit plane → PRNG positions → YUV444 LSBs → FFV1/MKV
MKV → extract bits (try seeds) → unpack_matching → secret
```

### DCT

```text
secret → VSTG → RS+interleave → sync ‖ bits → ×redundancy
      → QIM mid-DCT on Y → H.264 MP4
MP4 → QIM extract → majority → find sync → RS decode → unpack_matching
```

### Detect

```text
signatures → structure → self_probe → mp4_forensics → ffmpeg
         → [LSB stats → DCT stats] → score → verdict
```

---

## 10. Security and forensics notes (exam-useful)

1. **Steganography ≠ cryptography.** Hiding is not encryption; passwords add crypto *inside* the container.
2. **Append is detectable** — trailing `VSTG` / size mismatch vs media duration.
3. **LSB is detectable** — statistical tests; also “why is this FFV1/MKV?” metadata smell.
4. **DCT/QIM is stealthier but not invisible** — mid-band histogram / clustering can leak; capacity is small.
5. **Decoy** protects against password coercion, not against an analyst who sees two containers or runs detect.
6. **OpenPuff interoperability:** detect/compare can flag; reveal cannot extract (closed format).
7. **AES-GCM + scrypt** — industry-standard authenticated encryption; scrypt resists GPU/ASIC password guessing better than plain SHA hashes.

---

## 11. Testing strategy

Located under `tests/`:

| Area | What is proven |
|------|----------------|
| `test_container.py` | Pack/unpack, password, tamper, decoy password selection |
| `test_roundtrip.py` | Append / LSB / DCT encode→decode; DCT after extra H.264 pass; decoy append |
| `test_detect.py` | Flags own stego; OpenPuff slack sample when present |
| `test_compare.py` | OpenPuff pair `openpuff_like` |
| `test_reveal.py` | OpenPuff fail-fast (&lt; few seconds); append via `decode_payload` |
| `test_ecc.py` | Reed–Solomon round-trip |

Run:

```bash
pytest -q
```

---

## 12. Capacity cheat sheet

| Method | Binding limit | Order of magnitude |
|--------|---------------|--------------------|
| Append | Disk / FS max file size | Up to many GB |
| LSB | `0.9 × frames×W×H×3×bits/8` | Often multi‑GB on long HD clips |
| DCT | Robust slots / RS / `--max-robust` (default 1 MiB) | KB – ~1 MiB typical |

---

## 13. Suggested reading order in the codebase

1. `vsteg/__init__.py` — constants  
2. `vsteg/container.py` — format + crypto  
3. `vsteg/methods/append.py` — simplest path  
4. `vsteg/methods/lsb.py` — pixel embedding  
5. `vsteg/ecc.py` then `vsteg/methods/dct.py` — robust path  
6. `vsteg/detect/report.py` + `mp4_forensics.py` — steganalysis  
7. `vsteg/reveal.py` — auto decode policy  
8. `vsteg/compare.py` — pairwise forensics  
9. `vsteg/cli.py` / `web.py` — how users trigger the above  

---

## 14. Glossary

| Term | Meaning |
|------|---------|
| **Carrier / cover** | Innocent video that will hold the secret |
| **Stego object** | Output video that contains hidden data |
| **Payload / secret** | File or bytes being hidden |
| **LSB** | Least Significant Bit of a sample |
| **DCT** | Discrete Cosine Transform (JPEG/MPEG-style blocks) |
| **QIM** | Quantization Index Modulation — encode bits by quantizer parity |
| **ECC** | Error-correcting code (here Reed–Solomon) |
| **mdat / stsz** | MP4 media-data box / sample-size table |
| **mdat slack** | Bytes in `mdat` not accounted for by `stsz` (OpenPuff-like) |
| **Decoy** | Secondary payload unlocked by a different password |
| **Plausible deniability** | Ability to reveal a convincing fake secret under coercion |

---

*Generated for coursework from the `vsteg` implementation. If code and this document disagree, trust the source and update this file.*
