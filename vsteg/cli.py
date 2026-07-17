"""vsteg CLI — encode / decode / detect."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vsteg import __version__
from vsteg.container import ContainerError
from vsteg.detect.report import detect, format_json, format_text
from vsteg.methods import append, dct, lsb
from vsteg.reveal import RevealError, decode_payload
from vsteg.sniff import guess_extension


EXIT_OK = 0
EXIT_USAGE = 1
EXIT_DECODE = 2
EXIT_DETECT_POSITIVE = 3
EXIT_DETECT_CLEAN = 4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vsteg",
        description="Video steganography toolkit: encode, decode, detect",
    )
    parser.add_argument("--version", action="version", version=f"vsteg {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # encode
    p_enc = sub.add_parser("encode", help="Hide a payload in a carrier video")
    p_enc.add_argument("-i", "--input", required=True, help="Carrier video")
    p_enc.add_argument("-s", "--secret", required=True, help="Secret file to hide")
    p_enc.add_argument("-o", "--output", required=True, help="Output stego video")
    p_enc.add_argument("-p", "--password", default=None, help="Optional password")
    p_enc.add_argument(
        "--decoy",
        default=None,
        help="Optional decoy file (plausible deniability; requires passwords)",
    )
    p_enc.add_argument(
        "--decoy-password",
        default=None,
        help="Password that reveals the decoy (must differ from --password)",
    )
    p_enc.add_argument(
        "-m",
        "--method",
        choices=["append", "lsb", "dct"],
        default="append",
        help="Embedding method (default: append)",
    )
    p_enc.add_argument("--bits", type=int, default=1, help="LSB bits/channel (1-3)")
    p_enc.add_argument(
        "--strength", type=float, default=16.0, help="DCT QIM step Δ (default 16)"
    )
    p_enc.add_argument("--crf", type=int, default=18, help="H.264 CRF for dct method")
    p_enc.add_argument(
        "--max-robust",
        type=int,
        default=1 * 1024 * 1024,
        help="Max payload bytes for dct (default 1MiB)",
    )

    # decode
    p_dec = sub.add_parser("decode", help="Extract a hidden payload")
    p_dec.add_argument("-i", "--input", required=True, help="Stego video")
    p_dec.add_argument("-o", "--output", required=True, help="Recovered payload path")
    p_dec.add_argument("-p", "--password", default=None, help="Optional password")
    p_dec.add_argument(
        "-m",
        "--method",
        choices=["auto", "append", "lsb", "dct"],
        default="auto",
        help="Force method (default: auto-detect)",
    )
    p_dec.add_argument("--bits", type=int, default=1)
    p_dec.add_argument("--strength", type=float, default=16.0)

    # detect
    p_det = sub.add_parser("detect", help="Steganalysis / detection")
    p_det.add_argument("-i", "--input", required=True, help="Suspect video")
    p_det.add_argument("--json", action="store_true", help="JSON output")
    p_det.add_argument(
        "--fast",
        action="store_true",
        help="Skip deep statistical / DCT analysis",
    )

    # compare
    p_cmp = sub.add_parser("compare", help="Compare two videos and report differences")
    p_cmp.add_argument("-a", "--a", required=True, help="First video")
    p_cmp.add_argument("-b", "--b", required=True, help="Second video")
    p_cmp.add_argument("--json", action="store_true", help="JSON output")
    p_cmp.add_argument(
        "--html",
        default=None,
        help="Write an HTML report with graphs to this path",
    )
    p_cmp.add_argument(
        "--fast",
        action="store_true",
        help="Skip sampled frame analysis",
    )
    p_cmp.add_argument(
        "--samples",
        type=int,
        default=12,
        help="Number of frames to sample for deep compare (default 12)",
    )

    # web
    p_web = sub.add_parser("web", help="Start the beginner web UI")
    p_web.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    p_web.add_argument("--port", type=int, default=5000, help="Bind port (default 5000)")

    args = parser.parse_args(argv)

    try:
        if args.command == "encode":
            return _cmd_encode(args)
        if args.command == "decode":
            return _cmd_decode(args)
        if args.command == "detect":
            return _cmd_detect(args)
        if args.command == "compare":
            return _cmd_compare(args)
        if args.command == "web":
            return _cmd_web(args)
    except (ContainerError, append.AppendError, lsb.LSBError, dct.DCTError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_DECODE
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    return EXIT_USAGE


def _cmd_encode(args: argparse.Namespace) -> int:
    secret = Path(args.secret).read_bytes()
    method = args.method
    out = Path(args.output)
    decoy = Path(args.decoy).read_bytes() if args.decoy else None
    decoy_password = args.decoy_password

    if method == "append":
        append.encode(
            args.input,
            secret,
            out,
            password=args.password,
            decoy=decoy,
            decoy_password=decoy_password,
        )
    elif method == "lsb":
        if out.suffix.lower() not in {".mkv", ".avi"}:
            print(
                "warning: lsb method uses lossless FFV1; prefer a .mkv output",
                file=sys.stderr,
            )
        lsb.encode(
            args.input,
            secret,
            out,
            password=args.password,
            bits=args.bits,
            decoy=decoy,
            decoy_password=decoy_password,
        )
    elif method == "dct":
        dct.encode(
            args.input,
            secret,
            out,
            password=args.password,
            strength=args.strength,
            crf=args.crf,
            max_robust=args.max_robust,
            decoy=decoy,
            decoy_password=decoy_password,
        )
    else:
        print(f"unknown method: {method}", file=sys.stderr)
        return EXIT_USAGE

    note = " + decoy" if decoy is not None else ""
    print(f"encoded ({method}{note}) → {out}")
    return EXIT_OK


def _cmd_decode(args: argparse.Namespace) -> int:
    method = args.method
    try:
        if method == "lsb":
            plaintext = lsb.decode(
                args.input, password=args.password, bits=args.bits
            )
        elif method == "dct":
            plaintext = dct.decode(
                args.input, password=args.password, strength=args.strength
            )
        else:
            plaintext = decode_payload(
                args.input, password=args.password, method=method
            )
    except RevealError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_DECODE
    except Exception as exc:
        print(f"error: decode failed: {exc}", file=sys.stderr)
        return EXIT_DECODE

    out = Path(args.output)
    # If the user picked a generic name (.bin / no suffix), use sniffed type
    if out.suffix.lower() in {"", ".bin"}:
        out = out.with_suffix(guess_extension(plaintext))
    out.write_bytes(plaintext)
    print(f"decoded → {out} ({len(plaintext)} bytes)")
    return EXIT_OK


def _cmd_detect(args: argparse.Namespace) -> int:
    report = detect(args.input, deep=not args.fast)
    if args.json:
        print(format_json(report))
    else:
        print(format_text(report))
    if report.verdict == "clean":
        return EXIT_DETECT_CLEAN
    return EXIT_DETECT_POSITIVE


def _cmd_compare(args: argparse.Namespace) -> int:
    from vsteg.compare import compare, format_json as cmp_json
    from vsteg.compare import format_text as cmp_text
    from vsteg.compare import write_html

    report = compare(
        args.a,
        args.b,
        sample_frames=args.samples,
        deep=not args.fast,
    )
    if args.html:
        out = write_html(report, args.html)
        print(f"html report → {out}")
    if args.json:
        print(cmp_json(report))
    else:
        print(cmp_text(report))
    return EXIT_OK


def _cmd_web(args: argparse.Namespace) -> int:
    try:
        from vsteg.web import main as web_main
    except ImportError as exc:
        print(
            "error: web UI deps missing. Install with:\n"
            '  pip install -e ".[web]"\n'
            f"({exc})",
            file=sys.stderr,
        )
        return EXIT_USAGE
    web_main(host=args.host, port=args.port)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
