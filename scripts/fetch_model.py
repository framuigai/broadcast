#!/usr/bin/env python
"""Resumable model download, bypassing the Hugging Face cache.

Why this exists
---------------
huggingface_hub 1.32.0 names its partial files with a per-run suffix
(`<blob>.<random>.incomplete`), so a download interrupted on a slow link does NOT
resume — the next run starts a fresh file from zero. On a ~50 KB/s connection that
makes a 1.6 GB model effectively undownloadable.

This fetches each file directly from the repo's resolve URL with HTTP Range
resume, into a flat directory the CTranslate2 converter can read. An interrupted
run continues from exactly where it stopped.

    python scripts/fetch_model.py --model paza-whisper-large-v3-turbo-ct2
    python scripts/fetch_model.py --all
    python scripts/fetch_model.py --all --status
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Config              # noqa: E402
from app.services import registry          # noqa: E402

UA = "osint-monitor/0.1 (resumable model fetch)"

# Only what CTranslate2 + faster-whisper actually need. Skipping the .bin
# duplicates of the safetensors halves the bytes on the wire.
WANTED_SUFFIXES = (".json", ".txt", ".safetensors")
SKIP_NAMES = {"pytorch_model.bin", "model.bin", "flax_model.msgpack",
              "tf_model.h5", "model.onnx"}


def src_dir(model_ref: str) -> Path:
    """Flat download directory, separate from the converted output."""
    return Config.MODELS_DIR / "_src" / model_ref


def list_repo_files(repo: str, spec: dict | None = None) -> list[str]:
    # A spec may name its files explicitly — NLLB ships only pytorch_model.bin
    # and needs its sentencepiece model, both of which the default filter drops.
    if spec and spec.get("fetch_files"):
        return list(spec["fetch_files"])
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo)
    keep = []
    for f in files:
        if "/" in f:                       # skip subfolders (onnx variants, etc.)
            continue
        if f in SKIP_NAMES:
            continue
        if f.endswith(WANTED_SUFFIXES):
            keep.append(f)
    return keep


def remote_size(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
    except (urllib.error.URLError, ValueError, TimeoutError):
        return None


def fetch(repo: str, filename: str, dest_dir: Path, retries: int = 6) -> bool:
    """Download one file with curl's Range resume. Returns True when complete."""
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}?download=true"
    dest = dest_dir / filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    expected = remote_size(url)
    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if expected and have >= expected:
            print(f"    {filename}: complete ({have / 1e6:.1f} MB)")
            return True
        if have:
            print(f"    {filename}: resuming at {have / 1e6:.1f} MB"
                  f"{f' of {expected / 1e6:.1f} MB' if expected else ''}"
                  f" (attempt {attempt}/{retries})")
        else:
            print(f"    {filename}: starting"
                  f"{f' ({expected / 1e6:.1f} MB)' if expected else ''}")

        # -C - resumes from the current file length; --retry handles brief drops.
        cmd = ["curl", "-L", "-C", "-", "--fail", "--silent", "--show-error",
               "--retry", "5", "--retry-delay", "5", "--retry-all-errors",
               "--speed-time", "120", "--speed-limit", "1024",   # abort a dead stall
               "-A", UA, "-o", str(dest), url]
        result = subprocess.run(cmd, capture_output=True, text=True)

        now = dest.stat().st_size if dest.exists() else 0
        if expected and now >= expected:
            print(f"    {filename}: done ({now / 1e6:.1f} MB)")
            return True
        if result.returncode == 0 and not expected:
            print(f"    {filename}: done ({now / 1e6:.1f} MB)")
            return True
        if now > have:
            print(f"    {filename}: stalled at {now / 1e6:.1f} MB, retrying")
        else:
            err = (result.stderr or "").strip()[:180]
            print(f"    {filename}: no progress ({err or 'stalled'})")

    return False


def seed_from_hf_cache(model_ref: str, dest_dir: Path) -> None:
    """Reuse bytes already downloaded into the HF cache.

    The partial blobs there are contiguous prefixes, so copying one in lets the
    Range resume continue rather than re-fetching from zero.
    """
    cache = Config.MODELS_DIR / "_hf" / model_ref
    if not cache.exists():
        return
    partials = sorted(cache.rglob("*.incomplete"), key=lambda p: p.stat().st_size,
                      reverse=True)
    if not partials:
        return
    biggest = partials[0]
    target = dest_dir / "model.safetensors"
    if target.exists() and target.stat().st_size >= biggest.stat().st_size:
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    mb = biggest.stat().st_size / 1e6
    print(f"    reusing {mb:.1f} MB already downloaded into the HF cache")
    shutil.copy2(biggest, target)


def fetch_model(model_ref: str, spec: dict) -> bool:
    dest = src_dir(model_ref)
    dest.mkdir(parents=True, exist_ok=True)
    repo = spec["hf_repo"]

    print(f"  {model_ref}  <-  {repo}")
    try:
        files = list_repo_files(repo, spec)
    except Exception as exc:                     # noqa: BLE001 — network/API shape
        print(f"  [FAIL] could not list {repo}: {exc}")
        return False

    if any(f == "model.safetensors" for f in files):
        seed_from_hf_cache(model_ref, dest)

    ok = True
    # Small metadata first so a long transfer never blocks the cheap files.
    for f in sorted(files, key=lambda n: n.endswith((".safetensors", ".bin"))):
        ok &= fetch(repo, f, dest)
    return ok


def convert(model_ref: str, spec: dict) -> bool:
    out = Config.ROOT_DIR / spec["local_dir"]
    if (out / "model.bin").exists():
        print(f"  [skip] {model_ref} already converted")
        return True

    src = src_dir(model_ref)
    print(f"  converting -> {out}  ({spec.get('compute_type', 'int8')})")
    cmd = [sys.executable, "-m", "ctranslate2.converters.transformers",
           "--model", str(src), "--output_dir", str(out),
           "--quantization", spec.get("compute_type", "int8"), "--force"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [FAIL] convert:\n{result.stderr[-1200:]}")
        return False

    # faster-whisper expects the tokenizer and preprocessor beside the model;
    # other model kinds name what they need in the registry.
    wanted = spec.get("copy_files") or ("tokenizer.json", "preprocessor_config.json",
                                        "vocabulary.json")
    for name in wanted:
        s = src / name
        if s.exists() and not (out / name).exists():
            shutil.copy2(s, out / name)
    print(f"  [ok] {model_ref} ready")
    return True


def status() -> None:
    print(f"{'model_ref':<38} {'downloaded':>12} {'expected':>10}  state")
    for ref, spec in registry.all_model_specs().items():
        out = Config.ROOT_DIR / spec["local_dir"]
        d = src_dir(ref)
        got = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.exists() else 0
        state = "converted" if (out / "model.bin").exists() else \
                "downloading" if got else "not started"
        print(f"{ref:<38} {got / 1e6:>10.1f} MB {spec['approx_size_gb'] * 1000:>8.0f} MB  {state}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="single model_ref")
    ap.add_argument("--all", action="store_true", help="every registered model")
    ap.add_argument("--status", action="store_true", help="show progress and exit")
    ap.add_argument("--no-convert", action="store_true", help="download only")
    args = ap.parse_args()

    Config.enable_utf8_console()
    Config.ensure_dirs()

    if args.status:
        status()
        return 0

    specs = registry.all_model_specs()
    if args.model:
        if args.model not in specs:
            print(f"Unknown model_ref {args.model!r}. Registered: {', '.join(specs)}")
            return 2
        specs = {args.model: specs[args.model]}
    elif not args.all:
        ap.print_help()
        return 2

    ok = True
    for ref, spec in specs.items():
        got = fetch_model(ref, spec)
        if got and not args.no_convert:
            got = convert(ref, spec)
        ok &= got
        print()

    status()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
