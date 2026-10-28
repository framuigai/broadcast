#!/usr/bin/env python
"""Download and convert the ASR models declared in config/language_packs.json.

Both models are MIT-licensed and ungated — no Hugging Face account is required
for this step. (The Africa Next Voices *dataset* is gated, but that is Step 2.)

Conversion to CTranslate2 int8 is not optional on this machine: it has no CUDA
device, and CT2 int8 is several times faster than transformers on CPU while
halving the memory footprint.

    python scripts/download_models.py                  # everything
    python scripts/download_models.py --model paza-whisper-large-v3-turbo-ct2
    python scripts/download_models.py --list
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Config              # noqa: E402
from app.services import registry          # noqa: E402


def human(gb: float) -> str:
    return f"{gb:.1f} GB"


def dir_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def download_and_convert(model_ref: str, spec: dict, force: bool = False) -> bool:
    from huggingface_hub import snapshot_download

    target = Config.ROOT_DIR / spec["local_dir"]
    if target.exists() and not force:
        # A valid CT2 directory always has model.bin.
        if (target / "model.bin").exists():
            print(f"  [skip] {model_ref} already present ({human(dir_size_gb(target))})")
            return True
        print(f"  [warn] {target} exists but has no model.bin — reconverting")
        shutil.rmtree(target)

    repo = spec["hf_repo"]
    print(f"  [1/2] Downloading {repo} …")
    # One cache directory per model_ref. Keep this stable: changing it strands any
    # partially downloaded blobs, which on a slow link costs hours of re-download.
    cache = Config.MODELS_DIR / "_hf" / model_ref
    try:
        src = snapshot_download(
            repo_id=repo,
            cache_dir=str(cache),
            allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin",
                            "*.model", "tokenizer*", "preprocessor*"],
        )
    except Exception as exc:
        print(f"  [FAIL] download: {exc}")
        return False

    print(f"  [2/2] Converting to CTranslate2 ({spec.get('compute_type', 'int8')}) …")
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "ctranslate2.converters.transformers",
        "--model", src,
        "--output_dir", str(target),
        "--quantization", spec.get("compute_type", "int8"),
        "--copy_files", "tokenizer.json", "preprocessor_config.json",
        "--force",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Older/newer converter builds differ on --copy_files; retry without it.
        print("  [retry] converter rejected --copy_files; retrying minimally")
        result = subprocess.run([
            sys.executable, "-m", "ctranslate2.converters.transformers",
            "--model", src, "--output_dir", str(target),
            "--quantization", spec.get("compute_type", "int8"), "--force",
        ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  [FAIL] convert:\n{result.stderr[-1500:]}")
        return False

    # faster-whisper needs the tokenizer and preprocessor next to the model.
    for name in ("tokenizer.json", "preprocessor_config.json", "vocabulary.json"):
        s = Path(src) / name
        if s.exists() and not (target / name).exists():
            shutil.copy2(s, target / name)

    print(f"  [ok]   {model_ref} -> {target} ({human(dir_size_gb(target))})")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="single model_ref to fetch")
    ap.add_argument("--list", action="store_true", help="show registered models and exit")
    ap.add_argument("--force", action="store_true", help="re-download and reconvert")
    args = ap.parse_args()

    Config.enable_utf8_console()
    Config.ensure_dirs()
    specs = registry.all_model_specs()

    if args.list:
        print(f"{'model_ref':<40} {'repo':<44} {'size':>8}  status")
        for ref, s in specs.items():
            target = Config.ROOT_DIR / s["local_dir"]
            status = "present" if (target / "model.bin").exists() else "missing"
            print(f"{ref:<40} {s['hf_repo']:<44} {s['approx_size_gb']:>6.1f}G  {status}")
        return 0

    selected = {args.model: specs[args.model]} if args.model else specs
    if args.model and args.model not in specs:
        print(f"Unknown model_ref {args.model!r}. Registered: {', '.join(specs)}")
        return 2

    total = sum(s["approx_size_gb"] for s in selected.values())
    print(f"Fetching {len(selected)} model(s), roughly {human(total)} before conversion.")
    print(f"Target directory: {Config.MODELS_DIR}\n")

    ok = True
    for ref, spec in selected.items():
        print(f"{ref}  ({spec['licence']})")
        ok &= download_and_convert(ref, spec, force=args.force)
        print()

    if ok:
        print("All models ready. Next: python scripts/capture_worker.py")
    else:
        print("One or more models failed. Re-run with --model <ref> to retry individually.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
