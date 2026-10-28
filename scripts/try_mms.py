#!/usr/bin/env python
"""Sidecar experiment: is Meta MMS viable for Kikuyu / Dholuo?

NOT part of the pipeline. MMS is wav2vec2-CTC, which CTranslate2 cannot convert,
so this runs under plain `transformers` on CPU — slower than the deployed stack and
deliberately kept outside it (CLAUDE.md §7). The question is only whether its
per-language adapters produce better vernacular text than Paza, especially for
Dholuo where Paza is weak.

Known limitations of CTC output, expected before we start:
  * no punctuation, no capitalisation
  * no timestamps (breaks click-to-seek, reference alignment, routing arbitration)
  * one language per adapter, so no code-switching: English and Kiswahili inside a
    Kikuyu utterance will be rendered phonetically in Kikuyu characters

    python scripts/try_mms.py --fetch          # download (~3.7 GB, resumable)
    python scripts/try_mms.py --run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Config               # noqa: E402

REPO = "facebook/mms-1b-all"
DEST = Config.MODELS_DIR / "_src" / "mms-1b-all"
UA = "osint-monitor/0.1"

FILES = [
    "config.json", "preprocessor_config.json", "tokenizer_config.json",
    "special_tokens_map.json", "vocab.json",
    "adapter.kik.safetensors", "adapter.luo.safetensors",
    "model.safetensors",          # 3.7 GB, last so the small files land first
]

# Best captures per language, chosen by transcript density (see report_segments).
CLIPS = [
    ("kik", "data/audio/inooro_fm/2026/09/29/inooro_fm_20260929T080525Z_000.opus",
     "Inooro FM — best Kikuyu (Paza logprob -0.259)"),
    ("luo", "data/audio/ramogi_fm/2026/09/29/ramogi_fm_20260929T140612Z_000.opus",
     "Ramogi FM — best Dholuo (Paza logprob -0.47)"),
]


def fetch() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        url = f"https://huggingface.co/{REPO}/resolve/main/{name}?download=true"
        out = DEST / name
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
            expected = int(urllib.request.urlopen(req, timeout=30)
                           .headers.get("Content-Length", 0))
        except Exception:                       # noqa: BLE001 — size is advisory
            expected = 0
        if out.exists() and expected and out.stat().st_size >= expected:
            print(f"  skip {name} ({out.stat().st_size/1e6:.1f} MB)")
            continue
        print(f"  fetching {name} ({expected/1e6:.1f} MB)…", flush=True)
        # curl -C - so a 3.7 GB download on a slow link resumes (CLAUDE.md §6).
        r = subprocess.run(["curl", "-L", "-C", "-", "--fail", "--silent",
                            "--show-error", "--retry", "5", "--retry-all-errors",
                            "-A", UA, "-o", str(out), url],
                           capture_output=True, text=True)
        got = out.stat().st_size if out.exists() else 0
        print(f"    -> {got/1e6:.1f} MB rc={r.returncode} {r.stderr[:120]}")
    return 0


def run() -> int:
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    from faster_whisper.audio import decode_audio      # already installed

    if not (DEST / "model.safetensors").exists():
        print(f"Model not present at {DEST}. Run with --fetch first.")
        return 2

    print(f"Loading {REPO} from {DEST} (CPU, float32)…", flush=True)
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(DEST))
    model = Wav2Vec2ForCTC.from_pretrained(str(DEST))
    model.eval()
    print(f"  loaded in {time.perf_counter()-t0:.0f}s\n")

    results = []
    for lang, rel, label in CLIPS:
        path = Config.ROOT_DIR / rel
        if not path.exists():
            print(f"[{lang}] missing audio: {path}")
            continue

        print(f"[{lang}] {label}", flush=True)
        try:
            processor.tokenizer.set_target_lang(lang)
            model.load_adapter(lang)
        except Exception as exc:                 # noqa: BLE001 — report and continue
            print(f"   adapter load FAILED: {type(exc).__name__}: {exc}\n")
            results.append({"lang": lang, "clip": rel, "error": str(exc)[:300]})
            continue

        wav = decode_audio(str(path), sampling_rate=16_000)
        audio_seconds = len(wav) / 16_000

        t0 = time.perf_counter()
        inputs = processor(wav, sampling_rate=16_000, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        ids = torch.argmax(logits, dim=-1)[0]
        text = processor.decode(ids)
        wall = time.perf_counter() - t0

        rtf = audio_seconds / wall if wall else 0.0
        print(f"   audio {audio_seconds:.0f}s  wall {wall:.0f}s  RTF {rtf:.3f}x")
        print(f"   {text[:300]}\n", flush=True)
        results.append({
            "lang": lang, "clip": rel, "label": label,
            "audio_seconds": round(audio_seconds, 1),
            "wall_seconds": round(wall, 1), "realtime_factor": round(rtf, 3),
            "chars": len(text), "text": text,
        })

    outdir = Config.DATA_DIR / "comparisons" / "mms_trial"
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (outdir / "mms_results.json").write_text(
        json.dumps({"generated_utc": stamp, "model": REPO, "results": results},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    for r in results:
        if "text" in r:
            (outdir / f"mms_{r['lang']}.txt").write_text(r["text"] + "\n",
                                                         encoding="utf-8")
    print(f"Wrote {outdir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()

    Config.enable_utf8_console()
    Config.ensure_dirs()
    if args.fetch:
        return fetch()
    if args.run:
        return run()
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
