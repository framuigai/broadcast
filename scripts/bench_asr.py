#!/usr/bin/env python
"""Measure decode speed under different settings on this CPU.

Capacity planning for the whole POC rests on the realtime factor, so it needs to
be measured rather than assumed. RTF < 1.0 means slower than realtime: at 0.125x,
one hour of audio costs eight hours of CPU.

    python scripts/bench_asr.py --audio data/derived/bench20.opus --pack kik
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Config              # noqa: E402
from app.services import registry          # noqa: E402

# name, beam_size, word_timestamps, cpu_threads
CONFIGS = [
    ("baseline (beam5, words, 8t)", 5, True, 8),
    ("beam1, words, 8t", 1, True, 8),
    ("beam1, no words, 8t", 1, False, 8),
    ("beam1, no words, 12t", 1, False, 12),
    ("beam5, words, 12t", 5, True, 12),
]


def run(audio: str, pack_code: str, beam: int, words: bool, threads: int) -> tuple[float, float, int]:
    """Load a fresh model per config so thread count actually takes effect."""
    from faster_whisper import WhisperModel
    from app.services import asr

    pack = registry.pack_for(pack_code)
    asr._register_extra_language_tokens(pack.model_ref)
    spec = registry.model_spec(pack.model_ref)
    path = Config.ROOT_DIR / spec["local_dir"]

    model = WhisperModel(str(path), device="cpu",
                         compute_type=spec.get("compute_type", "int8"),
                         cpu_threads=threads)

    t0 = time.perf_counter()
    segments, info = model.transcribe(
        audio, language=pack.whisper_token, beam_size=beam,
        vad_filter=True, vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False, word_timestamps=words,
    )
    texts = [s.text for s in segments]           # generator: work happens here
    wall = time.perf_counter() - t0
    dur = getattr(info, "duration", 0.0) or 0.0
    del model
    return dur, wall, len(texts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--pack", default="kik")
    args = ap.parse_args()

    Config.enable_utf8_console()
    audio = str(Config.ROOT_DIR / args.audio) if not Path(args.audio).is_absolute() else args.audio
    if not Path(audio).exists():
        print(f"No such audio: {audio}")
        return 2

    print(f"Host      : {os.cpu_count()} logical cores, no CUDA")
    print(f"Audio     : {Path(audio).name}")
    print(f"Pack      : {args.pack}\n")
    print(f"{'config':<32} {'audio':>7} {'wall':>8} {'RTF':>7}  {'hrs/day':>8}")
    print("-" * 68)

    best = None
    for name, beam, words, threads in CONFIGS:
        try:
            dur, wall, n = run(audio, args.pack, beam, words, threads)
        except Exception as exc:                         # noqa: BLE001
            print(f"{name:<32} FAILED: {str(exc)[:40]}")
            continue
        rtf = dur / wall if wall else 0.0
        # What 24h of wall-clock buys, if the box did nothing else.
        hrs_per_day = 24 * rtf
        print(f"{name:<32} {dur:>6.1f}s {wall:>7.1f}s {rtf:>6.3f}x {hrs_per_day:>7.1f}h")
        if best is None or rtf > best[1]:
            best = (name, rtf)

    if best:
        print("-" * 68)
        print(f"Fastest: {best[0]} at {best[1]:.3f}x realtime "
              f"({24 * best[1]:.1f} audio-hours per 24h of wall clock)")
        print("\nNote: word_timestamps=False disables click-to-seek in the transcript UI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
