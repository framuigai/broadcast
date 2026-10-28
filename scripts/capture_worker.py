#!/usr/bin/env python
"""Continuous stream capture (Plan §6.2).

Capture runs 24/7 on every enabled channel — storage is cheap (~1.6 GB/day for all
six) and retrospective search has value. Transcription is what gets rationed, not
capture: every segment lands in the queue, and the web app's background pipeline
transcribes and translates in priority order.

    python scripts/capture_worker.py                      # all enabled channels
    python scripts/capture_worker.py --channel inooro_fm
    python scripts/capture_worker.py --once --duration 120 # one short test segment

This is only the scheduling loop. Capturing a segment — stream resolution, the
ffmpeg pull, the yt-dlp fallback, SHA-256, the `segments` row, the audit entry —
is `app.services.capture.capture_segment`, the same code the Live page's Record
button uses, so a segment is identical whichever way it was captured.

Two rules this worker exists to enforce:
  * Source audio is write-once. Nothing downstream ever modifies these files.
  * Every interruption is recorded as an explicit gap, never silently absorbed.
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db                                  # noqa: E402
from app.config import Config                       # noqa: E402
from app.services import capture, registry          # noqa: E402

ACTOR = "capture_worker"
MAX_BACKOFF = 300


def capture_one(ch: registry.Channel, duration: int,
                stop: threading.Event | None = None) -> None:
    """Capture and register one segment, then report it. Raises on failure."""
    seg_id = capture.capture_segment(ch, duration, source_type="broadcast",
                                     actor=ACTOR, cancel=stop)
    seg = db.query_one("SELECT path, priority FROM segments WHERE id=?", (seg_id,)) or {}
    path = Config.ROOT_DIR / seg.get("path", "")
    size_mb = path.stat().st_size / 1e6 if path.is_file() else 0.0
    print(f"  [{ch.id}] segment {seg_id}  {path.name}  {duration}s  {size_mb:.1f} MB"
          f"  priority={seg.get('priority')}")


def channel_loop(ch: registry.Channel, duration: int, stop: threading.Event) -> None:
    backoff = 5
    while not stop.is_set():
        gap_start = capture.utc_now()
        try:
            capture_one(ch, duration, stop)
            backoff = 5                                    # reset on success
        except Exception as exc:                            # noqa: BLE001 — log and retry
            # A capture cut short by Ctrl-C is still a hole in the archive, but
            # its cause is the operator, not the stream — say so in the gap.
            reason = "capture worker stopped by operator" if stop.is_set() else str(exc)
            print(f"  [{ch.id}] {'STOPPED' if stop.is_set() else 'ERROR'}: {reason}")
            capture.record_gap(ch.id, gap_start, reason)
            if stop.is_set():
                break
            print(f"  [{ch.id}] backing off {backoff}s")
            stop.wait(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", help="capture a single channel id")
    ap.add_argument("--duration", type=int, default=Config.SEGMENT_SECONDS,
                    help="segment length in seconds")
    ap.add_argument("--once", action="store_true", help="capture one segment then exit")
    args = ap.parse_args()

    Config.enable_utf8_console()
    Config.ensure_dirs()
    db.init_db()
    # capture.py logs its fallbacks and failures; surface them on the console.
    logging.basicConfig(level=logging.INFO, format="  %(levelname)s %(name)s: %(message)s")

    channels = ([registry.channel(args.channel)] if args.channel
                else registry.all_channels(enabled_only=True))
    if not channels:
        print("No enabled channels in config/channels.json")
        return 2

    print(f"Capture worker ({capture.SOFTWARE_VERSION}) — {len(channels)} channel(s), "
          f"{args.duration}s segments")
    print(f"Archive: {Config.AUDIO_DIR}")
    print("Storage cost is about 1.6 GB/day for all six channels.")
    print("Transcription + translation happen in the web app's background pipeline "
          "(python run.py).\n")

    if args.once:
        ok = True
        for ch in channels:
            gap_start = capture.utc_now()
            try:
                capture_one(ch, args.duration)
            except Exception as exc:                        # noqa: BLE001
                print(f"  [{ch.id}] ERROR: {exc}")
                capture.record_gap(ch.id, gap_start, str(exc))
                ok = False
        return 0 if ok else 1

    stop = threading.Event()
    threads = [threading.Thread(target=channel_loop, args=(ch, args.duration, stop),
                                daemon=True, name=ch.id) for ch in channels]
    for t in threads:
        t.start()
    print("Running. Ctrl-C to stop.\n")
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        # Ctrl-C reaches ffmpeg too (same console), so direct pulls end at once;
        # yt-dlp runs in its own process group, so the stop event is what ends
        # it. capture_segment then discards each partial file (never starting a
        # fallback) and the loop records the gap. Wait so none is left behind.
        print("\nStopping — discarding in-flight partial segments…")
        stop.set()
        for t in threads:
            t.join(timeout=60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
