#!/usr/bin/env python
"""Drain the queue from the command line: transcribe, then translate.

You normally do NOT need this — `python run.py` runs the same pipeline in the
background. Use it for batch runs, debugging, or a headless machine, and stop the
web app's worker first (OSINT_BACKGROUND_PIPELINE=0): only one pipeline may run
per machine, because each holds ~3 GB of models.

    python scripts/transcribe_worker.py                 # highest priority first
    python scripts/transcribe_worker.py --limit 5
    python scripts/transcribe_worker.py --segment 42 --pack kik
    python scripts/transcribe_worker.py --translate-only
    python scripts/transcribe_worker.py --watch

Throughput note. Transcription runs at roughly 0.2-0.3x realtime on this machine
(audio seconds per wall second — a 60 s clip takes ~4 minutes including routing;
see CLAUDE.md §3) while six channels produce 144 audio-hours a day, so the queue
is ordered by priority: analyst recordings first, then dayparts.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db                                  # noqa: E402
from app.config import Config                       # noqa: E402
from app.services import pipeline, registry         # noqa: E402


def drain_translations() -> int:
    total = 0
    try:
        while n := pipeline.translate_pending():
            total += n
            print(f"  translated {total} line(s)…")
    except pipeline.TranslationUnavailable as exc:
        print(f"  translation deferred: {exc}")
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=20, help="max segments this run")
    ap.add_argument("--segment", type=int, help="process one specific segment id")
    ap.add_argument("--pack", help="force a language pack, skipping routing")
    ap.add_argument("--translate-only", action="store_true",
                    help="only translate transcript lines still awaiting English")
    ap.add_argument("--watch", action="store_true", help="keep polling for new work")
    args = ap.parse_args()

    Config.enable_utf8_console()
    Config.ensure_dirs()
    db.init_db()

    lock = pipeline.acquire_process_lock()
    if not lock:
        print("Another pipeline is running on this machine (probably the web app's "
              "background worker).\nStop it, or start the app with "
              "OSINT_BACKGROUND_PIPELINE=0, then retry.")
        return 3

    pipeline.requeue_stale()
    pipeline.reconcile_translation_states()

    if args.pack and not registry.pack_exists(args.pack):
        print(f"No language pack {args.pack!r}. Registered: "
              f"{', '.join(p.code for p in registry.all_packs())}")
        return 2

    if args.translate_only:
        print(f"Translated {drain_translations()} line(s)")
        return 0

    if args.segment:
        seg = db.query_one("SELECT * FROM segments WHERE id=?", (args.segment,))
        if not seg:
            print(f"No segment {args.segment}")
            return 2
        db.execute("UPDATE segments SET status='transcribing' WHERE id=?", (seg["id"],))
        return 0 if pipeline.process_segment(seg, args.pack, actor="transcribe_worker") else 1

    done = 0
    t0 = time.perf_counter()
    while args.watch or done < args.limit:
        seg = pipeline.claim_next_segment()
        if not seg:
            drain_translations()
            if not args.watch:
                break
            time.sleep(Config.PIPELINE_POLL_SECONDS)
            continue
        print(f"[{seg['id']}] {Path(seg['path']).name}  ({seg['priority']})")
        ok = pipeline.process_segment(
            seg, args.pack, actor="transcribe_worker",
            on_stage=lambda s: print(f"  {s}…"))
        print(f"  {'done' if ok else 'FAILED'}")
        done += 1

    if not done:
        print("Queue is empty. Record a clip from the Live page, or run "
              "python scripts/capture_worker.py")
    else:
        print(f"Processed {done} segment(s) in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
