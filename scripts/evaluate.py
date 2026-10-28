#!/usr/bin/env python
"""Score a Language Pack against a frozen reference set (Plan §9).

    python scripts/evaluate.py --pack kik --set bcast_kik
    python scripts/evaluate.py --pack luo --set bcast_luo --by-channel
    python scripts/evaluate.py --pack kik --set bcast_kik --dry-run

Two things make these numbers mean something, and both are enforced here:

  * The normalisation signature is recorded with every score. Two runs with
    different normalisation are not comparable, and the tool says so rather than
    letting you read across them.
  * Results are broken down per station and per daypart. A single aggregate hides
    the difference between a studio bulletin and a telephone call-in, which is
    exactly the difference that matters operationally.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db                           # noqa: E402
from app.config import Config                # noqa: E402
from app.services import asr, evaluation, registry   # noqa: E402


def fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:5.1f}%"


def run(pack_code: str, set_name: str, by_channel: bool, by_daypart: bool,
        dry_run: bool, limit: int | None) -> int:
    pack = registry.pack_for(pack_code)

    try:
        items = evaluation.load_reference_set(set_name)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2

    items = [i for i in items if i.language == pack_code]
    if limit:
        items = items[:limit]
    if not items:
        print(f"No items for language {pack_code!r} in reference set {set_name!r}.")
        return 2

    print(f"Pack      : {pack.name} ({pack.code}) — {pack.autonym}")
    print(f"Model     : {pack.model_ref}  token <|{pack.whisper_token}|>")
    print(f"Set       : {set_name}  ({len(items)} items)")
    rules = pack.orthography.get("normalise", [])
    print(f"Normalise : {evaluation.normalisation_signature(rules)}")
    print()

    if dry_run:
        print("Dry run — no transcription performed. Items:")
        for i in items[:20]:
            print(f"  {i.channel_id or '—':<12} {i.daypart or '—':<16} {Path(i.audio_path).name}")
        return 0

    refs: list[str] = []
    hyps: list[str] = []
    groups: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)

    audio_seconds = 0.0
    t_start = time.perf_counter()

    for n, item in enumerate(items, 1):
        path = Path(item.audio_path)
        if not path.is_absolute():
            path = Config.ROOT_DIR / path
        if not path.exists():
            print(f"  [{n}/{len(items)}] MISSING {path}")
            continue

        try:
            utts, meta = asr.transcribe(path, pack_code)
        except (FileNotFoundError, RuntimeError, KeyError) as exc:
            print(f"  [{n}/{len(items)}] FAILED {path.name}: {exc}")
            continue

        hyp = " ".join(u.text for u in utts).strip()
        audio_seconds += meta["audio_seconds"]
        refs.append(item.reference)
        hyps.append(hyp)
        groups[(item.channel_id or "all", item.daypart or "all")].append((item.reference, hyp))

        one = evaluation.score_for_pack(pack_code, [item.reference], [hyp])
        print(f"  [{n}/{len(items)}] {path.name:<38} WER {fmt_pct(one.wer)}  "
              f"RTF {meta['realtime_factor']:.2f}x")

    if not refs:
        print("\nNothing scored — every item failed or was missing.")
        return 1

    wall = time.perf_counter() - t_start
    rtf = audio_seconds / wall if wall else 0.0
    overall = evaluation.score_for_pack(pack_code, refs, hyps)

    print("\n" + "=" * 72)
    print(f"OVERALL   WER {fmt_pct(overall.wer)}   CER {fmt_pct(overall.cer)}   "
          f"{overall.n_utterances} items, {overall.n_ref_words} ref words")
    print(f"          S={overall.substitutions}  D={overall.deletions}  I={overall.insertions}")
    print(f"          realtime factor {rtf:.2f}x  ({audio_seconds/60:.1f} min audio "
          f"in {wall/60:.1f} min wall)")
    print("=" * 72)

    if by_channel or by_daypart:
        print("\nBreakdown — read this, not the aggregate:")
        print(f"{'channel':<14} {'daypart':<18} {'items':>6} {'WER':>8} {'CER':>8}")
        for (chan, part), pairs in sorted(groups.items()):
            key_chan = chan if by_channel else "all"
            key_part = part if by_daypart else "all"
            s = evaluation.score_for_pack(pack_code, [p[0] for p in pairs],
                                          [p[1] for p in pairs])
            print(f"{key_chan:<14} {key_part:<18} {len(pairs):>6} "
                  f"{fmt_pct(s.wer):>8} {fmt_pct(s.cer):>8}")

    # Regression check before recording, so a worse run is visible immediately.
    check = evaluation.regression_check(pack_code, overall)
    print(f"\nRegression: {check['status'].upper()} — {check['message']}")

    evaluation.record_run(pack_code, pack.model_ref, set_name, overall,
                          realtime_factor=round(rtf, 3),
                          notes=f"{len(items)} items; {check['status']}")
    db.audit("evaluate.py", "eval_run", "pack", pack_code,
             f"{set_name} WER={overall.wer}")
    print("Recorded. View it at http://127.0.0.1:5000/evaluation")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, help="language pack code, e.g. kik")
    ap.add_argument("--set", dest="set_name", required=True, help="reference set name")
    ap.add_argument("--by-channel", action="store_true", default=True)
    ap.add_argument("--by-daypart", action="store_true", default=True)
    ap.add_argument("--dry-run", action="store_true", help="list items without transcribing")
    ap.add_argument("--limit", type=int, help="score only the first N items")
    args = ap.parse_args()

    Config.enable_utf8_console()
    Config.ensure_dirs()
    db.init_db()

    try:
        return run(args.pack, args.set_name, args.by_channel, args.by_daypart,
                   args.dry_run, args.limit)
    except KeyError as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
