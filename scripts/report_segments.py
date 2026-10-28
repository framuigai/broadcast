#!/usr/bin/env python
"""Dump the pipeline's output for one or more segments as a Markdown report.

    python scripts/report_segments.py 8 15
    python scripts/report_segments.py 8 15 --out data/comparisons/report.md

Reads what the pipeline actually produced — it does not re-run anything, so the
report reflects the deployed model configuration rather than an ad-hoc run.
Quarantined lines and untranslated lines are shown as such rather than hidden:
a report that silently drops withheld content is misleading about its own
coverage (CLAUDE.md §2.4).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import _to_eat                     # noqa: E402
from app import db                          # noqa: E402
from app.config import Config               # noqa: E402
from app.services import registry           # noqa: E402


def segment_block(seg_id: int) -> list[str]:
    seg = db.query_one("SELECT * FROM segments WHERE id=?", (seg_id,))
    if not seg:
        return [f"## Segment {seg_id}", "", "_not found_", ""]

    utts = db.query(
        "SELECT * FROM utterances WHERE segment_id=? ORDER BY seq", (seg_id,))
    try:
        ch = registry.channel(seg["channel_id"])
        ch_name, broadcaster = ch.name, ch.broadcaster
    except KeyError:
        ch_name, broadcaster = seg["channel_id"], ""

    langs = sorted({u["routed_language"] for u in utts})
    packs = {p.code: p for p in registry.all_packs()}
    lang_names = ", ".join(packs[c].name if c in packs else c for c in langs) or "—"
    asr_models = sorted({u["model_ref"] for u in utts if u.get("model_ref")})
    mt_models = sorted({u.get("translation_model_ref") for u in utts
                        if u.get("translation_model_ref")})
    n_quar = sum(1 for u in utts if u["quarantined"])
    speech = sum(u["end_seconds"] - u["start_seconds"] for u in utts)
    chars = sum(len(u["text_vernacular"] or "") for u in utts)

    out = [
        f"## Segment {seg_id} — {ch_name}",
        "",
        f"- **Station:** {ch_name} ({broadcaster})",
        f"- **Captured:** {_to_eat(seg['started_at_utc'])} EAT "
        f"(`{seg['started_at_utc']}`)",
        f"- **Duration:** {seg['duration_seconds']:.0f} s — "
        f"{speech:.1f} s of detected speech, {chars} characters transcribed",
        f"- **Language:** {lang_names}",
        f"- **ASR model:** `{', '.join(asr_models) or '—'}`",
        f"- **Translation model:** `{', '.join(mt_models) or '—'}`",
        f"- **Lines:** {len(utts)} ({n_quar} withheld by quality checks)",
        f"- **Source file:** `{Path(seg['path']).name}`",
        f"- **SHA-256:** `{(seg.get('sha256') or '')[:32]}…`",
        "",
    ]

    if not utts:
        out += ["_No transcript._", ""]
        return out

    out += ["| # | Time | logprob | Vernacular | English |",
            "|---|---|---|---|---|"]
    for u in utts:
        ts = f"{int(u['start_seconds']//60):02d}:{int(u['start_seconds']%60):02d}"
        vern = (u["text_vernacular"] or "").replace("|", "/").strip()
        status = u.get("translation_status")
        if u["quarantined"]:
            eng = f"_withheld — {(u['quarantine_reason'] or 'failed checks')[:60]}_"
        elif status == "skipped":
            eng = "_not translated (source quarantined)_"
        elif not u["text_english"]:
            eng = f"_{status or 'pending'}_"
        else:
            eng = (u["text_english"] or "").replace("|", "/").strip()
        flag = " ⚠️" if u["quarantined"] else ""
        out.append(f"| {u['seq']}{flag} | {ts} | {u['avg_logprob']:.3f} | "
                   f"{vern} | {eng} |")
    out.append("")

    good = [u for u in utts if not u["quarantined"] and u["text_english"]]
    if good:
        out += ["**Continuous reading (published lines only)**", "",
                "> " + " ".join((u["text_english"] or "").strip() for u in good),
                "",
                "**Vernacular source**", "",
                "> " + " ".join((u["text_vernacular"] or "").strip() for u in good),
                ""]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("segments", nargs="+", type=int)
    ap.add_argument("--out", help="output .md path")
    args = ap.parse_args()

    Config.enable_utf8_console()
    Config.ensure_dirs()

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# Pipeline output — Paza transcription + NLLB translation",
        "",
        f"Generated {stamp} from the live database. Nothing re-run: this is what the "
        "deployed pipeline produced.",
        "",
        "Lines marked ⚠️ failed the quality checks and are withheld from reports; "
        "they are shown here so the report is honest about what it excluded.",
        "",
        "---",
        "",
    ]
    for sid in args.segments:
        lines += segment_block(sid)
        lines += ["---", ""]

    out = Path(args.out) if args.out else (
        Config.DATA_DIR / "comparisons" / f"pipeline_report_{stamp[:10]}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
