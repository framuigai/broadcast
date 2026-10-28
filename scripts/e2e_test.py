#!/usr/bin/env python
"""End-to-end test on LIVE broadcast audio: capture -> transcribe -> translate.

Drives the running web app exactly as an analyst does — records a clip from each
station through the Live page's API, waits for the background pipeline, then
checks what came out. Requires `python run.py` to be running.

    python scripts/e2e_test.py                         # every enabled channel, 60 s clips
    python scripts/e2e_test.py --channels ramogi_fm inooro_fm --duration 30
    python scripts/e2e_test.py --segments 4 5           # verify already-captured segments

--segments skips recording and only verifies. Use it when a station is off air
(the TV channels are only live on YouTube part of the day).

Checks, per channel:
  * the clip was captured and went through the pipeline without failing
  * utterance timestamps are sane (end > start, within the clip)
  * every line reached a final translation state; every non-withheld line in a
    language that needs English has English
  * flagged English is withheld from the document API and the .txt export
  * Kikuyu diacritics (ĩ/ũ) survive to the API and the export
  * the transcript page renders, declaring UTF-8

Do not edit .py files while this runs: the dev server's reloader restarts the
process, which drops the in-memory recording jobs this test polls.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Config                       # noqa: E402
from app.services import registry                   # noqa: E402

FINAL_TX = {"done", "flagged", "skipped", "not_required", "corrected"}
DONE_JOB = {"done", "failed", "filtered"}


def http(base: str, path: str, body: dict | None = None, raw: bool = False):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(base + path, data=data, method="POST" if data else "GET",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        payload = r.read().decode("utf-8")
        return payload if raw else json.loads(payload)


class Report:
    def __init__(self, channel: str) -> None:
        self.channel = channel
        self.fails: list[str] = []
        self.warns: list[str] = []
        self.info: list[str] = []

    def check(self, ok: bool, msg: str) -> None:
        if not ok:
            self.fails.append(msg)

    @property
    def ok(self) -> bool:
        return not self.fails


def verify_segment(base: str, seg_id: int, duration: float, rep: Report) -> None:
    seg = http(base, f"/segment/{seg_id}")
    utts = seg.get("utterances", [])
    langs = sorted({u["routed_language"] for u in utts})
    rep.info.append(f"segment {seg_id}: {len(utts)} utterance(s), routed {', '.join(langs) or '-'}")
    if not utts:
        rep.warns.append("no utterances — clip was probably music/adverts/silence")
        return

    counts: dict[str, int] = {}
    for u in utts:
        st = u.get("translation_status")
        counts[st] = counts.get(st, 0) + 1

        rep.check(u["end_seconds"] > u["start_seconds"],
                  f"utt {u['id']}: end {u['end_seconds']} <= start {u['start_seconds']}")
        rep.check(u["end_seconds"] <= duration + 1.0,
                  f"utt {u['id']}: end {u['end_seconds']}s beyond {duration}s clip")
        rep.check(st in FINAL_TX, f"utt {u['id']}: translation_status={st!r} is not final")

        pack = registry.pack_for(u["routed_language"])
        if pack.needs_translation and not u["quarantined"]:
            rep.check(st in {"done", "flagged", "corrected"},
                      f"utt {u['id']} ({pack.code}): needs English but status={st!r}")
            rep.check(bool((u.get("text_english") or "").strip()),
                      f"utt {u['id']} ({pack.code}): no English text")
        if u["quarantined"]:
            rep.check(st == "skipped", f"utt {u['id']}: withheld transcript was translated ({st})")
    rep.info.append("translation states: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    # Withholding: flagged English must not appear in the document or export.
    doc = http(base, f"/document/{seg_id}")
    export = http(base, f"/document/{seg_id}/export.txt", raw=True)
    for u in utts:
        eng = (u.get("text_english") or "").strip()
        if u.get("translation_status") == "flagged" and eng:
            rep.check(eng not in doc.get("prose_english", ""),
                      f"utt {u['id']}: flagged English leaked into the document")
            # Match the EN: line itself — a copied-source flag means the same words
            # legitimately appear on the vernacular line above it.
            rep.check(f"EN: {eng}" not in export,
                      f"utt {u['id']}: flagged English leaked into the export")
            rep.check("translation withheld" in export,
                      f"utt {u['id']}: export does not mark withheld English")
        if u.get("translation_status") == "done" and eng and not u["quarantined"]:
            rep.check(eng in export, f"utt {u['id']}: checked English missing from export")
    rep.check("Translation  :" in export, "export has no Translation provenance line")

    # UTF-8: diacritics from any pack that declares them must survive to the export.
    # Only published lines count — withheld text is deliberately not exported.
    for code in langs:
        marks = registry.pack_for(code).orthography.get("diacritics", [])
        if not marks:
            continue
        text = " ".join(u["text_vernacular"] or "" for u in utts
                        if u["routed_language"] == code and not u["quarantined"])
        if not text.strip():
            rep.info.append(f"{code}: every line withheld — nothing to check in the export")
            continue
        if any(m in text for m in marks):
            exported = any(m in export for m in marks)
            rep.check(exported, f"{code} diacritics present in API but lost in export")
            rep.info.append(f"{code} diacritics {''.join(marks)}: API yes, export "
                            f"{'yes' if exported else 'NO'}")
        else:
            rep.warns.append(f"{code} lines contain no {''.join(marks)} — check the transcript")

    page = http(base.replace("/api", ""), f"/transcript/{seg_id}", raw=True)
    rep.check('charset="utf-8"' in page.lower() or "charset=utf-8" in page.lower(),
              "transcript page does not declare UTF-8")

    for u in [u for u in utts if not u["quarantined"]][:2]:
        rep.info.append(f"  [{u['routed_language']} {u['start_seconds']:.0f}s] "
                        f"{(u['text_vernacular'] or '')[:110]}")
        rep.info.append(f"      EN ({u.get('translation_status')}): "
                        f"{(u.get('text_english') or '—')[:110]}")


def print_reports(reports: dict[str, Report]) -> int:
    print("\n" + "=" * 78)
    for name, rep in reports.items():
        print(f"\n{'PASS' if rep.ok else 'FAIL'}  {name}")
        for m in rep.info:
            print(f"    {m}")
        for m in rep.warns:
            print(f"    WARN {m}")
        for m in rep.fails:
            print(f"    FAIL {m}")
    n_ok = sum(r.ok for r in reports.values())
    print(f"\n{n_ok}/{len(reports)} passed")
    return 0 if n_ok == len(reports) else 1


def verify_existing(base: str, seg_ids: list[int]) -> int:
    reports: dict[str, Report] = {}
    for sid in seg_ids:
        try:
            seg = http(base, f"/segment/{sid}")
        except urllib.error.HTTPError as exc:
            rep = Report(f"segment {sid}")
            rep.fails.append(f"HTTP {exc.code}")
            reports[rep.channel] = rep
            continue
        rep = Report(f"{seg['channel_id']} (segment {sid}, {seg['source_type']})")
        reports[rep.channel] = rep
        if seg["status"] != "transcribed":
            rep.fails.append(f"segment status is {seg['status']!r}, not 'transcribed'")
            continue
        verify_segment(base, sid, float(seg["duration_seconds"]), rep)
    return print_reports(reports)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:5000")
    ap.add_argument("--channels", nargs="*", help="channel ids (default: all enabled)")
    ap.add_argument("--duration", type=int, default=60, help="clip length in seconds")
    # One 60 s clip takes ~4 min through the pipeline, one at a time.
    ap.add_argument("--timeout", type=int, default=3600, help="overall wait, seconds")
    ap.add_argument("--segments", nargs="*", type=int,
                    help="verify these existing segment ids instead of recording")
    args = ap.parse_args()
    Config.enable_utf8_console()
    base = args.base.rstrip("/") + "/api"

    try:
        st = http(base, "/pipeline/status")
    except (urllib.error.URLError, OSError) as exc:
        print(f"App not reachable at {args.base} ({exc}). Start it: python run.py")
        return 2
    if st.get("state") in {"not_started", "locked_elsewhere", "stopped"}:
        print(f"Background pipeline is not running (state={st.get('state')}). "
              f"{st.get('last_error') or ''}")
        return 2

    if args.segments:
        return verify_existing(base, args.segments)

    channels = args.channels or [c.id for c in registry.all_channels(enabled_only=True)]
    reports = {ch: Report(ch) for ch in channels}
    jobs: dict[str, str] = {}
    for ch in channels:
        try:
            jobs[ch] = http(base, f"/record/{ch}", {"duration": args.duration,
                                                    "actor": "e2e_test"})["job_id"]
            print(f"recording {ch} ({args.duration}s)")
        except urllib.error.HTTPError as exc:
            reports[ch].fails.append(f"could not start recording: HTTP {exc.code}")

    t0 = time.time()
    finished: dict[str, dict] = {}
    last_line = ""
    while len(finished) < len(jobs) and time.time() - t0 < args.timeout:
        for ch, job_id in jobs.items():
            if ch in finished:
                continue
            try:
                job = http(base, f"/record/job/{job_id}")
            except urllib.error.HTTPError as exc:
                finished[ch] = {"state": "lost", "error": f"job vanished (HTTP {exc.code}) — "
                                "did the server restart?"}
                continue
            if job["state"] in DONE_JOB:
                job["elapsed"] = time.time() - t0
                finished[ch] = job
                print(f"  {ch}: {job['state']} after {job['elapsed']:.0f}s")
        pipe = http(base, "/pipeline/status")
        line = (f"  … pipeline {pipe['state']} seg={pipe.get('segment_id')} "
                f"queue={pipe['queue_depth']} backlog={pipe['translation_backlog']}")
        if line != last_line:
            print(line)
            last_line = line
        time.sleep(5)

    for ch in jobs:
        rep = reports[ch]
        job = finished.get(ch)
        if not job:
            rep.fails.append(f"timed out after {args.timeout}s")
            continue
        if job["state"] != "done":
            rep.fails.append(f"{job['state']}: {(job.get('error') or '')[:300]}")
            continue
        rep.info.append(f"captured + transcribed + translated in {job['elapsed']:.0f}s wall-clock")
        try:
            verify_segment(base, job["segment_id"], float(args.duration), rep)
        except (urllib.error.URLError, KeyError, ValueError) as exc:
            rep.fails.append(f"verification error: {exc}")

    return print_reports(reports)


if __name__ == "__main__":
    raise SystemExit(main())
