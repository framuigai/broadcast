"""The processing pipeline:  capture audio -> transcribe -> translate.

    capture      writes an immutable source file and a `segments` row (status
                 'captured'). Nothing else happens at capture time.
    transcribe   route to a language pack, ASR in that pack's language,
                 quarantine checks, store utterances.
    translate    vernacular transcript TEXT -> English text, per utterance, with
                 its own quality checks. Never from audio.

One code path, two drivers:
  * `BackgroundWorker` — a thread inside the web app, started by run.py, so
    anything captured (from the UI or by capture_worker.py) is transcribed and
    translated without anyone touching a terminal.
  * scripts/transcribe_worker.py — the same functions from the command line.

Only one of them may run per machine at a time: each loads ~3 GB of models, and
this machine has 13.7 GB. A process-level file lock enforces that.

The queue lives in SQLite, not in memory, so a restart (including the Flask
reloader restarting on a code edit) loses nothing — work in flight is requeued.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, IO

from app import db
from app.config import Config
from app.services import asr, quarantine, registry, routing, translation

log = logging.getLogger(__name__)

PRIORITY_ORDER = "CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END"
# After "model not downloaded yet". The check is only a file stat, so keep it
# short: a model that finishes downloading is picked up within seconds.
TRANSLATION_RETRY_SECONDS = 30


class TranslationUnavailable(RuntimeError):
    """A translation model is assigned in the registry but not present on disk."""


def utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_path(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else Config.ROOT_DIR / p


# --------------------------------------------------------------------------- #
#  One pipeline per machine
# --------------------------------------------------------------------------- #
def acquire_process_lock() -> IO[str] | None:
    """Take the machine-wide pipeline lock, or return None if another process has it.

    An OS-level lock rather than a PID file: it is released automatically when
    the holder dies, so a crash or a reloader restart never leaves it stale.
    Keep the returned handle open for as long as the lock is needed.
    """
    path = Config.PIPELINE_LOCK_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+", encoding="utf-8")
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


# --------------------------------------------------------------------------- #
#  Queue
# --------------------------------------------------------------------------- #
def queue_depth() -> int:
    row = db.query_one("SELECT COUNT(*) AS n FROM segments"
                       " WHERE status IN ('captured','queued')") or {}
    return row.get("n", 0)


def translation_backlog() -> int:
    row = db.query_one("SELECT COUNT(*) AS n FROM utterances"
                       " WHERE translation_status='pending'") or {}
    return row.get("n", 0)


def claim_next_segment() -> dict[str, Any] | None:
    """Atomically take the highest-priority waiting segment.

    Newest first within a priority: a clip an analyst just recorded is 'high'
    and wanted now; stale continuous capture can wait.
    """
    with db.session() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            f"SELECT * FROM segments WHERE status IN ('captured','queued')"
            f" ORDER BY {PRIORITY_ORDER}, started_at_utc DESC LIMIT 1").fetchone()
        if not row:
            return None
        conn.execute("UPDATE segments SET status='transcribing' WHERE id=?", (row["id"],))
        seg = dict(row)
        seg["status"] = "transcribing"
        return seg


def requeue_stale() -> int:
    """Recover work abandoned by a killed process. Call only while holding the lock.

    With the lock held no other pipeline is running, so anything marked in flight
    is orphaned. A segment that died mid-translation keeps its transcript; its
    untranslated lines are still 'pending' and the backlog pass picks them up.
    """
    n = 0
    with db.session() as conn:
        n += conn.execute("UPDATE segments SET status='queued'"
                          " WHERE status='transcribing'").rowcount
        n += conn.execute("UPDATE segments SET status='transcribed'"
                          " WHERE status='translating'").rowcount
    if n:
        log.info("Recovered %d segment(s) abandoned mid-pipeline", n)
    return n


def initial_translation_status(pack: registry.LanguagePack, quarantined: bool) -> str:
    if not pack.needs_translation:
        return "not_required"
    if not pack.translation_model_ref:
        return "unavailable"
    # Withheld transcripts are not translated: translating a hallucination only
    # produces a more convincing hallucination. Released later -> 'pending'.
    return "skipped" if quarantined else "pending"


def reconcile_translation_states() -> int:
    """Bring stored translation states in line with the current registry.

    Covers utterances transcribed before translation existed (NULL), and packs
    whose translation model was assigned after their audio was transcribed —
    enabling a language in config then back-fills its existing transcripts.
    """
    changed = 0
    for pack in registry.all_packs():
        rows = db.query(
            "SELECT id, quarantined, translation_status FROM utterances"
            " WHERE routed_language=? AND (translation_status IS NULL"
            "   OR translation_status IN ('unavailable','not_required'))",
            (pack.code,))
        for r in rows:
            want = initial_translation_status(pack, bool(r["quarantined"]))
            if want != r["translation_status"]:
                db.execute("UPDATE utterances SET translation_status=? WHERE id=?",
                           (want, r["id"]))
                changed += 1
    if changed:
        log.info("Reconciled translation state on %d utterance(s)", changed)
    return changed


# --------------------------------------------------------------------------- #
#  Stage 2: transcribe
# --------------------------------------------------------------------------- #
def transcribe_segment(seg: dict[str, Any], forced_pack: str | None = None,
                       actor: str = "pipeline") -> dict[str, Any]:
    """Route, transcribe, quarantine-check and store one segment.

    The caller must already have marked it 'transcribing' (claim_next_segment
    does). Raises on failure, having marked the segment 'failed' with a reason.
    """
    path = resolve_path(seg["path"])
    if not path.exists():
        _fail(seg["id"], "audio file missing")
        raise FileNotFoundError(path)

    try:
        if forced_pack:
            decision = routing.manual_override(forced_pack, actor=actor)
        else:
            decision = routing.route(path, seg["channel_id"])
        utts, meta = asr.transcribe(path, decision.language)
    except Exception as exc:
        _fail(seg["id"], str(exc))
        raise

    pack = registry.pack_for(decision.language)
    n_quarantined = 0
    with db.session() as conn:
        conn.execute("DELETE FROM utterances WHERE segment_id=?", (seg["id"],))
        for u in utts:
            verdict = quarantine.check(u, decision.language)
            n_quarantined += int(verdict.quarantined)
            conn.execute(
                "INSERT INTO utterances (segment_id, seq, start_seconds, end_seconds,"
                " routed_language, routing_confidence, routing_method, text_vernacular,"
                " avg_logprob, no_speech_prob, repetition_ratio, quarantined,"
                " quarantine_reason, model_ref, translation_status)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (seg["id"], u.seq, u.start, u.end, decision.language,
                 decision.confidence, decision.method, u.text,
                 u.avg_logprob, u.no_speech_prob, verdict.repetition_ratio,
                 int(verdict.quarantined), verdict.reason_text or None, u.model_ref,
                 initial_translation_status(pack, verdict.quarantined)))

    db.audit(actor, "transcribe", "segment", str(seg["id"]),
             f"{decision.language} via {decision.method}; {len(utts)} utterances;"
             f" {n_quarantined} quarantined")
    summary = {
        "segment_id": seg["id"], "language": decision.language,
        "method": decision.method, "confidence": decision.confidence,
        "utterances": len(utts), "quarantined": n_quarantined,
        "realtime_factor": meta.get("realtime_factor"),
    }
    log.info("Segment %s -> %s (%s, conf %.2f): %d utterances, %d quarantined, RTF %.2f",
             seg["id"], decision.language, decision.method, decision.confidence,
             len(utts), n_quarantined, meta.get("realtime_factor") or 0.0)
    return summary


def _fail(segment_id: int, reason: str) -> None:
    db.execute("UPDATE segments SET status='failed', filter_reason=? WHERE id=?",
               (reason[:500], segment_id))


# --------------------------------------------------------------------------- #
#  Stage 3: translate
# --------------------------------------------------------------------------- #
def translate_pending(segment_id: int | None = None, limit: int = 64) -> int:
    """Translate utterances whose translation_status is 'pending'.

    Returns how many were processed. Raises TranslationUnavailable if a model is
    assigned but not downloaded — those lines stay 'pending' and are picked up
    automatically once the model arrives.
    """
    where = "u.translation_status='pending' AND s.status NOT IN ('transcribing')"
    params: list[Any] = []
    if segment_id is not None:
        where += " AND u.segment_id=?"
        params.append(segment_id)
    rows = db.query(
        "SELECT u.id, u.routed_language, u.text_vernacular FROM utterances u"
        " JOIN segments s ON s.id=u.segment_id"
        f" WHERE {where} ORDER BY u.segment_id DESC, u.seq LIMIT ?",
        tuple(params) + (limit,))
    if not rows:
        return 0

    by_lang: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_lang.setdefault(r["routed_language"], []).append(r)

    done = 0
    for code, items in by_lang.items():
        pack = registry.pack_for(code)
        if not pack.translation_model_ref:
            # Config changed since these were queued.
            for r in items:
                db.execute("UPDATE utterances SET translation_status='unavailable'"
                           " WHERE id=?", (r["id"],))
            continue
        if not translation.is_available(pack.translation_model_ref):
            raise TranslationUnavailable(
                f"{pack.translation_model_ref} is assigned to {code} but not downloaded."
                f" Run: python scripts/fetch_model.py --model {pack.translation_model_ref}")

        try:
            results = translation.translate(
                [(r["text_vernacular"] or "") for r in items], code)
        except Exception as exc:          # noqa: BLE001 — recorded per line, never lost
            log.exception("Translation failed for %d %s line(s)", len(items), code)
            for r in items:
                db.execute(
                    "UPDATE utterances SET translation_status='failed',"
                    " translation_flag_reason=? WHERE id=?", (str(exc)[:300], r["id"]))
            continue

        now = utc_iso()
        for r, t in zip(items, results):
            verdict = quarantine.check_translation(r["text_vernacular"] or "", t.text, code)
            db.execute(
                "UPDATE utterances SET text_english=?, translation_status=?,"
                " translation_model_ref=?, translation_score=?,"
                " translation_flag_reason=?, translated_at_utc=? WHERE id=?",
                (t.text, "flagged" if verdict.quarantined else "done", t.model_ref,
                 t.score, verdict.reason_text or None, now, r["id"]))
            done += 1
    return done


# --------------------------------------------------------------------------- #
#  One segment, end to end
# --------------------------------------------------------------------------- #
def process_segment(seg: dict[str, Any], forced_pack: str | None = None,
                    actor: str = "pipeline",
                    on_stage: Callable[[str], None] | None = None) -> bool:
    """transcribe -> translate for one claimed segment. Returns True on success.

    A missing translation model does not fail the segment: the transcript is
    stored and the English follows once the model is present.
    """
    stage = on_stage or (lambda _s: None)
    stage("transcribing")
    try:
        transcribe_segment(seg, forced_pack, actor)
    except Exception as exc:              # noqa: BLE001 — segment already marked failed
        log.warning("Segment %s failed transcription: %s", seg["id"], exc)
        return False

    stage("translating")
    db.execute("UPDATE segments SET status='translating' WHERE id=?", (seg["id"],))
    try:
        while translate_pending(segment_id=seg["id"]):
            pass
    except TranslationUnavailable as exc:
        log.warning("Segment %s: transcript stored, translation deferred — %s",
                    seg["id"], exc)
    finally:
        db.execute("UPDATE segments SET status='transcribed' WHERE id=?", (seg["id"],))
    return True


# --------------------------------------------------------------------------- #
#  Background worker (runs inside the web app)
# --------------------------------------------------------------------------- #
class BackgroundWorker:
    """Single consumer of the segment queue. One per machine."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._lock_handle: IO[str] | None = None
        self._translate_retry_at = 0.0
        self._status: dict[str, Any] = {
            "state": "stopped", "segment_id": None, "channel_id": None,
            "since_utc": None, "last_error": None, "translation_note": None,
            "processed": 0, "failed": 0, "translated": 0, "started_utc": None,
        }

    # --- control -------------------------------------------------------
    def start(self) -> bool:
        if self._thread and self._thread.is_alive():
            return True
        self._lock_handle = acquire_process_lock()
        if not self._lock_handle:
            self._set(state="locked_elsewhere",
                      last_error="Another process holds the pipeline lock "
                                 "(is scripts/transcribe_worker.py running?)")
            log.warning("Background pipeline NOT started: another process holds %s",
                        Config.PIPELINE_LOCK_FILE)
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="pipeline-worker")
        self._set(state="starting", started_utc=utc_iso())
        self._thread.start()
        log.info("Background pipeline started (capture -> transcribe -> translate)")
        return True

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def wake(self) -> None:
        """Called after a capture so new audio is picked up immediately."""
        self._wake.set()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def _set(self, **kw: Any) -> None:
        with self._lock:
            self._status.update(kw)

    def _bump(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._status[key] = (self._status.get(key) or 0) + n

    # --- loop ----------------------------------------------------------
    def _run(self) -> None:
        try:
            requeue_stale()
            reconcile_translation_states()
        except Exception as exc:          # noqa: BLE001 — never kill the worker at boot
            log.exception("Pipeline start-up housekeeping failed")
            self._set(last_error=str(exc)[:300])

        while not self._stop.is_set():
            try:
                if self._step():
                    continue
            except Exception as exc:      # noqa: BLE001 — a background thread must survive
                log.exception("Pipeline step failed")
                self._set(last_error=str(exc)[:300])
                self._stop.wait(10)
            self._set(state="idle", segment_id=None, channel_id=None, since_utc=None)
            self._wake.wait(Config.PIPELINE_POLL_SECONDS)
            self._wake.clear()
        self._set(state="stopped")

    def _step(self) -> bool:
        """Do one unit of work. Returns False when there was nothing to do."""
        seg = claim_next_segment()
        if seg:
            self._set(segment_id=seg["id"], channel_id=seg["channel_id"],
                      since_utc=utc_iso())
            ok = process_segment(seg, on_stage=lambda s: self._set(state=s))
            self._bump("processed" if ok else "failed")
            return True

        # Queue empty: work through untranslated lines left from earlier runs.
        if time.time() < self._translate_retry_at:
            return False
        try:
            self._set(state="translating", segment_id=None)
            n = translate_pending()
        except TranslationUnavailable as exc:
            self._translate_retry_at = time.time() + TRANSLATION_RETRY_SECONDS
            self._set(translation_note=str(exc))
            return False
        self._set(translation_note=None)
        if n:
            self._bump("translated", n)
        return bool(n)


_worker: BackgroundWorker | None = None


def worker() -> BackgroundWorker:
    global _worker
    if _worker is None:
        _worker = BackgroundWorker()
    return _worker


def wake() -> None:
    if _worker is not None:
        _worker.wake()


def status() -> dict[str, Any]:
    st = worker().status() if _worker else {"state": "not_started"}
    st.update({
        "enabled": Config.BACKGROUND_PIPELINE,
        "queue_depth": queue_depth(),
        "translation_backlog": translation_backlog(),
        "loaded_asr_models": asr.loaded_models(),
        "loaded_translation_models": translation.loaded_models(),
    })
    return st
