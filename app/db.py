"""SQLite schema and helpers.

Audio lives on the filesystem; only metadata and text live here.
Language codes are ISO 639-3 everywhere (kik, luo, swh, eng).
All timestamps are UTC ISO-8601 with a trailing Z.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from app.config import Config

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- Immutable record of a captured audio segment (Plan §6.2, §10.2).
CREATE TABLE IF NOT EXISTS segments (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id        TEXT    NOT NULL,
    path              TEXT    NOT NULL UNIQUE,
    started_at_utc    TEXT    NOT NULL,
    duration_seconds  REAL    NOT NULL,
    sample_rate       INTEGER NOT NULL,
    sha256            TEXT    NOT NULL,
    source_type       TEXT    NOT NULL DEFAULT 'broadcast',  -- broadcast | user_upload
    uploaded_by       TEXT,
    capture_host      TEXT,
    software_version  TEXT,
    content_class     TEXT,      -- speech | music | advert | drama | unknown
    filter_reason     TEXT,      -- why it was dropped, if it was
    priority          TEXT    NOT NULL DEFAULT 'medium',
    status            TEXT    NOT NULL DEFAULT 'captured',
        -- captured | filtered | queued | transcribing | translating | transcribed | failed
    created_at_utc    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX IF NOT EXISTS ix_segments_channel_time ON segments(channel_id, started_at_utc DESC);
CREATE INDEX IF NOT EXISTS ix_segments_status       ON segments(status, priority);

-- One row per routed utterance within a segment.
CREATE TABLE IF NOT EXISTS utterances (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id        INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    seq               INTEGER NOT NULL,
    start_seconds     REAL    NOT NULL,   -- offset within the segment
    end_seconds       REAL    NOT NULL,
    routed_language   TEXT    NOT NULL,   -- ISO 639-3 pack code
    routing_confidence REAL,
    routing_method    TEXT,               -- prior | token_score | arbitration | manual
    text_vernacular   TEXT,
    text_english      TEXT,
    avg_logprob       REAL,
    no_speech_prob    REAL,
    repetition_ratio  REAL,
    quarantined       INTEGER NOT NULL DEFAULT 0,
    quarantine_reason TEXT,
    model_ref         TEXT,
    -- Translation (vernacular text -> English text; never from audio).
    translation_status TEXT,
        -- pending | done | flagged | corrected | failed | skipped | not_required | unavailable
    translation_model_ref TEXT,
    translation_score  REAL,              -- model log-prob score; for calibration only
    translation_flag_reason TEXT,         -- why the English is withheld, if it is
    translated_at_utc  TEXT,
    created_at_utc    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX IF NOT EXISTS ix_utt_segment ON utterances(segment_id, seq);
CREATE INDEX IF NOT EXISTS ix_utt_lang    ON utterances(routed_language);

-- Full-text search over both languages at once (Plan §10.6).
CREATE VIRTUAL TABLE IF NOT EXISTS utterances_fts USING fts5(
    text_vernacular, text_english, content='utterances', content_rowid='id',
    tokenize='unicode61 remove_diacritics 0'
);
-- remove_diacritics 0 is deliberate: Kikuyu ĩ/ũ are meaningful, not decoration.

CREATE TRIGGER IF NOT EXISTS utt_ai AFTER INSERT ON utterances BEGIN
    INSERT INTO utterances_fts(rowid, text_vernacular, text_english)
    VALUES (new.id, new.text_vernacular, new.text_english);
END;
CREATE TRIGGER IF NOT EXISTS utt_ad AFTER DELETE ON utterances BEGIN
    INSERT INTO utterances_fts(utterances_fts, rowid, text_vernacular, text_english)
    VALUES ('delete', old.id, old.text_vernacular, old.text_english);
END;
CREATE TRIGGER IF NOT EXISTS utt_au AFTER UPDATE ON utterances BEGIN
    INSERT INTO utterances_fts(utterances_fts, rowid, text_vernacular, text_english)
    VALUES ('delete', old.id, old.text_vernacular, old.text_english);
    INSERT INTO utterances_fts(rowid, text_vernacular, text_english)
    VALUES (new.id, new.text_vernacular, new.text_english);
END;

-- Analyst corrections. This is the training flywheel (Plan §10.4) —
-- every row is a perfectly in-domain, perfectly labelled training pair.
CREATE TABLE IF NOT EXISTS corrections (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    utterance_id     INTEGER NOT NULL REFERENCES utterances(id) ON DELETE CASCADE,
    field            TEXT    NOT NULL,   -- text_vernacular | text_english | routed_language
    original_value   TEXT,
    corrected_value  TEXT    NOT NULL,
    corrected_by     TEXT    NOT NULL DEFAULT 'analyst',
    note             TEXT,
    created_at_utc   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX IF NOT EXISTS ix_corr_utt ON corrections(utterance_id);

-- Capture interruptions are recorded explicitly, never silently absorbed (Plan §6.2).
CREATE TABLE IF NOT EXISTS capture_gaps (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id     TEXT NOT NULL,
    started_at_utc TEXT NOT NULL,
    ended_at_utc   TEXT,
    reason         TEXT
);

-- Self-populating advertisement/jingle fingerprint library (Plan §7.3).
-- Doubles as an advertisement airing log.
CREATE TABLE IF NOT EXISTS fingerprints (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint    TEXT NOT NULL UNIQUE,
    label          TEXT,
    kind           TEXT,            -- advert | jingle | ident | promo | unknown
    confirmed      INTEGER NOT NULL DEFAULT 0,
    first_seen_utc TEXT NOT NULL,
    hit_count      INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS fingerprint_hits (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint_id INTEGER NOT NULL REFERENCES fingerprints(id) ON DELETE CASCADE,
    channel_id     TEXT NOT NULL,
    aired_at_utc   TEXT NOT NULL
);

-- Append-only audit log — chain of custody (Plan §10.2).
CREATE TABLE IF NOT EXISTS audit_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    actor          TEXT NOT NULL,
    action         TEXT NOT NULL,
    object_type    TEXT,
    object_id      TEXT,
    detail         TEXT,
    created_at_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

-- Every evaluation run, so numbers are comparable over time (Plan §9).
CREATE TABLE IF NOT EXISTS eval_runs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    pack             TEXT NOT NULL,
    model_ref        TEXT NOT NULL,
    reference_set    TEXT NOT NULL,
    channel_id       TEXT,
    daypart          TEXT,
    wer              REAL,
    cer              REAL,
    n_utterances     INTEGER,
    n_words          INTEGER,
    normalisation    TEXT,   -- the exact rule set used; without it WER is meaningless
    realtime_factor  REAL,
    notes            TEXT,
    created_at_utc   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX IF NOT EXISTS ix_eval_pack ON eval_runs(pack, created_at_utc DESC);
"""


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or Config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def session(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# Columns added after the first release. CREATE TABLE IF NOT EXISTS never alters
# an existing table, so each addition is applied here, idempotently.
MIGRATIONS: list[tuple[str, str, str]] = [
    # 2026-09-29: translation stage (capture -> transcribe -> translate).
    ("utterances", "translation_status", "TEXT"),
    ("utterances", "translation_model_ref", "TEXT"),
    ("utterances", "translation_score", "REAL"),
    ("utterances", "translation_flag_reason", "TEXT"),
    ("utterances", "translated_at_utc", "TEXT"),
]


def _migrate(conn: sqlite3.Connection) -> None:
    for table, column, decl in MIGRATIONS:
        have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in have:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_utt_translation"
                 " ON utterances(translation_status)")


def init_db(db_path: Path | None = None) -> None:
    with session(db_path) as conn:
        # Migrate first: SCHEMA's CREATE TABLE is a no-op on an existing table.
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table'"
                              " AND name='utterances'").fetchone()
        if exists:
            _migrate(conn)
        conn.executescript(SCHEMA)
        _migrate(conn)


def query(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with session() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def query_one(sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: tuple[Any, ...] = ()) -> int:
    with session() as conn:
        cur = conn.execute(sql, params)
        return cur.lastrowid or cur.rowcount


def audit(actor: str, action: str, object_type: str = "", object_id: str = "",
          detail: str = "") -> None:
    """Append-only. Never update or delete rows in audit_log."""
    execute(
        "INSERT INTO audit_log (actor, action, object_type, object_id, detail)"
        " VALUES (?,?,?,?,?)",
        (actor, action, object_type, str(object_id), detail),
    )
