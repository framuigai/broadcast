"""Application settings. Environment-overridable, with defaults sane for this machine."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def enable_utf8_console() -> None:
    """Force UTF-8 on stdout/stderr.

    The Windows console defaults to cp1252, which cannot encode Kikuyu 'ĩ'/'ũ'.
    Without this, any CLI that prints a transcript dies with UnicodeEncodeError —
    and worse, a script that catches the error silently reports mangled text.
    UTF-8 end to end is not negotiable (CLAUDE.md §2.2).
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

ROOT = Path(__file__).resolve().parent.parent


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except ValueError:
        return default


class Config:
    # --- paths -----------------------------------------------------------
    ROOT_DIR = ROOT
    CONFIG_DIR = ROOT / "config"
    DATA_DIR = Path(os.environ.get("OSINT_DATA_DIR", ROOT / "data"))
    AUDIO_DIR = DATA_DIR / "audio"
    DERIVED_DIR = DATA_DIR / "derived"
    DB_PATH = DATA_DIR / "db" / "osint.sqlite3"
    MODELS_DIR = Path(os.environ.get("OSINT_MODELS_DIR", ROOT / "models"))
    EVAL_DIR = ROOT / "eval"
    REFERENCE_DIR = EVAL_DIR / "references"

    CHANNELS_FILE = CONFIG_DIR / "channels.json"
    LANGUAGE_PACKS_FILE = CONFIG_DIR / "language_packs.json"

    # --- capture ---------------------------------------------------------
    SEGMENT_SECONDS = _env_int("OSINT_SEGMENT_SECONDS", 300)   # 5 minutes
    SEGMENT_OVERLAP_SECONDS = 5
    # Upper bound on an on-demand recording from the Live page. The analyst picks
    # the length; this only stops a typo from pinning an ffmpeg process for days.
    # Transcription runs at ~0.2x realtime (CLAUDE.md §3), so a long clip sits in
    # the queue for about five times its own duration.
    MAX_RECORDING_SECONDS = _env_int("OSINT_MAX_RECORDING_SECONDS", 14_400)  # 4 h
    AUDIO_SAMPLE_RATE = 16_000
    AUDIO_CODEC = "libopus"
    AUDIO_BITRATE = "24k"

    # --- ASR -------------------------------------------------------------
    # Hard cap. 13.7 GB RAM total; one large model at a time is the safe budget.
    ASR_MAX_LOADED_MODELS = _env_int("OSINT_ASR_MAX_MODELS", 1)
    ASR_COMPUTE_TYPE = os.environ.get("OSINT_ASR_COMPUTE_TYPE", "int8")
    ASR_CPU_THREADS = _env_int("OSINT_ASR_THREADS", 8)
    ASR_VAD_FILTER = True
    # KEEP THIS ON. Disabling word timestamps is ~4x faster but silently corrupts
    # SEGMENT end times — measured on real Inooro FM audio, a 29-second utterance
    # came back as 0.0-0.8s. That breaks click-to-seek, the playback highlight,
    # reference-set alignment, and the duration-weighted scoring in routing
    # arbitration. The speed is not worth losing the timeline.
    ASR_WORD_TIMESTAMPS = os.environ.get("OSINT_WORD_TIMESTAMPS", "1") == "1"

    # --- translation -----------------------------------------------------
    # Text-to-text (vernacular transcript -> English). Its own pool slot, so an
    # ASR model and a translation model can be resident together (~1.6 + ~1.4 GB).
    TRANSLATION_CPU_THREADS = _env_int("OSINT_TRANSLATION_THREADS", 8)
    TRANSLATION_BATCH_SIZE = _env_int("OSINT_TRANSLATION_BATCH", 16)

    # --- background pipeline ----------------------------------------------
    # capture -> transcribe -> translate, run by one worker thread inside the web
    # app. One worker per machine: a second would double the model RAM.
    BACKGROUND_PIPELINE = os.environ.get("OSINT_BACKGROUND_PIPELINE", "1") == "1"
    PIPELINE_POLL_SECONDS = _env_int("OSINT_PIPELINE_POLL", 5)
    PIPELINE_LOCK_FILE = DATA_DIR / "pipeline.lock"

    # --- time ------------------------------------------------------------
    DISPLAY_TZ = "Africa/Nairobi"   # EAT, UTC+3. Storage is always UTC.

    # --- flask -----------------------------------------------------------
    SECRET_KEY = os.environ.get("OSINT_SECRET_KEY", "dev-only-change-in-production")
    JSON_SORT_KEYS = False
    TEMPLATES_AUTO_RELOAD = True

    enable_utf8_console = staticmethod(enable_utf8_console)

    @classmethod
    def ensure_dirs(cls) -> None:
        for d in (cls.DATA_DIR, cls.AUDIO_DIR, cls.DERIVED_DIR,
                  cls.DB_PATH.parent, cls.MODELS_DIR, cls.REFERENCE_DIR):
            d.mkdir(parents=True, exist_ok=True)
