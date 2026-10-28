"""Transcription: model pool plus the per-pack decode call.

CTranslate2 int8 on CPU, never `transformers` — several times faster on this
machine, which has no CUDA device.

The model pool is the only global state in the application. It is bounded:
this machine has 13.7 GB of RAM and one Whisper-turbo model occupies roughly
1.6 GB, so loading three at once alongside the rest of the pipeline is not safe.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from app.config import Config
from app.services import registry

log = logging.getLogger(__name__)

_pool: "OrderedDict[str, Any]" = OrderedDict()
_pool_lock = threading.Lock()



# --------------------------------------------------------------------------- #
#  Non-standard language tokens
# --------------------------------------------------------------------------- #
_registered_tokens: set[str] = set()


def _register_extra_language_tokens(model_ref: str) -> None:
    """Teach faster-whisper about language tokens Whisper never shipped with.

    Paza adds <|kik|>, <|luo|>, <|kln|>, <|mas|> and <|som|> to the tokenizer, but
    faster-whisper validates the `language` argument against a hardcoded list of
    Whisper's original 99 languages and raises ValueError for anything else. The
    tokens genuinely exist in the checkpoint — this is a client-side allow-list
    problem, not a model limitation.

    The extra tokens come from the registry (`extra_language_tokens`), so adding a
    language is still a config change. This is the one place in the codebase that
    touches Whisper's own naming, which is where CLAUDE.md §5 says it belongs.
    """
    spec = registry.model_spec(model_ref)
    extra = spec.get("extra_language_tokens") or []
    if not extra:
        return

    new = [t for t in extra if t not in _registered_tokens]
    if not new:
        return

    from faster_whisper import tokenizer as fw_tokenizer

    codes = getattr(fw_tokenizer, "_LANGUAGE_CODES", None)
    if codes is None:
        log.warning("faster-whisper has no _LANGUAGE_CODES to extend; "
                    "language validation may reject %s", new)
        return

    # The attribute is a tuple in some builds and a set in others.
    if isinstance(codes, (set, frozenset)):
        fw_tokenizer._LANGUAGE_CODES = set(codes) | set(extra)
    else:
        fw_tokenizer._LANGUAGE_CODES = tuple(codes) + tuple(
            t for t in extra if t not in codes)

    _registered_tokens.update(extra)
    log.info("Registered non-standard language tokens for %s: %s",
             model_ref, ", ".join(new))


# --------------------------------------------------------------------------- #
#  Model pool
# --------------------------------------------------------------------------- #
def _model_dir(model_ref: str) -> Path:
    spec = registry.model_spec(model_ref)
    path = Config.ROOT_DIR / spec["local_dir"]
    if not path.exists():
        raise FileNotFoundError(
            f"Model {model_ref!r} is not present at {path}.\n"
            f"Run:  python scripts/download_models.py --model {model_ref}"
        )
    return path


def load(model_ref: str):
    """Get a model from the pool, loading and evicting as needed (LRU)."""
    from faster_whisper import WhisperModel   # imported late: heavy

    with _pool_lock:
        if model_ref in _pool:
            _pool.move_to_end(model_ref)
            return _pool[model_ref]

        while len(_pool) >= Config.ASR_MAX_LOADED_MODELS:
            evicted, _ = _pool.popitem(last=False)
            log.info("Evicting model %s to stay within the RAM budget", evicted)

        spec = registry.model_spec(model_ref)
        _register_extra_language_tokens(model_ref)
        path = _model_dir(model_ref)
        log.info("Loading %s from %s (%s)", model_ref, path, Config.ASR_COMPUTE_TYPE)
        t0 = time.perf_counter()
        model = WhisperModel(
            str(path),
            device="cpu",
            compute_type=spec.get("compute_type", Config.ASR_COMPUTE_TYPE),
            cpu_threads=Config.ASR_CPU_THREADS,
        )
        log.info("Loaded %s in %.1fs", model_ref, time.perf_counter() - t0)
        _pool[model_ref] = model
        return model


def unload_all() -> None:
    with _pool_lock:
        _pool.clear()


def loaded_models() -> list[str]:
    with _pool_lock:
        return list(_pool.keys())


# --------------------------------------------------------------------------- #
#  Transcription
# --------------------------------------------------------------------------- #
@dataclass
class Utterance:
    seq: int
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    language: str
    model_ref: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def transcribe(audio_path: str | Path, pack_code: str,
               *, beam_size: int | None = None,
               initial_prompt: str | None = None,
               use_pack_prompt: bool = True) -> tuple[list[Utterance], dict[str, Any]]:
    """Transcribe a file with a specific Language Pack.

    Language is never auto-detected here. The caller decides, using the routing
    chain in `routing.py`, which is prior-weighted and candidate-restricted.
    Blind detection across Whisper's full language inventory is exactly what
    Plan §8 sets out to avoid.
    """
    pack = registry.pack_for(pack_code)
    model = load(pack.model_ref)
    _register_extra_language_tokens(pack.model_ref)

    # Whisper conditions its decoding on this text: it is how the model is told
    # what register, orthography and vocabulary to expect, and it is the largest
    # quality lever available without training. The text is registry data
    # (`pack.asr.initial_prompt`), so priming a new language stays a config
    # change. An explicit argument still wins, and routing opts out entirely.
    if initial_prompt is None and use_pack_prompt:
        initial_prompt = pack.asr.get("initial_prompt") or None

    t0 = time.perf_counter()
    segments, info = model.transcribe(
        str(audio_path),
        language=pack.whisper_token,          # ISO 639-3 -> Whisper token, here only
        beam_size=beam_size or pack.asr.get("beam_size", 5),
        vad_filter=Config.ASR_VAD_FILTER,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,     # limits runaway repetition hallucination
        initial_prompt=initial_prompt,
        # Segment-level timestamps drive click-to-seek; word-level ones cost ~2.2x
        # and are only needed for word-by-word playback highlighting. Registry can
        # override per pack.
        word_timestamps=pack.asr.get("word_timestamps", Config.ASR_WORD_TIMESTAMPS),
    )

    utterances: list[Utterance] = []
    for i, s in enumerate(segments):          # generator: work happens here
        utterances.append(Utterance(
            seq=i,
            start=round(s.start, 3),
            end=round(s.end, 3),
            text=(s.text or "").strip(),
            avg_logprob=getattr(s, "avg_logprob", 0.0),
            no_speech_prob=getattr(s, "no_speech_prob", 0.0),
            compression_ratio=getattr(s, "compression_ratio", 0.0),
            language=pack.code,
            model_ref=pack.model_ref,
        ))

    elapsed = time.perf_counter() - t0
    audio_seconds = getattr(info, "duration", 0.0) or 0.0
    meta = {
        "pack": pack.code,
        "model_ref": pack.model_ref,
        "audio_seconds": round(audio_seconds, 2),
        "prompted": bool(initial_prompt),
        "wall_seconds": round(elapsed, 2),
        # <1.0 means slower than realtime. Expect 0.5-1.0 on this CPU.
        "realtime_factor": round(audio_seconds / elapsed, 3) if elapsed > 0 else 0.0,
        "n_utterances": len(utterances),
    }
    log.info("Transcribed %s as %s: %d utterances, RTF %.2f",
             Path(audio_path).name, pack.code, len(utterances), meta["realtime_factor"])
    return utterances, meta


def transcribe_stream(audio_path: str | Path, pack_code: str) -> Iterator[Utterance]:
    """Yield utterances as they are decoded, for live UI updates."""
    utterances, _ = transcribe(audio_path, pack_code)
    yield from utterances


# --------------------------------------------------------------------------- #
#  Baseline harness
# --------------------------------------------------------------------------- #
def baseline_compare(audio_path: str | Path, pack_codes: list[str]) -> dict[str, Any]:
    """Run several packs over the same audio — the Step 1 bake-off (Plan §12.1 task 1.8).

    Note this loads each model in turn and evicts the previous one, so it is slow
    by design rather than by accident.
    """
    results: dict[str, Any] = {}
    for code in pack_codes:
        try:
            utts, meta = transcribe(audio_path, code)
            results[code] = {
                "meta": meta,
                "text": " ".join(u.text for u in utts),
                "utterances": [u.__dict__ for u in utts],
            }
        except (FileNotFoundError, KeyError, RuntimeError) as exc:
            log.error("Baseline run failed for pack %s: %s", code, exc)
            results[code] = {"error": str(exc)}
    return results
