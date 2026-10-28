"""Translation: vernacular transcript text -> English text.

The pipeline is  capture audio -> transcribe (ASR) -> translate (this module).
Translation never sees audio. It works on the transcript, so the English is
always a translation of the vernacular an analyst can see and correct — and a
corrected transcript can simply be re-translated.

Which model translates which language is registry data: a pack's
`translation.model_ref` names an entry under `models`, and its
`translation.source_token` is the model's language token (e.g. NLLB's
`kik_Latn`). Nothing here knows which languages exist.

CTranslate2 int8 on CPU, like ASR. The tokenizer is loaded with the `tokenizers`
library straight from tokenizer.json, so `transformers` stays a conversion-only
dependency (CLAUDE.md §7).
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Config
from app.services import registry

log = logging.getLogger(__name__)

# One translation model resident at a time, in its own slot beside the ASR pool.
_pool: dict[str, tuple[Any, Any]] = {}
_pool_lock = threading.Lock()


@dataclass
class Translation:
    text: str
    score: float | None          # CTranslate2 hypothesis score (log-prob)
    model_ref: str


def _model_dir(model_ref: str) -> Path:
    spec = registry.model_spec(model_ref)
    path = Config.ROOT_DIR / spec["local_dir"]
    if not (path / "model.bin").exists():
        raise FileNotFoundError(
            f"Translation model {model_ref!r} is not present at {path}.\n"
            f"Run:  python scripts/fetch_model.py --model {model_ref}")
    if not (path / "tokenizer.json").exists():
        raise FileNotFoundError(f"{path} has no tokenizer.json — re-run fetch_model.py")
    return path


def load(model_ref: str) -> tuple[Any, Any]:
    """Return (translator, tokenizer), loading and evicting as needed."""
    import ctranslate2                    # imported late: heavy
    from tokenizers import Tokenizer

    with _pool_lock:
        if model_ref in _pool:
            return _pool[model_ref]
        if _pool:
            log.info("Evicting translation model(s) %s", ", ".join(_pool))
            _pool.clear()

        spec = registry.model_spec(model_ref)
        path = _model_dir(model_ref)
        log.info("Loading translation model %s from %s", model_ref, path)
        t0 = time.perf_counter()
        translator = ctranslate2.Translator(
            str(path), device="cpu",
            compute_type=spec.get("compute_type", "int8"),
            intra_threads=Config.TRANSLATION_CPU_THREADS)
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        log.info("Loaded %s in %.1fs", model_ref, time.perf_counter() - t0)
        _pool[model_ref] = (translator, tokenizer)
        return _pool[model_ref]


def loaded_models() -> list[str]:
    with _pool_lock:
        return list(_pool)


def is_available(model_ref: str) -> bool:
    try:
        _model_dir(model_ref)
        return True
    except (FileNotFoundError, KeyError):
        return False


# Splitting is an OVERFLOW measure, not the default strategy.
#
# The intuition that NLLB — a sentence-level model — would do better on one
# sentence at a time was MEASURED AND REJECTED on 2026-10-07: on 8 real Kikuyu
# and Dholuo lines, splitting every line into sentences took flagged output from
# 1/8 to 3/8. Shorter fragments give the model less to anchor on, so on a
# language it cannot parse it falls back to its strongest prior — scripture —
# sooner. A line that translated coherently whole gained a Bible verse once cut.
#
# What splitting IS needed for: input longer than the model's token limit was
# silently truncated (`toks[:max_tokens]`), dropping the tail and then tripping
# the "translation much shorter than source" check on an otherwise sound line.
# So a line is passed through whole unless it would overflow, and only then is
# it broken up — losing some context, but never losing words.
_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")


def split_sentences(text: str, max_chars: int = 240) -> list[str]:
    """Break an over-long transcript line into pieces that fit the model.

    Splits on sentence punctuation first. A piece still over `max_chars` — long
    unpunctuated speech is common in broadcast vernacular — is split again at
    commas, and failing that at whitespace, so nothing is ever silently dropped.

    Callers should only reach for this when the whole line does not fit; see
    `translate`, and the measurement note above.
    """
    text = (text or "").strip()
    if not text:
        return []

    pieces: list[str] = []
    for part in _SENTENCE_END.split(text):
        part = part.strip()
        if not part:
            continue
        if len(part) <= max_chars:
            pieces.append(part)
            continue
        pieces.extend(_split_long(part, max_chars))
    return pieces or [text]


def _split_long(part: str, max_chars: int) -> list[str]:
    """Second-chance split for a run-on clause: commas, then whitespace."""
    out: list[str] = []
    buf = ""
    for chunk in re.split(r"(?<=,)\s+", part):
        candidate = f"{buf} {chunk}".strip() if buf else chunk
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        if buf:
            out.append(buf)
        # A single comma-free clause longer than the limit: fall back to words.
        if len(chunk) > max_chars:
            words, buf = chunk.split(), ""
            for w in words:
                candidate = f"{buf} {w}".strip() if buf else w
                if len(candidate) > max_chars and buf:
                    out.append(buf)
                    buf = w
                else:
                    buf = candidate
        else:
            buf = chunk
    if buf:
        out.append(buf)
    return out


def translate(texts: list[str], pack_code: str) -> list[Translation]:
    """Translate a batch of one pack's transcript lines into the target language.

    Raises KeyError if the pack has no translation model assigned, and
    FileNotFoundError if the model is assigned but not downloaded yet.
    """
    pack = registry.pack_for(pack_code)
    model_ref = pack.translation_model_ref
    if not model_ref:
        raise KeyError(f"Pack {pack.code!r} has no translation.model_ref")
    src_token = pack.translation.get("source_token")
    if not src_token:
        raise KeyError(f"Pack {pack.code!r} has no translation.source_token")
    tgt_token = registry.translation_target_token()

    spec = registry.model_spec(model_ref)
    max_tokens = int(spec.get("max_input_tokens", 400))
    translator, tokenizer = load(model_ref)

    # A line is sent whole whenever it fits — splitting measurably worsens output
    # (see the note above `split_sentences`). Only an overflowing line is broken
    # up, so no words are dropped. `spans` records which pieces belong to which
    # line so the English can be reassembled in order.
    max_chars = int(spec.get("max_input_chars", 240))

    def encode(piece: str) -> list[str]:
        # NLLB input: [src_lang] tokens </s>. Built by hand because
        # tokenizer.json's own template has no source language set and appends a
        # stray <unk>.
        toks = tokenizer.encode(piece, add_special_tokens=False).tokens
        return [src_token] + toks[:max_tokens] + ["</s>"]

    sources: list[list[str]] = []
    spans: list[tuple[int, int]] = []
    n_split = 0
    for text in texts:
        start = len(sources)
        text = (text or "").strip()
        if text:
            whole = tokenizer.encode(text, add_special_tokens=False).tokens
            if len(whole) <= max_tokens:
                sources.append([src_token] + whole + ["</s>"])
            else:
                n_split += 1
                for piece in split_sentences(text, max_chars):
                    sources.append(encode(piece))
        spans.append((start, len(sources)))

    if not sources:
        return [Translation(text="", score=None, model_ref=model_ref) for _ in texts]

    t0 = time.perf_counter()
    results = translator.translate_batch(
        sources,
        target_prefix=[[tgt_token]] * len(sources),
        beam_size=int(spec.get("beam_size", 4)),
        max_batch_size=Config.TRANSLATION_BATCH_SIZE,
        max_input_length=max_tokens + 2,       # + language token and </s>
        max_decoding_length=max_tokens,
        # CTranslate2's hypothesis score (log-prob, length-penalised). Stored for
        # calibration only — never trusted on its own to pass a translation.
        return_scores=True,
        # Same failure mode as Whisper: loops. Blocking repeated 3-grams is cheap.
        no_repeat_ngram_size=3,
    )

    decoded: list[tuple[str, float | None]] = []
    for res in results:
        hyp = res.hypotheses[0]
        if hyp and hyp[0] == tgt_token:
            hyp = hyp[1:]
        ids = [i for i in (tokenizer.token_to_id(t) for t in hyp) if i is not None]
        decoded.append((tokenizer.decode(ids, skip_special_tokens=True).strip(),
                        res.scores[0] if res.scores else None))

    # Reassemble: one English line per input line, pieces rejoined in order. The
    # stored score is the mean over the pieces, so it stays comparable with the
    # single-piece scores already in the database.
    out: list[Translation] = []
    for start, end in spans:
        parts = decoded[start:end]
        text = " ".join(t for t, _ in parts if t).strip()
        scores = [sc for _, sc in parts if sc is not None]
        out.append(Translation(text=text,
                               score=sum(scores) / len(scores) if scores else None,
                               model_ref=model_ref))

    log.info("Translated %d line(s) (%d split for length -> %d piece(s)) %s->%s in %.1fs",
             len(texts), n_split, len(sources), src_token, tgt_token,
             time.perf_counter() - t0)
    return out
