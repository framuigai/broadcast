"""Hallucination filtering (Plan §7.5).

Whisper-family models hallucinate fluently and confidently on music, silence and
noise. In a media-monitoring system that is not a quality issue, it is a
correctness issue: fabricated text about real, identifiable people can produce a
materially false intelligence assessment.

Quarantined utterances are never deleted. They are withheld from reports and sent
to the review queue, where a reviewer either corrects them (producing ASR training
data) or confirms them as noise (producing filter training data). Both outcomes
improve the system.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from app.services import registry

# Phrases Whisper emits on silence and music. Matched case-insensitively after
# stripping punctuation. Extend from observation rather than from guesswork.
HALLUCINATION_PHRASES = {
    "thanks for watching",
    "thank you for watching",
    "subscribe to my channel",
    "please subscribe",
    "like and subscribe",
    "see you in the next video",
    "subtitles by the amara org community",
    "amara org",
    "transcription by",
    "www mooji org",
    "sous-titrage société radio-canada",
    "copyright",
    "all rights reserved",
    "bye bye",
    "music",
    "applause",
    "foreign",
    "you",
}

_WORD = re.compile(r"\w+", re.UNICODE)
_PUNCT_STRIP = re.compile(r"[^\w\s]", re.UNICODE)


@dataclass
class Verdict:
    quarantined: bool
    reasons: list[str]
    repetition_ratio: float

    @property
    def reason_text(self) -> str:
        return "; ".join(self.reasons)


def repetition_ratio(text: str) -> float:
    """Share of tokens taken up by the single most frequent token.

    The classic Whisper failure mode is one phrase repeated many times, which
    scores near 1.0 here and near 0 for natural speech.
    """
    words = _WORD.findall(text.lower())
    if len(words) < 4:
        return 0.0
    most_common = Counter(words).most_common(1)[0][1]
    return most_common / len(words)


def ngram_loop_ratio(text: str, n: int = 4) -> float:
    """Detect repeated n-grams, which catches loops that vary the odd word and so
    slip past the unigram check."""
    words = _WORD.findall(text.lower())
    if len(words) < n * 2:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    return Counter(grams).most_common(1)[0][1] / len(grams)


def is_known_hallucination(text: str) -> bool:
    cleaned = _PUNCT_STRIP.sub(" ", text.lower())
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return True
    if cleaned in HALLUCINATION_PHRASES:
        return True
    # A very short utterance that is entirely a known phrase plus filler.
    return len(cleaned) < 40 and any(p in cleaned for p in HALLUCINATION_PHRASES
                                     if len(p) > 12)


def check(utterance: Any, pack_code: str) -> Verdict:
    """Apply the pack's thresholds to one utterance.

    `utterance` is an asr.Utterance or anything exposing text / avg_logprob /
    no_speech_prob / compression_ratio.
    """
    pack = registry.pack_for(pack_code)
    limits = pack.quarantine or {}
    reasons: list[str] = []

    text = (getattr(utterance, "text", "") or "").strip()
    avg_logprob = float(getattr(utterance, "avg_logprob", 0.0) or 0.0)
    no_speech = float(getattr(utterance, "no_speech_prob", 0.0) or 0.0)
    compression = float(getattr(utterance, "compression_ratio", 0.0) or 0.0)

    rep = max(repetition_ratio(text), ngram_loop_ratio(text))

    if not text:
        reasons.append("empty transcript")

    min_lp = limits.get("min_avg_logprob")
    if min_lp is not None and avg_logprob < min_lp:
        reasons.append(f"low confidence (avg_logprob {avg_logprob:.2f} < {min_lp})")

    max_ns = limits.get("max_no_speech_prob")
    if max_ns is not None and no_speech > max_ns:
        reasons.append(f"probably not speech (no_speech_prob {no_speech:.2f} > {max_ns})")

    max_rep = limits.get("max_repetition_ratio", 0.5)
    if rep > max_rep:
        reasons.append(f"repetition loop (ratio {rep:.2f} > {max_rep})")

    # Degenerate output: compression ratio well above natural language.
    if compression > 2.4:
        reasons.append(f"degenerate output (compression_ratio {compression:.2f})")

    if is_known_hallucination(text):
        reasons.append("matches known hallucination phrase")

    return Verdict(bool(reasons), reasons, round(rep, 3))


# Translation hallucination markers, from observation. Much of NLLB's Kikuyu
# training text is Bible translation, so on input it cannot parse it falls back to
# scripture — observed 2026-09-29 on Inooro/Kameme audio, e.g. "Do not be afraid,
# for I am the LORD your God" for a line about the Kenyan police. Broadcast news
# almost never uses this register, so it flags for review rather than discards.
BIBLICAL_REGISTER = re.compile(
    r"\bLORD\b|\b(?:unto|thee|thou|thy|thine|behold|verily|saith|hath|begat)\b"
    r"|\band he said (?:to|unto) them\b",
    re.IGNORECASE)
_LORD_CAPS = re.compile(r"\bLORD\b")


def biblical_register(output: str) -> bool:
    # "LORD" only in capitals (the KJV convention); "Lord" alone is ordinary.
    hits = BIBLICAL_REGISTER.findall(output)
    return bool(_LORD_CAPS.search(output)) or any(h.upper() != "LORD" for h in hits)


def letter_share(text: str) -> float:
    """Share of non-space characters that are letters. Digit/symbol runs that
    slip past ASR quarantine are not language, and translating them invents text."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if c.isalpha()) / len(chars)


def copy_ratio(source: str, output: str) -> float:
    """Share of output words that also appear in the source.

    Near 1.0 means the "translation" is the source passed through unchanged —
    NLLB's usual failure when it does not understand the input. Legitimately
    high for an all-English line routed as vernacular, which is why this flags
    for review rather than discarding.
    """
    out = _WORD.findall(output.lower())
    if not out:
        return 0.0
    src = set(_WORD.findall(source.lower()))
    return sum(1 for w in out if w in src) / len(out)


def check_translation(source: str, output: str, pack_code: str) -> Verdict:
    """Checks on vernacular -> English output (the §2.4 rule applied to translation).

    Translation hallucinates differently from ASR: it is fluent and confident
    when wrong, so model scores alone are not trusted here. Flagged English is
    kept for review but withheld from documents and exports.
    """
    limits = registry.translation_checks(pack_code)
    reasons: list[str] = []
    output = (output or "").strip()

    if not output:
        return Verdict(True, ["empty translation"], 0.0)

    n_src = len(_WORD.findall(source))
    n_out = len(_WORD.findall(output))
    if n_src >= limits.get("min_words_for_ratio", 4):
        ratio = n_out / n_src
        lo, hi = limits.get("min_length_ratio"), limits.get("max_length_ratio")
        if lo is not None and ratio < lo:
            reasons.append(f"translation much shorter than source (ratio {ratio:.2f} < {lo})")
        if hi is not None and ratio > hi:
            reasons.append(f"translation much longer than source (ratio {ratio:.2f} > {hi})")

    rep = max(repetition_ratio(output), ngram_loop_ratio(output))
    max_rep = limits.get("max_repetition_ratio", 0.5)
    if rep > max_rep:
        reasons.append(f"repetition loop in translation (ratio {rep:.2f} > {max_rep})")

    cp = copy_ratio(source, output)
    max_cp = limits.get("max_copy_ratio")
    if max_cp is not None and n_out >= 3 and cp > max_cp:
        reasons.append(f"possibly untranslated — copies the source (ratio {cp:.2f} > {max_cp})")

    min_letters = limits.get("min_source_letter_share")
    share = letter_share(source)
    if min_letters is not None and share < min_letters:
        reasons.append(f"source is mostly digits/symbols (letters {share:.2f} < {min_letters})")

    if limits.get("flag_biblical_register", True) and biblical_register(output):
        reasons.append("biblical register — typical NLLB hallucination on unparsed input")

    return Verdict(bool(reasons), reasons, round(rep, 3))


# Short labels for the reason strings above. The stored reason keeps its
# measured numbers — an analyst needs those — but a queue cannot be filtered or
# counted by free text, so each one also maps to a stable category.
REASON_CATEGORIES: list[tuple[str, str]] = [
    ("empty transcript", "Empty"),
    ("empty translation", "Empty (English)"),
    ("low confidence", "Low confidence"),
    ("probably not speech", "Not speech"),
    ("repetition loop in translation", "Repetition (English)"),
    ("repetition loop", "Repetition"),
    ("degenerate output", "Degenerate"),
    ("matches known hallucination phrase", "Known hallucination"),
    ("translation much shorter", "English too short"),
    ("translation much longer", "English too long"),
    ("possibly untranslated", "Untranslated"),
    ("source is mostly digits", "Not prose"),
    ("biblical register", "Biblical register"),
]


def reason_category(reason_text: str | None) -> str:
    """The short, filterable label for a stored quarantine reason.

    A row may carry several reasons joined by '; '; the first is the one that
    decided it, so that is the one shown.
    """
    first = (reason_text or "").split(";")[0].strip().lower()
    for prefix, label in REASON_CATEGORIES:
        if first.startswith(prefix):
            return label
    return "Other"


def check_all(utterances: Iterable[Any], pack_code: str) -> list[tuple[Any, Verdict]]:
    return [(u, check(u, pack_code)) for u in utterances]


def summarise(verdicts: Iterable[Verdict]) -> dict[str, Any]:
    """Quarantine rate is a reported metric in its own right (Plan §9.2)."""
    vs = list(verdicts)
    if not vs:
        return {"total": 0, "quarantined": 0, "rate": 0.0, "reasons": {}}
    counts: Counter[str] = Counter()
    for v in vs:
        for r in v.reasons:
            counts[r.split(" (")[0]] += 1
    n_q = sum(1 for v in vs if v.quarantined)
    return {
        "total": len(vs),
        "quarantined": n_q,
        "rate": round(n_q / len(vs), 4),
        "reasons": dict(counts.most_common()),
    }
