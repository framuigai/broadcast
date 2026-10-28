"""WER / CER measurement (Plan §9).

Two rules make these numbers mean anything:

  1. The normalisation rule set is recorded WITH every score. An unstated
     normalisation policy can move WER by ten points or more, which makes
     cross-run comparison meaningless.
  2. Reference sets under eval/references/ are never trained on, tuned against,
     or edited to flatter a model.

Scores are always broken down per language, per station and per daypart.
A single aggregate figure hides the difference between a studio bulletin and a
telephone call-in, which is exactly the difference that matters.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.config import Config
from app.services import registry

# --------------------------------------------------------------------------- #
#  Normalisation
# --------------------------------------------------------------------------- #
_PUNCT = re.compile(r"[^\w\sĩũĨŨ'-]", re.UNICODE)
_WS = re.compile(r"\s+")

# Dholuo ATR / vowel-harmony spelling varies across sources. Without this, WER
# measures orthographic convention rather than recognition (Plan §4.5).
_ATR_MAP = {"ɔ": "o", "ɛ": "e", "ʊ": "u", "ɪ": "i"}


def normalise(text: str | None, rules: Sequence[str]) -> str:
    """Apply a pack's normalisation rules. Order matters; it is fixed here so that
    two callers cannot produce different numbers from the same rule list."""
    if not text:
        return ""

    # NFC first: 'ĩ' must be one codepoint, not 'i' + combining tilde, or every
    # Kikuyu word with a diacritic scores as an error for cosmetic reasons.
    out = unicodedata.normalize("NFC", text)

    if "normalise_atr" in rules:
        for src, dst in _ATR_MAP.items():
            out = out.replace(src, dst)

    if "lowercase" in rules:
        out = out.lower()

    if "strip_punct" in rules:
        out = _PUNCT.sub(" ", out)

    if "preserve_diacritics" not in rules and "strip_diacritics" in rules:
        # Deliberately opt-in and never the default: stripping ĩ/ũ changes the word.
        out = "".join(c for c in unicodedata.normalize("NFD", out)
                      if not unicodedata.combining(c))
        out = unicodedata.normalize("NFC", out)

    if "collapse_ws" in rules:
        out = _WS.sub(" ", out)

    return out.strip()


def normalisation_signature(rules: Sequence[str]) -> str:
    """Stored alongside every score so runs are comparable."""
    return "+".join(sorted(rules)) or "none"


# --------------------------------------------------------------------------- #
#  Edit distance
# --------------------------------------------------------------------------- #
def _levenshtein(ref: Sequence[str], hyp: Sequence[str]) -> tuple[int, int, int, int]:
    """Return (substitutions, deletions, insertions, distance) via Wagner-Fischer
    with a backtrace-free counting pass. O(len(ref) * len(hyp)) time, O(len(hyp)) space
    for the distance, plus an op-counting row."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0, 0, m, m
    if m == 0:
        return 0, n, 0, n

    # Each cell carries (distance, subs, dels, ins)
    prev: list[tuple[int, int, int, int]] = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur: list[tuple[int, int, int, int]] = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur.append(prev[j - 1])
                cur[j] = (prev[j - 1][0], prev[j - 1][1], prev[j - 1][2], prev[j - 1][3])
                continue
            d_sub = prev[j - 1][0] + 1
            d_del = prev[j][0] + 1
            d_ins = cur[j - 1][0] + 1
            best = min(d_sub, d_del, d_ins)
            if best == d_sub:
                p = prev[j - 1]
                cur.append((best, p[1] + 1, p[2], p[3]))
            elif best == d_del:
                p = prev[j]
                cur.append((best, p[1], p[2] + 1, p[3]))
            else:
                p = cur[j - 1]
                cur.append((best, p[1], p[2], p[3] + 1))
        prev = cur
    d, s, dl, ins = prev[m]
    return s, dl, ins, d


# --------------------------------------------------------------------------- #
#  Scoring
# --------------------------------------------------------------------------- #
@dataclass
class Score:
    wer: float
    cer: float
    n_ref_words: int
    n_ref_chars: int
    substitutions: int
    deletions: int
    insertions: int
    n_utterances: int
    normalisation: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_pair(reference: str, hypothesis: str, rules: Sequence[str]) -> Score:
    return score_many([reference], [hypothesis], rules)


def score_many(references: Iterable[str], hypotheses: Iterable[str],
               rules: Sequence[str]) -> Score:
    """Corpus-level WER: total edits over total reference words.

    Not the mean of per-utterance WERs — that over-weights short utterances and is
    the single most common way to publish a number that cannot be reproduced.
    """
    refs = list(references)
    hyps = list(hypotheses)
    if len(refs) != len(hyps):
        raise ValueError(f"reference/hypothesis count mismatch: {len(refs)} vs {len(hyps)}")

    tot_s = tot_d = tot_i = 0
    tot_words = 0
    char_edits = 0
    tot_chars = 0

    for r, h in zip(refs, hyps):
        rn, hn = normalise(r, rules), normalise(h, rules)
        rw, hw = rn.split(), hn.split()
        s, d, i, _ = _levenshtein(rw, hw)
        tot_s += s
        tot_d += d
        tot_i += i
        tot_words += len(rw)

        _, _, _, cdist = _levenshtein(list(rn.replace(" ", "")), list(hn.replace(" ", "")))
        char_edits += cdist
        tot_chars += len(rn.replace(" ", ""))

    wer = (tot_s + tot_d + tot_i) / tot_words if tot_words else 0.0
    cer = char_edits / tot_chars if tot_chars else 0.0

    return Score(
        wer=round(wer, 4), cer=round(cer, 4),
        n_ref_words=tot_words, n_ref_chars=tot_chars,
        substitutions=tot_s, deletions=tot_d, insertions=tot_i,
        n_utterances=len(refs), normalisation=normalisation_signature(rules),
    )


def score_for_pack(pack_code: str, references: Iterable[str],
                   hypotheses: Iterable[str]) -> Score:
    """Score using the pack's own registered normalisation rules."""
    pack = registry.pack_for(pack_code)
    rules = pack.orthography.get("normalise", ["lowercase", "strip_punct", "collapse_ws"])
    return score_many(references, hypotheses, rules)


# --------------------------------------------------------------------------- #
#  Reference sets
# --------------------------------------------------------------------------- #
@dataclass
class ReferenceItem:
    audio_path: str
    reference: str
    language: str
    channel_id: str | None = None
    daypart: str | None = None
    programme_type: str | None = None
    notes: str | None = None


def load_reference_set(name: str) -> list[ReferenceItem]:
    """Reference sets are JSONL under eval/references/<name>.jsonl.

    One object per line so a set can be appended to and diffed cleanly in git.
    """
    path = Config.REFERENCE_DIR / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Reference set {name!r} not found at {path}. "
            "Create it from expert transcription — never from model output."
        )
    items: list[ReferenceItem] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                items.append(ReferenceItem(**json.loads(line)))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return items


def available_reference_sets() -> list[str]:
    if not Config.REFERENCE_DIR.exists():
        return []
    return sorted(p.stem for p in Config.REFERENCE_DIR.glob("*.jsonl"))


def is_reference_audio(path: str | Path) -> bool:
    """Guard: refuse to use reference audio as training data (Plan §2.6)."""
    try:
        Path(path).resolve().relative_to(Config.REFERENCE_DIR.resolve())
        return True
    except (ValueError, OSError):
        return False


# --------------------------------------------------------------------------- #
#  Persistence
# --------------------------------------------------------------------------- #
def record_run(pack: str, model_ref: str, reference_set: str, score: Score,
               channel_id: str | None = None, daypart: str | None = None,
               realtime_factor: float | None = None, notes: str = "") -> int:
    from app import db
    return db.execute(
        "INSERT INTO eval_runs (pack, model_ref, reference_set, channel_id, daypart,"
        " wer, cer, n_utterances, n_words, normalisation, realtime_factor, notes)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (pack, model_ref, reference_set, channel_id, daypart, score.wer, score.cer,
         score.n_utterances, score.n_ref_words, score.normalisation,
         realtime_factor, notes),
    )


def history(pack: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    from app import db
    if pack:
        return db.query(
            "SELECT * FROM eval_runs WHERE pack=? ORDER BY created_at_utc DESC LIMIT ?",
            (pack, limit))
    return db.query("SELECT * FROM eval_runs ORDER BY created_at_utc DESC LIMIT ?", (limit,))


def latest_by_pack() -> dict[str, dict[str, Any]]:
    """Most recent run per pack — what the dashboard shows."""
    from app import db
    rows = db.query(
        "SELECT e.* FROM eval_runs e"
        " JOIN (SELECT pack, MAX(created_at_utc) AS mx FROM eval_runs GROUP BY pack) m"
        "   ON e.pack = m.pack AND e.created_at_utc = m.mx"
    )
    return {r["pack"]: r for r in rows}


def regression_check(pack: str, new_score: Score, tolerance: float = 0.005) -> dict[str, Any]:
    """Compare a new score against the last recorded one for this pack.

    Used as a gate: a change that makes WER worse should be visible immediately,
    not discovered three weeks later.
    """
    prev = history(pack, limit=1)
    if not prev:
        return {"status": "baseline", "message": "No previous run — this becomes the baseline."}

    old = prev[0]
    if old["normalisation"] != new_score.normalisation:
        return {
            "status": "incomparable",
            "message": (f"Normalisation changed ({old['normalisation']} -> "
                        f"{new_score.normalisation}). Re-baseline before comparing."),
        }

    delta = new_score.wer - (old["wer"] or 0.0)
    if delta < -tolerance:
        status = "improved"
    elif delta > tolerance:
        status = "regressed"
    else:
        status = "unchanged"
    return {
        "status": status,
        "previous_wer": old["wer"],
        "new_wer": new_score.wer,
        "delta": round(delta, 4),
        "message": f"WER {old['wer']:.4f} -> {new_score.wer:.4f} ({delta:+.4f})",
    }
