"""Language routing (Plan §8).

Two problems that must not be conflated:

  * Segment-level language blocks — a Kiswahili headline read, an English advert,
    a twenty-minute Kikuyu bulletin. Solved here.
  * Intra-sentence code-switching — "nĩ kũrĩ budget allocation ya county".
    NOT solvable here, at any accuracy. The language changes mid-utterance, so
    there is no boundary to route on. That is fixed only by the vernacular model
    itself learning to emit English and Kiswahili words, which is what the
    code-switched training share buys in Step 2.

The chain is R0 prior -> R1 candidate restriction -> R2 token scoring ->
R3 temporal smoothing -> R4 arbitration. R0/R1 are free and do most of the work.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from app.services import registry

log = logging.getLogger(__name__)

# R4 fires only when the top two candidates are this close.
ARBITRATION_MARGIN = 0.15
# R3: how strongly we believe the language has NOT changed between adjacent segments.
SELF_TRANSITION = 0.85
# Tokens decoded when probing a candidate language. Short: we only need a score,
# not a transcript, and this runs on every segment.
PROBE_TOKENS = 12
# Sharpens the softmax over probe scores; log-likelihood gaps here are small in
# absolute terms but consistently ordered.
PROBE_SHARPNESS = 6.0


@dataclass
class RoutingDecision:
    language: str
    confidence: float
    method: str                      # prior | token_score | arbitration | manual | smoothed
    posteriors: dict[str, float]
    candidates: list[str]
    note: str = ""


# --------------------------------------------------------------------------- #
#  R0 + R1
# --------------------------------------------------------------------------- #
def candidates_for(channel_id: str) -> list[str]:
    """R1. Three candidates instead of Whisper's ninety-nine.

    This is the main accuracy win, and it is free.
    """
    return registry.channel(channel_id).candidate_languages


def prior_for(channel_id: str, daypart_name: str | None = None) -> dict[str, float]:
    """R0. The station prior, daypart-adjusted.

    News hours skew more Kiswahili than the station average. The adjustment is
    deliberately mild — priors are measured from Step 1 data, not invented, and
    an over-confident prior defeats the point of measuring at all.
    """
    ch = registry.channel(channel_id)
    prior = dict(ch.routing_prior)

    if daypart_name and "bulletin" in daypart_name.lower():
        if "swh" in prior:
            shift = min(0.10, prior.get(ch.primary_language, 0.0) * 0.15)
            prior["swh"] = prior["swh"] + shift
            prior[ch.primary_language] = max(0.0, prior[ch.primary_language] - shift)

    total = sum(prior.values()) or 1.0
    return {k: v / total for k, v in prior.items()}




def _score_language_tokens(model, audio_path: str | Path,
                           codes: Sequence[str]) -> dict[str, float]:
    """Score each candidate language token directly through the decoder.

    Why not `detect_language`: both faster-whisper's wrapper and CTranslate2's own
    `detect_language` evaluate only Whisper's ORIGINAL 100 language tokens. Paza's
    added <|kik|>, <|luo|>, <|kln|>, <|mas|> and <|som|> are simply absent from that
    output — measured on this checkpoint, it returns exactly 100 entries and none of
    them are the Kenyan vernaculars. Routing on it would score Kikuyu at 0.0 forever.

    Instead we encode once, then run a short constrained decode per candidate with
    the language token forced, and compare the returned sequence scores. That uses
    the full vocabulary, so the added languages are scored properly.

    Measured on Inooro FM audio: kik -0.10, luo -0.10, sw -0.64, en -0.88. Note the
    vernaculars sit close together — but candidate restriction (R1) means a station
    never has to separate kik from luo, only vernacular from Kiswahili/English, and
    that margin is decisive.
    """
    import numpy as np
    import ctranslate2
    from faster_whisper.audio import decode_audio

    waveform = decode_audio(str(audio_path),
                            sampling_rate=model.feature_extractor.sampling_rate)
    features = model.feature_extractor(waveform)
    window = features[:, :model.feature_extractor.nb_max_frames]
    batch = np.ascontiguousarray(window[np.newaxis, :, :]).astype(np.float32)
    encoded = model.model.encode(ctranslate2.StorageView.from_array(batch), to_cpu=False)

    raw: dict[str, float] = {}
    for code in codes:
        token = f"<|{registry.pack_for(code).whisper_token}|>"
        try:
            result = model.model.generate(
                encoded,
                [["<|startoftranscript|>", token, "<|transcribe|>", "<|notimestamps|>"]],
                max_length=PROBE_TOKENS, return_scores=True, beam_size=1)
            raw[code] = float(result[0].scores[0])
        except (RuntimeError, IndexError, ValueError) as exc:
            log.debug("Language probe failed for %s: %s", code, exc)

    if not raw:
        return {}

    # Scores are log-likelihoods; softmax them into comparable probabilities.
    top = max(raw.values())
    exps = {k: math.exp((v - top) * PROBE_SHARPNESS) for k, v in raw.items()}
    total = sum(exps.values()) or 1.0
    return {k: v / total for k, v in exps.items()}


# --------------------------------------------------------------------------- #
#  R2
# --------------------------------------------------------------------------- #
def score_candidates(audio_path: str | Path, channel_id: str,
                     daypart_name: str | None = None) -> RoutingDecision:
    """R2. Score only the candidate language tokens for this channel.

    Uses the ASR model's own language head rather than introducing a second large
    model: memory on this machine is tight, and a dedicated LID model would score
    across its full label inventory rather than the three languages this station
    can plausibly carry — which is precisely where the accuracy comes from.
    """
    from app.services import asr

    cands = candidates_for(channel_id)
    prior = prior_for(channel_id, daypart_name)

    # Candidates may not all share one physical model: Paza covers kik/luo/swh
    # while English uses stock Whisper. Rather than give up and route on the prior
    # alone, score the largest group that DOES share a model — for a vernacular
    # station that is the kik-vs-swh (or luo-vs-swh) discrimination which actually
    # matters — and let the prior carry the rest. Loading a second large model to
    # route one segment is not affordable on a 13.7 GB box.
    by_ref: dict[str, list[str]] = {}
    for c in cands:
        if registry.pack_exists(c):
            by_ref.setdefault(registry.pack_for(c).model_ref, []).append(c)

    if not by_ref:
        top = max(prior, key=prior.get)
        return RoutingDecision(top, prior[top], "prior", prior, cands,
                               "No registered packs among candidates.")

    # Prefer a model that is already resident, then the one covering most candidates.
    resident = set(asr.loaded_models())
    model_ref = max(by_ref, key=lambda r: (r in resident, len(by_ref[r])))
    scored_codes = by_ref[model_ref]
    unscored = [c for c in cands if c not in scored_codes]
    try:
        model = asr.load(model_ref)
        detected = _score_language_tokens(model, audio_path, scored_codes)
        if not detected:
            top = max(prior, key=prior.get)
            return RoutingDecision(top, prior[top], "prior", prior, cands,
                                   "Language scoring unavailable; fell back to prior.")

        # Posterior proportional to likelihood x prior, for the candidates we could
        # score. Candidates on another model keep their prior mass so they remain
        # reachable — an English advert on a Kikuyu station is still a possibility,
        # just one we did not get a likelihood for.
        combined = {}
        for c in cands:
            if c in detected:
                combined[c] = detected[c] * prior.get(c, 0.0)
            else:
                # Unscored: prior alone, discounted so a scored match wins ties.
                combined[c] = prior.get(c, 0.0) * 0.5
        total = sum(combined.values())
        if total <= 0:
            top = max(prior, key=prior.get)
            return RoutingDecision(top, prior[top], "prior", prior, cands,
                                   "Model returned no usable posterior; fell back to prior.")

        posteriors = {k: v / total for k, v in combined.items()}
        ranked = sorted(posteriors.items(), key=lambda kv: -kv[1])
        best, best_p = ranked[0]
        margin = best_p - (ranked[1][1] if len(ranked) > 1 else 0.0)

        method = "arbitration" if margin < ARBITRATION_MARGIN else "token_score"
        note = f"margin={margin:.3f}; scored={'+'.join(scored_codes)}"
        if unscored:
            note += f"; prior-only={'+'.join(unscored)}"
        return RoutingDecision(best, round(best_p, 4), method,
                               {k: round(v, 4) for k, v in posteriors.items()}, cands, note)

    except (FileNotFoundError, RuntimeError, KeyError, AttributeError, ValueError) as exc:
        log.warning("Routing scorer failed for %s: %s — falling back to prior", audio_path, exc)
        top = max(prior, key=prior.get)
        return RoutingDecision(top, prior[top], "prior", prior, cands, f"scorer error: {exc}")


# --------------------------------------------------------------------------- #
#  R3
# --------------------------------------------------------------------------- #
def smooth(decisions: Sequence[RoutingDecision],
           self_transition: float = SELF_TRANSITION) -> list[RoutingDecision]:
    """R3. Viterbi smoothing over a run of consecutive segments.

    Broadcast language blocks last minutes, not seconds. Without this, per-segment
    routing flaps between languages and that flapping dominates the error rate.
    Free, and one of the largest accuracy gains in the chain.
    """
    if len(decisions) < 2:
        return list(decisions)

    states = sorted({s for d in decisions for s in d.posteriors})
    if not states:
        return list(decisions)

    n_other = max(1, len(states) - 1)
    stay = math.log(self_transition)
    move = math.log((1.0 - self_transition) / n_other)
    eps = 1e-9

    # Forward pass
    v: list[dict[str, float]] = [{s: math.log(decisions[0].posteriors.get(s, eps) + eps)
                                  for s in states}]
    back: list[dict[str, str]] = [{}]

    for t in range(1, len(decisions)):
        v_t: dict[str, float] = {}
        b_t: dict[str, str] = {}
        for s in states:
            emit = math.log(decisions[t].posteriors.get(s, eps) + eps)
            best_prev, best_score = None, -math.inf
            for p in states:
                score = v[t - 1][p] + (stay if p == s else move)
                if score > best_score:
                    best_prev, best_score = p, score
            v_t[s] = best_score + emit
            b_t[s] = best_prev or s
        v.append(v_t)
        back.append(b_t)

    # Backward pass
    path = [max(v[-1], key=v[-1].get)]
    for t in range(len(decisions) - 1, 0, -1):
        path.append(back[t][path[-1]])
    path.reverse()

    out: list[RoutingDecision] = []
    for d, lang in zip(decisions, path):
        if lang == d.language:
            out.append(d)
        else:
            out.append(RoutingDecision(
                lang, d.posteriors.get(lang, 0.0), "smoothed", d.posteriors, d.candidates,
                f"smoothed from {d.language} ({d.method})"))
    return out


# --------------------------------------------------------------------------- #
#  R4
# --------------------------------------------------------------------------- #
def arbitrate(audio_path: str | Path, decision: RoutingDecision) -> RoutingDecision:
    """R4. Decode with the top two packs and keep the higher-scoring transcript.

    Expensive per segment, so it is applied only to the ~5% of traffic where the
    top two posteriors fall within ARBITRATION_MARGIN.
    """
    from app.services import asr

    ranked = sorted(decision.posteriors.items(), key=lambda kv: -kv[1])[:2]
    if len(ranked) < 2:
        return decision

    scores: dict[str, float] = {}
    for code, _ in ranked:
        try:
            # No pack prompt here: arbitration compares each candidate's own
            # confidence, and priming one with its vocabulary would make it look
            # more fluent than a rival decoded cold. Prompts belong to the
            # transcription decode, not the language decision.
            utts, _meta = asr.transcribe(audio_path, code, use_pack_prompt=False)
            if not utts:
                continue
            # Length-normalised average log-probability: the model's own confidence
            # in its output, comparable across languages.
            total_dur = sum(u.duration for u in utts) or 1.0
            scores[code] = sum(u.avg_logprob * u.duration for u in utts) / total_dur
        except (FileNotFoundError, RuntimeError, KeyError) as exc:
            log.warning("Arbitration decode failed for %s: %s", code, exc)

    if not scores:
        return decision

    winner = max(scores, key=scores.get)
    return RoutingDecision(
        winner, decision.posteriors.get(winner, 0.0), "arbitration",
        decision.posteriors, decision.candidates,
        "dual-decode: " + ", ".join(f"{k}={v:.3f}" for k, v in scores.items()))


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #
def route(audio_path: str | Path, channel_id: str, daypart_name: str | None = None,
          *, allow_arbitration: bool = True) -> RoutingDecision:
    """Full chain for a single segment."""
    decision = score_candidates(audio_path, channel_id, daypart_name)
    if allow_arbitration and decision.method == "arbitration":
        decision = arbitrate(audio_path, decision)
    return decision


def route_batch(items: Sequence[tuple[str | Path, str]],
                daypart_name: str | None = None) -> list[RoutingDecision]:
    """Route consecutive segments from one channel, with smoothing applied.

    Prefer this over repeated `route()` calls: R3 is where a large part of the
    accuracy comes from, and it needs the sequence.
    """
    raw = [score_candidates(p, cid, daypart_name) for p, cid in items]
    return smooth(raw)


def manual_override(language: str, actor: str = "analyst") -> RoutingDecision:
    """An analyst forcing a segment to a pack. Logged as routing training data."""
    if not registry.pack_exists(language):
        raise KeyError(f"No language pack for {language!r}")
    return RoutingDecision(language, 1.0, "manual", {language: 1.0}, [language],
                           f"manual override by {actor}")
