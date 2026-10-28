"""Channel and Language Pack registries.

This module is the ONLY place that knows which languages and channels exist.
Pipeline code asks the registry; it never tests for a language by name.
Adding the 44th language must be a config change, not a code change (Plan §3.2).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from app.config import Config


# --------------------------------------------------------------------------- #
#  Data classes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LanguagePack:
    code: str                  # ISO 639-3
    name: str
    autonym: str
    asr: dict[str, Any]
    translation: dict[str, Any]
    orthography: dict[str, Any]
    quarantine: dict[str, Any]
    service_tier: int
    colour: str
    sheng: dict[str, Any] = field(default_factory=dict)

    @property
    def model_ref(self) -> str:
        return self.asr["model_ref"]

    @property
    def whisper_token(self) -> str:
        """Whisper's own language token. Converts ISO 639-3 -> Whisper at the
        model boundary only; nowhere else in the codebase uses this."""
        return self.asr["whisper_token"]

    @property
    def needs_translation(self) -> bool:
        return bool(self.translation.get("required"))

    @property
    def translation_model_ref(self) -> str | None:
        """The text-to-text model that turns this pack's transcripts into English.

        None means translation is wanted (or not) but no model is assigned yet —
        enabling a language is adding a model_ref here, never a code change.
        """
        return self.translation.get("model_ref") or None


@dataclass(frozen=True)
class Channel:
    id: str
    name: str
    broadcaster: str
    medium: str                # tv | fm
    primary_language: str
    region: str
    capture: dict[str, Any]
    routing_prior: dict[str, float]
    accent: str
    enabled: bool = True

    @property
    def candidate_languages(self) -> list[str]:
        """The ONLY languages considered when routing this channel.

        This is the main accuracy win in routing (Plan §8.2 R1): three candidates
        instead of Whisper's ninety-nine. It is the formal expression of
        'we know Ramogi broadcasts in Luo' — a prior, not a hard assumption, so a
        Kiswahili insert is still caught.
        """
        return sorted(self.routing_prior, key=lambda k: -self.routing_prior[k])


@dataclass(frozen=True)
class Daypart:
    name: str
    start: str      # HH:MM, EAT
    end: str
    priority: str   # high | medium | low


# --------------------------------------------------------------------------- #
#  Loading
# --------------------------------------------------------------------------- #
def _read_json(path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:      # encoding is not optional here
        return json.load(fh)


@lru_cache(maxsize=1)
def _channels_raw() -> dict[str, Any]:
    return _read_json(Config.CHANNELS_FILE)


@lru_cache(maxsize=1)
def _packs_raw() -> dict[str, Any]:
    return _read_json(Config.LANGUAGE_PACKS_FILE)


def reload() -> None:
    """Drop caches after a config edit."""
    _channels_raw.cache_clear()
    _packs_raw.cache_clear()


# --------------------------------------------------------------------------- #
#  Language packs
# --------------------------------------------------------------------------- #
def all_packs() -> list[LanguagePack]:
    out = []
    for p in _packs_raw()["packs"]:
        out.append(LanguagePack(
            code=p["code"], name=p["name"], autonym=p.get("autonym", p["name"]),
            asr=p["asr"], translation=p.get("translation", {}),
            orthography=p.get("orthography", {}), quarantine=p.get("quarantine", {}),
            service_tier=p.get("service_tier", 3), colour=p.get("colour", "#888888"),
            sheng=p.get("sheng", {}),
        ))
    return out


def pack_for(code: str) -> LanguagePack:
    for p in all_packs():
        if p.code == code:
            return p
    raise KeyError(
        f"No language pack registered for {code!r}. "
        f"Add it to {Config.LANGUAGE_PACKS_FILE.name} — do not special-case it in code."
    )


def pack_exists(code: str) -> bool:
    return any(p.code == code for p in all_packs())


def model_spec(model_ref: str) -> dict[str, Any]:
    models = _packs_raw()["models"]
    if model_ref not in models:
        raise KeyError(f"Unknown model_ref {model_ref!r}. Register it under 'models'.")
    return models[model_ref]


def all_model_specs() -> dict[str, dict[str, Any]]:
    return {k: v for k, v in _packs_raw()["models"].items() if not k.startswith("_")}


def translation_settings() -> dict[str, Any]:
    return _packs_raw().get("translation", {})


def translation_target_token() -> str:
    """The target-language token for translation, taken from the target pack.

    English is a registered pack like any other, so its NLLB token lives in its
    own entry rather than as a literal in the translation code.
    """
    target = pack_for(translation_settings().get("target_pack", "eng"))
    token = target.translation.get("source_token")
    if not token:
        raise KeyError(f"Target pack {target.code!r} has no translation.source_token")
    return token


def translation_checks(pack_code: str) -> dict[str, Any]:
    """Pipeline-wide translation check thresholds, with per-pack overrides."""
    merged = {k: v for k, v in translation_settings().get("checks", {}).items()
              if not k.startswith("_")}
    merged.update(pack_for(pack_code).translation.get("checks", {}))
    return merged


# --------------------------------------------------------------------------- #
#  Channels
# --------------------------------------------------------------------------- #
def all_channels(medium: str | None = None, enabled_only: bool = False) -> list[Channel]:
    out = []
    for c in _channels_raw()["channels"]:
        ch = Channel(
            id=c["id"], name=c["name"], broadcaster=c["broadcaster"],
            medium=c["medium"], primary_language=c["primary_language"],
            region=c.get("region", ""), capture=c.get("capture", {}),
            routing_prior=c.get("routing_prior", {}), accent=c.get("accent", "#888888"),
            enabled=c.get("enabled", True),
        )
        if medium and ch.medium != medium:
            continue
        if enabled_only and not ch.enabled:
            continue
        out.append(ch)
    return out


def channel(channel_id: str) -> Channel:
    for c in all_channels():
        if c.id == channel_id:
            return c
    raise KeyError(f"Unknown channel {channel_id!r}")


def dayparts() -> list[Daypart]:
    return [Daypart(**w) for w in _channels_raw()["dayparts"]["windows"]]


def daypart_for(hhmm_eat: str) -> Daypart | None:
    """Resolve an EAT HH:MM to its daypart, handling the window that wraps midnight."""
    def mins(s: str) -> int:
        h, m = s.split(":")
        return int(h) * 60 + int(m)

    t = mins(hhmm_eat)
    for d in dayparts():
        a, b = mins(d.start), mins(d.end)
        if a <= b:
            if a <= t < b:
                return d
        else:                       # wraps midnight, e.g. 21:30 -> 05:00
            if t >= a or t < b:
                return d
    return None


def languages_in_use() -> list[str]:
    """Every language any enabled channel might carry. Drives model pre-loading."""
    codes: set[str] = set()
    for ch in all_channels(enabled_only=True):
        codes.update(ch.routing_prior)
    return sorted(codes)
