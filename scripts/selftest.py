#!/usr/bin/env python
"""Self-test for the pieces that must be right before any number is trusted.

Covers WER/CER arithmetic, Kikuyu diacritic preservation, Unicode NFC handling,
Dholuo ATR normalisation, and the hallucination quarantine thresholds.

    python scripts/selftest.py

Run this after touching evaluation.py, quarantine.py or any orthography rule.
"""
import sys, unicodedata
sys.path.insert(0, ".")
from app.config import Config
Config.enable_utf8_console()
from app.services import evaluation as ev
from app.services import quarantine as qu
from app.services import registry

print("=== packs ===")
for p in registry.all_packs():
    print(f"  {p.code:4} {p.name:12} {p.autonym:10} token=<|{p.whisper_token}|> model={p.model_ref}")

print("\n=== WER: identical ===")
s = ev.score_for_pack("kik", ["Nĩ kũrĩ ũhoro mwega"], ["Nĩ kũrĩ ũhoro mwega"])
print(" ", s.wer, s.normalisation)
assert s.wer == 0.0

print("=== WER: one substitution in 4 words ===")
s = ev.score_for_pack("kik", ["Nĩ kũrĩ ũhoro mwega"], ["Nĩ kũrĩ ũhoro mũru"])
print(" ", s.wer, f"S={s.substitutions} D={s.deletions} I={s.insertions}")
assert abs(s.wer - 0.25) < 1e-9

print("=== WER: deletion + insertion ===")
s = ev.score_for_pack("luo", ["ber ahinya winjo wach"], ["ber winjo wach moko"])
print(" ", s.wer, f"S={s.substitutions} D={s.deletions} I={s.insertions}")

print("\n=== diacritics must NOT be stripped for Kikuyu ===")
rules = registry.pack_for("kik").orthography["normalise"]
out = ev.normalise("Nĩ Kũrĩ Ũhoro", rules)
print("  normalised:", out)
assert "ĩ" in out and "ũ" in out, "Kikuyu diacritics were lost"

print("=== NFC: decomposed input must score as identical ===")
decomposed = unicodedata.normalize("NFD", "Nĩ kũrĩ")
composed  = unicodedata.normalize("NFC", "Nĩ kũrĩ")
print("  raw equal?", decomposed == composed, "(expected False)")
s = ev.score_for_pack("kik", [composed], [decomposed])
print("  WER after NFC normalisation:", s.wer, "(expected 0.0)")
assert s.wer == 0.0, "NFC normalisation is not being applied"

print("\n=== Dholuo ATR normalisation ===")
luo_rules = registry.pack_for("luo").orthography["normalise"]
print("  'wɛr ɔkɔ' ->", ev.normalise("wɛr ɔkɔ", luo_rules))

print("\n=== normalisation signature comparability ===")
print(" ", ev.normalisation_signature(rules))

print("\n=== quarantine ===")
class U:
    def __init__(self, text, lp=-0.3, ns=0.1, cr=1.5):
        self.text, self.avg_logprob, self.no_speech_prob, self.compression_ratio = text, lp, ns, cr

cases = [
    ("clean speech",        U("Nĩ kũrĩ ũhoro mwega kuuma Nyeri ũmũthĩ")),
    ("repetition loop",     U("asante sana asante sana asante sana asante sana asante sana")),
    ("known hallucination", U("Thanks for watching!")),
    ("low confidence",      U("mumble mumble", lp=-2.4)),
    ("not speech",          U("hmm", ns=0.95)),
    ("empty",               U("")),
]
verdicts = []
for label, u in cases:
    v = qu.check(u, "kik")
    verdicts.append(v)
    print(f"  {label:22} quarantined={str(v.quarantined):5} rep={v.repetition_ratio:.2f}  {v.reason_text[:70]}")

assert verdicts[0].quarantined is False, "clean speech must not be quarantined"
assert all(v.quarantined for v in verdicts[1:]), "a bad case slipped through"

print("\n  summary:", qu.summarise(verdicts))

print("\n=== translation registry ===")
# Every pack that needs English must name a registered translation model and a
# source token, or its lines silently sit at 'unavailable' forever.
specs = registry.all_model_specs()
print(f"  target token: {registry.translation_target_token()}")
for p in registry.all_packs():
    ref = p.translation_model_ref
    print(f"  {p.code:4} required={str(p.needs_translation):5} "
          f"token={p.translation.get('source_token')} model={ref}")
    if p.needs_translation:
        assert ref, f"{p.code} needs translation but has no translation.model_ref"
        assert ref in specs, f"{p.code}: model_ref {ref!r} is not registered under models"
        assert specs[ref].get("kind") == "translation", f"{ref} is not a translation model"
        assert p.translation.get("source_token"), f"{p.code} has no translation.source_token"

print("=== translation quality checks ===")
kik_src = "Nĩ kũrĩ ũhoro mwega wa mũgũnda ũyũ"
luo_src = "Ber ahinya winjo wach mag gweng'"
tcases = [
    # (label, source, output, pack, expect_flag)
    ("plausible kik",       kik_src, "There is good news about this farm", "kik", False),
    ("plausible luo",       luo_src, "It is very good to hear news from the village", "luo", False),
    ("copied source",       kik_src, kik_src, "kik", True),
    ("far too short",       kik_src, "farm", "kik", True),
    ("repetition loop",     kik_src, "the the the the the the the", "kik", True),
    ("empty",               kik_src, "", "kik", True),
    ("biblical register",   kik_src, "And he said to them, Do not be afraid, for I am the LORD", "kik", True),
    ("digits as source",    "0777.0376.0166.0376", "This is the first time I have been here", "kik", True),
    ("ordinary 'Lord'",     luo_src, "The Lord Mayor opened the market in the village", "luo", False),
]
for label, src, out, code, expect in tcases:
    v = qu.check_translation(src, out, code)
    print(f"  {label:20} flagged={str(v.quarantined):5} {v.reason_text[:70]}")
    assert v.quarantined is expect, f"translation check wrong for {label!r}"

print("\nALL CHECKS PASSED")
