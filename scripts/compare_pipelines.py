#!/usr/bin/env python
"""Run every transcription/translation route over one audio file and save the
results side by side for human comparison.

    python scripts/compare_pipelines.py data/audio/.../file.opus

Routes compared:
  1. Paza  transcribe  <|kik|>          -> vernacular transcript
  2. NLLB  kik_Latn -> eng_Latn         -> English, from route 1's text
  3. NLLB  + loanword lexicon           -> English, phonetic loanwords mapped first
  4. Paza  translate  <|kik|>           -> English, straight from audio

Everything lands in data/comparisons/<stem>/ as plain text plus a metadata
JSON, so the outputs can be read without the app running.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Config                      # noqa: E402
from app.services import asr, registry             # noqa: E402

# Phonetic renderings of English/Swahili loanwords, observed in our own
# transcripts. This is the seed of the per-pack lexicon.
LEXICON = {
    "thirikari": "government", "abori": "police", "borithi": "police",
    "fainzi": "fines", "risaasi": "bullets", "biacara": "business",
    "njohi": "alcohol", "kitambulisho": "identity card",
    "shuuri": "school", "thukuru": "school", "hosibitari": "hospital",
    "bathi": "bus", "mbeca": "money", "gavana": "governor",
    "kaunti": "county", "seneta": "senator", "mbunge": "member of parliament",
}


def normalise(text: str) -> tuple[str, list[str]]:
    """Map phonetic loanwords back to English. Returns text and what changed."""
    out, hits = text, []
    for src, dst in LEXICON.items():
        if re.search(rf"\b{src}\b", out, flags=re.IGNORECASE):
            hits.append(f"{src} -> {dst}")
            out = re.sub(rf"\b{src}\b", dst, out, flags=re.IGNORECASE)
    return out, hits


def paza_run(audio: Path, pack: str, task: str) -> dict:
    """One Paza pass. task is 'transcribe' or 'translate'."""
    from faster_whisper import WhisperModel

    p = registry.pack_for(pack)
    spec = registry.model_spec(p.model_ref)
    asr._register_extra_language_tokens(p.model_ref)
    path = str(Config.ROOT_DIR / spec["local_dir"])

    model = WhisperModel(path, device="cpu",
                         compute_type=spec.get("compute_type", "int8"),
                         cpu_threads=Config.ASR_CPU_THREADS)
    t0 = time.perf_counter()
    segments, info = model.transcribe(
        str(audio), language=p.whisper_token, task=task, beam_size=5,
        vad_filter=True, vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False, word_timestamps=True)
    segs = list(segments)
    wall = time.perf_counter() - t0
    del model

    dur = getattr(info, "duration", 0.0) or 0.0
    return {
        "text": " ".join(s.text.strip() for s in segs).strip(),
        "lines": [{"start": round(s.start, 2), "end": round(s.end, 2),
                   "avg_logprob": round(s.avg_logprob, 3),
                   "no_speech_prob": round(s.no_speech_prob, 3),
                   "text": s.text.strip()} for s in segs],
        "avg_logprob": round(sum(s.avg_logprob for s in segs) / len(segs), 3) if segs else None,
        "n_segments": len(segs),
        "audio_seconds": round(dur, 2),
        "wall_seconds": round(wall, 1),
        "realtime_factor": round(dur / wall, 3) if wall else 0.0,
    }


def nllb_translate(texts: dict[str, str], src_token: str) -> dict[str, dict]:
    """Translate several variants with one model load."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_dir = Config.MODELS_DIR / "_src" / "nllb-200-distilled-600M"
    if not model_dir.exists():
        return {k: {"error": f"NLLB not present at {model_dir}"} for k in texts}

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir), dtype=torch.float32)
    model.eval()
    tok.src_lang = src_token
    bos = tok.convert_tokens_to_ids("eng_Latn")

    out: dict[str, dict] = {}
    for key, text in texts.items():
        if not text.strip():
            out[key] = {"text": "", "wall_seconds": 0.0}
            continue
        t0 = time.perf_counter()
        # Translate sentence by sentence: NLLB degrades on long inputs, and our
        # measurements showed short units hallucinate far less.
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()] or [text]
        pieces = []
        for part in parts:
            enc = tok(part, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                gen = model.generate(**enc, forced_bos_token_id=bos,
                                     max_new_tokens=200, num_beams=4)
            pieces.append(tok.batch_decode(gen, skip_special_tokens=True)[0])
        out[key] = {"text": " ".join(pieces),
                    "n_sentences": len(parts),
                    "wall_seconds": round(time.perf_counter() - t0, 1)}
    del model
    return out


def paza_novad(audio: Path, pack_code: str) -> dict:
    """Same as paza_run transcribe, but with VAD off.

    If VAD is discarding speech (common on music-bedded or heavily limited
    broadcast audio) the no-VAD run will differ sharply.
    """
    from faster_whisper import WhisperModel

    p = registry.pack_for(pack_code)
    spec = registry.model_spec(p.model_ref)
    asr._register_extra_language_tokens(p.model_ref)
    model = WhisperModel(str(Config.ROOT_DIR / spec["local_dir"]), device="cpu",
                         compute_type=spec.get("compute_type", "int8"),
                         cpu_threads=Config.ASR_CPU_THREADS)
    t0 = time.perf_counter()
    segs, info = model.transcribe(str(audio), language=p.whisper_token,
                                  task="transcribe", beam_size=5, vad_filter=False,
                                  condition_on_previous_text=False,
                                  word_timestamps=True)
    out = list(segs)
    wall = time.perf_counter() - t0
    del model
    dur = getattr(info, "duration", 0.0) or 0.0
    return {
        "text": " ".join(x.text.strip() for x in out).strip(),
        "avg_logprob": round(sum(x.avg_logprob for x in out) / len(out), 3) if out else None,
        "n_segments": len(out), "wall_seconds": round(wall, 1),
        "realtime_factor": round(dur / wall, 3) if wall else 0.0,
    }


def stock_autodetect(audio: Path) -> dict:
    """Stock Whisper with language auto-detection.

    Diagnostic only. Stock Whisper cannot represent Kikuyu or Dholuo, so what it
    guesses tells us whether the audio is speech at all and whether it is in one
    of Whisper's 99 languages (i.e. actually Swahili or English rather than
    vernacular).
    """
    from faster_whisper import WhisperModel

    spec = registry.model_spec("whisper-large-v3-turbo-ct2")
    path = Config.ROOT_DIR / spec["local_dir"]
    if not (path / "model.bin").exists():
        return {"error": "stock whisper not present"}
    model = WhisperModel(str(path), device="cpu", compute_type="int8",
                         cpu_threads=Config.ASR_CPU_THREADS)
    t0 = time.perf_counter()
    segs, info = model.transcribe(str(audio), beam_size=5, vad_filter=True,
                                  condition_on_previous_text=False)
    out = list(segs)
    wall = time.perf_counter() - t0
    del model
    return {
        "detected_language": getattr(info, "language", None),
        "language_probability": round(getattr(info, "language_probability", 0.0), 3),
        "text": " ".join(x.text.strip() for x in out).strip(),
        "avg_logprob": round(sum(x.avg_logprob for x in out) / len(out), 3) if out else None,
        "n_segments": len(out), "wall_seconds": round(wall, 1),
    }



def _emit(outdir: Path, name: str, content: str) -> None:
    """Write one result immediately.

    Routes can take 20+ minutes each on this CPU. Buffering everything until the
    end means an interrupted run loses completed work — so each route is flushed
    to disk as soon as it returns.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / name).write_text((content or "") + chr(10), encoding="utf-8")
    print(f"      -> wrote {name}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--pack", default="kik")
    ap.add_argument("--out", help="output directory (default data/comparisons/<stem>)")
    ap.add_argument("--diagnose", action="store_true",
                    help="add diagnostic routes: no-VAD, alternate pack, stock Whisper auto-detect")
    args = ap.parse_args()

    Config.enable_utf8_console()
    Config.ensure_dirs()

    audio = Path(args.audio)
    if not audio.is_absolute():
        audio = Config.ROOT_DIR / audio
    if not audio.exists():
        print(f"No such audio: {audio}")
        return 2

    outdir = Path(args.out) if args.out else (
        Config.DATA_DIR / "comparisons" / audio.stem)
    outdir.mkdir(parents=True, exist_ok=True)

    pack = registry.pack_for(args.pack)
    print(f"Audio : {audio.name}")
    print(f"Pack  : {pack.name} ({pack.code})  token <|{pack.whisper_token}|>")
    print(f"Out   : {outdir}\n")

    # --- 1. Paza transcribe -------------------------------------------
    print("[1/4] Paza transcribe (vernacular)…", flush=True)
    r_tx = paza_run(audio, args.pack, "transcribe")
    print(f"      {r_tx['n_segments']} segs, logprob {r_tx['avg_logprob']}, "
          f"RTF {r_tx['realtime_factor']}x, {r_tx['wall_seconds']}s")
    _emit(outdir, "1_paza_transcribe_vernacular.txt", r_tx["text"])

    # --- 2. Paza translate --------------------------------------------
    print("[2/4] Paza translate (audio -> English)…", flush=True)
    r_st = paza_run(audio, args.pack, "translate")
    print(f"      {r_st['n_segments']} segs, logprob {r_st['avg_logprob']}, "
          f"RTF {r_st['realtime_factor']}x, {r_st['wall_seconds']}s")
    _emit(outdir, "2_paza_translate_audio_to_english.txt", r_st["text"])

    # --- 3/4. NLLB, raw and lexicon-normalised ------------------------
    norm_text, hits = normalise(r_tx["text"])
    print(f"[3/4] Lexicon: {len(hits)} substitution(s) {hits if hits else ''}")
    print("[4/4] NLLB translate (raw + normalised)…", flush=True)
    src_token = pack.translation.get("source_token", "kik_Latn")
    mt = nllb_translate({"raw": r_tx["text"], "normalised": norm_text}, src_token)
    print(f"      raw {mt['raw'].get('wall_seconds')}s, "
          f"normalised {mt['normalised'].get('wall_seconds')}s")
    _emit(outdir, "3_nllb_from_transcript_raw.txt", mt["raw"].get("text", ""))
    _emit(outdir, "4_nllb_from_transcript_lexicon.txt", mt["normalised"].get("text", ""))
    _emit(outdir, "0_transcript_after_lexicon.txt", norm_text)

    diag: dict = {}
    if args.diagnose:
        print("[5] Paza transcribe, VAD off…", flush=True)
        diag["5_paza_novad"] = paza_novad(audio, args.pack)
        print(f"      logprob {diag['5_paza_novad']['avg_logprob']}, "
              f"{diag['5_paza_novad']['n_segments']} segs")

        other = "swh" if args.pack != "swh" else "kik"
        print(f"[6] Paza transcribe as {other} (is it actually that language?)…", flush=True)
        diag[f"6_paza_as_{other}"] = paza_run(audio, other, "transcribe")
        diag[f"6_paza_as_{other}"].pop("lines", None)
        print(f"      logprob {diag[f'6_paza_as_{other}']['avg_logprob']}")

        print("[7] Stock Whisper, language auto-detect…", flush=True)
        diag["7_stock_autodetect"] = stock_autodetect(audio)
        d = diag["7_stock_autodetect"]
        print(f"      detected={d.get('detected_language')} "
              f"p={d.get('language_probability')} logprob={d.get('avg_logprob')}")
        files_extra = {
            "5_paza_transcribe_novad.txt": diag["5_paza_novad"].get("text", ""),
            f"6_paza_as_{other}.txt": diag[f"6_paza_as_{other}"].get("text", ""),
            "7_stock_whisper_autodetect.txt":
                f"detected: {d.get('detected_language')} "
                f"(p={d.get('language_probability')})" + chr(10) * 2 + d.get("text", ""),
        }
    else:
        files_extra = {}

    # --- write outputs -------------------------------------------------
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    files = {
        "1_paza_transcribe_vernacular.txt": r_tx["text"],
        "2_paza_translate_audio_to_english.txt": r_st["text"],
        "3_nllb_from_transcript_raw.txt": mt["raw"].get("text", ""),
        "4_nllb_from_transcript_lexicon.txt": mt["normalised"].get("text", ""),
        "0_transcript_after_lexicon.txt": norm_text,
    }
    files.update(files_extra)
    for name, content in files.items():
        (outdir / name).write_text(content + "\n", encoding="utf-8")

    meta = {
        "generated_utc": stamp,
        "audio": str(audio.relative_to(Config.ROOT_DIR)),
        "audio_seconds": r_tx["audio_seconds"],
        "pack": {"code": pack.code, "name": pack.name,
                 "whisper_token": pack.whisper_token, "model": pack.model_ref},
        "routes": {
            "1_paza_transcribe": {k: v for k, v in r_tx.items() if k != "lines"},
            "2_paza_translate": {k: v for k, v in r_st.items() if k != "lines"},
            "3_nllb_raw": mt["raw"],
            "4_nllb_lexicon": mt["normalised"],
        },
        "lexicon_hits": hits,
        "diagnostics": diag,
        "segments_transcribe": r_tx["lines"],
        "segments_translate": r_st["lines"],
    }
    (outdir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- side-by-side markdown ----------------------------------------
    md = [
        f"# Pipeline comparison — {audio.name}",
        "",
        f"- Audio: `{meta['audio']}` · {r_tx['audio_seconds']}s",
        f"- Pack: {pack.name} (`{pack.code}`, token `<|{pack.whisper_token}|>`)",
        f"- Generated: {stamp}",
        "",
        "| Route | What it does | Wall | RTF | avg_logprob |",
        "|---|---|---|---|---|",
        f"| 1 Paza transcribe | audio -> vernacular | {r_tx['wall_seconds']}s | "
        f"{r_tx['realtime_factor']}x | {r_tx['avg_logprob']} |",
        f"| 2 Paza translate | audio -> English | {r_st['wall_seconds']}s | "
        f"{r_st['realtime_factor']}x | {r_st['avg_logprob']} |",
        f"| 3 NLLB raw | transcript -> English | {mt['raw'].get('wall_seconds')}s | — | — |",
        f"| 4 NLLB + lexicon | normalised -> English | "
        f"{mt['normalised'].get('wall_seconds')}s | — | — |",
        "",
        "---",
        "",
        "## 1. Paza transcribe — vernacular",
        "", r_tx["text"] or "_(empty)_", "",
        "## 2. Paza translate — audio straight to English",
        "", r_st["text"] or "_(empty)_", "",
        "## 3. NLLB — translated from transcript 1, raw",
        "", mt["raw"].get("text") or "_(empty)_", "",
        "## 4. NLLB — translated after loanword normalisation",
        "",
        f"Substitutions applied: {', '.join(hits) if hits else 'none'}",
        "", mt["normalised"].get("text") or "_(empty)_", "",
        "---",
        "",
        "## Per-segment detail — transcribe",
        "",
        "| Start | End | logprob | Text |",
        "|---|---|---|---|",
    ]
    for ln in r_tx["lines"]:
        md.append(f"| {ln['start']} | {ln['end']} | {ln['avg_logprob']} | "
                  f"{ln['text'].replace('|', '/')} |")
    md += ["", "## Per-segment detail — translate", "",
           "| Start | End | logprob | Text |", "|---|---|---|---|"]
    for ln in r_st["lines"]:
        md.append(f"| {ln['start']} | {ln['end']} | {ln['avg_logprob']} | "
                  f"{ln['text'].replace('|', '/')} |")

    (outdir / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\nWrote {len(files) + 2} files to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
