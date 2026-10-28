# Kikuyu & Luo POC — Build Plan

Companion to `OSINT_POC_Kikuyu_Luo_Plan.docx` (the programme plan). That document
says *what and why*; this one says *what we build, in what order, and how we know
it works*. Section references below (§N) point into the docx.

**Current position: Step 1 — baseline. No datasets, no training.**

---

## 0. Why baseline before data

The instinct is to get training data first. That is the wrong order here, for a
reason specific to this problem:

Africa Next Voices (754 h Kikuyu, 723 h Luo) is prompted and read speech recorded in
controlled conditions. Inooro and Ramogi are spontaneous, overlapped, compressed,
music-bedded and telephone-degraded. **The domain gap will very likely dominate the
error rate — more than the amount of training data does.** Paza was already trained on
Africa Next Voices, so fine-tuning on the same corpus is not automatically an
improvement over it.

So the sequence is:

1. Measure what off-the-shelf models do on **real broadcast audio**. (Step 1)
2. Look at *where* they fail — which station, which daypart, which programme type.
3. Let that error analysis decide what data to acquire. (Steps 2–3)

Step 1 also builds the correction workflow, which is what turns analyst review time
into in-domain training data. Ten hours of corrected broadcast audio will probably
beat another hundred hours of studio-read speech. That mechanism must exist *before*
the baseline is run, or the highest-value data in the whole POC is generated and
thrown away.

---

## 1. Models — the decision, and why

The requirement is **a separate model per language**, because the system scales to
44+ languages and later fine-tuning must be independent per language. Paza is a single
six-language checkpoint, so it is the **baseline to beat**, not the deployed product
(§4.1).

### 1a. English / Kiswahili

| | Choice | Reasoning |
|---|---|---|
| English | `openai/whisper-large-v3-turbo`, stock | Already strong on Kenyan-accented English. Also the shared backbone for everything else, so it must be present regardless. |
| Kiswahili | Paza `<\|sw\|>` for the baseline | Fine-tuned on Common Voice Swahili and Kenya-focused. |
| Kiswahili challengers | `openchs/asr-whisper-largev3-finetuned`, `Sunbird/asr-whisper-51-african-languages` | Bake-off in Step 1 (task 1.8). Let the numbers decide rather than the reputation. |

They are one pack for the POC, split into two as soon as their training data diverges —
which happens the moment Sheng data arrives.

**Sheng** is not a routing target. It is a Kiswahili-matrix urban register, so the
router will call it `swh`, which is correct. It is handled as a *text layer* on the
Kiswahili branch: lexicon + normaliser flags the spans, translation consumes the gloss.
No public Sheng ASR corpus exists; collection is a Phase 2 budget line (§4.3).

### 1b. Kikuyu

- **Baseline:** Paza `<|kik|>`.
- **Step 2 target:** independently trained `kik` adapter on `openai/whisper-large-v3-turbo`,
  Kikuyu data only.
- Orthography: `ĩ`/`ũ` preserved everywhere, enforced at the normaliser (`preserve_diacritics`).

### 1c. Dholuo

- **Baseline:** Paza `<|luo|>`.
- **Step 2 target:** independently trained `luo` adapter, Dholuo data only.
- Dholuo tone is unwritten — a genuine ceiling on achievable WER, documented in the
  evaluation rather than treated as a model defect. ATR/vowel-harmony spelling varies by
  source, so `normalise_atr` runs before scoring or WER measures spelling, not recognition.

### 1d. How "separate models" is implemented

Per-language **LoRA adapters over a shared frozen backbone**. Independently trained,
independently evaluated, independently versioned and rolled back — and mergeable to
standalone weights at any time. At 44 languages this is ~1.6 GB base + ~50–150 MB per
language, against ~70 GB for 44 full checkpoints. It also fits this machine's 13.7 GB
of RAM, which three separate full checkpoints would not (§4.2).

The registry indirection (`config/language_packs.json` → `model_ref` → `models` table)
is what makes the Step 2 swap a config change. Preserve it.

---

## 2. Datasets — where they are, and when we touch them

**Not in Step 1.** Listed here so the Step 2 blocker is actioned early, because it sits
on the critical path of the whole programme.

| Dataset | Location | Kikuyu | Dholuo | Licence | Access |
|---|---|---|---|---|---|
| Africa Next Voices — Kenya | `MCAA1-MSU/anv_data_ke` | 754 h | 723 h | CC-BY-4.0 | **Gated** — needs an HF account + agreement to share name and institution |
| WAXAL | `google/WaxalNLP` | `kik_tts` | `luo_tts` | CC-BY-4.0 / CC-BY-SA-4.0 | Open |
| Common Voice Swahili | `mozilla-foundation/common_voice_*` | — | — | CC-0 | Terms acceptance |
| DigiGreen Kikuyu | Digital Green | yes | — | verify | Contact publisher |

Two things to note now rather than later:

- **WAXAL's Kikuyu and Luo are TTS configs** — single-speaker, clean. Excellent for
  lexicon, pronunciation and as a clean-condition test set; a poor primary training
  corpus, because one speaker invites overfitting. The 1,250 h ASR pool is other languages.
- **CC-BY-SA may extend to trained weights.** Legal review before any share-alike audio
  enters a training set. ANV alone is CC-BY-4.0 and sufficient, so the safe path costs
  nothing (§5.2).

---

## 3. Data sources — the six POC channels

Registered in `config/channels.json`. This is the authoritative list.

| Channel | Lang | Type | Source | Capture |
|---|---|---|---|---|
| Inooro TV | kik | TV | `youtube.com/watch?v=FKxVC3NHSp4` | yt-dlp, audio only |
| Kameme TV | kik | TV | `youtube.com/watch?v=3sreKsrmJVY` | yt-dlp, audio only |
| Ramogi TV | luo | TV | `youtube.com/watch?v=qfmHPApvQL8` | yt-dlp, audio only |
| Inooro FM | kik | FM | `radio.or.ke/inooro/` · alt `radio.ke/357-inooro-fm.html` | resolve → ffmpeg |
| Kameme FM | kik | FM | `onlineradiobox.com/ke/kamemefm/` · alt `radio.garden/listen/kameme-fm-101-1/xhDmAfyc` | resolve → ffmpeg |
| Ramogi FM | luo | FM | `radio.or.ke/ramogi/` · alt `radio.ke/434-ramogi-fm.html` | resolve → ffmpeg |

**Open task:** the three FM entries point at web *pages*, not stream endpoints. Resolve
the direct ICY/HLS endpoint once and store it in `resolved_stream_url`. Prefer the
broadcaster's own endpoint over an aggregator — aggregators add a re-hosting layer, are
less stable, and carry their own terms. One hour of work that removes a permanent source
of fragility.

YouTube live video IDs rotate, which is why `resolve_live` is set: the worker resolves
the current live endpoint each time rather than trusting the stored watch URL.

**Capacity:** Opus 24 kbps mono @ 16 kHz ≈ 10.8 MB per channel-hour → ~1.6 GB/day for
all six → ~47 GB/month against 288 GB free. About six months before pruning.

---

## 4. Filtering — ads, music, soaps

Vernacular FM is 30–50% music and advertising; TV carries ad blocks and imported drama.
Filtering is a **correctness control**, not an optimisation: Whisper hallucinates
fluently on music, and fabricated text about real people is worse than no text (§7).

| Layer | Mechanism | Status |
|---|---|---|
| 0 — Schedule | Daypart priority table in `channels.json`. Capture 24/7, transcribe ~5 h/day of high-value windows. | Implemented (registry) |
| 1 — Music/speech | Spectral flatness, ZCR, RMS stability + small classifier, 5–10 s hysteresis. Must distinguish "music only" from "speech over a music bed" — the latter is retained. | Step 1 task 1.5 |
| 2 — Ad fingerprinting | Chromaprint over retained segments; any 10–60 s block recurring 3+ times in 48 h is an ad/jingle. Self-populating library; all later airings drop free. | Schema ready (`fingerprints`) |
| 3 — Programme type | News / talk / advert / music / drama / religious / sports classification. | Step 1 task 1.5 |
| 4 — Hallucination | Post-ASR: logprob, no-speech prob, repetition ratio, n-gram loops, compression ratio, known-phrase list, unexpected language. | **Implemented** (`quarantine.py`) |

The fingerprint library pays for itself twice: the same data that suppresses ads *is* a
complete ad-airing log — which advertiser, which station, when, how often. Politically
valuable during an election cycle, and free.

**Filter accuracy is reported separately from WER.** Target: ≥90% ad/music airtime
removed, <2% genuine speech lost. A false negative here loses intelligence permanently,
so speech loss is the metric that matters most.

---

## 5. The application

Flask + vanilla JS, no build step, single node. VS Code is the dev environment;
`python run.py` serves on `127.0.0.1:5000`.

### Implemented

```
Sidebar ── Monitor:  Dashboard · Television · FM Radio
           Content:  Library · Quarantine
           System:   Evaluation · Language Packs
```

| Page | What it does |
|---|---|
| **Dashboard** | Stat tiles (channels live, hours captured, archive, utterances, quarantine rate, corrections), channel cards, baseline WER summary |
| **Television / FM Radio** | Channel cards with live/idle state, language chip with autonym, and the **routing prior bar** — visualising the candidate restriction that drives routing accuracy |
| **Channel detail** | Per-channel stats, routing profile, capture configuration (flags unresolved stream endpoints), recent segments |
| **Library** | Browse + full-text search across vernacular **and** English simultaneously, with `<mark>` hit highlighting; filters by channel, language, status |
| **Transcript** | Audio player, click-a-timestamp-to-seek, playback follows the transcript, dual-pane vernacular/English, routing-confidence bars, inline editing → corrections API |
| **Quarantine** | Review queue; release as genuine speech or confirm as noise. Both feed training data |
| **Evaluation** | Current WER/CER per pack, reference set inventory, run history with per-run deltas and normalisation signatures, ad-hoc scorer |
| **Language Packs** | Every registered pack with its model ref, Whisper token, diacritics, service tier; model artefact table |

Design notes: dark palette tuned for long sessions, all colour in `:root` custom
properties, station identity carried by a per-channel accent, EAT clock in the topbar,
UTF-8 verified end to end (the Kikuyu card renders `Gĩkũyũ` and the `ĩ ũ` chip).

### Still to build

- Waveform rendering in the transcript player (currently the native audio element).
- Upload page for analyst-supplied media (API and schema support it; `source_type='user_upload'`).
- Watchlists, saved queries and the report builder.
- Ingest monitor with the gap register surfaced (schema exists: `capture_gaps`).

---

## 6. How we monitor WER

Detailed in §9. The mechanics, as built:

### The rules

1. **Reference sets are frozen.** `eval/references/*.jsonl` is never trained on, tuned
   against, or edited to flatter a model. `evaluation.is_reference_audio()` is the guard.
2. **Normalisation travels with the score.** Every recorded run stores its normalisation
   signature. Two runs with different signatures are marked *incomparable* rather than
   silently compared — an unstated normalisation policy can move WER by ten points.
3. **Corpus-level WER**, not the mean of per-utterance WERs. The latter over-weights
   short utterances and is the commonest way to publish an irreproducible number.
4. **Never one aggregate.** Always per language, per station, per daypart.
5. **NFC first.** `ĩ` must be one codepoint, not `i` + combining tilde, or every Kikuyu
   word with a diacritic scores as an error for cosmetic reasons.

### The mechanics

```powershell
python scripts/evaluate.py --pack kik --set bcast_kik      # score + record
python scripts/evaluate.py --pack luo --set bcast_luo
python scripts/evaluate.py --pack kik --set bcast_kik --dry-run   # list items only
```

Each run prints per-item WER, an overall line with S/D/I breakdown and realtime factor,
a per-channel × per-daypart breakdown, and a **regression verdict** against the previous
run of the same pack — `improved` / `regressed` / `unchanged` / `incomparable`. It then
records to `eval_runs`, which is what the Evaluation page reads.

### Metrics tracked

| Metric | Where |
|---|---|
| WER, CER per pack / station / daypart | `eval_runs`, Evaluation page |
| Routing accuracy + confusion matrix | reported separately — a misrouted segment is a *routing* failure with a catastrophic WER, and fixing the wrong component wastes weeks |
| Filter recall, speech-loss rate | filter evaluation, reported separately |
| Quarantine rate | `quarantine.summarise()`, dashboard tile |
| Realtime factor | recorded per run — the capacity planning number |
| Translation adequacy | human-rated sample; automatic MT metrics are unreliable for these pairs |

### Reference sets needed

| Set | Content | Size |
|---|---|---|
| `bcast_kik` | Inooro + Kameme, TV and FM, across dayparts | 2 h transcribed, 2 h language-labelled |
| `bcast_luo` | Ramogi TV and FM, across dayparts | 2 h transcribed, 2 h language-labelled |
| `bcast_swh_eng` | Kiswahili/English segments from all six | 1 h transcribed |
| `clean_kik` / `clean_luo` | Held-out ANV test split + WAXAL TTS | publisher-defined |

**This is the one unavoidable human task in Step 1.** Four to six hours of expert
native-speaker transcription per language — a small, bounded engagement, but everything
measurable waits on it. Commission it now.

---

## 6b. Measured reality — 2026-09-23

Three findings from the first real run that change assumptions in the plan.

### Throughput is close to plan, but only if measured properly

Paza large-v3-turbo int8 measures **RTF 0.17–0.21x on realistic 60 s segments** —
**4.2–5.0 audio-hours per 24 h**, against the 5 the plan assumed and the 144 the six
channels produce. Earlier 20-second test clips suggested 2–3 h/day; that was fixed
per-call overhead dominating a short clip, not the steady-state rate. Always benchmark
at production segment length.

Raising thread count made it *slower* — 2 P-cores + 8 weak E-cores. Keep 8 threads.

**Do not disable word timestamps to go faster.** It is ~4x quicker and silently corrupts
segment end times (a 29 s utterance returned as `0.0-0.8s`). The text still looks correct,
so the damage surfaces later — in click-to-seek, the playback highlight, reference-set
alignment, and the duration-weighted scoring in routing arbitration.

Routing adds ~30–40 s per 60 s segment; include it in capacity planning.

### Paza's added languages are invisible to standard language detection

Both faster-whisper's `detect_language` and CTranslate2's own return **exactly Whisper's
original 100 language tokens**. `<|kik|>` and `<|luo|>` are absent, so routing built on
them scores Kikuyu at 0.0 forever.

Fixed by scoring language tokens **directly through the decoder** (`_score_language_tokens`
in `routing.py`): encode once, then a short forced-language probe decode per candidate.
Measured on Inooro FM audio — kik -0.10, luo -0.10, sw -0.64, en -0.88.

Live results: Inooro FM -> kik (0.95), Ramogi FM -> luo (0.95), Kameme FM -> kik (0.93).

Note kik and luo score almost identically. That does not matter: candidate restriction
(R1) means a station never has to separate kik from luo, only vernacular from
Kiswahili/English — and that margin is decisive. It is a direct vindication of the
prior-plus-restriction design.

### A shared multilingual checkpoint leaks languages into each other

Decoding Ramogi FM audio with `<|luo|>` produced text containing Kikuyu orthography
(`kũgerera`, `mũthemba`, `mũtano`) — Paza bleeding one training language into another.
The quarantine filter caught it (avg_logprob -3.94, well past the -1.1 threshold).

This is empirical support for the per-language-model decision in §4.1: a shared
checkpoint does not merely dilute capacity, it actively contaminates output.

---

## 7. Step 1 task board

| # | Task | Status |
|---|---|---|
| 1.1 | Python 3.11 venv, inference stack | **done** |
| 1.2 | Download + convert Paza and Whisper-turbo to CT2 int8 | in progress |
| 1.3 | Capture agent with reconnect, gap logging, provenance | **done** (`capture_worker.py`) |
| 1.4 | Immutable store + schema + SHA-256 + audit log | **done** (`db.py`) |
| 1.5 | Content filter layers 0–3 | layer 0 done; 1–3 outstanding |
| 1.6 | Routing chain R0–R4 | **done** (`routing.py`) |
| 1.7 | Commission expert transcription of reference sets | **blocked — human** |
| 1.8 | Model bake-off on the reference sets | waits on 1.2 + 1.7 |
| 1.9 | UI: monitor, library, transcript, corrections | **done** |
| 1.10 | Translation + enrichment + report builder | outstanding |

### Exit criteria

- Published baseline WER table, per station and daypart.
- Measured routing accuracy on a hand-labelled set.
- Measured filter recall and speech-loss rate.
- 14 days of continuous capture with a completeness report, <2% loss.
- An analyst can go from a topic to a cited report in the UI in under five minutes.

---

## 8. The three blockers worth chasing now

1. **Hugging Face account + Africa Next Voices gate.** Minutes of work, on the critical
   path of Step 2 and therefore of the programme.
2. **Cloud account with billing** for the Step 2 GPU training node. LoRA on a 24 GB L4
   is hours, not days — but it needs an account.
3. **Expert transcription** of the broadcast reference sets. Everything measurable waits
   on this, and it cannot be parallelised away.

Nothing else blocks progress: capture, filtering, routing, the UI and the evaluation
harness all run on this machine today.
