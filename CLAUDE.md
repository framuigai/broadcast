# CLAUDE.md — OSINT Vernacular Broadcast Monitoring

Guidance for Claude Code when working in this repository.

---

## 1. What this project is

A media-monitoring system that captures Kenyan broadcast channels, filters out
non-intelligence content, transcribes speech with **per-language ASR models**,
translates vernacular to English, and serves it to analysts through a Flask web UI.

**Phase 1 (current): proof of concept — Kikuyu and Luo only.**
The system is designed from the outset to scale to **44+ Kenyan languages**.
Kikuyu and Luo are the first two entries in a registry, *not* special cases.

Authoritative plan: `OSINT_POC_Kikuyu_Luo_Plan.docx` (sections referenced as §N below).

### The pipeline

```
capture audio  ->  transcribe (Paza / Whisper)  ->  translate (NLLB)
  immutable         original language, with        transcript TEXT -> English,
  source file       routing + quarantine checks    with its own quality checks
```

- **Three separate stages, each with its own model.** ASR only turns speech into text
  in the language spoken. Translation only turns that text into English — it never
  sees audio. Do **not** use Whisper's `task="translate"` (audio -> English): the English
  would not be a translation of the transcript the analyst sees, could not follow an
  analyst's correction, and would tie translation to the ASR model that Step 2 replaces.
- **It runs in the background.** `python run.py` starts one worker thread
  (`app/services/pipeline.py`) that drains the queue: anything captured — from the Live
  page or by `capture_worker.py` — is transcribed and then translated with no terminal
  involvement. The queue is the `segments` table, so restarts lose nothing.
- **Translation is enabled per language in the registry**, via
  `packs[].translation.model_ref`. Currently enabled: `kik`, `luo` and `swh` — every
  language any station routes to, so all TV and radio audio yields a transcript and
  English. `eng` needs no translation. Enabling a new language is adding the same
  `model_ref` line; existing transcripts are back-filled on the next worker start.

### Current stage: Step 1 — baseline

We are measuring what **off-the-shelf models** achieve on real broadcast audio.
**No training, no fine-tuning, and no dataset downloads at this stage.**
Africa Next Voices and WAXAL come in Step 2. Do not add dataset-download code yet.

---

## 2. The rules that matter most

These are the ones that are expensive to get wrong. Read them before writing code.

### 2.1 Never hard-code a language

Kikuyu and Luo must never appear as literals in pipeline code. Everything flows from
the **Language Pack registry** (`config/language_packs.json`) and the **channel registry**
(`config/channels.json`).

```python
# WRONG — will have to be rewritten 42 more times
if station == "inooro":
    model = load_model("kikuyu")

# RIGHT
pack = registry.pack_for(segment.routed_language)
model = model_pool.get(pack.asr_model_ref)
```

If a feature cannot be expressed as a registry entry, that is a design bug, not a
reason to hard-code. Raise it before implementing.

### 2.2 UTF-8 end to end, always

Kikuyu uses **ĩ** and **ũ**. These must survive capture → database → model →
translation → UI → DOCX export. This is the most common silent failure in
multilingual systems.

- Open every file with an explicit `encoding="utf-8"`.
- SQLite connections: text factory left at default (Python 3 handles UTF-8 natively).
- Every HTML template declares `<meta charset="utf-8">`.
- Test fixture string, use it liberally: `Nĩ kũrĩ ũhoro mwega` (Kikuyu), `Ber ahinya` (Dholuo).

### 2.3 Source audio is immutable

Captured audio is **write-once**. No stage ever modifies a source file. Derived audio
(resampled, segmented, denoised) goes to separate paths under `data/derived/`.
Every source file gets a SHA-256 recorded at capture time.

- **One capture implementation:** `app/services/capture.py::capture_segment`. The Live
  page's Record button and `scripts/capture_worker.py` both call it; the script is
  only a scheduling loop. Never copy capture logic into a script — add a parameter.
  (Until 2026-09-29 the worker had its own copy, `capture/0.1`, which had drifted.)
- **Output names are reserved atomically** (`reserve_output_path`, `O_EXCL`) before
  ffmpeg starts, so two captures of one channel in the same second get `_000` and
  `_001`. Cleanup only ever deletes the file its own capture reserved. The previous
  fixed `_000` name let a same-second collision delete the other capture's audio.
- `segments.software_version` records which capture code wrote each file
  (`capture/0.3` is current).

### 2.4 Never publish an unverified transcript

Whisper-family models hallucinate fluently on music and silence. Any segment failing the
quarantine checks (`app/services/quarantine.py`) is flagged and withheld from reports —
never silently published. Every reported item links back to its audio timestamp.

The same rule applies to translation, which fails *fluently*: English that fails
`quarantine.check_translation()` (copied source, implausible length, repetition loop)
is stored with `translation_status='flagged'` and withheld from documents and exports.
Quarantined transcripts are not translated at all (`'skipped'`) — translating a
hallucination only makes it more convincing. Analyst-written English
(`'corrected'`) is authoritative and is never overwritten by re-translation.

### 2.5 UTC in storage, EAT in the UI

Store every timestamp as UTC (`...Z`). Convert to Africa/Nairobi (UTC+3) only at render
time. Broadcast analysis is time-sensitive and off-by-three-hours bugs are easy to make
and hard to notice.

### 2.6 Reference sets are sacred

Anything under `eval/references/` is **never** used for training, tuning, augmentation,
or prompt examples. It exists solely to produce comparable WER numbers across runs.

### 2.7 Standing architecture decisions (decided 2026-09-29 — do not re-open)

These were decided by the project owner after discussion. Build toward them; do not
re-propose alternatives unless a measurement shows a guardrail below has been hit.
Both are Step 2 work — they fix the *direction*, not permission to add training code now.

**Decision 1 — ASR: separate weights per vernacular.**
Each vernacular gets its own LoRA adapter on a shared Whisper base: separately trained,
separately versioned, separately rolled back. Training Meru must never be able to
regress Kikuyu. Each pack's `asr.model_ref` points at its own model.

Guardrails:
- **Merged for deployment.** CTranslate2/faster-whisper cannot apply LoRA at runtime,
  so each adapter is merged into a full model before int8 conversion (~1.6 GB per
  language on disk; switching language = a full model load). Adapters are compact as
  training/versioning artefacts, not as deployed ones. Budget disk and load time for it.
- **Routing stays on a shared multilingual model.** `routing.py` probes every candidate
  language per segment. Per-language models there would load/evict a model per
  candidate per segment. Route with the shared base (today: Paza); use the
  per-language adapter only for the transcription decode. Resident set: routing model
  + one adapter model + translation model (~4.6 GB) — check the RAM budget before
  changing `ASR_MAX_LOADED_MODELS`.
- **Train on code-switched speech.** Broadcast vernacular is full of Kiswahili and
  English. An adapter trained on "pure" vernacular will drop or mangle them. Each
  adapter's training mix must include code-switched speech.
- **Family-sharing is allowed when data is thin.** Separate per language is the
  default. A language with only a few hours of data may share an adapter with close
  relatives (e.g. Kikuyu/Embu/Meru) or be initialised from a relative's adapter — but
  only if measured WER beats a separate adapter. Two packs sharing one `model_ref` is
  already valid registry config.

**Decision 2 — Translation: one shared multilingual model.**
All languages translate through one shared model (currently
`nllb-200-distilled-1.3B`), extended language by language. Related languages share
what they learn, which matters most for the low-resource ones. Each pack names it via
`translation.model_ref`.

Guardrails:
- **"Shared model", not "NLLB forever".** NLLB-200 is CC-BY-NC-4.0 and fine-tuned
  derivatives inherit it. The licence must be decided before Step 2 invests in
  fine-tuning; a permissively licensed shared model is an acceptable swap.
- **Per-language regression eval on every retrain.** This decision trades away the
  isolation Decision 1 buys: adding language #12 retrains the model Kikuyu uses. Every
  retrain reports quality for *every* enabled language, before/after, not an average.
- **Split by family only on measured interference.** If adding languages measurably
  degrades existing ones, split into per-family translation models. That is a
  `model_ref` change, not a code change — keep it that way.
- **Coverage and quality are not implied.** NLLB covers ~6 of the 44+ languages today;
  the rest need parallel text we do not yet have. First measurement on real Kikuyu
  broadcast (2026-09-29): 4 of 10 lines hallucinated (mostly Bible verses). This decision
  is about architecture; it says nothing about the output being good enough.

---

## 3. Environment

| Item | Value |
|---|---|
| Python | 3.11 in `.venv/` (NOT the system 3.13 — wheel coverage) |
| Activate | `.\.venv\Scripts\Activate.ps1` (PowerShell) |
| Shell | Windows 11, PowerShell primary; Git Bash available |
| Hardware | i7-1355U, 13.7 GB RAM, **no CUDA GPU** |
| Inference | `faster-whisper` / CTranslate2, **int8** — never plain `transformers` for inference |
| Models | `models/` (gitignored, ~4 GB) |
| ffmpeg | Already on PATH (v9.0) |

### Hardware consequences — do not design around a GPU

**Measured, 2026-09-23** (Paza large-v3-turbo, int8, real Inooro FM audio, idle machine):

| Config | RTF | audio-hours per 24 h |
|---|---|---|
| beam 5, word timestamps, 8 threads, **60 s segment** | **0.17–0.21x** | **4.2–5.0 h** |
| same config, 20 s clip | 0.08–0.12x | 2.0–3.0 h |
| beam 1, word timestamps | 0.105x | 2.5 h |
| beam 1, **no** word timestamps | 0.23–0.83x | — **do not use, see below** |

Two things that are easy to get wrong here:

- **Measure on realistic segment lengths.** Short clips are dominated by fixed
  per-call overhead and understate throughput by ~2x. The 5-minute production
  segments will do better than any of these numbers.
- **More threads is slower.** This CPU is 2 P-cores + 8 E-cores; spreading a
  latency-bound decode over weak E-cores costs more than it gains. Leave
  `ASR_CPU_THREADS` at 8.

### Never disable word timestamps

`word_timestamps=False` is ~4x faster and **silently corrupts segment end times** —
measured on real audio, a 29-second utterance came back as `0.0-0.8s`. The text is
fine, so this does not look like a bug until something downstream breaks: click-to-seek,
the playback highlight, reference-set alignment, and the duration-weighted scoring in
`routing.arbitrate()` all depend on that timeline. `ASR_WORD_TIMESTAMPS` defaults to on
and should stay on.

Language routing adds ~27–38 s per 60 s segment (one encode + one short probe decode
per candidate).

- Six channels × 24 h = 144 audio-hours/day against ~4-5 transcribable — a ~30x
  shortfall, which is roughly what the plan originally assumed after all.
- Therefore: **capture continuously, transcribe selectively.** The plan's ~5
  audio-hours/day target is achievable on realistic segment lengths, but only with the
  machine doing nothing else. Budget for less in practice.
- Routing costs ~30-40 s per segment on top, so include it in capacity planning.
- Never load more than one large ASR model at a time without checking the RAM budget.
  Model pool is LRU with a hard cap (`ASR_MAX_LOADED_MODELS`, default 1). The
  translation model has its own single slot, so the resident set is one ASR model
  (~1.6 GB) + one translation model (~1.4 GB).
- **One pipeline per machine.** The web app's background worker and
  `scripts/transcribe_worker.py` share a file lock (`data/pipeline.lock`); the second
  one refuses to start. Two would double the model RAM. To use the CLI worker, start the
  app with `OSINT_BACKGROUND_PIPELINE=0`.
- Stages never run concurrently (one worker, one segment at a time), so ASR and
  translation do not compete for the CPU.
- Training belongs on a rented GPU, not here. Do not add training code to this repo yet.

---

## 4. Layout

```
osint/
├── CLAUDE.md                  ← this file
├── OSINT_POC_Kikuyu_Luo_Plan.docx  ← authoritative plan
├── run.py                     ← dev entrypoint
├── config/
│   ├── channels.json          ← the 6 POC channels; single source of truth
│   └── language_packs.json    ← per-language model + routing + orthography config
├── app/
│   ├── __init__.py            ← create_app() factory
│   ├── config.py              ← settings from env, sane defaults
│   ├── db.py                  ← SQLite schema + connection helpers
│   ├── routes/
│   │   ├── views.py           ← HTML pages
│   │   └── api.py             ← JSON API consumed by the frontend
│   ├── services/
│   │   ├── registry.py        ← channel + language pack loading
│   │   ├── capture.py         ← THE capture implementation (ffmpeg / yt-dlp), shared by UI + worker
│   │   ├── pipeline.py        ← queue + background worker: transcribe -> translate
│   │   ├── asr.py             ← model pool + transcription
│   │   ├── translation.py     ← text-to-text translation (NLLB, CT2 int8)
│   │   ├── routing.py         ← language routing (§8)
│   │   ├── quarantine.py      ← hallucination filters (§7.5)
│   │   └── evaluation.py      ← WER/CER scoring (§9)
│   ├── static/{css,js}/
│   └── templates/
├── scripts/                   ← operational CLI entrypoints
├── data/                      ← gitignored: audio, db, derived
└── eval/references/           ← frozen reference sets — NEVER train on these
```

---

## 5. Conventions

### Python
- Standard library first, then third-party, then local. No wildcard imports.
- Type hints on every function signature that crosses a module boundary.
- Services are plain functions and small classes — no framework magic, no global state
  except the explicitly managed model pool.
- Raise specific exceptions; never `except:` bare. Log with context, not just the message.

### Database
- SQLite via `app/db.py` helpers. Always parameterised queries.
- Schema changes go in `app/db.py` `SCHEMA` with a migration note in the commit.
- Language codes are **ISO 639-3**: `kik`, `luo`, `swh`, `eng`. Not `ki`, not `sw`,
  not `kikuyu`. The one exception is Whisper's decoder tokens, which use its own
  convention (`sw` for Swahili) — convert at the model boundary only, in `asr.py`.

### Frontend
- No build step. Vanilla JS + a single stylesheet. Do not add npm, React, Tailwind CLI
  or a bundler without asking — the deployment target is a single-node local app.
- CSS custom properties in `:root` for all colours. Never hard-code a hex outside `:root`.
- Fetch from `/api/*`; render client-side. Templates provide the shell, not the data.

### Naming
Audio files: `{station_id}/{YYYY}/{MM}/{DD}/{station_id}_{YYYYMMDD}T{HHMMSS}Z_{seq}.opus`

---

## 6. Running it

```powershell
.\.venv\Scripts\Activate.ps1
python run.py                              # UI on http://127.0.0.1:5000 + background pipeline
python scripts/fetch_model.py --all        # fetch/convert ASR + translation models (once)
python scripts/fetch_model.py --status     # download progress
python scripts/capture_worker.py           # optional: continuous stream capture
python scripts/evaluate.py --pack kik      # WER against the frozen reference set
python scripts/selftest.py                 # no models needed: WER maths, orthography, all quality checks
python scripts/e2e_test.py                 # LIVE: record every station, verify transcript + English
```

`e2e_test.py` needs `python run.py` running. A 60 s clip takes **~4 minutes** through
the pipeline (≈30 s routing + ≈3.5 min ASR at ~0.25x realtime + ~10 s translation;
measured 2026-09-29), one segment at a time — so six stations take ~25 min. Do not
edit files under `app/` while it runs: a reload drops the in-memory recording jobs it
polls (`scripts/`, `data/`, `models/`, `eval/`, `docs/` are excluded from the reloader).
TV channels are only live on YouTube part of the day; when one is off air, use
`--segments <ids>` to verify already-captured audio. Run it after any pipeline,
capture, translation or registry change.

**Reading the RTF numbers in §3:** RTF here is *audio seconds per wall-clock second*.
0.2x means one minute of audio takes about five minutes to transcribe — not twelve
seconds. The "audio-hours per 24 h" column is the unambiguous one.

Day to day, only `python run.py` is needed: record from the **Live** page and the clip
is transcribed and translated in the background (status strip on the same page;
`GET /api/pipeline/status`). `scripts/transcribe_worker.py` runs the same pipeline from
the command line for batch runs or debugging (`--translate-only` re-drains English),
and only with the app's worker off.

The dev server's reloader restarts the process on every `.py` edit, which aborts an
in-flight transcription. That is safe — the segment is requeued on restart — but it
costs the work done so far.

### Use `fetch_model.py`, not `download_models.py`

`huggingface_hub` 1.32.0 names its partial files with a **per-run suffix**
(`<blob>.<random>.incomplete`), so an interrupted download does **not** resume — the
next run starts from zero. Combined with its Xet transfer path this measured
~45 KB/s here, making a 1.6 GB model effectively undownloadable.

`scripts/fetch_model.py` fetches straight from the repo resolve URL with `curl -C -`
Range resume into `models/_src/<model_ref>/`, then converts. Measured **~2-6 MB/s** on
the same connection — a ~100x difference, and genuinely resumable.

Do not "simplify" this back to `snapshot_download`.

---

## 7. What not to do

- **Do not download datasets.** Step 1 is baseline-only. ANV is gated and needs a human
  to accept terms; WAXAL is Step 2. Adding dataset code now is premature.
- **Do not add training/fine-tuning code.** Step 2, on a rented GPU, in a separate repo path.
- **Do not use `transformers` for inference.** CTranslate2 int8 is several times faster on
  this CPU. `transformers` is only for the one-off model conversion in `scripts/`.
  The NLLB tokenizer is loaded with the `tokenizers` library from `tokenizer.json`.
- **Do not translate from audio** (Whisper `task="translate"`), and do not transcribe or
  translate inline in a request handler or a per-recording thread. Capture only
  enqueues; the pipeline does the rest.
- **Do not commit** anything in `data/`, `models/`, or `.venv/`.
- **Do not "improve" WER by touching the reference sets.** If a reference transcript is
  genuinely wrong, fix it, record the correction in `eval/CHANGELOG.md`, and re-baseline
  *every* previously reported number. Never silently edit.
- **Do not report an aggregate WER.** Always per-language, per-station, per-daypart (§9.1).
  An average hides the difference between studio bulletin and telephone call-in.

---

## 8. Definition of done for a pipeline change

1. Works on a real captured segment, not a synthetic fixture.
2. Kikuyu diacritics verified intact at the UI layer.
3. New behaviour is registry-driven, not language-specific.
4. If it affects transcription output: WER re-run on the reference set, before/after
   numbers stated in the commit message.
5. Timestamps stored UTC, displayed EAT.

---

## 9. Model reference

| Pack | ASR model | Whisper token | Translation (-> English) | NLLB token | Role |
|---|---|---|---|---|---|
| `eng` | `openai/whisper-large-v3-turbo` | `en` | — (target) | `eng_Latn` | English baseline |
| `swh` | `microsoft/paza-whisper-large-v3-turbo` | `sw` | `facebook/nllb-200-distilled-1.3B` | `swh_Latn` | Kiswahili + Sheng carrier |
| `kik` | `microsoft/paza-whisper-large-v3-turbo` | `kik` | `facebook/nllb-200-distilled-1.3B` | `kik_Latn` | Kikuyu **baseline** |
| `luo` | `microsoft/paza-whisper-large-v3-turbo` | `luo` | `facebook/nllb-200-distilled-1.3B` | `luo_Latn` | Dholuo **baseline** |

Paza is the **baseline to beat**, not the final product (§4.1). In Step 2 the `kik` and
`luo` packs are replaced by independently trained per-language adapters. The registry
indirection is what makes that swap a config change rather than a code change — preserve it.

NLLB is likewise a **baseline**: Kikuyu and Luo are among its weaker languages, and it
was not trained on code-switched (vernacular + Kiswahili + English) sentences. Language
tokens are the pack's `translation.source_token`; the target comes from the `eng` pack.
Translation stays a separate model from ASR so either can be replaced independently.

**Licence: NLLB-200 is CC-BY-NC-4.0 (non-commercial), and fine-tuned derivatives
inherit it.** Acceptable for the POC; must be decided before Step 2 invests in
fine-tuning it. `fetch_model.py` fetches it via the spec's `fetch_files` (the repo ships
only `pytorch_model.bin`, ~5.5 GB).
