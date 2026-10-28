# OSINT Vernacular Broadcast Monitoring

Captures Kenyan broadcast channels, filters out non-intelligence content, transcribes
speech with **per-language ASR models**, translates vernacular to English, and serves it
to analysts through a web UI.

**Phase 1 — proof of concept: Kikuyu and Luo.** Designed from the outset to scale to
44+ Kenyan languages; these two are the first entries in a registry, not special cases.

- Programme plan: `OSINT_POC_Kikuyu_Luo_Plan.docx`
- Build plan: `docs/POC_PLAN.md`
- Working agreement for Claude Code: `CLAUDE.md`

---

## Quick start

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/download_models.py --list    # what is registered, what is present
python scripts/download_models.py           # fetch + convert to CT2 int8 (~4 GB, once)
python run.py                               # http://127.0.0.1:5000
```

Then, once models are in place:

```powershell
python scripts/capture_worker.py --once --duration 120   # one test segment per channel
python scripts/transcribe_worker.py --limit 5            # drain the queue
python scripts/evaluate.py --pack kik --set bcast_kik    # score + record WER
```

---

## Channels

| Channel | Language | Medium | Broadcaster |
|---|---|---|---|
| Inooro TV / Inooro FM | Kikuyu | TV / FM | Royal Media Services |
| Kameme TV / Kameme FM | Kikuyu | TV / FM | Mediamax Network |
| Ramogi TV / Ramogi FM | Dholuo | TV / FM | Royal Media Services |

Defined in `config/channels.json`. Adding a channel is a config change, never a code change.

## Language packs

| Pack | Model | Token | Role |
|---|---|---|---|
| `eng` | `openai/whisper-large-v3-turbo` | `en` | English |
| `swh` | `microsoft/paza-whisper-large-v3-turbo` | `sw` | Kiswahili, carries Sheng |
| `kik` | `microsoft/paza-whisper-large-v3-turbo` | `kik` | Kikuyu — **baseline** |
| `luo` | `microsoft/paza-whisper-large-v3-turbo` | `luo` | Dholuo — **baseline** |

Paza is the baseline to beat. Step 2 replaces `kik` and `luo` with independently trained
per-language adapters. Defined in `config/language_packs.json`.

---

## Layout

```
config/     channel + language pack registries — the only place languages are named
app/        Flask application
  services/ registry · capture · asr · routing · quarantine · evaluation
  routes/   views (HTML) · api (JSON)
scripts/    download_models · capture_worker · transcribe_worker · evaluate
eval/       frozen reference sets — NEVER trained on
data/       captured audio + SQLite (gitignored)
models/     CT2 model artefacts (gitignored)
```

## Constraints worth knowing before you change anything

- **No GPU.** i7-1355U, 13.7 GB RAM. Inference runs here; training does not.
  Roughly 0.5–1.0× realtime per stream, so capture is continuous but transcription is
  rationed by daypart priority (~5 audio-hours/day against the 144 produced).
- **UTF-8 end to end.** Kikuyu `ĩ`/`ũ` must survive every layer. Windows consoles default
  to cp1252 — scripts call `Config.enable_utf8_console()` for this reason.
- **UTC in storage, EAT in the UI.**
- **Source audio is immutable**, SHA-256 recorded at capture.
- **Reference sets are frozen** and never trained on.
