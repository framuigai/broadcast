# Expert Transcription Brief — Kikuyu & Dholuo Reference Sets

**What this is:** the instructions you hand to the native-speaker transcribers who
produce the reference sets. Everything measurable in Step 1 depends on this work.

---

## 1. What "commission the expert transcription" means

It means **paying two or three fluent native speakers per language to listen to our
captured broadcast audio and type out exactly what was said** — then using their text as
the yardstick the machine is measured against.

That is all it is. It is not data collection, it is not recording anyone in a studio, and
it is not the paid respondent programme from the original concept note (which is deferred
to Phase 4). No new audio is created: we transcribe the Inooro, Kameme and Ramogi audio
we have already captured.

### Why it cannot be skipped

WER — Word Error Rate — is *machine output* compared against *known-correct text*. Without
the known-correct text there is no error rate, and therefore:

- no way to state what Paza actually achieves on Kenyan broadcast audio;
- no way to prove that a Step 2 trained model is better than the Step 1 baseline;
- no way to tell a routing failure apart from a transcription failure;
- no defensible number to put in front of a decision-maker.

### Why we cannot produce it ourselves

The transcript has to be right. A model cannot check its own homework — using Whisper
output as the reference would simply measure how consistently Whisper repeats itself, and
would bake its errors in as "truth". It has to be a human who actually speaks Kikuyu or
Dholuo.

### Size and shape of the engagement

| Item | Amount |
|---|---|
| Audio to transcribe | 4–6 hours per language (Kikuyu, Dholuo), 1 hour Kiswahili/English |
| People | 2–3 per language: one primary transcriber, one reviewer |
| Elapsed time | Roughly 1–2 weeks |
| Effort ratio | Broadcast audio runs 5–10× realtime to transcribe carefully. 5 hours of audio is ~30–50 hours of work |
| Skills | Native fluency; literate in the standard orthography; comfortable with headphones and a text editor |
| Where to find them | University language departments (Kikuyu: Dedan Kimathi, Karatina; Dholuo: Maseno), the broadcasters' own staff, professional transcription agencies, KenCorpus-affiliated researchers |

It is a small, bounded, one-off piece of work. It is on the critical path purely because
nothing downstream can be measured until it exists.

---

## 2. What we need transcribed

Selected from real captured broadcast audio, **spread deliberately across conditions**.
A reference set made only of clean studio anchor reads produces a flattering number that
collapses the first time a call-in show is processed.

| Set | Source channels | Audio | Must include |
|---|---|---|---|
| `bcast_kik` | Inooro FM, Inooro TV, Kameme FM, Kameme TV | 2 h transcribed + 2 h language-labelled | News bulletin, morning drive, call-in, advertising break |
| `bcast_luo` | Ramogi FM, Ramogi TV | 2 h transcribed + 2 h language-labelled | Same spread |
| `bcast_swh_eng` | All six | 1 h | Kiswahili and English inserts on vernacular stations |

Target mix within each set: roughly 40% studio/news, 40% talk and call-in (including
telephone-quality callers), 20% other — advertising, music-bedded links, outside broadcast.

---

## 3. Rules for transcribers

### Transcribe what was said, not what should have been said
- Do not correct the speaker's grammar, repetitions or false starts.
- Do not summarise, tidy, or skip filler.
- If a passage is genuinely inaudible, mark it `[inaudible]` rather than guessing.
- If it is unclear but you have a reading, mark it `[unclear: your best guess]`.

### Orthography
- **Kikuyu:** standard orthography with **ĩ** and **ũ**. These are separate letters, not
  decorations — `kuria` and `kũria` are different words. Type them properly; do not
  substitute `i`/`u`. Save as UTF-8.
- **Dholuo:** standard orthography. Tone is not written — that is expected, and it is a
  documented limit on achievable accuracy, not something to compensate for.
- Numbers: write as spoken (`ishirini na tano`, not `25`).
- Punctuation: normal sentence punctuation. It is removed before scoring, so do not agonise.

### Code-switching — important
Kenyan broadcast constantly mixes languages. Transcribe **each word in its own language**
and spell it that way:

> `Nĩ kũrĩ budget allocation ya county`

English words in English spelling, Kiswahili in Kiswahili spelling, vernacular in
vernacular spelling. Do not translate the inserted words, and do not drop them.

### Sheng
Write phonetically using standard Kiswahili spelling, with English-origin words spelled as
English. Mark the span `[sheng: ...]` where you can. This is the least-documented part of
the work and the most valuable.

### Non-speech
Mark with explicit tokens rather than leaving a silent gap:
`[music]`, `[jingle]`, `[applause]`, `[laughter]`, `[phone]`, `[silence]`, `[crosstalk]`.
Labelled non-speech is what teaches the system not to hallucinate over it.

### Speakers
Label turns generically — `Anchor:`, `Caller 1:`, `Guest:`. Do **not** attempt to identify
real individuals by name.

---

## 4. Deliverable format

One line per utterance, JSONL, UTF-8, following `eval/references/_TEMPLATE.jsonl`:

```json
{"audio_path": "data/audio/inooro_fm/2026/09/23/inooro_fm_20260923T060000Z_000.opus",
 "reference": "Nĩ kũrĩ ũhoro mwega kuuma Nyeri ũmũthĩ",
 "language": "kik", "channel_id": "inooro_fm", "daypart": "Morning drive",
 "programme_type": "news", "notes": ""}
```

Transcribers who are not comfortable with JSONL can deliver plain text or a spreadsheet
with the columns *audio file · start time · end time · language · transcript · programme
type*; we convert it. Do not let the file format become a barrier to hiring the right
person.

A second reviewer checks each file against the audio before it is accepted.

---

## 5. Quality control

| Tier | Check |
|---|---|
| 1 — Automated | UTF-8 validity, diacritics present in Kikuyu, character-set violations, empty rows, tag well-formedness |
| 2 — Peer review | A second native speaker verifies the transcript against the audio |
| 3 — Expert audit | A senior linguist spot-audits each transcriber's output |

Measure inter-transcriber agreement on a shared 15-minute sample before the bulk work
starts. If two fluent speakers disagree significantly on the same audio, the orthography
convention needs settling first — otherwise that disagreement becomes noise in every WER
number produced afterwards.

---

## 6. Once delivered — the set is frozen

From acceptance onward the reference set is **immutable**:

- Never used for training, fine-tuning, augmentation or prompt examples (CLAUDE.md §2.6).
- Never edited to make a model look better.
- If a reference transcript is genuinely wrong: correct it, log it in `eval/CHANGELOG.md`,
  and **re-baseline every previously reported number** against the corrected set.

Then:

```powershell
python scripts/evaluate.py --pack kik --set bcast_kik
python scripts/evaluate.py --pack luo --set bcast_luo
```

which prints per-item WER, an overall figure with S/D/I breakdown and realtime factor, a
per-channel × per-daypart table, and a regression verdict — and records the run so the
Evaluation page can chart it.

That is the Step 1 baseline. Everything in Step 2 is measured against it.
