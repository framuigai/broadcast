# Reference Sets

Frozen, expert-transcribed audio used to measure model performance. See
`docs/POC_PLAN.md` §6 and the programme plan §9.

## Sets required for Step 1

| Set | Content | Target size |
|---|---|---|
| `bcast_kik.jsonl` | Inooro + Kameme, TV and FM, across dayparts | 2 h transcribed |
| `bcast_luo.jsonl` | Ramogi TV and FM, across dayparts | 2 h transcribed |
| `bcast_swh_eng.jsonl` | Kiswahili/English segments from all six channels | 1 h transcribed |
| `clean_kik.jsonl` / `clean_luo.jsonl` | Held-out ANV test split + WAXAL TTS | publisher-defined |

## Producing one

1. Capture real broadcast audio across dayparts — not just the easy studio bulletin.
   A set made only of clean anchor reads produces a flattering number that collapses
   the first time a call-in show is processed.
2. Commission expert native-speaker transcription. Four to six hours per language.
3. Write one JSONL row per utterance following `_TEMPLATE.jsonl`.
4. Commit it. From that point it is frozen.

## Scoring against one

```powershell
python scripts/evaluate.py --pack kik --set bcast_kik
```

The run prints per-item WER, an overall figure with S/D/I breakdown and realtime factor,
a per-channel × per-daypart table, and a regression verdict against the previous run of
the same pack. It records to `eval_runs`, which the Evaluation page reads.

## Why the breakdown matters more than the headline

Studio bulletin WER and telephone call-in WER differ enormously. A single aggregate
figure conceals both, and it is the call-in number that determines whether the system is
actually usable on the content that carries the most sentiment signal.
