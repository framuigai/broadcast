"""JSON API consumed by the frontend."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

from app import db
from app.config import Config
from app.services import evaluation, pipeline, quarantine, registry

bp = Blueprint("api", __name__)


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
#  Registry
# --------------------------------------------------------------------------- #
@bp.get("/channels")
def channels():
    medium = request.args.get("medium")
    out = []
    for ch in registry.all_channels(medium=medium):
        stats = db.query_one(
            "SELECT COUNT(*) AS n, COALESCE(SUM(duration_seconds),0) AS secs,"
            "       MAX(started_at_utc) AS last_seen"
            "  FROM segments WHERE channel_id=?", (ch.id,)) or {}
        transcribed = db.query_one(
            "SELECT COUNT(*) AS n FROM segments WHERE channel_id=? AND status='transcribed'",
            (ch.id,)) or {}
        # Capture health: gaps are recorded explicitly, never silently absorbed,
        # so an operator can see at a glance which feeds are actually working.
        gap = db.query_one(
            "SELECT reason, started_at_utc FROM capture_gaps WHERE channel_id=?"
            " ORDER BY id DESC LIMIT 1", (ch.id,)) or {}
        n_gaps = (db.query_one(
            "SELECT COUNT(*) AS n FROM capture_gaps WHERE channel_id=?", (ch.id,)) or {}).get("n", 0)
        pack = registry.pack_for(ch.primary_language)
        out.append({
            "id": ch.id, "name": ch.name, "broadcaster": ch.broadcaster,
            "medium": ch.medium, "region": ch.region, "accent": ch.accent,
            "enabled": ch.enabled,
            "language": {"code": pack.code, "name": pack.name, "autonym": pack.autonym,
                         "colour": pack.colour},
            "candidates": ch.candidate_languages,
            "prior": ch.routing_prior,
            "capture_method": ch.capture.get("method"),
            "endpoint": ch.capture.get("resolved_stream_url") or ch.capture.get("url"),
            "endpoint_verified_utc": ch.capture.get("verified_utc"),
            "known_issue": ch.capture.get("known_issue"),
            "health": {
                "gaps": n_gaps,
                "last_gap_utc": gap.get("started_at_utc"),
                "last_gap_reason": (gap.get("reason") or "")[:240] or None,
                "state": ("ok" if stats.get("n") and not n_gaps
                          else "degraded" if stats.get("n")
                          else "failing" if n_gaps else "untested"),
            },
            "stats": {
                "segments": stats.get("n", 0),
                "hours": round((stats.get("secs") or 0) / 3600, 2),
                "transcribed": transcribed.get("n", 0),
                "last_seen_utc": stats.get("last_seen"),
            },
        })
    return jsonify(out)


@bp.post("/channels/<channel_id>/capture-url")
def set_capture_url(channel_id: str):
    """Rewrite a channel's capture URL in config/channels.json.

    Live stream URLs — YouTube's above all — go stale without warning, so the
    analyst needs to fix one without editing JSON by hand. The registry remains
    the single source of truth: this edits the file, then drops the cache.
    """
    payload = request.get_json(silent=True) or {}
    url = (payload.get("url") or "").strip()
    actor = payload.get("actor", "analyst")

    # Which key holds the capture source depends on the method: 'yt-dlp' channels
    # are captured from capture.url, 'stream' ones from capture.resolved_stream_url
    # (see capture.py). Editing the wrong key would write something nothing reads.
    field = payload.get("field", "url")
    index = payload.get("index", 0)
    if field not in {"url", "resolved_stream_url", "fallback_urls"}:
        return jsonify({"error": f"field {field!r} is not editable"}), 400

    if not url:
        return jsonify({"error": "url is required"}), 400
    if not url.startswith(("http://", "https://")):
        return jsonify({"error": "url must start with http:// or https://"}), 400

    with open(Config.CHANNELS_FILE, encoding="utf-8") as fh:
        raw = json.load(fh)

    entry = next((c for c in raw.get("channels", []) if c.get("id") == channel_id), None)
    if entry is None:
        return jsonify({"error": "not found"}), 404

    capture = entry.setdefault("capture", {})

    if field == "fallback_urls":
        fallbacks = capture.setdefault("fallback_urls", [])
        if not isinstance(index, int) or not 0 <= index < len(fallbacks):
            return jsonify({"error": f"no fallback at index {index}"}), 400
        previous = fallbacks[index]
        if previous == url:
            return jsonify({"ok": True, "url": url, "unchanged": True})
        fallbacks[index] = url
    else:
        previous = capture.get(field, "")
        if previous == url:
            return jsonify({"ok": True, "url": url, "unchanged": True})
        capture[field] = url
        if field == "url":
            # The stored endpoint was resolved from the old URL; it cannot
            # outlive it. Stream channels have no 'url' and are unaffected.
            capture.pop("resolved_stream_url", None)

    capture["verified_utc"] = _utc_now_iso()

    # Write via a temp file in the same directory so a crash mid-write cannot
    # leave channels.json truncated — it is the single source of truth.
    tmp = Config.CHANNELS_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(raw, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    tmp.replace(Config.CHANNELS_FILE)
    registry.reload()

    db.audit(actor, "edit_capture_url", "channel", channel_id,
             f"{field}[{index}]: {previous} -> {url}" if field == "fallback_urls"
             else f"{field}: {previous} -> {url}")
    return jsonify({"ok": True, "url": url, "previous": previous, "field": field})


@bp.get("/packs")
def packs():
    return jsonify([{
        "code": p.code, "name": p.name, "autonym": p.autonym, "colour": p.colour,
        "model_ref": p.model_ref, "whisper_token": p.whisper_token,
        "service_tier": p.service_tier, "needs_translation": p.needs_translation,
        "translation_model_ref": p.translation_model_ref,
        "diacritics": p.orthography.get("diacritics", []),
        "baseline_note": p.asr.get("_baseline_note", ""),
    } for p in registry.all_packs()])


@bp.get("/dayparts")
def dayparts():
    return jsonify([{"name": d.name, "start": d.start, "end": d.end,
                     "priority": d.priority} for d in registry.dayparts()])


# --------------------------------------------------------------------------- #
#  Dashboard
# --------------------------------------------------------------------------- #
@bp.get("/stats")
def stats():
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    seg = db.query_one(
        "SELECT COUNT(*) AS n, COALESCE(SUM(duration_seconds),0) AS secs FROM segments") or {}
    today_seg = db.query_one(
        "SELECT COUNT(*) AS n, COALESCE(SUM(duration_seconds),0) AS secs"
        "  FROM segments WHERE started_at_utc LIKE ?", (f"{today}%",)) or {}
    utt = db.query_one(
        "SELECT COUNT(*) AS n, SUM(quarantined) AS q FROM utterances") or {}
    queue = db.query_one(
        "SELECT COUNT(*) AS n FROM segments WHERE status IN ('captured','queued')") or {}
    tx = db.query_one(
        "SELECT SUM(translation_status='done') AS done,"
        "       SUM(translation_status='flagged') AS flagged,"
        "       SUM(translation_status='pending') AS pending FROM utterances") or {}
    corrections = db.query_one("SELECT COUNT(*) AS n FROM corrections") or {}

    n_utt = utt.get("n", 0) or 0
    n_q = utt.get("q", 0) or 0

    by_lang = db.query(
        "SELECT routed_language AS code, COUNT(*) AS n FROM utterances"
        " GROUP BY routed_language ORDER BY n DESC")

    return jsonify({
        "segments_total": seg.get("n", 0),
        "hours_total": round((seg.get("secs") or 0) / 3600, 2),
        "segments_today": today_seg.get("n", 0),
        "hours_today": round((today_seg.get("secs") or 0) / 3600, 2),
        "utterances": n_utt,
        "quarantined": n_q,
        "quarantine_rate": round(n_q / n_utt, 4) if n_utt else 0.0,
        "queue_depth": queue.get("n", 0),
        "translations": {"done": tx.get("done") or 0, "flagged": tx.get("flagged") or 0,
                         "pending": tx.get("pending") or 0},
        "corrections": corrections.get("n", 0),
        "channels_enabled": len(registry.all_channels(enabled_only=True)),
        "channels_total": len(registry.all_channels()),
        "by_language": by_lang,
        "generated_at_utc": _utc_now_iso(),
    })


# --------------------------------------------------------------------------- #
#  Library and search
# --------------------------------------------------------------------------- #
@bp.get("/segments")
def segments():
    channel_id = request.args.get("channel")
    status = request.args.get("status")
    limit = min(int(request.args.get("limit", 60)), 500)
    offset = int(request.args.get("offset", 0))

    where, params = [], []
    if channel_id:
        where.append("channel_id=?")
        params.append(channel_id)
    if status:
        where.append("status=?")
        params.append(status)
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    rows = db.query(
        f"SELECT id, channel_id, path, started_at_utc, duration_seconds, status,"
        f"       content_class, priority, source_type"
        f"  FROM segments {clause} ORDER BY started_at_utc DESC LIMIT ? OFFSET ?",
        tuple(params) + (limit, offset))

    for r in rows:
        u = db.query_one(
            "SELECT COUNT(*) AS n, GROUP_CONCAT(DISTINCT routed_language) AS langs"
            "  FROM utterances WHERE segment_id=?", (r["id"],)) or {}
        r["utterance_count"] = u.get("n", 0)
        r["languages"] = (u.get("langs") or "").split(",") if u.get("langs") else []
        r["filename"] = Path(r["path"]).name
    return jsonify(rows)


@bp.get("/search")
def search():
    """Full-text across vernacular and English simultaneously (Plan §10.6)."""
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({"query": "", "hits": [], "count": 0})

    limit = min(int(request.args.get("limit", 50)), 200)
    lang = request.args.get("language")
    channel_id = request.args.get("channel")

    extra, params = "", [q]
    if lang:
        extra += " AND u.routed_language=?"
        params.append(lang)
    if channel_id:
        extra += " AND s.channel_id=?"
        params.append(channel_id)
    params.append(limit)

    try:
        rows = db.query(
            "SELECT u.id, u.segment_id, u.routed_language, u.start_seconds, u.end_seconds,"
            "       u.quarantined, u.routing_confidence,"
            "       snippet(utterances_fts, 0, '<mark>', '</mark>', '…', 12) AS snip_vern,"
            "       snippet(utterances_fts, 1, '<mark>', '</mark>', '…', 12) AS snip_eng,"
            "       s.channel_id, s.started_at_utc"
            "  FROM utterances_fts"
            "  JOIN utterances u ON u.id = utterances_fts.rowid"
            "  JOIN segments   s ON s.id = u.segment_id"
            " WHERE utterances_fts MATCH ?" + extra +
            " ORDER BY rank LIMIT ?", tuple(params))
    except Exception as exc:                       # malformed FTS query
        return jsonify({"query": q, "hits": [], "count": 0, "error": str(exc)}), 400

    return jsonify({"query": q, "count": len(rows), "hits": rows})


@bp.get("/segment/<int:segment_id>")
def segment_detail(segment_id: int):
    seg = db.query_one("SELECT * FROM segments WHERE id=?", (segment_id,))
    if not seg:
        return jsonify({"error": "not found"}), 404
    seg["utterances"] = db.query(
        "SELECT * FROM utterances WHERE segment_id=? ORDER BY seq", (segment_id,))
    seg["filename"] = Path(seg["path"]).name
    try:
        ch = registry.channel(seg["channel_id"])
        seg["channel_name"] = ch.name
        seg["accent"] = ch.accent
    except KeyError:
        seg["channel_name"] = seg["channel_id"]
        seg["accent"] = "#888"
    return jsonify(seg)


@bp.get("/audio/<int:segment_id>")
def audio(segment_id: int):
    seg = db.query_one("SELECT path FROM segments WHERE id=?", (segment_id,))
    if not seg:
        return jsonify({"error": "not found"}), 404
    path = Path(seg["path"])
    if not path.is_absolute():
        path = Config.ROOT_DIR / path
    if not path.exists():
        return jsonify({"error": "audio file missing", "path": str(path)}), 410
    # Confine reads to the data directory — the path column must never escape it.
    try:
        path.resolve().relative_to(Config.DATA_DIR.resolve())
    except ValueError:
        return jsonify({"error": "path outside data directory"}), 403
    return send_file(path, conditional=True)


# --------------------------------------------------------------------------- #
#  Corrections — the training flywheel (Plan §10.4)
# --------------------------------------------------------------------------- #
@bp.post("/utterance/<int:utterance_id>/correct")
def correct(utterance_id: int):
    payload = request.get_json(silent=True) or {}
    field = payload.get("field", "text_vernacular")
    value = payload.get("value")
    actor = payload.get("actor", "analyst")

    if field not in {"text_vernacular", "text_english", "routed_language"}:
        return jsonify({"error": f"field {field!r} is not correctable"}), 400
    if value is None:
        return jsonify({"error": "value is required"}), 400

    current = db.query_one("SELECT * FROM utterances WHERE id=?", (utterance_id,))
    if not current:
        return jsonify({"error": "not found"}), 404

    if field == "routed_language" and not registry.pack_exists(value):
        return jsonify({"error": f"no language pack for {value!r}"}), 400

    db.execute(
        "INSERT INTO corrections (utterance_id, field, original_value, corrected_value,"
        " corrected_by, note) VALUES (?,?,?,?,?,?)",
        (utterance_id, field, current.get(field), value, actor, payload.get("note", "")))

    updates = [f"{field}=?"]
    params: list = [value]
    if field == "routed_language":
        updates.append("routing_method='manual'")
        updates.append("routing_confidence=1.0")
    # A corrected utterance is, by definition, no longer quarantined.
    updates.append("quarantined=0")
    updates.append("quarantine_reason=NULL")
    # Keep the English in step with the transcript. Analyst-written English is
    # authoritative and never overwritten; otherwise a changed transcript (or a
    # changed language) sends the line back through translation.
    if field == "text_english":
        updates.append("translation_status='corrected'")
        updates.append("translation_flag_reason=NULL")
    elif current.get("translation_status") != "corrected":
        lang = value if field == "routed_language" else current["routed_language"]
        updates.append("translation_status=?")
        params.append(pipeline.initial_translation_status(registry.pack_for(lang), False))
        updates.append("translation_flag_reason=NULL")
    params.append(utterance_id)
    db.execute(f"UPDATE utterances SET {', '.join(updates)} WHERE id=?", tuple(params))

    db.audit(actor, "correct_utterance", "utterance", str(utterance_id), f"{field} updated")
    pipeline.wake()
    return jsonify({"ok": True, "utterance_id": utterance_id, "field": field})


@bp.get("/corrections/export")
def export_corrections():
    """Export corrections as a training manifest.

    This is the in-domain broadcast corpus that Step 2 needs and that studio-read
    datasets cannot provide.
    """
    rows = db.query(
        "SELECT c.id, c.field, c.corrected_value, c.created_at_utc,"
        "       u.start_seconds, u.end_seconds, u.routed_language,"
        "       s.path, s.channel_id, s.started_at_utc"
        "  FROM corrections c"
        "  JOIN utterances u ON u.id = c.utterance_id"
        "  JOIN segments   s ON s.id = u.segment_id"
        " WHERE c.field='text_vernacular' ORDER BY c.created_at_utc DESC")
    return jsonify({"count": len(rows), "manifest": rows,
                    "note": "In-domain broadcast training pairs. Step 2 input."})


# --------------------------------------------------------------------------- #
#  Quarantine
# --------------------------------------------------------------------------- #
@bp.get("/quarantine")
def quarantine_list():
    rows = db.query(
        "SELECT u.*, s.channel_id, s.started_at_utc, s.path"
        "  FROM utterances u JOIN segments s ON s.id=u.segment_id"
        " WHERE u.quarantined=1 ORDER BY u.id DESC LIMIT 200")

    # The stored reason keeps its measured numbers; the category is what the
    # queue filters and counts by. Channel names come from the registry so the
    # queue reads like the rest of the UI rather than showing raw ids.
    names = {c.id: c.name for c in registry.all_channels()}
    for r in rows:
        r["channel_name"] = names.get(r["channel_id"], r["channel_id"])
        r["reason_category"] = quarantine.reason_category(r.get("quarantine_reason"))
    return jsonify(rows)


@bp.post("/quarantine/<int:utterance_id>/resolve")
def quarantine_resolve(utterance_id: int):
    payload = request.get_json(silent=True) or {}
    action = payload.get("action")          # confirm_noise | release
    actor = payload.get("actor", "analyst")
    if action not in {"confirm_noise", "release"}:
        return jsonify({"error": "action must be confirm_noise or release"}), 400

    if action == "release":
        db.execute("UPDATE utterances SET quarantined=0, quarantine_reason=NULL WHERE id=?",
                   (utterance_id,))
        # Withheld lines were never translated; a released one now needs English.
        u = db.query_one("SELECT routed_language, translation_status FROM utterances"
                         " WHERE id=?", (utterance_id,))
        if u and u["translation_status"] == "skipped":
            db.execute("UPDATE utterances SET translation_status=? WHERE id=?",
                       (pipeline.initial_translation_status(
                           registry.pack_for(u["routed_language"]), False), utterance_id))
            pipeline.wake()
    else:
        db.execute("UPDATE utterances SET quarantine_reason=COALESCE(quarantine_reason,'')"
                   " || ' [confirmed noise]' WHERE id=?", (utterance_id,))
    db.audit(actor, f"quarantine_{action}", "utterance", str(utterance_id))
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
#  Evaluation (Plan §9)
# --------------------------------------------------------------------------- #
@bp.get("/evaluation/history")
def eval_history():
    return jsonify(evaluation.history(request.args.get("pack"), limit=200))


@bp.get("/evaluation/latest")
def eval_latest():
    return jsonify(evaluation.latest_by_pack())


@bp.get("/evaluation/reference-sets")
def reference_sets():
    out = []
    for name in evaluation.available_reference_sets():
        try:
            items = evaluation.load_reference_set(name)
            langs = sorted({i.language for i in items})
            out.append({"name": name, "items": len(items), "languages": langs,
                        "channels": sorted({i.channel_id for i in items if i.channel_id})})
        except (ValueError, FileNotFoundError) as exc:
            out.append({"name": name, "error": str(exc)})
    return jsonify(out)


@bp.post("/evaluation/score")
def score_text():
    """Ad-hoc scoring: paste a reference and a hypothesis, get WER back.

    Useful for spot checks. The pack's own normalisation rules are applied, so the
    number is comparable with the recorded runs rather than being a different metric.
    """
    payload = request.get_json(silent=True) or {}
    pack = payload.get("pack", "kik")
    ref = payload.get("reference", "")
    hyp = payload.get("hypothesis", "")
    if not ref:
        return jsonify({"error": "reference is required"}), 400
    try:
        score = evaluation.score_for_pack(pack, [ref], [hyp])
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(score.as_dict())


@bp.get("/health")
def health():
    from app.services import asr, translation
    return jsonify({
        "ok": True,
        "loaded_models": asr.loaded_models() + translation.loaded_models(),
        "packs": [p.code for p in registry.all_packs()],
        "channels": len(registry.all_channels()),
        "db": str(Config.DB_PATH),
        "time_utc": _utc_now_iso(),
    })
