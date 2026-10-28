"""Live monitoring, on-demand recording, and transcript documents.

Kept separate from api.py because these are the operator-facing endpoints — the
ones that make the system usable rather than merely correct.
"""
from __future__ import annotations

from pathlib import Path

from flask import Blueprint, Response, jsonify, request

from app import db
from app.config import Config
from app.services import capture, pipeline, registry

bp = Blueprint("api_live", __name__)


# --------------------------------------------------------------------------- #
#  Live monitoring
# --------------------------------------------------------------------------- #
@bp.get("/listen/<channel_id>")
def listen(channel_id: str):
    """Proxy a channel as a continuous MP3 stream for a browser audio element.

    Proxied rather than linked directly because the CDN endpoints are not
    CORS-enabled, YouTube's signed URLs expire mid-listen, and TV feeds are HLS,
    which a plain <audio> element cannot play.
    """
    try:
        registry.channel(channel_id)
    except KeyError:
        return jsonify({"error": "unknown channel"}), 404

    try:
        stream = capture.live_stream(channel_id)
    except (RuntimeError, KeyError, OSError) as exc:
        return jsonify({"error": str(exc)[:300]}), 503

    return Response(stream, mimetype="audio/mpeg", headers={
        "Cache-Control": "no-cache, no-store",
        "X-Accel-Buffering": "no",
    })


@bp.get("/listen-check/<channel_id>")
def listen_check(channel_id: str):
    """Can this channel be listened to right now? Resolves without transcoding."""
    try:
        ch = registry.channel(channel_id)
    except KeyError:
        return jsonify({"error": "unknown channel"}), 404
    try:
        url = capture.resolve_stream_url(ch)
    except (RuntimeError, OSError) as exc:
        return jsonify({"ok": False, "error": str(exc)[:300]}), 503
    kind = "hls" if (".m3u8" in url or "manifest" in url) else "direct"
    return jsonify({"ok": True, "channel": ch.id, "kind": kind})


@bp.get("/watch/<channel_id>")
def watch(channel_id: str):
    """Where to embed this channel's live picture, for the TV preview dialog.

    Audio is proxied because a browser cannot play HLS; video is handed to the
    platform's own player instead of being re-encoded, which would cost an
    ffmpeg video transcode this machine has no headroom for (CLAUDE.md §3).
    """
    try:
        ch = registry.channel(channel_id)
    except KeyError:
        return jsonify({"error": "unknown channel"}), 404
    if ch.medium != "tv":
        return jsonify({"error": "this channel has no video feed"}), 400

    try:
        video_id = capture.resolve_video_id(ch)
    except (RuntimeError, OSError) as exc:
        return jsonify({"error": str(exc)[:300]}), 503

    return jsonify({
        "channel": ch.id,
        "name": ch.name,
        "video_id": video_id,
        "embed_url": f"https://www.youtube.com/embed/{video_id}?autoplay=1",
        "watch_url": f"https://www.youtube.com/watch?v={video_id}",
    })


# --------------------------------------------------------------------------- #
#  On-demand recording
# --------------------------------------------------------------------------- #
@bp.post("/record/<channel_id>")
def record(channel_id: str):
    payload = request.get_json(silent=True) or {}
    try:
        duration = int(payload.get("duration", 60))
    except (TypeError, ValueError):
        return jsonify({"error": "duration must be a number"}), 400
    # The analyst decides how long to record; the ceiling only stops a typo from
    # pinning an ffmpeg process indefinitely. Raise it with OSINT_MAX_RECORDING_SECONDS.
    if duration < 5 or duration > Config.MAX_RECORDING_SECONDS:
        return jsonify({"error": f"duration must be 5-{Config.MAX_RECORDING_SECONDS} "
                                 f"seconds"}), 400

    try:
        job_id = capture.start_recording(
            channel_id, duration, actor=payload.get("actor", "analyst"))
    except KeyError:
        return jsonify({"error": "unknown channel"}), 404
    return jsonify({"job_id": job_id, "state": "recording", "duration": duration})


@bp.get("/record/job/<job_id>")
def record_job(job_id: str):
    job = capture.job_status(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404
    return jsonify(job)


@bp.get("/record/jobs")
def record_jobs():
    return jsonify(capture.all_jobs())


# --------------------------------------------------------------------------- #
#  Background pipeline
# --------------------------------------------------------------------------- #
@bp.get("/pipeline/status")
def pipeline_status():
    return jsonify(pipeline.status())


@bp.post("/segment/<int:segment_id>/retranslate")
def retranslate(segment_id: int):
    """Send a segment's transcript back through translation (not transcription).

    Lines an analyst has corrected in English are left alone.
    """
    if not db.query_one("SELECT id FROM segments WHERE id=?", (segment_id,)):
        return jsonify({"error": "not found"}), 404
    n = db.execute(
        "UPDATE utterances SET translation_status='pending', translation_flag_reason=NULL"
        " WHERE segment_id=? AND translation_status IN ('done','flagged','failed')",
        (segment_id,))
    actor = (request.get_json(silent=True) or {}).get("actor", "analyst")
    db.audit(actor, "retranslate", "segment", str(segment_id), f"{n} line(s) requeued")
    pipeline.wake()
    return jsonify({"ok": True, "requeued": n})


# --------------------------------------------------------------------------- #
#  Transcript documents
# --------------------------------------------------------------------------- #
def _document_for(segment_id: int) -> dict | None:
    seg = db.query_one("SELECT * FROM segments WHERE id=?", (segment_id,))
    if not seg:
        return None
    seg["utterances"] = db.query(
        "SELECT * FROM utterances WHERE segment_id=? ORDER BY seq", (segment_id,))
    try:
        ch = registry.channel(seg["channel_id"])
        seg["channel_name"] = ch.name
        seg["broadcaster"] = ch.broadcaster
        seg["accent"] = ch.accent
    except KeyError:
        seg["channel_name"] = seg["channel_id"]
        seg["broadcaster"] = ""
        seg["accent"] = "#888888"
    seg["filename"] = Path(seg["path"]).name
    return seg


PUBLISHABLE_TRANSLATION = {"done", "corrected"}


def english_publishable(u: dict) -> bool:
    """Only checked (or analyst-written) English reaches a document or export.

    Flagged translations are stored for review but withheld, exactly as
    quarantined transcripts are (CLAUDE.md §2.4).
    """
    status = u.get("translation_status")
    if status is None:                   # pre-translation rows: analyst-entered text
        return bool(u.get("text_english"))
    return status in PUBLISHABLE_TRANSLATION


@bp.get("/document/<int:segment_id>")
def document(segment_id: int):
    """A transcript as a readable document rather than a table of rows."""
    doc = _document_for(segment_id)
    if not doc:
        return jsonify({"error": "not found"}), 404

    include_withheld = request.args.get("include_withheld") == "1"
    body = [u for u in doc["utterances"] if include_withheld or not u["quarantined"]]
    doc["prose_vernacular"] = " ".join(
        (u["text_vernacular"] or "").strip() for u in body).strip()
    doc["prose_english"] = " ".join(
        (u["text_english"] or "").strip() for u in body if english_publishable(u)).strip()
    for u in doc["utterances"]:
        u["english_publishable"] = english_publishable(u)
    doc["withheld"] = sum(1 for u in doc["utterances"] if u["quarantined"])
    return jsonify(doc)


def _timestamp(seconds: float) -> str:
    return f"[{int(seconds // 60):02d}:{int(seconds % 60):02d}]"


@bp.get("/document/<int:segment_id>/export.<fmt>")
def export_document(segment_id: int, fmt: str):
    """Download a transcript as plain text or Markdown.

    Every export carries provenance — station, UTC and EAT timestamps, model,
    SHA-256 — because a transcript without a chain of custody is not evidence.
    Withheld passages are listed as withheld rather than silently dropped, so the
    reader can see that something was excluded and why.
    """
    if fmt not in {"txt", "md"}:
        return jsonify({"error": "format must be txt or md"}), 400

    doc = _document_for(segment_id)
    if not doc:
        return jsonify({"error": "not found"}), 404

    from app import _to_eat

    packs = {p.code: p for p in registry.all_packs()}
    langs = sorted({u["routed_language"] for u in doc["utterances"]})
    lang_names = ", ".join(packs[c].name if c in packs else c for c in langs) or "-"
    withheld = sum(1 for u in doc["utterances"] if u["quarantined"])
    rule = "---" if fmt == "md" else "-" * 64

    lines = [
        ("# " if fmt == "md" else "") + f"Broadcast transcript - {doc['channel_name']}",
        "",
        f"Station      : {doc['channel_name']} ({doc['broadcaster']})",
        f"Captured     : {_to_eat(doc['started_at_utc'])} EAT",
        f"Stored (UTC) : {doc['started_at_utc']}",
        f"Duration     : {doc['duration_seconds']:.0f}s",
        f"Language(s)  : {lang_names}",
        f"Model        : {(doc['utterances'][0]['model_ref'] if doc['utterances'] else '-')}",
        f"Translation  : {next((u['translation_model_ref'] for u in doc['utterances'] if u.get('translation_model_ref')), '-')}",
        f"Source file  : {doc['filename']}",
        f"SHA-256      : {doc.get('sha256', '')}",
        f"Utterances   : {len(doc['utterances'])} ({withheld} withheld)",
        "",
        rule,
        "",
    ]

    for u in doc["utterances"]:
        ts = _timestamp(u["start_seconds"])
        if u["quarantined"]:
            reason = u["quarantine_reason"] or "failed automated quality checks"
            note = f"WITHHELD - {reason}"
            lines.append(f"{ts} *{note}*" if fmt == "md" else f"{ts} [{note}]")
            lines.append("")
            continue
        lines.append(f"{ts} {(u['text_vernacular'] or '').strip()}")
        prefix = "> " if fmt == "md" else "      EN: "
        if u["text_english"] and english_publishable(u):
            lines.append(prefix + u["text_english"].strip())
        elif u.get("translation_status") == "flagged":
            reason = u.get("translation_flag_reason") or "failed automated checks"
            lines.append(prefix + f"[translation withheld - {reason}]")
        lines.append("")

    lines += [
        rule,
        "Machine-generated transcript and machine translation. Withheld passages",
        "failed automated quality checks and are excluded pending human review.",
        "English is translated from the transcript above, not from the audio;",
        "verify against the source audio before relying on any passage.",
    ]

    stamp = doc["started_at_utc"].replace(":", "").replace("-", "")
    name = f"transcript_{doc['channel_id']}_{stamp}.{fmt}"
    return Response("\n".join(lines), mimetype="text/plain; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})
