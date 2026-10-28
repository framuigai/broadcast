"""HTML pages. Data comes from /api/* — templates provide the shell, not the payload."""
from __future__ import annotations

from flask import Blueprint, abort, render_template, request

from app import db
from app.config import Config
from app.services import evaluation, registry

bp = Blueprint("views", __name__)


def _nav_counts() -> dict[str, int]:
    """Badge counts for the sidebar. Cheap queries only — this runs on every page."""
    try:
        return {
            "tv": len(registry.all_channels(medium="tv")),
            "fm": len(registry.all_channels(medium="fm")),
            "library": (db.query_one("SELECT COUNT(*) c FROM segments") or {}).get("c", 0),
            "quarantine": (db.query_one(
                "SELECT COUNT(*) c FROM utterances WHERE quarantined=1") or {}).get("c", 0),
        }
    except Exception:          # a fresh database should never break navigation
        return {"tv": 0, "fm": 0, "library": 0, "quarantine": 0}


@bp.app_context_processor
def inject_globals():
    return {"nav_counts": _nav_counts(), "packs": registry.all_packs()}


@bp.route("/")
def dashboard():
    return render_template("dashboard.html", page="dashboard",
                           channels=registry.all_channels(),
                           latest_eval=evaluation.latest_by_pack())


@bp.route("/tv")
def television():
    return render_template("channels.html", page="tv", medium="tv",
                           title="Television", subtitle="Vernacular TV channels under monitoring",
                           channels=registry.all_channels(medium="tv"))


@bp.route("/fm")
def radio():
    return render_template("channels.html", page="fm", medium="fm",
                           title="FM Radio", subtitle="Vernacular radio stations under monitoring",
                           channels=registry.all_channels(medium="fm"))


@bp.route("/channel/<channel_id>")
def channel_detail(channel_id: str):
    try:
        ch = registry.channel(channel_id)
    except KeyError:
        abort(404)
    return render_template("channel_detail.html", page=ch.medium, channel=ch)


@bp.route("/live")
def live_monitor():
    return render_template("live.html", page="live",
                           channels=registry.all_channels(enabled_only=True),
                           max_recording_seconds=Config.MAX_RECORDING_SECONDS)


@bp.route("/document/<int:segment_id>")
def document(segment_id: int):
    seg = db.query_one("SELECT * FROM segments WHERE id=?", (segment_id,))
    if not seg:
        abort(404)
    return render_template("document.html", page="library", segment=seg)


@bp.route("/library")
def library():
    return render_template("library.html", page="library",
                           channels=registry.all_channels(),
                           q=request.args.get("q", ""))


@bp.route("/transcript/<int:segment_id>")
def transcript(segment_id: int):
    seg = db.query_one("SELECT * FROM segments WHERE id=?", (segment_id,))
    if not seg:
        abort(404)
    return render_template("transcript.html", page="library", segment=seg)


@bp.route("/quarantine")
def quarantine_queue():
    return render_template("quarantine.html", page="quarantine")


@bp.route("/evaluation")
def evaluation_page():
    return render_template("evaluation.html", page="evaluation",
                           reference_sets=evaluation.available_reference_sets(),
                           latest=evaluation.latest_by_pack())


@bp.route("/packs")
def packs_page():
    return render_template("packs.html", page="packs",
                           models=registry.all_model_specs())


@bp.errorhandler(404)
def not_found(_e):
    return render_template("error.html", code=404,
                           message="That page does not exist."), 404
