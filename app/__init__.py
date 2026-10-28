"""Flask application factory."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from flask import Flask

from app.config import Config

EAT = timezone(timedelta(hours=3))   # Africa/Nairobi. Storage is UTC; this is display only.


def create_app(config_object: type[Config] = Config) -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config.from_object(config_object)
    config_object.ensure_dirs()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
    )

    from app import db
    db.init_db()

    from app.routes.views import bp as views_bp
    from app.routes.api import bp as api_bp
    from app.routes.api_live import bp as api_live_bp
    app.register_blueprint(views_bp)
    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(api_live_bp, url_prefix="/api")

    app.jinja_env.filters["eat"] = _to_eat
    app.jinja_env.filters["eat_time"] = lambda s: _to_eat(s, "%H:%M")
    app.jinja_env.filters["duration"] = _human_duration
    app.jinja_env.globals["now_eat"] = lambda: datetime.now(EAT)

    return app


def _to_eat(utc_iso: str | None, fmt: str = "%d %b %Y, %H:%M") -> str:
    """UTC in storage, EAT in the interface — never the other way round."""
    if not utc_iso:
        return "—"
    try:
        dt = datetime.fromisoformat(str(utc_iso).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(EAT).strftime(fmt)
    except (ValueError, TypeError):
        return str(utc_iso)


def _human_duration(seconds: float | None) -> str:
    if not seconds:
        return "0s"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"
