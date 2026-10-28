#!/usr/bin/env python
"""Development entrypoint.  python run.py  ->  http://127.0.0.1:5000

Also starts the background pipeline (capture -> transcribe -> translate), so any
audio captured — from the Live page or by capture_worker.py — is transcribed and
translated without a terminal. Set OSINT_BACKGROUND_PIPELINE=0 to run the UI
alone (e.g. while running scripts/transcribe_worker.py instead).
"""
import os

from app import create_app
from app.config import Config

app = create_app()

DEBUG = True
USE_RELOADER = DEBUG

if __name__ == "__main__":
    # With the reloader on, this file runs twice: a parent that only watches for
    # code changes, and a child (WERKZEUG_RUN_MAIN=true) that serves requests.
    # Start the worker in the child only, or the models load twice.
    serving_process = not USE_RELOADER or os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    if serving_process and Config.BACKGROUND_PIPELINE:
        from app.services import pipeline
        pipeline.worker().start()

    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        print("OSINT Monitoring — http://127.0.0.1:5000")
        print("Background pipeline: "
              + ("on (capture -> transcribe -> translate)" if Config.BACKGROUND_PIPELINE
                 else "off (OSINT_BACKGROUND_PIPELINE=0)"))
    # The reloader watches every .py under the project, not just app/. Editing a
    # script must not restart the server: a restart drops in-memory recording
    # jobs and aborts the segment being transcribed (it is requeued, but the
    # minutes of work are lost).
    root = Config.ROOT_DIR
    exclude = [str(root / d / "*") for d in ("scripts", "data", "models", "eval", "docs")]
    app.run(host="127.0.0.1", port=5000, debug=DEBUG, use_reloader=USE_RELOADER,
            exclude_patterns=exclude)
