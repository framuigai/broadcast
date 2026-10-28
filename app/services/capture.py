"""Stream resolution, live listening and recording.

The ONLY capture implementation. scripts/capture_worker.py (continuous capture)
and the Live page's Record button (on-demand) both call `capture_segment`, so a
clip recorded from the browser is indistinguishable from one the worker captured
— same path layout, same provenance, same immutability guarantees. Do not copy
this logic into a script; add a parameter here instead.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from app import db
from app.config import Config
from app.services import registry

log = logging.getLogger(__name__)

EAT = timezone(timedelta(hours=3))
# Recorded on every segment row. 0.3: single shared implementation; unique
# write-once output names (no overwrite on a same-second collision); cancellable.
SOFTWARE_VERSION = "capture/0.3"
MIN_SEGMENT_BYTES = 1024

# Resolved stream URLs are cached briefly: yt-dlp resolution takes seconds and a
# googlevideo URL stays valid for a while, but not forever.
_url_cache: dict[str, tuple[str, float]] = {}
_video_id_cache: dict[str, tuple[str, float]] = {}
_URL_TTL = 240.0
_cache_lock = threading.Lock()


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%SZ")


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(binary: str) -> str:
    found = shutil.which(binary)
    if not found:
        raise RuntimeError(
            f"{binary} is not on PATH. Install it before capturing "
            f"(yt-dlp: pip install yt-dlp; ffmpeg is already present on this machine).")
    return found


def reserve_output_path(out_dir: Path, channel_id: str, started: datetime) -> Path:
    """Atomically claim a new, unused archive filename and return it.

    `{station}_{YYYYMMDD}T{HHMMSS}Z_{seq}.opus` — seq exists for exactly this:
    two captures of one channel starting in the same second (the worker plus an
    analyst, or a double-click) get _000 and _001. The name is claimed by
    creating an empty file with O_EXCL, so the check and the claim cannot race.

    Without this, the second ffmpeg found the first one's file, stalled on an
    overwrite prompt, and the failure cleanup then deleted the OTHER capture's
    audio — a direct breach of the write-once rule (CLAUDE.md §2.3).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for seq in range(1000):
        path = out_dir / f"{channel_id}_{stamp(started)}_{seq:03d}.opus"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.close(fd)
        return path
    raise RuntimeError(f"No free sequence number for {channel_id} at {stamp(started)}")


def _reset(path: Path) -> None:
    """Empty a reserved file without releasing its name (between capture attempts)."""
    with open(path, "wb"):
        pass


def discard_partial(path: Path) -> None:
    """Remove a failed capture's output so the archive holds only real segments."""
    try:
        if path.exists():
            path.unlink()
    except OSError as exc:
        log.warning("Could not remove partial file %s: %s", path, exc)


def record_gap(channel_id: str, started: datetime, reason: str) -> None:
    db.execute(
        "INSERT INTO capture_gaps (channel_id, started_at_utc, ended_at_utc, reason)"
        " VALUES (?,?,?,?)",
        (channel_id, iso(started), iso(utc_now()), reason[:500]))


# --------------------------------------------------------------------------- #
#  Stream resolution
# --------------------------------------------------------------------------- #
def resolve_stream_url(ch: registry.Channel, use_cache: bool = True) -> str:
    """Return a URL ffmpeg can read directly.

    YouTube live channels rotate their video id, so we ask yt-dlp for the current
    endpoint rather than trusting a stored watch URL.
    """
    if use_cache:
        with _cache_lock:
            hit = _url_cache.get(ch.id)
            if hit and time.time() - hit[1] < _URL_TTL:
                return hit[0]

    method = ch.capture.get("method")
    if method == "yt-dlp":
        result = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--no-warnings",
             "-f", "bestaudio", "-g", ch.capture["url"]],
            capture_output=True, text=True, timeout=120)
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(f"yt-dlp could not resolve {ch.capture['url']}: "
                               f"{result.stderr.strip()[:300]}")
        url = result.stdout.strip().splitlines()[0]
    else:
        url = ch.capture.get("resolved_stream_url") or ""
        if not url:
            raise RuntimeError(
                f"{ch.id}: no resolved_stream_url. The registry URL is a web page, "
                "not a stream endpoint — resolve the direct ICY/HLS endpoint once "
                "and store it in config/channels.json.")

    with _cache_lock:
        _url_cache[ch.id] = (url, time.time())
    return url


def resolve_video_id(ch: registry.Channel, use_cache: bool = True) -> str:
    """The id of the video a yt-dlp channel is currently streaming.

    For watching the picture rather than hearing the audio: the audio path
    proxies through ffmpeg because a browser cannot play HLS, but video is best
    left to the platform's own player, which needs the live video id. A 24/7
    channel rotates that id, so it is resolved live and cached briefly.
    """
    if ch.capture.get("method") != "yt-dlp":
        raise RuntimeError(f"{ch.id}: only yt-dlp channels have a watchable video")

    if use_cache:
        with _cache_lock:
            hit = _video_id_cache.get(ch.id)
            if hit and time.time() - hit[1] < _URL_TTL:
                return hit[0]

    result = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--no-warnings", "--get-id",
         ch.capture["url"]],
        capture_output=True, text=True, timeout=120)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"yt-dlp could not resolve {ch.capture['url']}: "
                           f"{result.stderr.strip()[:300]}")
    video_id = result.stdout.strip().splitlines()[0]

    with _cache_lock:
        _video_id_cache[ch.id] = (video_id, time.time())
    return video_id


# --------------------------------------------------------------------------- #
#  Live listening
# --------------------------------------------------------------------------- #
def live_stream(channel_id: str, bitrate: str = "64k") -> Iterator[bytes]:
    """Yield a continuous MP3 stream of a channel, for a browser audio element.

    Proxying through ffmpeg rather than pointing the browser at the source solves
    three problems at once: the CDN endpoints are not CORS-enabled, YouTube's
    signed URLs expire mid-listen, and TV feeds are HLS which a plain <audio>
    element cannot play. MP3 is used because every browser handles a chunked
    MP3 stream without a media-source shim.
    """
    ch = registry.channel(channel_id)
    url = resolve_stream_url(ch)
    ffmpeg = require("ffmpeg")

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
        "-i", url,
        "-vn", "-ac", "1", "-ar", "22050",
        "-f", "mp3", "-acodec", "libmp3lame", "-b:a", bitrate,
        "-flush_packets", "1",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            bufsize=0)
    log.info("Live stream opened for %s (pid %s)", channel_id, proc.pid)
    try:
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            yield chunk
    except GeneratorExit:
        # Browser closed the connection — stop transcoding immediately rather
        # than leaving an ffmpeg process running for the life of the server.
        log.info("Live stream closed by client for %s", channel_id)
        raise
    finally:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass


# --------------------------------------------------------------------------- #
#  Recording — shared by the worker and the UI
# --------------------------------------------------------------------------- #
def capture_segment(ch: registry.Channel, duration: int,
                    source_type: str = "broadcast",
                    actor: str | None = None,
                    cancel: threading.Event | None = None) -> int:
    """Capture one segment, register it, and return its row id.

    `cancel`, when set, stops the capture early and skips the yt-dlp fallback —
    used by the capture worker on Ctrl-C so a stop never starts a new download.
    Raises RuntimeError on failure (or cancellation), having removed any partial
    file. Never touches a file it did not create.
    """
    started = utc_now()
    out_dir = (Config.AUDIO_DIR / ch.id / started.strftime("%Y") /
               started.strftime("%m") / started.strftime("%d"))
    out_path = reserve_output_path(out_dir, ch.id, started)
    try:
        _pull(ch, duration, out_path, cancel)
    except BaseException:
        discard_partial(out_path)          # ours: reserved above, never someone else's
        raise

    part = registry.daypart_for(started.astimezone(EAT).strftime("%H:%M"))
    priority = part.priority if part else "medium"
    # A clip an analyst asked for is wanted now, not when the queue drains.
    if source_type == "on_demand":
        priority = "high"

    seg_id = db.execute(
        "INSERT INTO segments (channel_id, path, started_at_utc, duration_seconds,"
        " sample_rate, sha256, source_type, uploaded_by, capture_host,"
        " software_version, priority, status)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (ch.id, str(out_path.relative_to(Config.ROOT_DIR)), iso(started), float(duration),
         Config.AUDIO_SAMPLE_RATE, sha256_of(out_path), source_type, actor,
         "localhost", SOFTWARE_VERSION, priority, "captured"))

    db.audit(actor or "capture", "capture_segment", "segment", str(seg_id),
             f"{ch.id} {duration}s {source_type}")

    # Hand off to the background pipeline (no-op if it is not running in this
    # process — it polls the queue anyway).
    from app.services import pipeline
    pipeline.wake()
    return seg_id


def _pull(ch: registry.Channel, duration: int, out_path: Path,
          cancel: threading.Event | None) -> None:
    """Fill the reserved `out_path` with `duration` seconds of the channel.

    Direct ffmpeg pull first; for yt-dlp sources, fall back to letting yt-dlp
    fetch. Raises RuntimeError if nothing usable was captured.
    """
    url = resolve_stream_url(ch)
    fallback_ok = ch.capture.get("method") == "yt-dlp"
    ffmpeg = require("ffmpeg")

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "10",
        "-i", url, "-t", str(duration),
        "-vn", "-ac", "1", "-ar", str(Config.AUDIO_SAMPLE_RATE),
        "-c:a", Config.AUDIO_CODEC, "-b:a", Config.AUDIO_BITRATE,
        # -y overwrites only the empty file reserve_output_path() just created
        # for this capture; without it ffmpeg stalls on an overwrite prompt.
        "-y", str(out_path),
    ]

    err = ""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                stdin=subprocess.DEVNULL, timeout=duration + 120)
        if (result.returncode == 0 and out_path.exists()
                and out_path.stat().st_size >= MIN_SEGMENT_BYTES):
            return
        size = out_path.stat().st_size if out_path.exists() else 0
        err = (f"ffmpeg failed (exit {result.returncode}, {size} bytes): "
               f"{result.stderr.strip()[:300]}")
    except subprocess.TimeoutExpired:
        err = f"ffmpeg timed out after {duration + 120}s"

    if cancel is not None and cancel.is_set():
        raise RuntimeError(f"capture cancelled ({err})")
    if not fallback_ok:
        raise RuntimeError(err)

    log.info("%s: direct pull failed (%s); trying yt-dlp fallback", ch.id, err[:120])
    _reset(out_path)                       # keep the reservation, drop the partial
    if not _capture_via_ytdlp(ch, duration, out_path, cancel):
        raise RuntimeError(f"direct pull and yt-dlp fallback both failed: {err}")

    part = registry.daypart_for(started.astimezone(EAT).strftime("%H:%M"))
    priority = part.priority if part else "medium"
    # A clip an analyst asked for is wanted now, not when the queue drains.
    if source_type == "on_demand":
        priority = "high"

    seg_id = db.execute(
        "INSERT INTO segments (channel_id, path, started_at_utc, duration_seconds,"
        " sample_rate, sha256, source_type, uploaded_by, capture_host,"
        " software_version, priority, status)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (ch.id, str(out_path.relative_to(Config.ROOT_DIR)), iso(started), float(duration),
         Config.AUDIO_SAMPLE_RATE, sha256_of(out_path), source_type, actor,
         "localhost", SOFTWARE_VERSION, priority, "captured"))

    db.audit(actor or "capture", "capture_segment", "segment", str(seg_id),
             f"{ch.id} {duration}s {source_type}")

    # Hand off to the background pipeline (no-op if it is not running in this
    # process — it polls the queue anyway).
    from app.services import pipeline
    pipeline.wake()
    return seg_id


def _capture_via_ytdlp(ch: registry.Channel, duration: int, out_path: Path,
                       cancel: threading.Event | None = None) -> bool:
    """Fallback for live streams ffmpeg's HLS demuxer cannot pull directly.

    Some YouTube live channels (Ramogi TV at time of writing) stall ffmpeg
    indefinitely while others on the same platform work, so this is a per-stream
    quirk. yt-dlp does the fetching — it handles YouTube's headers, throttling and
    fragment retries — and is stopped with a break signal, not killed, so its
    buffer is flushed to disk before remuxing. A force-kill loses the buffer.

    `out_path` is this capture's reserved (empty) file; the caller discards it
    on a False return.
    """
    ts_path = out_path.with_suffix(".ts")
    # yt-dlp skips a download whose output already exists — never let a stale
    # .ts from a crashed run be remuxed in as new audio.
    discard_partial(ts_path)
    cmd = [sys.executable, "-m", "yt_dlp", "--no-warnings", "--no-part",
           "--hls-use-mpegts", "-f", "bestaudio", "-o", str(ts_path),
           ch.capture["url"]]
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, creationflags=flags)

    # Wait for the segment duration (plus startup slack), waking each second so a
    # cancel stops it promptly instead of after a full segment.
    deadline = time.monotonic() + duration + 20
    while proc.poll() is None and time.monotonic() < deadline:
        if cancel is not None and cancel.is_set():
            break
        time.sleep(1)
    if proc.poll() is None:
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
            proc.wait(timeout=25)
        except (subprocess.TimeoutExpired, OSError):
            proc.kill()
            proc.wait(timeout=10)

    if cancel is not None and cancel.is_set():
        discard_partial(ts_path)
        return False
    if not ts_path.exists() or ts_path.stat().st_size < MIN_SEGMENT_BYTES:
        discard_partial(ts_path)
        return False

    # -y overwrites only this capture's reserved file.
    remux = subprocess.run(
        [require("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(ts_path), "-t", str(duration), "-vn", "-ac", "1",
         "-ar", str(Config.AUDIO_SAMPLE_RATE),
         "-c:a", Config.AUDIO_CODEC, "-b:a", Config.AUDIO_BITRATE, str(out_path)],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=180)
    discard_partial(ts_path)

    return (remux.returncode == 0 and out_path.exists()
            and out_path.stat().st_size >= MIN_SEGMENT_BYTES)


# --------------------------------------------------------------------------- #
#  On-demand recording jobs
# --------------------------------------------------------------------------- #
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def start_recording(channel_id: str, duration: int, actor: str = "analyst") -> str:
    """Kick off a recording in the background and return a job id.

    Recording blocks for `duration` seconds, so it cannot run inside the request.
    It only captures: the segment lands in the queue as 'high' priority and the
    background pipeline transcribes and translates it (app/services/pipeline.py).
    Transcribing here, in a thread per recording, would load a model per clip.
    """
    ch = registry.channel(channel_id)
    job_id = uuid.uuid4().hex[:12]

    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id, "channel_id": ch.id, "channel_name": ch.name,
            "duration": duration, "state": "recording", "segment_id": None,
            "error": None, "started_utc": iso(utc_now()),
        }

    def worker() -> None:
        try:
            seg_id = capture_segment(ch, duration, source_type="on_demand", actor=actor)
            with _jobs_lock:
                _jobs[job_id].update(state="captured", segment_id=seg_id)
        except Exception as exc:                        # noqa: BLE001 — surfaced to UI
            log.warning("On-demand recording failed for %s: %s", channel_id, exc)
            record_gap(ch.id, utc_now(), str(exc))
            with _jobs_lock:
                _jobs[job_id].update(state="failed", error=str(exc)[:400])

    threading.Thread(target=worker, daemon=True, name=f"rec-{job_id}").start()
    return job_id


# Segment status -> what the operator sees for a recording job.
_JOB_STATE = {"captured": "queued", "queued": "queued", "transcribing": "transcribing",
              "translating": "translating", "transcribed": "done", "failed": "failed",
              "filtered": "filtered"}


def _with_pipeline_state(job: dict[str, Any]) -> dict[str, Any]:
    """Once captured, a job's progress is its segment's progress through the pipeline."""
    job = dict(job)
    if job.get("segment_id") and job["state"] != "failed":
        seg = db.query_one("SELECT status, filter_reason FROM segments WHERE id=?",
                           (job["segment_id"],))
        if seg:
            job["state"] = _JOB_STATE.get(seg["status"], seg["status"])
            if seg["status"] == "failed":
                job["error"] = seg["filter_reason"]
    return job


def job_status(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        job = dict(job) if job else None
    return _with_pipeline_state(job) if job else None


def all_jobs(limit: int = 20) -> list[dict[str, Any]]:
    with _jobs_lock:
        jobs = sorted((dict(j) for j in _jobs.values()), key=lambda j: j["started_utc"],
                      reverse=True)[:limit]
    return [_with_pipeline_state(j) for j in jobs]
