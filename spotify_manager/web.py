"""
web.py - Local web UI for spotify_manager.

This module keeps the existing CLI intact and adds a server-rendered browser
interface for the most common library cleanup and discovery flows.
"""

from __future__ import annotations

import datetime
import math
import time
from collections import defaultdict
from html import escape
from typing import Any
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

import spotipy

from .auth import get_spotify
from .models.taste_model import TasteModel
from .tasks.organise import GENRE_BUCKETS, MOOD_LABELS
from .utils.duplicates import build_duplicate_removal_payload, find_duplicate_entries
from .utils.audio import HAS_ML, _batch_audio_features, _batch_audio_features_with_ids, _cluster_tracks, _similarity_score
from .utils.display import console
from .utils.spotify import get_all_playlists, get_playlist_tracks, paginate

DISCOVERY_FEATURE_KEYS = [
    "energy",
    "valence",
    "danceability",
    "tempo",
    "acousticness",
    "instrumentalness",
    "speechiness",
]
GENRE_OTHER = "Other / Mixed"

APP_STYLES = """
:root {
  --bg: #f6f0e6;
  --bg-wash: #efe2cb;
  --card: rgba(255, 250, 242, 0.82);
  --card-strong: #fff7ed;
  --ink: #1f2933;
  --muted: #59636e;
  --line: rgba(31, 41, 51, 0.12);
  --accent: #c05c3c;
  --accent-soft: rgba(192, 92, 60, 0.14);
  --sea: #1f6f78;
  --sea-soft: rgba(31, 111, 120, 0.14);
  --gold: #c89b2a;
  --ok: #2f855a;
  --warn: #9a6700;
  --danger: #b42318;
  --shadow: 0 20px 45px rgba(45, 36, 28, 0.10);
  --radius-xl: 28px;
  --radius-lg: 18px;
  --radius-md: 12px;
}

* {
  box-sizing: border-box;
}

html {
  scroll-behavior: smooth;
}

body {
  margin: 0;
  color: var(--ink);
  font-family: "Avenir Next", "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at top left, rgba(200, 155, 42, 0.22), transparent 30%),
    radial-gradient(circle at top right, rgba(31, 111, 120, 0.18), transparent 32%),
    linear-gradient(180deg, var(--bg) 0%, var(--bg-wash) 100%);
}

body.is-loading {
  overflow: hidden;
}

.shell {
  max-width: 1260px;
  margin: 0 auto;
  padding: 32px 22px 64px;
}

.hero {
  position: relative;
  overflow: hidden;
  padding: 34px;
  border: 1px solid rgba(255, 255, 255, 0.5);
  border-radius: var(--radius-xl);
  background:
    linear-gradient(135deg, rgba(255, 247, 237, 0.95), rgba(247, 235, 219, 0.82)),
    linear-gradient(180deg, rgba(255, 255, 255, 0.4), rgba(255, 255, 255, 0));
  box-shadow: var(--shadow);
}

.hero::after {
  content: "";
  position: absolute;
  inset: auto -120px -150px auto;
  width: 360px;
  height: 360px;
  border-radius: 999px;
  background: radial-gradient(circle, rgba(31, 111, 120, 0.22), transparent 70%);
}

.eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 7px 12px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.55);
  color: var(--sea);
  font-size: 12px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.hero h1 {
  margin: 16px 0 10px;
  max-width: 700px;
  font-family: "Iowan Old Style", "Palatino Linotype", serif;
  font-size: clamp(2.4rem, 4vw, 4.2rem);
  line-height: 0.98;
}

.hero p {
  margin: 0;
  max-width: 740px;
  color: var(--muted);
  font-size: 1.02rem;
  line-height: 1.65;
}

.stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 14px;
  margin-top: 24px;
}

.stat {
  padding: 18px 18px 16px;
  border: 1px solid rgba(255, 255, 255, 0.6);
  border-radius: var(--radius-lg);
  background: rgba(255, 255, 255, 0.52);
  backdrop-filter: blur(10px);
}

.stat-label {
  color: var(--muted);
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.stat-value {
  margin-top: 10px;
  font-size: 1.9rem;
  font-weight: 700;
}

.stat-subtle {
  margin-top: 6px;
  color: var(--muted);
  font-size: 0.92rem;
}

.flash {
  margin-top: 22px;
  padding: 16px 18px;
  border-radius: var(--radius-md);
  border: 1px solid rgba(255, 255, 255, 0.7);
  box-shadow: var(--shadow);
}

.flash.success {
  background: rgba(47, 133, 90, 0.12);
  color: #174a32;
}

.flash.error {
  background: rgba(180, 35, 24, 0.10);
  color: #7a1d17;
}

.loading-overlay {
  position: fixed;
  inset: 0;
  display: grid;
  place-items: center;
  padding: 18px;
  background: rgba(31, 41, 51, 0.32);
  backdrop-filter: blur(12px);
  opacity: 0;
  visibility: hidden;
  pointer-events: none;
  transition: opacity 180ms ease, visibility 180ms ease;
  z-index: 999;
}

body.is-loading .loading-overlay {
  opacity: 1;
  visibility: visible;
  pointer-events: auto;
}

.loading-card {
  width: min(100%, 520px);
  padding: 24px;
  border-radius: var(--radius-xl);
  background:
    linear-gradient(135deg, rgba(255, 247, 237, 0.98), rgba(247, 235, 219, 0.94));
  border: 1px solid rgba(255, 255, 255, 0.72);
  box-shadow: var(--shadow);
}

.loading-kicker {
  color: var(--sea);
  font-size: 0.76rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.loading-title {
  margin: 10px 0 6px;
  font-size: 1.3rem;
}

.loading-detail {
  margin: 0;
  color: var(--muted);
  line-height: 1.6;
}

.loading-bar {
  position: relative;
  overflow: hidden;
  width: 100%;
  height: 14px;
  margin-top: 18px;
  border-radius: 999px;
  background: rgba(31, 41, 51, 0.08);
}

.loading-fill {
  width: 10%;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #c05c3c, #c89b2a, #1f6f78);
  transition: width 180ms ease;
}

.loading-meta {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  margin-top: 10px;
  color: var(--muted);
  font-size: 0.92rem;
}

.loading-note {
  margin-top: 14px;
  color: var(--muted);
  font-size: 0.9rem;
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
  gap: 18px;
  margin-top: 24px;
}

.panel {
  border: 1px solid rgba(255, 255, 255, 0.65);
  border-radius: var(--radius-xl);
  background: var(--card);
  box-shadow: var(--shadow);
  overflow: hidden;
}

.panel-head {
  padding: 24px 24px 12px;
}

.panel h2 {
  margin: 0;
  font-size: 1.45rem;
}

.panel-copy {
  margin: 9px 0 0;
  color: var(--muted);
  line-height: 1.6;
}

.panel-body {
  padding: 8px 24px 24px;
}

form {
  display: grid;
  gap: 12px;
}

.field-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px;
}

label {
  display: grid;
  gap: 6px;
  font-size: 0.92rem;
  color: var(--muted);
}

input,
select,
button,
textarea {
  width: 100%;
  padding: 12px 13px;
  border: 1px solid var(--line);
  border-radius: 12px;
  font: inherit;
  color: var(--ink);
  background: rgba(255, 255, 255, 0.96);
}

input:focus,
select:focus,
textarea:focus {
  outline: 2px solid rgba(192, 92, 60, 0.24);
  outline-offset: 1px;
  border-color: rgba(192, 92, 60, 0.35);
}

.button-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

button {
  width: auto;
  min-width: 138px;
  cursor: pointer;
  border: 0;
  transition: transform 120ms ease, opacity 120ms ease, box-shadow 120ms ease;
}

button:hover {
  transform: translateY(-1px);
}

.primary {
  color: #fff;
  background: linear-gradient(135deg, #c05c3c, #9f4226);
  box-shadow: 0 10px 18px rgba(192, 92, 60, 0.24);
}

.secondary {
  color: var(--sea);
  background: linear-gradient(135deg, rgba(31, 111, 120, 0.18), rgba(31, 111, 120, 0.10));
}

.subtle {
  color: var(--ink);
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid var(--line);
}

.tag {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--accent);
  font-size: 0.84rem;
}

.results {
  margin-top: 18px;
  padding: 16px;
  border-radius: var(--radius-lg);
  background: rgba(255, 255, 255, 0.58);
  border: 1px solid rgba(255, 255, 255, 0.65);
}

.results h3 {
  margin: 0 0 10px;
  font-size: 1.05rem;
}

.table-wrap {
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.94rem;
}

th,
td {
  padding: 10px 10px;
  text-align: left;
  border-bottom: 1px solid rgba(31, 41, 51, 0.08);
}

th {
  color: var(--muted);
  font-size: 0.78rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.muted {
  color: var(--muted);
}

.mono {
  font-family: "SFMono-Regular", "Menlo", monospace;
}

.stack {
  display: grid;
  gap: 12px;
}

.mini-card {
  padding: 14px;
  border-radius: 14px;
  border: 1px solid rgba(31, 41, 51, 0.08);
  background: rgba(255, 255, 255, 0.72);
}

.mini-card strong {
  display: block;
  margin-bottom: 5px;
}

.chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.chip {
  padding: 7px 10px;
  border-radius: 999px;
  background: rgba(31, 111, 120, 0.10);
  color: var(--sea);
  font-size: 0.85rem;
}

.radio-list {
  display: grid;
  gap: 10px;
  margin-top: 14px;
}

.radio-item {
  display: flex;
  gap: 10px;
  align-items: flex-start;
  padding: 12px;
  border-radius: 14px;
  border: 1px solid rgba(31, 41, 51, 0.08);
  background: rgba(255, 255, 255, 0.76);
}

.radio-item input {
  width: auto;
  margin-top: 2px;
}

.footer-note {
  margin-top: 22px;
  padding: 20px 22px;
  border-radius: var(--radius-xl);
  background: rgba(31, 111, 120, 0.10);
  color: #24454a;
}

.footer-note p {
  margin: 0;
  line-height: 1.6;
}

details {
  padding: 12px 0;
  border-bottom: 1px solid rgba(31, 41, 51, 0.08);
}

details:last-child {
  border-bottom: 0;
}

summary {
  cursor: pointer;
  font-weight: 600;
}

.summary-line {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
  margin-top: 8px;
}

.warning {
  color: var(--warn);
}

@media (max-width: 720px) {
  .shell {
    padding-inline: 14px;
  }

  .hero {
    padding: 24px;
  }

  .panel-head,
  .panel-body {
    padding-inline: 18px;
  }

  .button-row {
    flex-direction: column;
  }

  button {
    width: 100%;
  }
}
"""

APP_SCRIPT = """
(() => {
  const overlay = document.querySelector("[data-loading-overlay]");
  if (!overlay) return;

  const title = overlay.querySelector("[data-loading-title]");
  const detail = overlay.querySelector("[data-loading-detail]");
  const bar = overlay.querySelector("[role='progressbar']");
  const fill = overlay.querySelector("[data-loading-fill]");
  const percent = overlay.querySelector("[data-loading-percent]");
  const elapsed = overlay.querySelector("[data-loading-elapsed]");

  let intervalId = null;
  let startTime = 0;
  let progressValue = 10;

  const stopTicker = () => {
    if (intervalId) {
      window.clearInterval(intervalId);
      intervalId = null;
    }
  };

  const setProgress = (value) => {
    progressValue = Math.max(8, Math.min(94, value));
    fill.style.width = `${progressValue}%`;
    bar.setAttribute("aria-valuenow", `${Math.round(progressValue)}`);
    percent.textContent = `${Math.round(progressValue)}%`;
  };

  const startTicker = () => {
    stopTicker();
    startTime = Date.now();
    intervalId = window.setInterval(() => {
      const seconds = Math.max(0, Math.floor((Date.now() - startTime) / 1000));
      elapsed.textContent = `${seconds}s elapsed`;
      const next = progressValue + (progressValue < 55 ? 7 : progressValue < 80 ? 3 : 1.2);
      setProgress(next);
    }, 220);
  };

  const startLoading = (message, subcopy) => {
    title.textContent = message || "Working on your Spotify library...";
    detail.textContent = subcopy || "This progress bar shows that the request is still active while the local server works.";
    setProgress(10);
    elapsed.textContent = "0s elapsed";
    document.body.classList.add("is-loading");
    startTicker();
  };

  document.addEventListener("submit", (event) => {
    const form = event.target;
    if (!(form instanceof HTMLFormElement)) return;
    if ((form.method || "").toLowerCase() !== "post") return;

    const submitter = event.submitter;
    const message =
      (submitter && submitter.dataset.loadingMessage) ||
      form.dataset.loadingMessage ||
      "Working on your Spotify library...";
    const subcopy =
      (submitter && submitter.dataset.loadingDetail) ||
      form.dataset.loadingDetail ||
      "This progress bar shows that the request is still active while the local server works.";

    form.querySelectorAll("button, input[type='submit']").forEach((element) => {
      element.disabled = true;
    });

    startLoading(message, subcopy);
  }, true);

  window.addEventListener("pageshow", () => {
    stopTicker();
    document.body.classList.remove("is-loading");
  });
})();
"""


def run_web_app(host: str = "127.0.0.1", port: int = 8000):
    """Serve the local browser UI until interrupted."""
    app = build_app()
    console.print(f"[bold green]spotify_manager web UI[/bold green] running at http://{host}:{port}")
    console.print("Press Ctrl+C to stop the server.")
    with make_server(host, port, app) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            console.print("\n[dim]Web UI stopped.[/dim]")


def build_app():
    def app(environ, start_response):
        status = "200 OK"
        try:
            sp = get_spotify()
            overview, playlists = load_dashboard_context(sp)
            active_section = None
            flash = None
            sections: dict[str, Any] = {}

            method = environ.get("REQUEST_METHOD", "GET").upper()
            path = environ.get("PATH_INFO", "/")

            if method == "POST":
                form = parse_form(environ)
                try:
                    if path == "/duplicates":
                        active_section = "duplicates"
                        playlist_id = get_value(form, "playlist_id")
                        op = get_value(form, "op", "scan")
                        if op == "remove":
                            removal = remove_duplicates(sp, playlist_id)
                            flash = {"kind": "success", "message": removal["message"]}
                        sections["duplicates"] = {
                            "playlist_id": playlist_id,
                            "results": scan_duplicates(sp, playlist_id),
                        }
                    elif path == "/never-played":
                        active_section = "never-played"
                        playlist_id = get_value(form, "playlist_id")
                        days = to_int(get_value(form, "cutoff_days", "30"), 30, minimum=1, maximum=3650)
                        sections["never_played"] = {
                            "playlist_id": playlist_id,
                            "cutoff_days": days,
                            "report": scan_never_played(sp, playlist_id, days),
                        }
                    elif path == "/size-audit":
                        active_section = "size-audit"
                        playlist_id = get_value(form, "playlist_id")
                        threshold = to_int(get_value(form, "threshold", "80"), 80, minimum=1, maximum=10000)
                        op = get_value(form, "op", "scan")
                        sections["size_audit"] = {
                            "playlist_id": playlist_id,
                            "threshold": threshold,
                            "report": audit_playlist_sizes(sp, playlist_id, threshold),
                        }
                        if op == "preview":
                            sections["size_audit"]["preview"] = build_smart_split_preview(sp, playlist_id)
                    elif path == "/organize-liked":
                        active_section = "organize-liked"
                        strategy = get_value(form, "strategy", "both")
                        prefix = get_value(form, "prefix", "Liked Songs")
                        mood_clusters = to_int(get_value(form, "mood_clusters", "4"), 4, minimum=2, maximum=8)
                        create_playlists = get_value(form, "create_playlists") == "yes"
                        sections["organize"] = {
                            "strategy": strategy,
                            "prefix": prefix,
                            "mood_clusters": mood_clusters,
                            "create_playlists": create_playlists,
                            "result": organize_liked_web(
                                sp,
                                strategy=strategy,
                                prefix=prefix,
                                n_clusters=mood_clusters,
                                create_playlists=create_playlists,
                            ),
                        }
                        if create_playlists:
                            flash = {
                                "kind": "success",
                                "message": "Playlist organization finished. New playlists were created where groups had tracks.",
                            }
                    elif path == "/discovery":
                        active_section = "discovery"
                        op = get_value(form, "op", "search")
                        state = {
                            "mode": get_value(form, "mode", "song"),
                            "song_query": get_value(form, "song_query"),
                            "playlist_id": get_value(form, "playlist_id"),
                            "n_results": to_int(get_value(form, "n_results", "20"), 20, minimum=1, maximum=100),
                            "save_playlist_name": get_value(form, "save_playlist_name"),
                        }
                        sections["discovery"] = {"state": state}
                        if op == "search":
                            if not state["song_query"].strip():
                                raise ValueError("Enter a song search before trying to find seed tracks.")
                            sections["discovery"]["search_results"] = search_seed_tracks(sp, state["song_query"])
                        else:
                            selected_track_id = get_value(form, "selected_track_id")
                            sections["discovery"]["result"] = run_discovery_web(
                                sp,
                                mode=state["mode"],
                                seed_track_id=selected_track_id,
                                playlist_id=state["playlist_id"],
                                n_results=state["n_results"],
                                save_playlist_name=state["save_playlist_name"],
                            )
                            if sections["discovery"]["result"].get("saved_to"):
                                flash = {
                                    "kind": "success",
                                    "message": f"Recommendations saved to '{sections['discovery']['result']['saved_to']}'.",
                                }
                except Exception as exc:
                    flash = {"kind": "error", "message": str(exc)}

            html = render_dashboard(overview, playlists, sections, flash, active_section)
        except SystemExit as exc:  # pragma: no cover - auth helper exits on missing credentials
            status = "500 Internal Server Error"
            html = render_error_page(RuntimeError(f"Spotify credentials are missing or invalid (exit code {exc.code})."))
        except Exception as exc:  # pragma: no cover - best effort browser diagnostics
            status = "500 Internal Server Error"
            html = render_error_page(exc)

        start_response(
            status,
            [
                ("Content-Type", "text/html; charset=utf-8"),
                ("Cache-Control", "no-store"),
            ],
        )
        return [html.encode("utf-8")]

    return app


def parse_form(environ) -> dict[str, list[str]]:
    try:
        length = int(environ.get("CONTENT_LENGTH", "0") or "0")
    except ValueError:
        length = 0
    raw = environ["wsgi.input"].read(length).decode("utf-8")
    return parse_qs(raw, keep_blank_values=True)


def get_value(form: dict[str, list[str]], key: str, default: str = "") -> str:
    values = form.get(key)
    return values[0] if values else default


def to_int(value: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def load_dashboard_context(sp: spotipy.Spotify) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    me = sp.current_user()
    playlists = sorted(get_all_playlists(sp), key=lambda item: item["name"].lower())
    saved = sp.current_user_saved_tracks(limit=1)
    recent = sp.current_user_recently_played(limit=10)
    model = TasteModel.load()

    overview = {
        "user_name": me.get("display_name") or me.get("id") or "Spotify user",
        "profile_url": me.get("external_urls", {}).get("spotify", ""),
        "owned_playlists": len(playlists),
        "liked_tracks": saved.get("total", 0),
        "followers": me.get("followers", {}).get("total", 0),
        "recent_tracks": len(recent.get("items", [])),
        "taste_model_ready": bool(model.clusters),
        "taste_sessions": model.session_count,
    }
    return overview, playlists


def require_playlist(playlists: list[dict[str, Any]], playlist_id: str) -> dict[str, Any]:
    if not playlist_id:
        raise ValueError("Choose a playlist first.")
    playlist = next((item for item in playlists if item["id"] == playlist_id), None)
    if not playlist:
        raise ValueError("That playlist was not found in your owned playlists.")
    return playlist


def scan_duplicates(sp: spotipy.Spotify, playlist_id: str) -> list[dict[str, Any]]:
    playlists = get_all_playlists(sp)
    playlist = require_playlist(playlists, playlist_id)
    results = []
    tracks = get_playlist_tracks(sp, playlist["id"])
    duplicates = find_duplicate_entries(tracks)
    if duplicates:
        results.append({
            "playlist_id": playlist["id"],
            "playlist_name": playlist["name"],
            "duplicate_count": len(duplicates),
            "tracks": duplicates,
        })
    return results


def remove_duplicates(sp: spotipy.Spotify, playlist_id: str) -> dict[str, Any]:
    playlists = get_all_playlists(sp)
    playlist = require_playlist(playlists, playlist_id)

    tracks = get_playlist_tracks(sp, playlist_id)
    duplicates = find_duplicate_entries(tracks)
    if not duplicates:
        return {"removed": 0, "message": f"'{playlist['name']}' is already clean."}

    payload = build_duplicate_removal_payload(duplicates)
    for start in range(0, len(payload), 100):
        sp.playlist_remove_specific_occurrences_of_items(playlist_id, payload[start:start + 100])

    removed = len(duplicates)
    return {"removed": removed, "message": f"Removed {removed} duplicate tracks from '{playlist['name']}'."}


def scan_never_played(sp: spotipy.Spotify, playlist_id: str, cutoff_days: int) -> dict[str, Any]:
    recent = sp.current_user_recently_played(limit=50)
    played_ids = {item["track"]["id"] for item in recent.get("items", []) if item.get("track")}
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=cutoff_days)
    playlist = require_playlist(get_all_playlists(sp), playlist_id)

    rows = []
    for item in get_playlist_tracks(sp, playlist["id"]):
        track = item.get("track") or {}
        track_id = track.get("id")
        added_at = item.get("added_at")
        if not track_id or not added_at:
            continue
        try:
            added_dt = datetime.datetime.strptime(added_at, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
        if track_id not in played_ids and added_dt < cutoff:
            rows.append({
                "playlist": playlist["name"],
                "track": track.get("name", "Unknown track"),
                "artist": (track.get("artists") or [{"name": "Unknown artist"}])[0]["name"],
                "added": added_at[:10],
            })

    rows.sort(key=lambda row: (row["playlist"].lower(), row["added"], row["track"].lower()))
    visible_rows = rows[:250]
    return {
        "count": len(rows),
        "truncated": len(rows) > len(visible_rows),
        "rows": visible_rows,
    }


def audit_playlist_sizes(sp: spotipy.Spotify, playlist_id: str, threshold: int) -> list[dict[str, Any]]:
    playlist = require_playlist(get_all_playlists(sp), playlist_id)
    count = (playlist.get("tracks") or {}).get("total", 0)
    if count <= threshold:
        return []
    return [{
        "playlist_id": playlist["id"],
        "playlist_name": playlist["name"],
        "track_count": count,
        "suggested_splits": math.ceil(count / threshold),
    }]


def build_smart_split_preview(sp: spotipy.Spotify, playlist_id: str) -> dict[str, Any]:
    if not HAS_ML:
        raise ValueError("Smart split preview needs numpy and scikit-learn installed.")

    playlist = require_playlist(get_all_playlists(sp), playlist_id)

    tracks = get_playlist_tracks(sp, playlist_id)
    ids = [item["track"]["id"] for item in tracks if item.get("track") and item["track"].get("id")]
    feature_rows = _batch_audio_features_with_ids(sp, ids)
    features = [features for _, features in feature_rows]
    if len(features) < 2:
        raise ValueError("This playlist does not have enough audio features for a split preview.")

    n_clusters = max(2, min(5, max(2, len(features) // 20)))
    labels, cluster_info = _cluster_tracks(features, n_clusters)
    suggestions = []
    for index, info in enumerate(cluster_info):
        track_count = sum(1 for (_, _), label in zip(feature_rows, labels) if label == index)
        suggestions.append({
            "label": f"Set {index + 1}",
            "track_count": track_count,
            "energy": round(info["energy"], 2),
            "valence": round(info["valence"], 2),
            "tempo": round(info["tempo"]),
        })
    return {
        "playlist_name": playlist["name"],
        "clusters": suggestions,
    }


def organize_liked_web(
    sp: spotipy.Spotify,
    *,
    strategy: str,
    prefix: str,
    n_clusters: int,
    create_playlists: bool,
) -> dict[str, Any]:
    liked_raw = paginate(sp.current_user_saved_tracks)
    liked_tracks = [
        item["track"] for item in liked_raw
        if item and item.get("track") and item["track"].get("id") and not item["track"].get("is_local")
    ]
    if not liked_tracks:
        raise ValueError("No liked songs were found on this account.")

    do_mood = strategy in {"mood", "both"}
    do_genre = strategy in {"genre", "both"}
    result: dict[str, Any] = {
        "total_liked": len(liked_tracks),
        "mood_groups": [],
        "genre_groups": [],
        "created_playlists": [],
    }
    ids = [track["id"] for track in liked_tracks]

    if do_mood:
        if not HAS_ML:
            raise ValueError("Mood clustering needs numpy and scikit-learn installed.")
        feature_rows = _batch_audio_features_with_ids(sp, ids)
        features = [features for _, features in feature_rows]
        if not features:
            raise ValueError("Spotify did not return audio features for the liked songs scan.")
        labels, cluster_info = _cluster_tracks(features, n_clusters)
        mood_clusters: dict[int, list[dict[str, Any]]] = defaultdict(list)
        tracks_by_id = {track["id"]: track for track in liked_tracks}
        for (track_id, _), label in zip(feature_rows, labels):
            track = tracks_by_id.get(track_id)
            if track:
                mood_clusters[int(label)].append(track)

        result["mood_groups"] = [
            {
                "label": MOOD_LABELS.get(index, f"Mood {index + 1}"),
                "track_count": len(mood_clusters[index]),
                "energy": round(info["energy"], 2),
                "valence": round(info["valence"], 2),
                "tempo": round(info["tempo"]),
            }
            for index, info in enumerate(cluster_info)
        ]

        if create_playlists:
            grouped_tracks = {
                MOOD_LABELS.get(index, f"Mood {index + 1}"): tracks
                for index, tracks in mood_clusters.items()
                if tracks
            }
            result["created_playlists"].extend(
                create_group_playlists(
                    sp,
                    prefix=prefix,
                    groups=grouped_tracks,
                    description_prefix="Auto-organized by mood",
                )
            )

    if do_genre:
        artist_ids = []
        track_primary_artist: dict[str, str] = {}
        for track in liked_tracks:
            artists = track.get("artists") or []
            if not artists or not artists[0].get("id"):
                continue
            artist_id = artists[0]["id"]
            artist_ids.append(artist_id)
            track_primary_artist[track["id"]] = artist_id

        genre_map = batch_artist_genres(sp, artist_ids)
        genre_clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for track in liked_tracks:
            artist_id = track_primary_artist.get(track["id"], "")
            bucket = assign_genre_bucket(genre_map.get(artist_id, []))
            genre_clusters[bucket].append(track)

        result["genre_groups"] = [
            {"label": label, "track_count": len(tracks)}
            for label, tracks in sorted(genre_clusters.items(), key=lambda item: (-len(item[1]), item[0].lower()))
            if len(tracks) >= 5
        ]

        if create_playlists:
            grouped_tracks = {
                label: tracks for label, tracks in genre_clusters.items() if len(tracks) >= 5
            }
            result["created_playlists"].extend(
                create_group_playlists(
                    sp,
                    prefix=prefix,
                    groups=grouped_tracks,
                    description_prefix="Auto-organized by genre",
                )
            )

    return result


def batch_artist_genres(sp: spotipy.Spotify, artist_ids: list[str]) -> dict[str, list[str]]:
    genre_map = {}
    unique_ids = list(dict.fromkeys(artist_ids))
    for start in range(0, len(unique_ids), 50):
        batch = unique_ids[start:start + 50]
        artists = sp.artists(batch).get("artists", [])
        for artist in artists:
            if artist:
                genre_map[artist["id"]] = artist.get("genres", [])
        time.sleep(0.05)
    return genre_map


def assign_genre_bucket(genres: list[str]) -> str:
    lowered = [genre.lower() for genre in genres]
    for bucket, keywords in GENRE_BUCKETS.items():
        for keyword in keywords:
            if any(keyword in genre for genre in lowered):
                return bucket
    return GENRE_OTHER


def create_group_playlists(
    sp: spotipy.Spotify,
    *,
    prefix: str,
    groups: dict[str, list[dict[str, Any]]],
    description_prefix: str,
) -> list[str]:
    me = sp.current_user()
    created = []
    for label, tracks in groups.items():
        if not tracks:
            continue
        playlist_name = f"{prefix} - {label}"
        playlist = sp.user_playlist_create(
            me["id"],
            playlist_name,
            public=False,
            description=f"{description_prefix} - {len(tracks)} tracks",
        )
        uris = [track["uri"] for track in tracks if track.get("uri")]
        for start in range(0, len(uris), 100):
            sp.playlist_add_items(playlist["id"], uris[start:start + 100])
            time.sleep(0.1)
        created.append(playlist_name)
    return created


def search_seed_tracks(sp: spotipy.Spotify, query: str) -> list[dict[str, Any]]:
    results = sp.search(q=query, type="track", limit=8)
    tracks = []
    for item in results.get("tracks", {}).get("items", []):
        tracks.append({
            "id": item["id"],
            "name": item["name"],
            "artist": (item.get("artists") or [{"name": "Unknown artist"}])[0]["name"],
            "album": item.get("album", {}).get("name", ""),
        })
    if not tracks:
        raise ValueError("No seed tracks matched that search.")
    return tracks


def run_discovery_web(
    sp: spotipy.Spotify,
    *,
    mode: str,
    seed_track_id: str,
    playlist_id: str,
    n_results: int,
    save_playlist_name: str,
) -> dict[str, Any]:
    do_song = mode in {"song", "both"}
    do_playlist = mode in {"playlist", "both"}

    seed_features: dict[str, Any] = {}
    playlist_features: dict[str, Any] = {}
    seed_track_name = ""
    playlist_name = ""

    if do_song:
        if not seed_track_id:
            raise ValueError("Choose a seed track first for song-based discovery.")
        seed_track = sp.track(seed_track_id)
        seed_track_name = seed_track.get("name", "Selected track")
        seed_audio = sp.audio_features([seed_track_id])
        if seed_audio and seed_audio[0]:
            seed_features = {
                key: seed_audio[0][key]
                for key in DISCOVERY_FEATURE_KEYS
                if key in seed_audio[0]
            }

    if do_playlist:
        if not playlist_id:
            raise ValueError("Choose a playlist when using playlist-based discovery.")
        playlist = next((item for item in get_all_playlists(sp) if item["id"] == playlist_id), None)
        if not playlist:
            raise ValueError("That playlist was not found in your library.")
        playlist_name = playlist["name"]
        playlist_features = average_playlist_profile(sp, playlist_id)

    if do_song and do_playlist and seed_features and playlist_features:
        target_features = {}
        for key in DISCOVERY_FEATURE_KEYS:
            song_value = seed_features.get(key)
            playlist_value = playlist_features.get(key)
            if song_value is not None and playlist_value is not None:
                target_features[key] = round(0.4 * song_value + 0.6 * playlist_value, 4)
            elif song_value is not None:
                target_features[key] = song_value
            elif playlist_value is not None:
                target_features[key] = playlist_value
    elif do_song and seed_features:
        target_features = seed_features
    elif do_playlist and playlist_features:
        target_features = playlist_features
    else:
        raise ValueError("A target profile could not be built from the selected discovery settings.")

    seed_tracks = [seed_track_id] if seed_track_id else []
    if do_playlist and not do_song:
        seed_tracks = pick_representative_seeds(sp, target_features)

    request_kwargs: dict[str, Any] = {
        "limit": min(n_results, 100),
    }
    if seed_tracks:
        request_kwargs["seed_tracks"] = seed_tracks

    feature_param_map = {
        "energy": ("target_energy", "min_energy", "max_energy"),
        "valence": ("target_valence", "min_valence", "max_valence"),
        "danceability": ("target_danceability", "min_danceability", "max_danceability"),
        "tempo": ("target_tempo", "min_tempo", "max_tempo"),
        "acousticness": ("target_acousticness", "min_acousticness", "max_acousticness"),
        "instrumentalness": ("target_instrumentalness", "min_instrumentalness", "max_instrumentalness"),
    }
    tolerance = {
        "energy": 0.15,
        "valence": 0.15,
        "danceability": 0.15,
        "tempo": 20,
        "acousticness": 0.2,
        "instrumentalness": 0.2,
    }
    for feature, keys in feature_param_map.items():
        value = target_features.get(feature)
        if value is None:
            continue
        target_key, min_key, max_key = keys
        request_kwargs[target_key] = value
        window = tolerance[feature]
        if feature == "tempo":
            request_kwargs[min_key] = max(40, round(value - window))
            request_kwargs[max_key] = min(220, round(value + window))
        else:
            request_kwargs[min_key] = max(0.0, round(value - window, 3))
            request_kwargs[max_key] = min(1.0, round(value + window, 3))

    recommendations = sp.recommendations(**request_kwargs).get("tracks", [])
    if not recommendations:
        raise ValueError("Spotify did not return any recommendations for that setup.")

    recommendation_ids = [track["id"] for track in recommendations if track.get("id")]
    recommendation_features = {
        features["id"]: features for features in _batch_audio_features(sp, recommendation_ids) if features
    }

    rows = []
    for track in recommendations:
        track_id = track["id"]
        features = recommendation_features.get(track_id, {})
        rows.append({
            "name": track["name"],
            "artist": (track.get("artists") or [{"name": "Unknown artist"}])[0]["name"],
            "url": track.get("external_urls", {}).get("spotify", ""),
            "energy": round(features.get("energy", 0.0), 2),
            "valence": round(features.get("valence", 0.0), 2),
            "danceability": round(features.get("danceability", 0.0), 2),
            "tempo": round(features.get("tempo", 0.0)),
            "match_pct": round(_similarity_score(target_features, features), 1),
            "uri": track["uri"],
        })

    saved_to = ""
    if save_playlist_name.strip():
        me = sp.current_user()
        new_playlist = sp.user_playlist_create(
            me["id"],
            save_playlist_name.strip(),
            public=False,
            description="Discovered via spotify_manager web UI",
        )
        uris = [row["uri"] for row in rows]
        for start in range(0, len(uris), 100):
            sp.playlist_add_items(new_playlist["id"], uris[start:start + 100])
        saved_to = save_playlist_name.strip()

    return {
        "rows": rows,
        "seed_track_name": seed_track_name,
        "playlist_name": playlist_name,
        "saved_to": saved_to,
    }


def average_playlist_profile(sp: spotipy.Spotify, playlist_id: str) -> dict[str, float]:
    tracks = get_playlist_tracks(sp, playlist_id)
    ids = [item["track"]["id"] for item in tracks if item.get("track") and item["track"].get("id")]
    features = _batch_audio_features(sp, ids)
    if not features:
        raise ValueError("Spotify did not return audio features for that playlist.")

    profile = {}
    for key in DISCOVERY_FEATURE_KEYS:
        values = [feature[key] for feature in features if key in feature]
        profile[key] = float(sum(values) / len(values)) if values else 0.0
    return profile


def pick_representative_seeds(sp: spotipy.Spotify, target: dict[str, Any], n: int = 2) -> list[str]:
    top = sp.current_user_top_tracks(limit=50, time_range="medium_term")
    candidates = top.get("items", [])
    ids = [track["id"] for track in candidates if track.get("id")]
    if not ids:
        return []
    features = {row["id"]: row for row in _batch_audio_features(sp, ids) if row}
    scored = sorted(
        (
            (track_id, _similarity_score(target, features[track_id]))
            for track_id in ids
            if track_id in features
        ),
        key=lambda item: -item[1],
    )
    return [track_id for track_id, _ in scored[:n]]


def render_dashboard(
    overview: dict[str, Any],
    playlists: list[dict[str, Any]],
    sections: dict[str, Any],
    flash: dict[str, str] | None,
    active_section: str | None,
) -> str:
    taste_status = "Ready" if overview["taste_model_ready"] else "Not built"
    flash_html = render_flash(flash)
    section_jump = (
        f"<script>window.location.hash = '#{escape(active_section)}';</script>"
        if active_section else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>spotify_manager web</title>
  <style>{APP_STYLES}</style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="eyebrow">spotify_manager web</div>
      <h1>Move the toolkit out of the terminal and into the browser.</h1>
      <p>
        This local dashboard keeps the cleanup and discovery tools close to the Spotify data,
        but gives them a calmer UI with forms, previews, and action buttons instead of stacked prompts.
      </p>
      <div class="stats">
        <article class="stat">
          <div class="stat-label">Account</div>
          <div class="stat-value">{escape(str(overview["user_name"]))}</div>
          <div class="stat-subtle">{render_profile_link(overview["profile_url"])}</div>
        </article>
        <article class="stat">
          <div class="stat-label">Owned Playlists</div>
          <div class="stat-value">{overview["owned_playlists"]}</div>
          <div class="stat-subtle">Only playlists you own are modified here.</div>
        </article>
        <article class="stat">
          <div class="stat-label">Liked Songs</div>
          <div class="stat-value">{overview["liked_tracks"]}</div>
          <div class="stat-subtle">Available to organize by mood or genre.</div>
        </article>
        <article class="stat">
          <div class="stat-label">Taste Engine</div>
          <div class="stat-value">{escape(taste_status)}</div>
          <div class="stat-subtle">{overview["taste_sessions"]} recorded sessions.</div>
        </article>
      </div>
    </section>
    {flash_html}
    <section class="grid">
      {render_duplicates_section(playlists, sections.get("duplicates"))}
      {render_never_played_section(playlists, sections.get("never_played"))}
      {render_size_audit_section(playlists, sections.get("size_audit"))}
      {render_organize_section(sections.get("organize"))}
      {render_discovery_section(playlists, sections.get("discovery"))}
    </section>
    <section class="footer-note">
      <p>
        The browser UI covers the most form-friendly flows first. The full Taste Engine rating loop
        still fits better in the CLI today because it is a multi-step session with per-track feedback.
      </p>
    </section>
  </main>
  <div class="loading-overlay" data-loading-overlay aria-live="polite" aria-busy="true">
    <div class="loading-card">
      <div class="loading-kicker">Working</div>
      <h2 class="loading-title" data-loading-title>Working on your Spotify library...</h2>
      <p class="loading-detail" data-loading-detail>This progress bar shows that the request is still active while the local server works.</p>
      <div class="loading-bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="10">
        <div class="loading-fill" data-loading-fill></div>
      </div>
      <div class="loading-meta">
        <span data-loading-percent>10%</span>
        <span data-loading-elapsed>0s elapsed</span>
      </div>
      <p class="loading-note">The percentage is approximate, but it confirms the scan is still actively running.</p>
    </div>
  </div>
  {section_jump}
  <script>{APP_SCRIPT}</script>
</body>
</html>"""


def render_profile_link(url: str) -> str:
    if not url:
        return "Spotify account linked locally."
    return f'<a href="{escape(url)}" target="_blank" rel="noreferrer">Open profile</a>'


def render_flash(flash: dict[str, str] | None) -> str:
    if not flash:
        return ""
    kind = "success" if flash.get("kind") == "success" else "error"
    return f'<div class="flash {kind}">{escape(flash.get("message", ""))}</div>'


def render_loading_attrs(message: str, detail: str) -> str:
    return (
        f'data-loading-message="{escape(message, quote=True)}" '
        f'data-loading-detail="{escape(detail, quote=True)}"'
    )


def render_playlist_select(name: str, playlists: list[dict[str, Any]], selected_id: str, placeholder: str = "Choose a playlist") -> str:
    options = [f"<option value=''>{escape(placeholder)}</option>"]
    for playlist in playlists:
        is_selected = " selected" if playlist["id"] == selected_id else ""
        track_total = (playlist.get("tracks") or {}).get("total")
        track_total_label = f"{track_total} tracks" if isinstance(track_total, int) else "track count unknown"
        options.append(
            f"<option value='{escape(playlist['id'])}'{is_selected}>{escape(playlist['name'])} ({track_total_label})</option>"
        )
    return f"<select name=\"{escape(name, quote=True)}\">{''.join(options)}</select>"


def render_duplicates_section(playlists: list[dict[str, Any]], data: dict[str, Any] | None) -> str:
    playlist_id = data.get("playlist_id", "") if data else ""
    results_html = ""
    if data and data.get("results") is not None:
        results = data["results"]
        if not results:
            results_html = '<div class="results"><h3>No duplicates found.</h3><p class="muted">This playlist looks clean right now.</p></div>'
        else:
            blocks = []
            for playlist in results:
                track_items = "".join(
                    f"<li>{escape(track['name'])} — {escape(track['artist'])} "
                    f"<span class='muted'>(position {track['position'] + 1}, matches kept track at "
                    f"position {track['kept_position'] + 1} via {escape(track['match_label'])})</span></li>"
                    for track in playlist["tracks"][:20]
                )
                extra_note = ""
                if len(playlist["tracks"]) > 20:
                    extra_note = f"<p class='muted'>Showing the first 20 of {playlist['duplicate_count']} duplicate entries.</p>"
                blocks.append(
                    f"""
                    <details open>
                      <summary>{escape(playlist['playlist_name'])}</summary>
                      <div class="summary-line">
                        <span class="tag">{playlist['duplicate_count']} duplicates</span>
                        <form method="post" action="/duplicates" {render_loading_attrs("Removing duplicate songs from this playlist...", "Spotify has to remove the matching positions precisely, so this can take a moment on larger playlists.")}>
                          <input type="hidden" name="op" value="remove">
                          <input type="hidden" name="playlist_id" value="{escape(playlist['playlist_id'])}">
                          <button class="primary" type="submit">Remove duplicates</button>
                        </form>
                      </div>
                      <ul class="stack">{track_items}</ul>
                      {extra_note}
                    </details>
                    """
                )
            results_html = f'<div class="results"><h3>Duplicate scan</h3>{"".join(blocks)}</div>'

    return f"""
    <article class="panel" id="duplicates">
      <div class="panel-head">
        <div class="tag">Task 1</div>
        <h2>Duplicate cleaner</h2>
        <p class="panel-copy">Scan one playlist for the same song appearing more than once, even when Spotify IDs differ between versions.</p>
      </div>
      <div class="panel-body">
        <form method="post" action="/duplicates" {render_loading_attrs("Scanning the selected playlist for duplicate songs...", "Comparing songs by ISRC when available, otherwise by normalized title, artist, and duration.")}>
          <input type="hidden" name="op" value="scan">
          <div class="field-grid">
            <label>Playlist
              {render_playlist_select("playlist_id", playlists, playlist_id)}
            </label>
          </div>
          <div class="button-row">
            <button class="primary" type="submit">Scan playlist</button>
          </div>
        </form>
        {results_html}
      </div>
    </article>
    """


def render_never_played_section(playlists: list[dict[str, Any]], data: dict[str, Any] | None) -> str:
    playlist_id = data.get("playlist_id", "") if data else ""
    cutoff_days = 30
    results_html = ""
    if data:
        cutoff_days = data.get("cutoff_days", 30)
        report = data.get("report") or {}
        if report:
            if report["count"] == 0:
                results_html = '<div class="results"><h3>No stale tracks found.</h3><p class="muted">Nothing matched the current recency cutoff.</p></div>'
            else:
                rows = "".join(
                    f"<tr><td>{escape(row['playlist'])}</td><td>{escape(row['track'])}</td><td>{escape(row['artist'])}</td><td>{escape(row['added'])}</td></tr>"
                    for row in report["rows"]
                )
                truncation = (
                    "<p class='warning'>Showing the first 250 rows to keep the page usable.</p>"
                    if report.get("truncated") else ""
                )
                results_html = f"""
                <div class="results">
                  <h3>{report['count']} tracks look stale</h3>
                  <p class="muted">Spotify only exposes your last 50 played tracks, so treat this as a review list rather than a perfect history.</p>
                  {truncation}
                  <div class="table-wrap">
                    <table>
                      <thead>
                        <tr><th>Playlist</th><th>Track</th><th>Artist</th><th>Added</th></tr>
                      </thead>
                      <tbody>{rows}</tbody>
                    </table>
                  </div>
                </div>
                """

    return f"""
    <article class="panel" id="never-played">
      <div class="panel-head">
        <div class="tag">Task 2</div>
        <h2>Never-played finder</h2>
        <p class="panel-copy">Flag tracks that are not in your recent Spotify history and have been sitting in playlists for a while.</p>
      </div>
      <div class="panel-body">
        <form method="post" action="/never-played" {render_loading_attrs("Building your never-played review list...", "Checking the selected playlist against your recent listening history and added dates.")}>
          <div class="field-grid">
            <label>Playlist
              {render_playlist_select("playlist_id", playlists, playlist_id)}
            </label>
            <label>Older than how many days?
              <input type="number" min="1" max="3650" name="cutoff_days" value="{cutoff_days}">
            </label>
          </div>
          <div class="button-row">
            <button class="primary" type="submit">Generate review list</button>
          </div>
        </form>
        {results_html}
      </div>
    </article>
    """


def render_size_audit_section(playlists: list[dict[str, Any]], data: dict[str, Any] | None) -> str:
    playlist_id = data.get("playlist_id", "") if data else ""
    threshold = 80
    report = []
    preview = None
    if data:
        threshold = data.get("threshold", 80)
        report = data.get("report") or []
        preview = data.get("preview")

    report_html = ""
    if data:
        if not report:
            report_html = '<div class="results"><h3>No oversized playlists.</h3><p class="muted">Everything is currently under the chosen threshold.</p></div>'
        else:
            rows = []
            for item in report:
                rows.append(
                    f"""
                    <tr>
                      <td>{escape(item['playlist_name'])}</td>
                      <td>{item['track_count']}</td>
                      <td>{item['suggested_splits']}</td>
                      <td>
                        <form method="post" action="/size-audit" {render_loading_attrs("Preparing a smart split preview...", "Fetching playlist audio features and sketching rough mood-based groupings.")}>
                          <input type="hidden" name="op" value="preview">
                          <input type="hidden" name="threshold" value="{threshold}">
                          <input type="hidden" name="playlist_id" value="{escape(item['playlist_id'])}">
                          <button class="secondary" type="submit">Smart split</button>
                        </form>
                      </td>
                    </tr>
                    """
                )
            report_html = f"""
            <div class="results">
              <h3>Oversized playlists</h3>
              <div class="table-wrap">
                <table>
                  <thead>
                    <tr><th>Playlist</th><th>Tracks</th><th>Suggested sets</th><th>Preview</th></tr>
                  </thead>
                  <tbody>{''.join(rows)}</tbody>
                </table>
              </div>
            </div>
            """

    preview_html = ""
    if preview:
        cards = "".join(
            f"<div class='mini-card'><strong>{escape(item['label'])}</strong><div class='muted'>{item['track_count']} tracks</div><div class='muted'>energy {item['energy']} | valence {item['valence']} | tempo {item['tempo']} bpm</div></div>"
            for item in preview["clusters"]
        )
        preview_html = f"""
        <div class="results">
          <h3>Smart split preview for {escape(preview['playlist_name'])}</h3>
          <div class="stack">{cards}</div>
        </div>
        """

    return f"""
    <article class="panel" id="size-audit">
      <div class="panel-head">
        <div class="tag">Task 3</div>
        <h2>Playlist size audit</h2>
        <p class="panel-copy">Check whether a selected playlist is oversized, then preview a rough mood-based split before you reorganize anything.</p>
      </div>
      <div class="panel-body">
        <form method="post" action="/size-audit" {render_loading_attrs("Auditing the selected playlist...", "Checking whether this playlist is over your chosen size threshold.")}>
          <input type="hidden" name="op" value="scan">
          <div class="field-grid">
            <label>Playlist
              {render_playlist_select("playlist_id", playlists, playlist_id)}
            </label>
            <label>Oversized over this track count
              <input type="number" min="1" max="10000" name="threshold" value="{threshold}">
            </label>
          </div>
          <div class="button-row">
            <button class="primary" type="submit">Audit playlist</button>
          </div>
        </form>
        {report_html}
        {preview_html}
      </div>
    </article>
    """


def render_organize_section(data: dict[str, Any] | None) -> str:
    strategy = "both"
    prefix = "Liked Songs"
    mood_clusters = 4
    create_playlists = False
    result = None
    if data:
        strategy = data.get("strategy", strategy)
        prefix = data.get("prefix", prefix)
        mood_clusters = data.get("mood_clusters", mood_clusters)
        create_playlists = data.get("create_playlists", create_playlists)
        result = data.get("result")

    result_html = ""
    if result:
        group_cards = []
        if result["mood_groups"]:
            mood_cards = "".join(
                f"<div class='mini-card'><strong>{escape(group['label'])}</strong><div class='muted'>{group['track_count']} tracks</div><div class='muted'>energy {group['energy']} | valence {group['valence']} | tempo {group['tempo']} bpm</div></div>"
                for group in result["mood_groups"]
            )
            group_cards.append(f"<div><h3>Mood groups</h3><div class='stack'>{mood_cards}</div></div>")
        if result["genre_groups"]:
            genre_cards = "".join(
                f"<div class='mini-card'><strong>{escape(group['label'])}</strong><div class='muted'>{group['track_count']} tracks</div></div>"
                for group in result["genre_groups"]
            )
            group_cards.append(f"<div><h3>Genre groups</h3><div class='stack'>{genre_cards}</div></div>")
        created_html = ""
        if result["created_playlists"]:
            chips = "".join(f"<span class='chip'>{escape(name)}</span>" for name in result["created_playlists"])
            created_html = f"<div><h3>Created</h3><div class='chip-row'>{chips}</div></div>"
        result_html = f"""
        <div class="results">
          <h3>{result['total_liked']} liked songs scanned</h3>
          <div class="stack">
            {''.join(group_cards)}
            {created_html}
          </div>
        </div>
        """

    return f"""
    <article class="panel" id="organize-liked">
      <div class="panel-head">
        <div class="tag">Task 4</div>
        <h2>Liked songs organizer</h2>
        <p class="panel-copy">Preview mood and genre clusters for your liked songs, then optionally create the playlists directly from the UI.</p>
      </div>
      <div class="panel-body">
        <form method="post" action="/organize-liked" {render_loading_attrs("Organizing your liked songs...", "Spotify data and audio features are being grouped into mood and genre buckets.")}>
          <div class="field-grid">
            <label>Grouping strategy
              <select name="strategy">
                <option value="mood"{" selected" if strategy == "mood" else ""}>Mood</option>
                <option value="genre"{" selected" if strategy == "genre" else ""}>Genre</option>
                <option value="both"{" selected" if strategy == "both" else ""}>Both</option>
              </select>
            </label>
            <label>Playlist prefix
              <input type="text" name="prefix" value="{escape(prefix)}">
            </label>
            <label>Mood playlist count
              <input type="number" min="2" max="8" name="mood_clusters" value="{mood_clusters}">
            </label>
            <label>Create playlists now?
              <select name="create_playlists">
                <option value="no"{" selected" if not create_playlists else ""}>Preview only</option>
                <option value="yes"{" selected" if create_playlists else ""}>Create playlists</option>
              </select>
            </label>
          </div>
          <div class="button-row">
            <button class="primary" type="submit">Run organizer</button>
          </div>
        </form>
        {result_html}
      </div>
    </article>
    """


def render_discovery_section(playlists: list[dict[str, Any]], data: dict[str, Any] | None) -> str:
    state = {
        "mode": "song",
        "song_query": "",
        "playlist_id": "",
        "n_results": 20,
        "save_playlist_name": "",
    }
    search_results = []
    result = None
    if data:
        state.update(data.get("state") or {})
        search_results = data.get("search_results") or []
        result = data.get("result")

    options = ["<option value=''>Choose a playlist</option>"]
    for playlist in playlists:
        selected = " selected" if playlist["id"] == state["playlist_id"] else ""
        track_total = (playlist.get("tracks") or {}).get("total")
        track_total_label = f"{track_total} tracks" if isinstance(track_total, int) else "track count unknown"
        options.append(
            f"<option value='{escape(playlist['id'])}'{selected}>{escape(playlist['name'])} ({track_total_label})</option>"
        )

    search_html = ""
    if search_results:
        items = "".join(
            f"""
            <label class="radio-item">
              <input type="radio" name="selected_track_id" value="{escape(track['id'])}"{" checked" if index == 0 else ""}>
              <span><strong>{escape(track['name'])}</strong><br><span class="muted">{escape(track['artist'])} | {escape(track['album'])}</span></span>
            </label>
            """
            for index, track in enumerate(search_results)
        )
        search_html = f"""
        <div class="results">
          <h3>Choose a seed track</h3>
          <form method="post" action="/discovery" {render_loading_attrs("Running music discovery...", "Fetching recommendations that match your chosen seed track and playlist energy profile.")}>
            <input type="hidden" name="op" value="run">
            <input type="hidden" name="mode" value="{escape(state['mode'])}">
            <input type="hidden" name="song_query" value="{escape(state['song_query'])}">
            <input type="hidden" name="playlist_id" value="{escape(state['playlist_id'])}">
            <input type="hidden" name="n_results" value="{state['n_results']}">
            <input type="hidden" name="save_playlist_name" value="{escape(state['save_playlist_name'])}">
            <div class="radio-list">{items}</div>
            <div class="button-row">
              <button class="primary" type="submit">Run discovery</button>
            </div>
          </form>
        </div>
        """

    result_html = ""
    if result:
        rows = "".join(
            f"<tr><td>{escape(row['name'])}</td><td>{escape(row['artist'])}</td><td>{row['energy']}</td><td>{row['valence']}</td><td>{row['danceability']}</td><td>{row['tempo']}</td><td>{row['match_pct']}%</td><td>{render_track_link(row['url'])}</td></tr>"
            for row in result["rows"]
        )
        seed_bits = []
        if result.get("seed_track_name"):
            seed_bits.append(f"seed song: {escape(result['seed_track_name'])}")
        if result.get("playlist_name"):
            seed_bits.append(f"playlist profile: {escape(result['playlist_name'])}")
        context = " | ".join(seed_bits) if seed_bits else "recommendation results"
        result_html = f"""
        <div class="results">
          <h3>Discovery results</h3>
          <p class="muted">{context}</p>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>Track</th><th>Artist</th><th>Energy</th><th>Valence</th><th>Dance</th><th>Tempo</th><th>Match</th><th>Open</th></tr>
              </thead>
              <tbody>{rows}</tbody>
            </table>
          </div>
        </div>
        """

    return f"""
    <article class="panel" id="discovery">
      <div class="panel-head">
        <div class="tag">Task 5</div>
        <h2>One-shot discovery</h2>
        <p class="panel-copy">Blend a seed song, a playlist energy profile, or both to generate recommendations without dropping back into prompt-by-prompt CLI mode.</p>
      </div>
      <div class="panel-body">
        <form method="post" action="/discovery" {render_loading_attrs("Starting music discovery...", "Depending on the button you clicked, this will either search for seed tracks or fetch recommendations.")}>
          <div class="field-grid">
            <label>Mode
              <select name="mode">
                <option value="song"{" selected" if state["mode"] == "song" else ""}>By song</option>
                <option value="playlist"{" selected" if state["mode"] == "playlist" else ""}>By playlist</option>
                <option value="both"{" selected" if state["mode"] == "both" else ""}>Blend both</option>
              </select>
            </label>
            <label>Song search
              <input type="text" name="song_query" value="{escape(state['song_query'])}" placeholder="Artist and title">
            </label>
            <label>Playlist reference
              <select name="playlist_id">
                {''.join(options)}
              </select>
            </label>
            <label>Recommendation count
              <input type="number" min="1" max="100" name="n_results" value="{state['n_results']}">
            </label>
            <label>Optional playlist name for saving
              <input type="text" name="save_playlist_name" value="{escape(state['save_playlist_name'])}" placeholder="Leave blank to preview only">
            </label>
          </div>
          <div class="button-row">
            <button class="secondary" type="submit" name="op" value="search" data-loading-message="Searching for seed tracks..." data-loading-detail="Looking through Spotify's catalog so you can choose the right starting song.">Find seed tracks</button>
            <button class="primary" type="submit" name="op" value="run" data-loading-message="Running music discovery..." data-loading-detail="Fetching recommendations and scoring how closely they match your target sound.">Run discovery</button>
          </div>
        </form>
        {search_html}
        {result_html}
      </div>
    </article>
    """


def render_track_link(url: str) -> str:
    if not url:
        return '<span class="muted">n/a</span>'
    safe_url = escape(url)
    return f'<a href="{safe_url}" target="_blank" rel="noreferrer">Open</a>'


def render_error_page(exc: Exception) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>spotify_manager web error</title>
  <style>{APP_STYLES}</style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="eyebrow">spotify_manager web</div>
      <h1>Something needs attention before the page can continue.</h1>
      <p>{escape(str(exc))}</p>
    </section>
  </main>
</body>
</html>"""
