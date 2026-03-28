"""
discovery.py — Task 5: one-shot music discovery seeded by a song or playlist energy.
"""

import json
import math
from typing import Optional

import spotipy

from ..utils.display import HAS_RICH, console, confirm, ask
from ..utils.spotify import get_all_playlists, get_playlist_tracks
from ..utils.audio import _batch_audio_features, _similarity_score, _print_audio_profile

DISCOVERY_FEATURE_KEYS = ["energy", "valence", "danceability", "tempo",
                           "acousticness", "instrumentalness", "speechiness"]


def task_discovery(sp: spotipy.Spotify, dry_run: bool):
    console.rule("[bold cyan]Task 5 · Music Discovery[/bold cyan]")

    if HAS_RICH:
        console.print("[bold]Discovery mode:[/bold]")
        console.print("  [cyan]1[/cyan] · By Song     — find tracks similar to a song you love")
        console.print("  [cyan]2[/cyan] · By Playlist — find tracks matching a playlist's energy profile")
        console.print("  [cyan]3[/cyan] · Both        — song similarity filtered to match playlist energy")
        mode = ask("Choose", default="1")
    else:
        mode = ask("Discovery mode? 1=By Song, 2=By Playlist, 3=Both", default="1")

    do_song     = mode in ("1", "3")
    do_playlist = mode in ("2", "3")

    n_results = int(ask("How many recommendations to fetch?", default="20"))

    seed_track_id     = None
    seed_track_name   = None
    seed_features     = {}
    playlist_features = {}

    # ── Seed: Song ─────────────────────────────────────────────────────────────
    if do_song:
        seed_track_id, seed_track_name = _pick_seed_track(sp)
        if seed_track_id:
            feats = sp.audio_features([seed_track_id])
            if feats and feats[0]:
                seed_features = {k: feats[0][k] for k in DISCOVERY_FEATURE_KEYS if k in feats[0]}
                _print_audio_profile(f"Audio profile · {seed_track_name}", seed_features)

    # ── Seed: Playlist energy ──────────────────────────────────────────────────
    if do_playlist:
        playlist_features = _pick_playlist_profile(sp)

    # ── Merge features when mode=3 ─────────────────────────────────────────────
    if do_song and do_playlist and seed_features and playlist_features:
        merged = {}
        for k in DISCOVERY_FEATURE_KEYS:
            s = seed_features.get(k)
            p = playlist_features.get(k)
            if s is not None and p is not None:
                merged[k] = round(0.4 * s + 0.6 * p, 4)
            elif s is not None:
                merged[k] = s
            elif p is not None:
                merged[k] = p
        target_features = merged
        _print_audio_profile("Blended target profile (40% song · 60% playlist)", target_features)
    elif do_playlist and playlist_features:
        target_features = playlist_features
    elif do_song and seed_features:
        target_features = seed_features
    else:
        console.print("[red]Could not build a target profile — aborting discovery.[/red]")
        return

    # ── Build recommendation request ───────────────────────────────────────────
    seed_tracks  = [seed_track_id] if seed_track_id else []
    seed_artists = []
    seed_genres  = []

    if do_playlist and not do_song:
        seed_tracks = _pick_representative_seeds(sp, target_features)

    kwargs: dict = dict(
        seed_tracks  = seed_tracks  or None,
        seed_artists = seed_artists or None,
        seed_genres  = seed_genres  or None,
        limit        = min(n_results, 100),
    )

    feature_param_map = {
        "energy":           ("target_energy",           "min_energy",           "max_energy"),
        "valence":          ("target_valence",           "min_valence",          "max_valence"),
        "danceability":     ("target_danceability",      "min_danceability",     "max_danceability"),
        "tempo":            ("target_tempo",             "min_tempo",            "max_tempo"),
        "acousticness":     ("target_acousticness",      "min_acousticness",     "max_acousticness"),
        "instrumentalness": ("target_instrumentalness",  "min_instrumentalness", "max_instrumentalness"),
    }
    TOLERANCE = {
        "energy": 0.15, "valence": 0.15, "danceability": 0.15,
        "tempo": 20, "acousticness": 0.2, "instrumentalness": 0.2,
    }
    for feat, (target_key, min_key, max_key) in feature_param_map.items():
        val = target_features.get(feat)
        if val is None:
            continue
        tol = TOLERANCE.get(feat, 0.15)
        kwargs[target_key] = val
        if feat != "tempo":
            kwargs[min_key] = max(0.0, round(val - tol, 3))
            kwargs[max_key] = min(1.0, round(val + tol, 3))
        else:
            kwargs[min_key] = max(40, round(val - tol))
            kwargs[max_key] = min(220, round(val + tol))

    if not kwargs["seed_tracks"]:  del kwargs["seed_tracks"]
    if not kwargs["seed_artists"]: del kwargs["seed_artists"]
    if not kwargs["seed_genres"]:  del kwargs["seed_genres"]

    console.print("\nFetching recommendations from Spotify…")
    try:
        recs = sp.recommendations(**kwargs)
    except Exception as e:
        console.print(f"[red]Recommendations API error: {e}[/red]")
        return

    rec_tracks = recs.get("tracks", [])
    if not rec_tracks:
        console.print("[yellow]No recommendations returned. Try loosening the target parameters.[/yellow]")
        return

    # ── Display results ────────────────────────────────────────────────────────
    rec_ids   = [t["id"] for t in rec_tracks]
    rec_feats = {f["id"]: f for f in _batch_audio_features(sp, rec_ids) if f}

    console.print(f"\n[bold]Top {len(rec_tracks)} recommendations:[/bold]")
    if HAS_RICH:
        from rich.table import Table
        table = Table(show_lines=True)
        table.add_column("#",       width=3,  justify="right")
        table.add_column("Track",   style="white")
        table.add_column("Artist",  style="dim")
        table.add_column("Energy",  justify="right", width=7)
        table.add_column("Valence", justify="right", width=7)
        table.add_column("Dance",   justify="right", width=6)
        table.add_column("BPM",     justify="right", width=5)
        table.add_column("Match %", justify="right", style="green", width=8)
        for idx, t in enumerate(rec_tracks, 1):
            tid   = t["id"]
            feats = rec_feats.get(tid, {})
            score = _similarity_score(target_features, feats)
            table.add_row(
                str(idx),
                t["name"][:40],
                t["artists"][0]["name"] if t["artists"] else "?",
                f"{feats.get('energy', 0):.2f}",
                f"{feats.get('valence', 0):.2f}",
                f"{feats.get('danceability', 0):.2f}",
                f"{feats.get('tempo', 0):.0f}",
                f"{score:.0f}%",
            )
        console.print(table)
    else:
        for idx, t in enumerate(rec_tracks, 1):
            feats = rec_feats.get(t["id"], {})
            score = _similarity_score(target_features, feats)
            print(f"  {idx:>2}. {t['name']} — {t['artists'][0]['name'] if t['artists'] else '?'} ({score:.0f}% match)")

    # Save to JSON
    report = [
        {
            "rank":           i + 1,
            "name":           t["name"],
            "artist":         t["artists"][0]["name"] if t["artists"] else "?",
            "uri":            t["uri"],
            "url":            t["external_urls"].get("spotify", ""),
            "match_pct":      round(_similarity_score(target_features, rec_feats.get(t["id"], {})), 1),
            "audio_features": {k: rec_feats.get(t["id"], {}).get(k) for k in DISCOVERY_FEATURE_KEYS},
        }
        for i, t in enumerate(rec_tracks)
    ]
    report_path = "discovery_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    console.print(f"\n[green]Report saved → {report_path}[/green]")

    # ── Offer to save to a new playlist ───────────────────────────────────────
    if dry_run:
        console.print("[dim](dry-run — playlist not created)[/dim]")
        return

    if confirm("Save these recommendations to a new playlist?", dry_run=False):
        mode_label = "Song" if mode == "1" else ("Playlist Energy" if mode == "2" else "Song + Playlist")
        pl_name    = ask("New playlist name", default=f"Discovered via {mode_label}")
        me         = sp.current_user()
        new_pl     = sp.user_playlist_create(
            me["id"], pl_name, public=False,
            description=f"Auto-discovered by spotify_manager · seeded from {seed_track_name or 'playlist energy'}"
        )
        uris = [t["uri"] for t in rec_tracks]
        for chunk_start in range(0, len(uris), 100):
            sp.playlist_add_items(new_pl["id"], uris[chunk_start:chunk_start + 100])
        console.print(f"[green]✓ Playlist '{pl_name}' created with {len(rec_tracks)} tracks![/green]")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pick_seed_track(sp: spotipy.Spotify) -> tuple[Optional[str], Optional[str]]:
    """Search for a track by name and let the user pick one."""
    query = ask("Search for a seed song (name or artist + name)")
    if not query:
        return None, None

    results = sp.search(q=query, type="track", limit=8)
    tracks  = results.get("tracks", {}).get("items", [])
    if not tracks:
        console.print("[yellow]No tracks found for that query.[/yellow]")
        return None, None

    console.print("\n[bold]Search results:[/bold]")
    for i, t in enumerate(tracks, 1):
        artist = t["artists"][0]["name"] if t["artists"] else "?"
        console.print(f"  [cyan]{i}[/cyan] · {t['name']} — {artist}")

    choice = ask("Pick a track number", default="1")
    try:
        idx    = int(choice) - 1
        chosen = tracks[idx]
        console.print(f"  → Seeding on: [bold]{chosen['name']}[/bold] by {chosen['artists'][0]['name']}")
        return chosen["id"], chosen["name"]
    except (ValueError, IndexError):
        console.print("[red]Invalid choice.[/red]")
        return None, None


def _pick_playlist_profile(sp: spotipy.Spotify) -> dict:
    """Let user pick one of their playlists and compute its avg audio feature profile."""
    playlists = get_all_playlists(sp)
    console.print("\n[bold]Your playlists:[/bold]")
    for i, pl in enumerate(playlists, 1):
        console.print(f"  [cyan]{i:>2}[/cyan] · {pl['name']} ({pl['tracks']['total']} tracks)")

    choice = ask("Pick a playlist number to use as energy reference")
    try:
        idx = int(choice) - 1
        pl  = playlists[idx]
    except (ValueError, IndexError):
        console.print("[red]Invalid choice.[/red]")
        return {}

    console.print(f"  → Computing energy profile for: [bold]{pl['name']}[/bold]…")
    tracks   = get_playlist_tracks(sp, pl["id"])
    ids      = [t["track"]["id"] for t in tracks]
    features = _batch_audio_features(sp, ids)
    if not features:
        console.print("[red]Could not fetch audio features for this playlist.[/red]")
        return {}

    # Plain Python mean — no numpy dependency here
    profile = {}
    for k in DISCOVERY_FEATURE_KEYS:
        vals = [f[k] for f in features if k in f]
        profile[k] = float(sum(vals) / len(vals)) if vals else 0.0

    _print_audio_profile(f"Energy profile · {pl['name']}", profile)
    return profile


def _pick_representative_seeds(sp: spotipy.Spotify, target: dict, n: int = 2) -> list[str]:
    """
    From the user's top tracks, pick N tracks whose audio features are
    closest to the target profile — used as API seeds when there is no
    explicit song seed.
    """
    try:
        top        = sp.current_user_top_tracks(limit=50, time_range="medium_term")
        candidates = top.get("items", [])
    except Exception:
        return []

    ids   = [t["id"] for t in candidates]
    feats = {f["id"]: f for f in _batch_audio_features(sp, ids) if f}

    scored = sorted(
        [(tid, _similarity_score(target, feats[tid])) for tid in ids if tid in feats],
        key=lambda x: -x[1],
    )
    return [tid for tid, _ in scored[:n]]
