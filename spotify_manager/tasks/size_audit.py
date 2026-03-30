"""
size_audit.py — Task 3: flag oversized playlists and optionally preview smart splits.
"""

import math

import spotipy

from ..utils.display import HAS_RICH, console, ask
from ..utils.spotify import get_all_playlists, get_playlist_tracks
from ..utils.audio import HAS_ML, _batch_audio_features_with_ids, _cluster_tracks


def task_size_audit(sp: spotipy.Spotify, dry_run: bool):
    console.rule("[bold cyan]Task 3 · Playlist Size Audit[/bold cyan]")

    threshold = int(ask("Flag playlists with more than how many tracks?", default="80"))
    playlists = get_all_playlists(sp)

    oversized = []
    for pl in playlists:
        count = pl["tracks"]["total"]
        if count > threshold:
            oversized.append((pl["name"], count, pl["id"], pl.get("snapshot_id")))

    if not oversized:
        console.print(f"[green]All playlists are under {threshold} tracks. 🎉[/green]")
        return

    oversized.sort(key=lambda x: -x[1])

    if HAS_RICH:
        from rich.table import Table
        table = Table(title=f"Oversized Playlists (>{threshold} tracks)", show_lines=True)
        table.add_column("Playlist",        style="yellow")
        table.add_column("Tracks",          justify="right", style="bold red")
        table.add_column("Suggested Split", style="dim")
        for name, count, _, _ in oversized:
            splits = math.ceil(count / threshold)
            table.add_row(name, str(count), f"~{splits} sub-playlists of ~{threshold}")
        console.print(table)
    else:
        for name, count, _, _ in oversized:
            splits = math.ceil(count / threshold)
            print(f"  {name}: {count} tracks → suggest {splits} sub-playlists")

    console.print("\n[dim]To split by era/mood, run Task 4 on specific playlists.[/dim]")

    if not dry_run and HAS_ML:
        pl_name = ask("\nEnter a playlist name to get a smart split preview (or leave blank to skip)", default="")
        if pl_name:
            match = next((p for p in oversized if p[0].lower() == pl_name.lower()), None)
            if match:
                _smart_split_preview(sp, match[2], match[0], match[3])
            else:
                console.print("[red]Playlist not found in oversized list.[/red]")


def _smart_split_preview(sp: spotipy.Spotify, pl_id: str, pl_name: str, snapshot_id: str | None = None):
    console.print(f"\n[bold]Analyzing '{pl_name}' for smart split…[/bold]")
    tracks = get_playlist_tracks(sp, pl_id, snapshot_id)
    ids    = [t["track"]["id"] for t in tracks]

    feature_rows = _batch_audio_features_with_ids(sp, ids)
    features = [features for _, features in feature_rows]
    if not features:
        console.print("[red]Could not fetch audio features.[/red]")
        return

    n_clusters = max(2, min(5, len(features) // 20))
    labels, cluster_info = _cluster_tracks(features, n_clusters)

    console.print(f"\nSuggested {n_clusters} sub-playlists:")
    for i, info in enumerate(cluster_info):
        tracks_in = [track_id for (track_id, _), label in zip(feature_rows, labels) if label == i]
        console.print(
            f"  Sub-playlist {i+1}: ~{len(tracks_in)} tracks | "
            f"energy={info['energy']:.2f}, valence={info['valence']:.2f}, "
            f"tempo={info['tempo']:.0f}bpm"
        )
