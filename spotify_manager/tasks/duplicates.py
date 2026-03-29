"""
duplicates.py — Task 1: remove duplicate tracks from every owned playlist.
"""

import spotipy

from ..utils.display import console, confirm
from ..utils.duplicates import build_duplicate_removal_payload, find_duplicate_entries
from ..utils.spotify import get_all_playlists, get_playlist_tracks


def task_duplicates(sp: spotipy.Spotify, dry_run: bool):
    console.rule("[bold cyan]Task 1 · Duplicate Removal[/bold cyan]")

    playlists     = get_all_playlists(sp)
    total_removed = 0

    for pl in playlists:
        pl_id   = pl["id"]
        pl_name = pl["name"]
        tracks  = get_playlist_tracks(sp, pl_id)
        dupes = find_duplicate_entries(tracks)

        if not dupes:
            continue

        console.print(f"\n[bold yellow]{pl_name}[/bold yellow] — {len(dupes)} duplicate(s) found:")
        for entry in dupes:
            console.print(
                f"  pos {entry['position'] + 1:>3}  {entry['name']} — {entry['artist']}  "
                f"[dim](matches '{entry['kept_name']}' — {entry['kept_artist']} at "
                f"pos {entry['kept_position'] + 1} via {entry['match_label']})[/dim]"
            )

        if dry_run:
            console.print("  [dim](dry-run — no changes made)[/dim]")
            continue

        if confirm(f"  Remove {len(dupes)} duplicate(s) from '{pl_name}'?", dry_run=False):
            tracks_payload = build_duplicate_removal_payload(dupes)
            # Spotify allows max 100 tracks per remove call
            for chunk_start in range(0, len(tracks_payload), 100):
                chunk = tracks_payload[chunk_start:chunk_start + 100]
                sp.playlist_remove_specific_occurrences_of_items(pl_id, chunk)
            console.print(f"  [green]✓ Removed {len(dupes)} duplicate(s)[/green]")
            total_removed += len(dupes)

    console.print(f"\n[bold]Summary:[/bold] {total_removed} duplicates removed across all playlists.")
