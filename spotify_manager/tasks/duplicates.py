"""
duplicates.py — Task 1: remove duplicate tracks from every owned playlist.
"""

import spotipy

from ..utils.display import HAS_RICH, console, confirm, ask
from ..utils.duplicates import build_duplicate_removal_payload, find_duplicate_entries
from ..utils.spotify import get_all_playlists, get_liked_tracks, get_playlist_tracks


def task_duplicates(sp: spotipy.Spotify, dry_run: bool):
    console.rule("[bold cyan]Task 1 · Duplicate Removal[/bold cyan]")

    total_removed = 0
    include_liked = False

    if HAS_RICH:
        include_liked = ask("Also scan Liked Songs?", default="y").lower().startswith("y")
    else:
        include_liked = ask("Also scan Liked Songs? (y/n)", default="y").lower().startswith("y")

    sources = [("playlist", pl["id"], pl["name"], get_playlist_tracks(sp, pl["id"])) for pl in get_all_playlists(sp)]
    if include_liked:
        sources.append(("liked", "__liked_songs__", "Liked Songs", get_liked_tracks(sp)))

    for source_kind, source_id, source_name, tracks in sources:
        dupes = find_duplicate_entries(tracks)

        if not dupes:
            continue

        console.print(f"\n[bold yellow]{source_name}[/bold yellow] — {len(dupes)} duplicate(s) found:")
        for entry in dupes:
            console.print(
                f"  pos {entry['position'] + 1:>3}  {entry['name']} — {entry['artist']}  "
                f"[dim](matches '{entry['kept_name']}' — {entry['kept_artist']} at "
                f"pos {entry['kept_position'] + 1} via {entry['match_label']})[/dim]"
            )

        if dry_run:
            console.print("  [dim](dry-run — no changes made)[/dim]")
            continue

        if confirm(f"  Remove {len(dupes)} duplicate(s) from '{source_name}'?", dry_run=False):
            if source_kind == "liked":
                duplicate_ids = [entry["track_id"] for entry in dupes]
                for chunk_start in range(0, len(duplicate_ids), 50):
                    chunk = duplicate_ids[chunk_start:chunk_start + 50]
                    sp.current_user_saved_tracks_delete(chunk)
            else:
                tracks_payload = build_duplicate_removal_payload(dupes)
                # Spotify allows max 100 tracks per remove call
                for chunk_start in range(0, len(tracks_payload), 100):
                    chunk = tracks_payload[chunk_start:chunk_start + 100]
                    sp.playlist_remove_specific_occurrences_of_items(source_id, chunk)
            console.print(f"  [green]✓ Removed {len(dupes)} duplicate(s)[/green]")
            total_removed += len(dupes)

    console.print(f"\n[bold]Summary:[/bold] {total_removed} duplicates removed across scanned sources.")
