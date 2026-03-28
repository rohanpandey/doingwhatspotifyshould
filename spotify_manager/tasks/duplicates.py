"""
duplicates.py — Task 1: remove duplicate tracks from every owned playlist.
"""

from collections import defaultdict

import spotipy

from ..utils.display import console, confirm
from ..utils.spotify import get_all_playlists, get_playlist_tracks


def task_duplicates(sp: spotipy.Spotify, dry_run: bool):
    console.rule("[bold cyan]Task 1 · Duplicate Removal[/bold cyan]")

    playlists     = get_all_playlists(sp)
    total_removed = 0

    for pl in playlists:
        pl_id   = pl["id"]
        pl_name = pl["name"]
        tracks  = get_playlist_tracks(sp, pl_id)

        seen  = {}   # track_id -> first position
        dupes = []   # list of (position, track_id, name)

        for i, item in enumerate(tracks):
            tid  = item["track"]["id"]
            name = item["track"]["name"]
            if tid in seen:
                dupes.append((i, tid, name))
            else:
                seen[tid] = i

        if not dupes:
            continue

        console.print(f"\n[bold yellow]{pl_name}[/bold yellow] — {len(dupes)} duplicate(s) found:")
        for pos, tid, name in dupes:
            console.print(f"  pos {pos:>3}  {name}")

        if dry_run:
            console.print("  [dim](dry-run — no changes made)[/dim]")
            continue

        if confirm(f"  Remove {len(dupes)} duplicate(s) from '{pl_name}'?", dry_run=False):
            by_tid = defaultdict(list)
            for pos, tid, _ in dupes:
                by_tid[tid].append(pos)

            tracks_payload = [
                {"uri": f"spotify:track:{tid}", "positions": positions}
                for tid, positions in by_tid.items()
            ]
            # Spotify allows max 100 tracks per remove call
            for chunk_start in range(0, len(tracks_payload), 100):
                chunk = tracks_payload[chunk_start:chunk_start + 100]
                sp.playlist_remove_specific_occurrences_of_items(pl_id, chunk)
            console.print(f"  [green]✓ Removed {len(dupes)} duplicate(s)[/green]")
            total_removed += len(dupes)

    console.print(f"\n[bold]Summary:[/bold] {total_removed} duplicates removed across all playlists.")
