"""
never_played.py — Task 2: find tracks not in recent play history.
"""

import datetime
import json

import spotipy

from ..utils.display import HAS_RICH, console, ask
from ..utils.spotify import get_all_playlists, get_playlist_tracks


def task_never_played(sp: spotipy.Spotify, dry_run: bool):
    console.rule("[bold cyan]Task 2 · Never-Played Tracks[/bold cyan]")
    console.print("[dim]Note: Spotify API only exposes the last 50 recently played tracks.[/dim]")
    console.print("[dim]'Never played' here = not in your recent history + added >N days ago.[/dim]\n")

    recent     = sp.current_user_recently_played(limit=50)
    played_ids = {item["track"]["id"] for item in recent.get("items", [])}

    playlists   = get_all_playlists(sp)
    cutoff_days = int(ask("Flag tracks added more than how many days ago?", default="30"))
    cutoff      = datetime.datetime.utcnow() - datetime.timedelta(days=cutoff_days)

    never_played_report = []

    for pl in playlists:
        pl_name = pl["name"]
        tracks  = get_playlist_tracks(sp, pl["id"], pl.get("snapshot_id"))
        stale   = []

        for item in tracks:
            tid    = item["track"]["id"]
            name   = item["track"]["name"]
            artist = item["track"]["artists"][0]["name"] if item["track"]["artists"] else "?"
            added  = item.get("added_at", "")
            try:
                added_dt = datetime.datetime.strptime(added, "%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                continue

            if tid not in played_ids and added_dt < cutoff:
                stale.append({"playlist": pl_name, "track": name, "artist": artist, "added": added[:10]})

        never_played_report.extend(stale)

    if not never_played_report:
        console.print("[green]No never-played tracks found![/green]")
        return

    if HAS_RICH:
        from rich.table import Table
        table = Table(title=f"Never-Played Tracks (added >{cutoff_days}d ago)", show_lines=True)
        table.add_column("Playlist", style="yellow")
        table.add_column("Track",    style="white")
        table.add_column("Artist",   style="dim")
        table.add_column("Added",    style="dim")
        for row in never_played_report:
            table.add_row(row["playlist"], row["track"], row["artist"], row["added"])
        console.print(table)
    else:
        for r in never_played_report:
            print(f"  [{r['playlist']}] {r['track']} — {r['artist']} (added {r['added']})")

    report_path = "never_played_report.json"
    with open(report_path, "w") as f:
        json.dump(never_played_report, f, indent=2)
    console.print(f"\n[green]Report saved to {report_path}[/green]")
