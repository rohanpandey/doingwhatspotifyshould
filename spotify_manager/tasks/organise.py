"""
organise.py — Task 4: cluster liked songs into mood and/or genre playlists.
"""

import time
from collections import defaultdict

import spotipy

from ..utils.display import HAS_RICH, console, confirm, ask
from ..utils.spotify import paginate, get_all_playlists
from ..utils.audio import HAS_ML, _batch_audio_features_with_ids, _cluster_tracks

# ── Constants ─────────────────────────────────────────────────────────────────

MOOD_LABELS = {
    0: "🔥 High Energy · Happy",
    1: "⚡ High Energy · Dark",
    2: "🌅 Chill · Upbeat",
    3: "🌙 Low Energy · Introspective",
    4: "💃 Dance Floor",
}

GENRE_BUCKETS = {
    "🎤 Hip-Hop / R&B":          ["hip hop", "rap", "r&b", "trap", "drill", "soul", "neo soul"],
    "🎸 Rock / Alt":              ["rock", "alternative", "indie", "grunge", "punk", "emo", "metal"],
    "🎹 Electronic / Dance":      ["electronic", "edm", "house", "techno", "dance", "trance",
                                    "electro", "dubstep", "drum and bass", "dnb", "jungle",
                                    "ambient", "synthwave", "chillwave"],
    "🎷 Jazz / Blues / Soul":     ["jazz", "blues", "soul", "funk", "gospel", "swing", "bop"],
    "🎻 Classical / Soundtrack":  ["classical", "orchestra", "opera", "soundtrack", "score",
                                    "chamber", "baroque", "piano"],
    "🌍 World / Latin":           ["latin", "reggaeton", "salsa", "bossa nova", "afrobeats",
                                    "afropop", "k-pop", "j-pop", "bollywood", "flamenco",
                                    "cumbia", "samba", "dancehall", "reggae"],
    "🎵 Pop":                     ["pop", "synth pop", "dream pop", "art pop", "bubblegum"],
    "🪕 Folk / Country":          ["folk", "country", "bluegrass", "americana", "singer-songwriter",
                                    "acoustic"],
}

GENRE_OTHER = "🎶 Other / Mixed"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _assign_genre_bucket(genres: list[str]) -> str:
    """Return the first matching genre bucket for a list of Spotify genre tags."""
    lowered = [g.lower() for g in genres]
    for bucket, keywords in GENRE_BUCKETS.items():
        for kw in keywords:
            if any(kw in g for g in lowered):
                return bucket
    return GENRE_OTHER


def _batch_artist_genres(sp: spotipy.Spotify, artist_ids: list[str]) -> dict[str, list[str]]:
    """Fetch genres for a list of artist IDs. Returns {artist_id: [genres]}."""
    genre_map  = {}
    unique_ids = list(set(artist_ids))
    for start in range(0, len(unique_ids), 50):
        batch  = unique_ids[start:start + 50]
        result = sp.artists(batch)
        for artist in result.get("artists", []):
            if artist:
                genre_map[artist["id"]] = artist.get("genres", [])
        time.sleep(0.05)
    return genre_map


def _create_playlists_from_groups(
    sp: spotipy.Spotify,
    user_id: str,
    groups: dict[str, list],
    prefix: str,
    description_fn,
):
    """Create one Spotify playlist per group, adding tracks in batches of 100."""
    for label, tracks in groups.items():
        pl_name = f"{prefix} · {label}"
        console.print(f"  Creating '{pl_name}' ({len(tracks)} tracks)…")
        new_pl    = sp.user_playlist_create(
            user_id, pl_name, public=False,
            description=description_fn(label, tracks),
        )
        new_pl_id  = new_pl["id"]
        track_uris = [f"spotify:track:{t['id']}" for t in tracks]
        for chunk_start in range(0, len(track_uris), 100):
            sp.playlist_add_items(new_pl_id, track_uris[chunk_start:chunk_start + 100])
            time.sleep(0.1)
        console.print(f"  [green]✓ '{pl_name}'[/green]")


# ── Task 4 ────────────────────────────────────────────────────────────────────

def task_organize_liked(sp: spotipy.Spotify, dry_run: bool):
    console.rule("[bold cyan]Task 4 · Organize Liked Songs[/bold cyan]")

    if not HAS_ML:
        console.print("[red]scikit-learn and numpy are required for mood clustering.[/red]")
        console.print("pip install scikit-learn numpy")
        return

    console.print("Fetching your Liked Songs (this may take a while)…")
    liked_raw    = paginate(sp.current_user_saved_tracks)
    liked_tracks = [
        it["track"] for it in liked_raw
        if it and it.get("track") and it["track"].get("id") and not it["track"].get("is_local")
    ]
    console.print(f"  Found {len(liked_tracks)} liked tracks.\n")

    if HAS_RICH:
        console.print("[bold]Grouping strategy:[/bold]")
        console.print("  [cyan]1[/cyan] · Mood / Vibe  (energy + valence + tempo via audio features)")
        console.print("  [cyan]2[/cyan] · Genre        (from Spotify artist tags)")
        console.print("  [cyan]3[/cyan] · Both         (create two separate sets of playlists)")
        strategy = ask("Choose", default="3")
    else:
        strategy = ask("Grouping? 1=Mood, 2=Genre, 3=Both", default="3")

    do_mood  = strategy in ("1", "3")
    do_genre = strategy in ("2", "3")

    me      = sp.current_user()
    user_id = me["id"]
    ids     = [t["id"] for t in liked_tracks]
    prefix  = ask("Playlist name prefix", default="Liked Songs")

    # ── Mood clustering ────────────────────────────────────────────────────────
    if do_mood:
        n_clusters = int(ask("How many mood playlists? (4–6 recommended)", default="4"))
        console.print("\nFetching audio features for mood clustering…")
        feature_rows = _batch_audio_features_with_ids(sp, ids)
        features = [features for _, features in feature_rows]

        if not features:
            console.print("[red]Could not fetch audio features — skipping mood grouping.[/red]")
        else:
            labels, cluster_info = _cluster_tracks(features, n_clusters)

            mood_clusters: dict[int, list] = defaultdict(list)
            tracks_by_id = {track["id"]: track for track in liked_tracks}
            for (track_id, _), label in zip(feature_rows, labels):
                track = tracks_by_id.get(track_id)
                if track:
                    mood_clusters[int(label)].append(track)

            console.print("\n[bold]Proposed mood playlists:[/bold]")
            if HAS_RICH:
                from rich.table import Table
                table = Table(show_lines=True)
                table.add_column("#",       style="bold")
                table.add_column("Label",   style="yellow")
                table.add_column("Tracks",  justify="right")
                table.add_column("Energy",  justify="right")
                table.add_column("Valence", justify="right")
                table.add_column("BPM",     justify="right")
                for i, info in enumerate(cluster_info):
                    table.add_row(
                        str(i + 1), MOOD_LABELS.get(i, f"Mood {i+1}"),
                        str(len(mood_clusters[i])),
                        f"{info['energy']:.2f}", f"{info['valence']:.2f}",
                        f"{info['tempo']:.0f}",
                    )
                console.print(table)
            else:
                for i, info in enumerate(cluster_info):
                    print(f"  {MOOD_LABELS.get(i, f'Mood {i+1}')}: {len(mood_clusters[i])} tracks")

            if dry_run:
                console.print("[dim](dry-run — mood playlists not created)[/dim]")
            elif confirm(f"Create {n_clusters} mood playlists?", dry_run=False):
                _create_playlists_from_groups(
                    sp, user_id,
                    {MOOD_LABELS.get(i, f"Mood {i+1}"): tracks for i, tracks in mood_clusters.items()},
                    prefix=prefix,
                    description_fn=lambda label, tracks: f"Auto-organized by mood · {len(tracks)} tracks",
                )

    # ── Genre grouping ─────────────────────────────────────────────────────────
    if do_genre:
        console.print("\nFetching artist genres for genre grouping…")

        artist_ids: list[str]            = []
        track_primary_artist: dict[str, str] = {}
        for t in liked_tracks:
            if t.get("artists"):
                aid = t["artists"][0]["id"]
                artist_ids.append(aid)
                track_primary_artist[t["id"]] = aid

        genre_map = _batch_artist_genres(sp, artist_ids)

        genre_clusters: dict[str, list] = defaultdict(list)
        for t in liked_tracks:
            aid    = track_primary_artist.get(t["id"], "")
            genres = genre_map.get(aid, [])
            bucket = _assign_genre_bucket(genres)
            genre_clusters[bucket].append(t)

        MIN_GENRE_SIZE = 5
        genre_clusters = {
            k: v for k, v in sorted(genre_clusters.items(), key=lambda x: -len(x[1]))
            if len(v) >= MIN_GENRE_SIZE
        }

        console.print("\n[bold]Proposed genre playlists:[/bold]")
        if HAS_RICH:
            from rich.table import Table
            table = Table(show_lines=True)
            table.add_column("Genre",  style="yellow")
            table.add_column("Tracks", justify="right")
            for genre, tracks in genre_clusters.items():
                table.add_row(genre, str(len(tracks)))
            console.print(table)
        else:
            for genre, tracks in genre_clusters.items():
                print(f"  {genre}: {len(tracks)} tracks")

        if dry_run:
            console.print("[dim](dry-run — genre playlists not created)[/dim]")
        elif confirm(f"Create {len(genre_clusters)} genre playlists?", dry_run=False):
            _create_playlists_from_groups(
                sp, user_id, genre_clusters,
                prefix=prefix,
                description_fn=lambda label, tracks: f"Auto-organized by genre · {len(tracks)} tracks",
            )

    console.print("\n[bold green]Task 4 complete! 🎵[/bold green]")
