"""
spotify.py — Pagination and library-level Spotify helpers.
"""

import time

import spotipy


def paginate(sp_func, *args, limit=50, **kwargs):
    """Collect all pages from a Spotify paged endpoint."""
    results = []
    offset = 0
    while True:
        page = sp_func(*args, limit=limit, offset=offset, **kwargs)
        items = page.get("items", [])
        results.extend(items)
        if len(items) < limit:
            break
        offset += limit
        time.sleep(0.05)  # gentle rate-limiting
    return results


def get_all_playlists(sp: spotipy.Spotify):
    """Return all playlists owned by the current user."""
    me = sp.current_user()["id"]
    all_pls = paginate(sp.current_user_playlists)
    return [p for p in all_pls if p and p.get("owner", {}).get("id") == me]


def get_playlist_tracks(sp: spotipy.Spotify, playlist_id: str):
    """Return all track items in a playlist, filtering out local/null tracks."""
    fields = "items(added_at,track(id,name,artists(name,id),duration_ms,is_local,external_ids)),next"
    items = paginate(sp.playlist_items, playlist_id, fields=fields)
    return [it for it in items if it and it.get("track") and it["track"].get("id")]
