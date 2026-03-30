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
    fields = "items(added_at,item(id,name,artists(name,id),duration_ms,is_local,external_ids,uri)),next"
    items = paginate(sp.playlist_items, playlist_id, fields=fields)
    normalized = []
    for item in items:
        if not item:
            continue
        track = item.get("track") or item.get("item")
        if not track or not track.get("id") or track.get("is_local"):
            continue
        normalized.append({
            "added_at": item.get("added_at"),
            "track": track,
        })
    return normalized


def get_liked_tracks(sp: spotipy.Spotify):
    """Return normalized liked-song items in the same shape as playlist tracks."""
    items = paginate(sp.current_user_saved_tracks)
    normalized = []
    for item in items:
        if not item:
            continue
        track = item.get("track")
        if not track or not track.get("id") or track.get("is_local"):
            continue
        normalized.append({
            "added_at": item.get("added_at"),
            "track": track,
        })
    return normalized


def remove_liked_tracks(sp: spotipy.Spotify, track_ids: list[str]):
    """Remove saved tracks via the current library endpoint in 40-URI batches."""
    for start in range(0, len(track_ids), 40):
        chunk = [f"spotify:track:{track_id}" for track_id in track_ids[start:start + 40] if track_id]
        if not chunk:
            continue
        sp._delete("me/library", uris=",".join(chunk))
        time.sleep(0.05)
