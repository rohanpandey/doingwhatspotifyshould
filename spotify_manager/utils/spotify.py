"""
spotify.py — Pagination and library-level Spotify helpers.
"""

import copy
import time

import spotipy

PLAYLIST_TRACK_CACHE_TTL_SECONDS = 6 * 60 * 60
LIKED_TRACK_CACHE_TTL_SECONDS = 20 * 60
_playlist_track_cache: dict[str, dict[str, object]] = {}
_liked_track_cache: dict[str, object] = {
    "tracks": None,
    "expires_at": 0.0,
}


def invalidate_playlist_tracks_cache(playlist_id: str | None = None):
    if playlist_id:
        _playlist_track_cache.pop(playlist_id, None)
        return
    _playlist_track_cache.clear()


def invalidate_liked_tracks_cache():
    _liked_track_cache["tracks"] = None
    _liked_track_cache["expires_at"] = 0.0


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


def get_playlist_tracks(sp: spotipy.Spotify, playlist_id: str, snapshot_id: str | None = None):
    """Return all track items in a playlist, reusing cached normalized results when unchanged."""
    cached = _playlist_track_cache.get(playlist_id)
    now = time.time()
    if cached:
        cache_snapshot = cached.get("snapshot_id")
        cache_expires_at = float(cached.get("expires_at", 0.0) or 0.0)
        if snapshot_id and cache_snapshot == snapshot_id:
            return copy.deepcopy(cached.get("tracks", []))
        if not snapshot_id and now < cache_expires_at:
            return copy.deepcopy(cached.get("tracks", []))

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
    _playlist_track_cache[playlist_id] = {
        "snapshot_id": snapshot_id,
        "tracks": copy.deepcopy(normalized),
        "expires_at": now + PLAYLIST_TRACK_CACHE_TTL_SECONDS,
    }
    return normalized


def get_liked_tracks(sp: spotipy.Spotify):
    """Return normalized liked-song items in the same shape as playlist tracks."""
    now = time.time()
    cached_tracks = _liked_track_cache.get("tracks")
    if cached_tracks and now < float(_liked_track_cache.get("expires_at", 0.0) or 0.0):
        return copy.deepcopy(cached_tracks)

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
    _liked_track_cache["tracks"] = copy.deepcopy(normalized)
    _liked_track_cache["expires_at"] = now + LIKED_TRACK_CACHE_TTL_SECONDS
    return normalized


def remove_liked_tracks(sp: spotipy.Spotify, track_ids: list[str]):
    """Remove saved tracks via the current library endpoint in 40-URI batches."""
    for start in range(0, len(track_ids), 40):
        chunk = [f"spotify:track:{track_id}" for track_id in track_ids[start:start + 40] if track_id]
        if not chunk:
            continue
        sp._delete("me/library", uris=",".join(chunk))
        time.sleep(0.05)
    invalidate_liked_tracks_cache()
