"""
spotify_client.py — Shared Spotify API throttling, cooldown, and caching.
"""

from __future__ import annotations

import copy
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import spotipy


@dataclass(frozen=True)
class CachePolicy:
    ttl_seconds: int
    stale_seconds: int


@dataclass
class CacheEntry:
    value: Any
    expires_at: float
    stale_at: float


CACHE_POLICIES: dict[str, CachePolicy] = {
    "current_user": CachePolicy(ttl_seconds=6 * 60 * 60, stale_seconds=24 * 60 * 60),
    "current_user_playlists": CachePolicy(ttl_seconds=30 * 60, stale_seconds=6 * 60 * 60),
    "playlist_items": CachePolicy(ttl_seconds=30 * 60, stale_seconds=6 * 60 * 60),
    "current_user_saved_tracks": CachePolicy(ttl_seconds=30 * 60, stale_seconds=6 * 60 * 60),
    "current_user_recently_played": CachePolicy(ttl_seconds=2 * 60, stale_seconds=10 * 60),
    "current_user_top_tracks": CachePolicy(ttl_seconds=60 * 60, stale_seconds=6 * 60 * 60),
    "current_user_top_artists": CachePolicy(ttl_seconds=60 * 60, stale_seconds=6 * 60 * 60),
    "audio_features": CachePolicy(ttl_seconds=24 * 60 * 60, stale_seconds=7 * 24 * 60 * 60),
    "artists": CachePolicy(ttl_seconds=12 * 60 * 60, stale_seconds=3 * 24 * 60 * 60),
    "artist_related_artists": CachePolicy(ttl_seconds=12 * 60 * 60, stale_seconds=3 * 24 * 60 * 60),
    "artist_top_tracks": CachePolicy(ttl_seconds=12 * 60 * 60, stale_seconds=3 * 24 * 60 * 60),
    "track": CachePolicy(ttl_seconds=24 * 60 * 60, stale_seconds=7 * 24 * 60 * 60),
    "search": CachePolicy(ttl_seconds=10 * 60, stale_seconds=60 * 60),
    "recommendations": CachePolicy(ttl_seconds=10 * 60, stale_seconds=60 * 60),
}

WRITE_METHODS = {
    "_delete",
    "_post",
    "_put",
    "playlist_add_items",
    "playlist_remove_specific_occurrences_of_items",
    "user_playlist_create",
}


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(subvalue)) for key, subvalue in value.items()))
    if isinstance(value, (list, tuple, set)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def _build_cache_key(method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
    return method_name, _freeze(args), _freeze(kwargs)


class SpotifyApiController:
    def __init__(self):
        self._lock = threading.RLock()
        self._cache: dict[tuple[Any, ...], CacheEntry] = {}
        self._next_request_at = 0.0
        self._cooldown_until = 0.0
        self._cooldown_message = ""
        self._min_interval_seconds = 0.14

    def _clone(self, value: Any) -> Any:
        return copy.deepcopy(value)

    def _get_cache_entry(self, cache_key: tuple[Any, ...] | None) -> CacheEntry | None:
        if cache_key is None:
            return None
        with self._lock:
            return self._cache.get(cache_key)

    def _store_cache_entry(self, cache_key: tuple[Any, ...], policy: CachePolicy, value: Any):
        now = time.time()
        entry = CacheEntry(
            value=self._clone(value),
            expires_at=now + policy.ttl_seconds,
            stale_at=now + policy.stale_seconds,
        )
        with self._lock:
            self._cache[cache_key] = entry

    def _respect_cooldown(self):
        with self._lock:
            remaining = self._cooldown_until - time.time()
            message = self._cooldown_message or "Spotify asked us to slow down."
        if remaining > 0:
            headers = {"Retry-After": str(max(1, math.ceil(remaining)))}
            raise spotipy.SpotifyException(429, -1, message, headers=headers)

    def _reserve_request_slot(self):
        wait_time = 0.0
        with self._lock:
            now = time.time()
            wait_time = max(0.0, self._next_request_at - now)
            self._next_request_at = max(now, self._next_request_at) + self._min_interval_seconds
        if wait_time > 0:
            time.sleep(wait_time)

    def _note_exception(self, exc: spotipy.SpotifyException):
        if exc.http_status != 429:
            return
        retry_after = 30
        if exc.headers:
            try:
                retry_after = max(1, int(exc.headers.get("Retry-After", "30") or "30"))
            except (TypeError, ValueError):
                retry_after = 30
        with self._lock:
            self._cooldown_until = max(self._cooldown_until, time.time() + retry_after)
            self._cooldown_message = str(exc)

    def _clear_cache(self, methods: set[str]):
        if not methods:
            return
        with self._lock:
            doomed = [key for key in self._cache if key and key[0] in methods]
            for key in doomed:
                self._cache.pop(key, None)

    def invalidate_after_write(self, method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]):
        methods_to_clear: set[str] = set()
        if method_name in {"playlist_add_items", "playlist_remove_specific_occurrences_of_items", "user_playlist_create"}:
            methods_to_clear.update({"current_user_playlists", "playlist_items"})
        if method_name == "_delete":
            path = str(args[0]) if args else ""
            if path == "me/library":
                methods_to_clear.add("current_user_saved_tracks")
            elif path.startswith("playlists/"):
                methods_to_clear.update({"current_user_playlists", "playlist_items"})
        if method_name == "_put":
            path = str(args[0]) if args else ""
            if path == "me/library":
                methods_to_clear.add("current_user_saved_tracks")
        self._clear_cache(methods_to_clear)

    def get_rate_limit_status(self) -> dict[str, Any]:
        with self._lock:
            remaining = max(0, math.ceil(self._cooldown_until - time.time()))
            return {
                "cooldown_until": self._cooldown_until,
                "retry_after_seconds": remaining,
                "cooldown_active": remaining > 0,
                "message": self._cooldown_message,
            }

    def call(self, method_name: str, target: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        policy = CACHE_POLICIES.get(method_name)
        cache_key = _build_cache_key(method_name, args, kwargs) if policy else None
        cached = self._get_cache_entry(cache_key)
        now = time.time()

        if cached and now < cached.expires_at:
            return self._clone(cached.value)
        if cached and self.get_rate_limit_status()["cooldown_active"] and now < cached.stale_at:
            return self._clone(cached.value)

        self._respect_cooldown()
        self._reserve_request_slot()

        try:
            result = target(*args, **kwargs)
        except spotipy.SpotifyException as exc:
            self._note_exception(exc)
            if cached and exc.http_status in {429, 500, 502, 503, 504} and time.time() < cached.stale_at:
                return self._clone(cached.value)
            raise

        if method_name in WRITE_METHODS:
            self.invalidate_after_write(method_name, args, kwargs)
        elif policy is not None:
            self._store_cache_entry(cache_key, policy, result)

        return result


_CONTROLLER = SpotifyApiController()


class SpotifyClientProxy:
    """Proxy a Spotipy client through the shared controller."""

    def __init__(self, client: spotipy.Spotify):
        self._client = client

    def get_rate_limit_status(self) -> dict[str, Any]:
        return _CONTROLLER.get_rate_limit_status()

    def invalidate_write_caches(self):
        _CONTROLLER._clear_cache({"current_user_playlists", "playlist_items", "current_user_saved_tracks"})

    def __getattr__(self, name: str):
        target = getattr(self._client, name)
        if not callable(target):
            return target

        def wrapped(*args, **kwargs):
            return _CONTROLLER.call(name, target, args, kwargs)

        return wrapped


def wrap_spotify_client(client: spotipy.Spotify) -> SpotifyClientProxy:
    return SpotifyClientProxy(client)
