"""
duplicates.py - Shared duplicate-detection helpers.

The goal is to catch "same song twice" cases even when Spotify track IDs differ
between album, single, clean/explicit, or region-specific variants.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

MATCH_LABEL_ISRC = "ISRC"
MATCH_LABEL_META = "name + artist + duration"
_DURATION_BUCKET_MS = 2000


def _normalize_text(value: str) -> str:
    value = (value or "").lower().strip()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def primary_artist_name(track: dict[str, Any]) -> str:
    artists = track.get("artists") or []
    if not artists:
        return "?"
    return artists[0].get("name", "?")


def duplicate_match_key(track: dict[str, Any]) -> tuple[Any, ...]:
    """
    Prefer ISRC when available. Otherwise fall back to a metadata signature
    based on normalized title, primary artist, and a coarse duration bucket.
    """
    external_ids = track.get("external_ids") or {}
    isrc = (external_ids.get("isrc") or "").strip().lower()
    if isrc:
        return ("isrc", isrc)

    duration_ms = int(track.get("duration_ms") or 0)
    duration_bucket = int(round(duration_ms / _DURATION_BUCKET_MS) * _DURATION_BUCKET_MS)
    return (
        "meta",
        _normalize_text(track.get("name", "")),
        _normalize_text(primary_artist_name(track)),
        duration_bucket,
    )


def duplicate_match_label(match_key: tuple[Any, ...]) -> str:
    return MATCH_LABEL_ISRC if match_key and match_key[0] == "isrc" else MATCH_LABEL_META


def find_duplicate_entries(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Return duplicate playlist entries, keeping the first occurrence of each
    semantic song key and flagging later occurrences for removal.
    """
    seen_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    duplicates = []

    for index, item in enumerate(tracks):
        track = item.get("track") or {}
        track_id = track.get("id")
        if not track_id:
            continue

        match_key = duplicate_match_key(track)
        if match_key in seen_by_key:
            first = seen_by_key[match_key]
            duplicates.append({
                "position": index,
                "track_id": track_id,
                "name": track.get("name", "Unknown track"),
                "artist": primary_artist_name(track),
                "kept_position": first["position"],
                "match_label": duplicate_match_label(match_key),
            })
        else:
            seen_by_key[match_key] = {
                "position": index,
                "track_id": track_id,
            }

    return duplicates


def build_duplicate_removal_payload(duplicates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped_positions: dict[str, list[int]] = defaultdict(list)
    for entry in duplicates:
        grouped_positions[entry["track_id"]].append(entry["position"])
    return [
        {"uri": f"spotify:track:{track_id}", "positions": positions}
        for track_id, positions in grouped_positions.items()
    ]
