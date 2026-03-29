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
    value = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", value)
    value = re.sub(r"\b(feat|featuring|ft)\b.*", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def primary_artist_name(track: dict[str, Any]) -> str:
    artists = track.get("artists") or []
    if not artists:
        return "?"
    return artists[0].get("name", "?")


def duplicate_isrc_key(track: dict[str, Any]) -> tuple[Any, ...] | None:
    external_ids = track.get("external_ids") or {}
    isrc = (external_ids.get("isrc") or "").strip().lower()
    if isrc:
        return ("isrc", isrc)
    return None


def duplicate_meta_key(track: dict[str, Any]) -> tuple[Any, ...]:
    """Metadata signature based on normalized title, primary artist, and duration bucket."""
    duration_ms = int(track.get("duration_ms") or 0)
    duration_bucket = int(round(duration_ms / _DURATION_BUCKET_MS) * _DURATION_BUCKET_MS)
    return (
        "meta",
        _normalize_text(track.get("name", "")),
        _normalize_text(primary_artist_name(track)),
        duration_bucket,
    )


def duplicate_match_key(track: dict[str, Any]) -> tuple[Any, ...]:
    """
    Backwards-compatible single-key helper.
    Prefer ISRC when present, otherwise use metadata.
    """
    return duplicate_isrc_key(track) or duplicate_meta_key(track)


def duplicate_match_label(match_key: tuple[Any, ...] | None) -> str:
    return MATCH_LABEL_ISRC if match_key and match_key[0] == "isrc" else MATCH_LABEL_META


def find_duplicate_entries(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Return duplicate playlist entries, keeping the first occurrence of each
    semantic song key and flagging later occurrences for removal.
    """
    seen_by_isrc: dict[tuple[Any, ...], dict[str, Any]] = {}
    seen_by_meta: dict[tuple[Any, ...], dict[str, Any]] = {}
    duplicates = []

    for index, item in enumerate(tracks):
        track = item.get("track") or {}
        track_id = track.get("id")
        if not track_id:
            continue

        isrc_key = duplicate_isrc_key(track)
        meta_key = duplicate_meta_key(track)

        first = None
        match_key = None
        if isrc_key and isrc_key in seen_by_isrc:
            first = seen_by_isrc[isrc_key]
            match_key = isrc_key
        elif meta_key in seen_by_meta:
            first = seen_by_meta[meta_key]
            match_key = meta_key

        if first:
            duplicates.append({
                "position": index,
                "track_id": track_id,
                "name": track.get("name", "Unknown track"),
                "artist": primary_artist_name(track),
                "kept_position": first["position"],
                "kept_name": first["name"],
                "kept_artist": first["artist"],
                "match_label": duplicate_match_label(match_key),
            })
        else:
            entry = {
                "position": index,
                "track_id": track_id,
                "name": track.get("name", "Unknown track"),
                "artist": primary_artist_name(track),
            }
            if isrc_key:
                seen_by_isrc[isrc_key] = entry
            seen_by_meta[meta_key] = entry

    return duplicates


def build_duplicate_removal_payload(duplicates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped_positions: dict[str, list[int]] = defaultdict(list)
    for entry in duplicates:
        grouped_positions[entry["track_id"]].append(entry["position"])
    return [
        {"uri": f"spotify:track:{track_id}", "positions": positions}
        for track_id, positions in grouped_positions.items()
    ]
