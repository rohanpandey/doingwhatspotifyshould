"""
audio.py — Audio feature helpers, k-means clustering, and similarity scoring.

`HAS_ML` is re-exported from here so that task modules have a single place
to check whether scikit-learn and numpy are available.
"""

import math
import time

try:
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    HAS_ML = True
except ImportError:
    HAS_ML = False

from .display import console


# ── Batch audio feature fetching ──────────────────────────────────────────────

def _batch_audio_features(sp, ids: list[str]) -> list[dict]:
    """Fetch audio features for all track IDs in batches of 100."""
    all_features = []
    for start in range(0, len(ids), 100):
        batch = ids[start:start + 100]
        result = sp.audio_features(batch)
        if result:
            all_features.extend([f for f in result if f])
        time.sleep(0.1)
    return all_features


def _batch_audio_features_with_ids(sp, ids: list[str]) -> list[tuple[str, dict]]:
    """
    Fetch audio features while preserving the requested track IDs.

    Spotify can return `None` for some tracks; callers that need to keep
    features aligned with specific source tracks should use this helper.
    """
    rows = []
    for start in range(0, len(ids), 100):
        batch = ids[start:start + 100]
        result = sp.audio_features(batch) or []
        for track_id, features in zip(batch, result):
            if features:
                rows.append((track_id, features))
        time.sleep(0.1)
    return rows


# ── K-means clustering ─────────────────────────────────────────────────────────

def _cluster_tracks(features: list[dict], n_clusters: int):
    """
    K-means cluster on energy, valence, danceability, tempo, acousticness.
    Returns (labels, cluster_info) where cluster_info is a list of per-cluster
    mean-feature dicts, sorted by energy descending.

    Requires HAS_ML — callers must check before calling.
    """
    FEATURE_KEYS = ["energy", "valence", "danceability", "tempo", "acousticness"]
    X = np.array([[f.get(k, 0) for k in FEATURE_KEYS] for f in features])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    cluster_info = []
    for i in range(n_clusters):
        mask = labels == i
        subset = X[mask]
        if len(subset) == 0:
            cluster_info.append({k: 0 for k in FEATURE_KEYS})
        else:
            info = {k: float(subset[:, j].mean()) for j, k in enumerate(FEATURE_KEYS)}
            cluster_info.append(info)

    # Sort by energy descending for consistent mood labelling
    order = sorted(range(n_clusters), key=lambda i: -cluster_info[i]["energy"])
    label_map = {old: new for new, old in enumerate(order)}
    labels = np.array([label_map[l] for l in labels])
    cluster_info = [cluster_info[i] for i in order]

    return labels, cluster_info


# ── Similarity scoring ─────────────────────────────────────────────────────────

def _similarity_score(target: dict, actual: dict) -> float:
    """
    0–100 similarity score using normalised Euclidean distance on
    energy, valence, danceability, acousticness, and tempo (÷200).
    """
    keys = ["energy", "valence", "danceability", "acousticness"]
    diffs = []
    for k in keys:
        t = target.get(k)
        a = actual.get(k)
        if t is not None and a is not None:
            diffs.append((t - a) ** 2)
    tt = target.get("tempo")
    at = actual.get("tempo")
    if tt is not None and at is not None:
        diffs.append(((tt - at) / 200) ** 2)
    if not diffs:
        return 0.0
    dist = math.sqrt(sum(diffs) / len(diffs))
    return max(0.0, min(100.0, (1 - dist) * 100))


def _weighted_similarity(target: dict, actual: dict, weights: dict) -> float:
    """
    Weighted Euclidean similarity using the model's learned feature weights.
    Falls back gracefully to equal weighting if weights dict is empty.
    """
    keys = ["energy", "valence", "danceability", "acousticness"]
    diffs = []
    for k in keys:
        t, a = target.get(k), actual.get(k)
        w = weights.get(k, 1.0)
        if t is not None and a is not None:
            diffs.append(w * (t - a) ** 2)
    tt, at = target.get("tempo"), actual.get("tempo")
    if tt is not None and at is not None:
        w = weights.get("tempo", 1.0)
        diffs.append(w * ((tt - at) / 200) ** 2)
    if not diffs:
        return 0.0
    w_total = sum(weights.get(k, 1.0) for k in (keys + ["tempo"]))
    dist = math.sqrt(sum(diffs) / (w_total or len(diffs)))
    return max(0.0, min(100.0, (1 - dist) * 100))


# ── Display helpers ────────────────────────────────────────────────────────────

def _print_audio_profile(title: str, profile: dict):
    """Pretty-print a set of audio feature values as bar charts."""
    bar_width = 20
    console.print(f"\n[bold]{title}[/bold]")
    for k, v in profile.items():
        if k == "tempo":
            console.print(f"  {k:<18} {v:.1f} BPM")
        else:
            filled = int(round(v * bar_width))
            bar = "█" * filled + "░" * (bar_width - filled)
            console.print(f"  {k:<18} {bar}  {v:.2f}")
