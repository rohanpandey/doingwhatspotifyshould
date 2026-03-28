"""
taste_model.py — Persistent taste model and session log.

TasteModel stores cluster centroids, bandit state, feature weights, and
exclusion sets between discovery sessions.  Session records are appended to
session_log.jsonl as newline-delimited JSON.
"""

import datetime
import json
import math
import os
import time

try:
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    HAS_ML = True
except ImportError:
    HAS_ML = False

from ..utils.display import console

# ── Constants ─────────────────────────────────────────────────────────────────

TASTE_MODEL_PATH  = "taste_model.json"
SESSION_LOG_PATH  = "session_log.jsonl"
FEATURE_KEYS_FULL = ["energy", "valence", "danceability", "tempo", "acousticness", "instrumentalness"]
ALGO_NAMES        = ["centroid", "boundary", "frontier", "artist_graph"]


# ── Session log ───────────────────────────────────────────────────────────────

def log_session(algo: str, cluster_idx: int, rated_tracks: list[dict]):
    """Append one session record to session_log.jsonl (append-only)."""
    record = {
        "session_id":  f"s{int(time.time())}",
        "timestamp":   datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "algorithm":   algo,
        "cluster_idx": cluster_idx,
        "tracks":      rated_tracks,
    }
    with open(SESSION_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_session_log() -> list[dict]:
    """Load all session records from session_log.jsonl."""
    sessions = []
    if not os.path.exists(SESSION_LOG_PATH):
        return sessions
    with open(SESSION_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sessions.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return sessions


# ── TasteModel ────────────────────────────────────────────────────────────────

class TasteModel:
    """Persistent, session-aware taste model stored as a local JSON file."""

    def __init__(self):
        self.known_ids: set       = set()
        self.seen_ids: set        = set()
        self.disliked_ids: set    = set()
        self.saved_ids: set       = set()
        self.clusters: list       = []
        self.top_artist_ids: list = []
        self.built_at: str        = ""
        self.session_count: int   = 0

        # UCB1 bandit state: {algo_name: {tries, rewards}}
        self.bandit: dict = {a: {"tries": 0, "rewards": 0.0} for a in ALGO_NAMES}

        # Feature weights — uniform until enough rated tracks accumulate
        self.feature_weights: dict = {k: 1.0 for k in FEATURE_KEYS_FULL}

    # ── Persistence ───────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "known_ids":       list(self.known_ids),
            "seen_ids":        list(self.seen_ids),
            "disliked_ids":    list(self.disliked_ids),
            "saved_ids":       list(self.saved_ids),
            "clusters":        self.clusters,
            "top_artist_ids":  self.top_artist_ids,
            "built_at":        self.built_at,
            "session_count":   self.session_count,
            "bandit":          self.bandit,
            "feature_weights": self.feature_weights,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TasteModel":
        m = cls()
        m.known_ids       = set(d.get("known_ids", []))
        m.seen_ids        = set(d.get("seen_ids", []))
        m.disliked_ids    = set(d.get("disliked_ids", []))
        m.saved_ids       = set(d.get("saved_ids", []))
        m.clusters        = d.get("clusters", [])
        m.top_artist_ids  = d.get("top_artist_ids", [])
        m.built_at        = d.get("built_at", "")
        m.session_count   = d.get("session_count", 0)
        m.bandit          = d.get("bandit", {a: {"tries": 0, "rewards": 0.0} for a in ALGO_NAMES})
        m.feature_weights = d.get("feature_weights", {k: 1.0 for k in FEATURE_KEYS_FULL})
        # Back-fill any missing algo keys (e.g. on model upgrade)
        for a in ALGO_NAMES:
            m.bandit.setdefault(a, {"tries": 0, "rewards": 0.0})
        return m

    def save(self):
        with open(TASTE_MODEL_PATH, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        console.print(f"[dim]Model saved → {TASTE_MODEL_PATH}[/dim]")

    @classmethod
    def load(cls) -> "TasteModel":
        if os.path.exists(TASTE_MODEL_PATH):
            try:
                with open(TASTE_MODEL_PATH) as f:
                    return cls.from_dict(json.load(f))
            except Exception as e:
                console.print(f"[yellow]Could not load existing model ({e}), starting fresh.[/yellow]")
        return cls()

    # ── Exclusion ─────────────────────────────────────────────────────────────

    def is_excluded(self, track_id: str) -> bool:
        return (track_id in self.known_ids or
                track_id in self.seen_ids  or
                track_id in self.disliked_ids)

    # ── UCB1 bandit ───────────────────────────────────────────────────────────

    def bandit_pick(self) -> str:
        """UCB1: pick the algorithm with the highest upper confidence bound."""
        total_tries = sum(s["tries"] for s in self.bandit.values())
        if total_tries == 0:
            return ALGO_NAMES[0]  # cold start

        best_algo, best_score = None, -1.0
        for algo in ALGO_NAMES:
            s = self.bandit[algo]
            if s["tries"] == 0:
                return algo  # try everything at least once first
            avg_reward = s["rewards"] / s["tries"]
            exploration = math.sqrt(2 * math.log(total_tries) / s["tries"])
            score = avg_reward + exploration
            if score > best_score:
                best_score, best_algo = score, algo
        return best_algo

    def bandit_update(self, algo: str, loved: int, liked: int, shown: int):
        """Update bandit arm reward after a session."""
        if shown == 0 or algo not in self.bandit:
            return
        reward = (loved * 2 + liked * 1) / shown  # 0–2 per track shown
        self.bandit[algo]["tries"]   += 1
        self.bandit[algo]["rewards"] += reward

    def bandit_summary(self) -> list[dict]:
        """Return sorted list of algo performance dicts for display."""
        rows = []
        for algo in ALGO_NAMES:
            s = self.bandit[algo]
            rows.append({
                "algo":       algo,
                "tries":      s["tries"],
                "avg_reward": round(s["rewards"] / s["tries"], 3) if s["tries"] else None,
            })
        rows.sort(key=lambda r: -(r["avg_reward"] or -1))
        return rows

    # ── Clustering ────────────────────────────────────────────────────────────

    def build_clusters(self, features: list[dict], track_ids: list[str], n_clusters: int = 4):
        if not HAS_ML or not features:
            return
        X = np.array([[f.get(k, 0) for k in FEATURE_KEYS_FULL] for f in features])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n = min(n_clusters, len(X))
        km = KMeans(n_clusters=n, random_state=42, n_init=12)
        labels = km.fit_predict(X_scaled)
        self.clusters = []
        for i in range(n):
            mask = labels == i
            subset_raw = X[mask]
            subset_ids = [track_ids[j] for j, m in enumerate(mask) if m and j < len(track_ids)]
            if len(subset_raw) == 0:
                continue
            center = {k: float(subset_raw[:, j].mean()) for j, k in enumerate(FEATURE_KEYS_FULL)}
            spread = {k: float(subset_raw[:, j].std())  for j, k in enumerate(FEATURE_KEYS_FULL)}
            dists = [
                (sum((features[ti].get(k, 0) - center[k]) ** 2 for k in FEATURE_KEYS_FULL), track_ids[ti])
                for ti, label in enumerate(labels) if label == i and ti < len(track_ids)
            ]
            dists.sort(key=lambda x: x[0])
            seed_ids = [tid for _, tid in dists[:2]]
            self.clusters.append({
                "center": center, "spread": spread,
                "size": int(mask.sum()), "seed_track_ids": seed_ids,
            })
        self.clusters.sort(key=lambda c: -c["center"].get("energy", 0))

    def update_clusters_from_feedback(self, loved_features: list[dict], disliked_features: list[dict],
                                       ema_alpha: float = 0.10):
        """
        EMA pull: shift each cluster centroid 10% toward the mean of loved tracks
        that belong to it; nudge away from disliked tracks.
        """
        if not self.clusters or not loved_features:
            return

        loved_X    = np.array([[f.get(k, 0) for k in FEATURE_KEYS_FULL] for f in loved_features])
        loved_mean = loved_X.mean(axis=0)

        for cl in self.clusters:
            center_vec = np.array([cl["center"].get(k, 0) for k in FEATURE_KEYS_FULL])

            # Only pull this cluster if loved mean is closest to it
            dists_to_clusters = [
                np.linalg.norm(loved_mean - np.array([c["center"].get(k, 0) for k in FEATURE_KEYS_FULL]))
                for c in self.clusters
            ]
            if self.clusters.index(cl) != int(np.argmin(dists_to_clusters)):
                continue

            new_center = (1 - ema_alpha) * center_vec + ema_alpha * loved_mean

            if disliked_features:
                disliked_X    = np.array([[f.get(k, 0) for k in FEATURE_KEYS_FULL] for f in disliked_features])
                disliked_mean = disliked_X.mean(axis=0)
                new_center   += 0.05 * (new_center - disliked_mean)

            # Clamp features to valid ranges
            for j, k in enumerate(FEATURE_KEYS_FULL):
                if k == "tempo":
                    new_center[j] = max(50.0, min(200.0, new_center[j]))
                else:
                    new_center[j] = max(0.0, min(1.0, new_center[j]))

            cl["center"] = {k: float(new_center[j]) for j, k in enumerate(FEATURE_KEYS_FULL)}

    # ── Feature weight learning ───────────────────────────────────────────────

    def learn_feature_weights(self, session_log_path: str = SESSION_LOG_PATH):
        """
        Fit logistic regression on (audio features → loved/liked vs skipped/disliked)
        from the full session log, then store the resulting feature importance scores.
        Requires HAS_ML and ~20+ rated tracks with both positive and negative examples.
        """
        if not HAS_ML:
            return

        X_rows, y_rows = [], []
        if not os.path.exists(session_log_path):
            return

        with open(session_log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    session = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for t in session.get("tracks", []):
                    feats  = t.get("audio_features")
                    rating = t.get("rating", "skipped")
                    if not feats:
                        continue
                    row   = [feats.get(k, 0.0) for k in FEATURE_KEYS_FULL]
                    label = 1 if rating in ("loved", "liked") else 0
                    X_rows.append(row)
                    y_rows.append(label)

        if len(X_rows) < 20 or sum(y_rows) < 5 or sum(1 - y for y in y_rows) < 5:
            console.print("[dim]Not enough rated tracks yet to learn feature weights (need ~20+).[/dim]")
            return

        X = np.array(X_rows)
        y = np.array(y_rows)

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        lr = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
        lr.fit(X_s, y)

        coefs      = lr.coef_[0]
        coefs_pos  = coefs - coefs.min() + 0.01
        coefs_norm = coefs_pos / coefs_pos.mean()

        self.feature_weights = {k: round(float(coefs_norm[j]), 4) for j, k in enumerate(FEATURE_KEYS_FULL)}
        console.print(f"[green]Feature weights updated from {len(X_rows)} rated tracks.[/green]")
        console.print("  " + "  ".join(f"{k}={v:.2f}" for k, v in self.feature_weights.items()))

    # ── Target features for discovery ─────────────────────────────────────────

    def get_targets(self, mode: str, cluster_idx: int) -> dict:
        if not self.clusters:
            return {}
        cl     = self.clusters[cluster_idx % len(self.clusters)]
        center = cl["center"]
        spread = cl["spread"]
        if mode == "centroid":
            return dict(center)
        target     = dict(center)
        multiplier = 1.5 if mode == "boundary" else 3.0
        for k in FEATURE_KEYS_FULL:
            s         = spread.get(k, 0.1)
            direction = 1 if hash(k + str(cluster_idx)) % 2 == 0 else -1
            if k == "tempo":
                target[k] = max(50.0, min(200.0, target[k] + direction * s * multiplier))
            else:
                target[k] = max(0.0, min(1.0, target[k] + direction * s * multiplier))
        return target
