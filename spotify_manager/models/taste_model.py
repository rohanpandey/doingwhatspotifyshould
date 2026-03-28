"""
taste_model.py — Persistent taste model and session log.

TasteModel stores cluster centroids, bandit state, feature weights, loved-track
vectors, and exclusion sets between discovery sessions.  Session records are
appended to session_log.jsonl as newline-delimited JSON.

Learning pipeline
-----------------
Representation:   GaussianMixture (full covariance) on 6 audio features
Algorithm select: Thompson Sampling over 4 discovery modes
Cluster drift:    EMA pull toward loved tracks, repulsion from disliked
Feature weights:  GradientBoostingClassifier + temporal decay per session
KNN memory:       Rolling window of up to 200 loved feature vectors
Exclusions:       known (library) and disliked = permanent
                  seen = expires after SEEN_EXPIRY_DAYS (default 90)
"""

import datetime
import json
import math
import os
import random
import time

try:
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.mixture import GaussianMixture
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    HAS_ML = True
except ImportError:
    HAS_ML = False

from ..utils.display import console

# ── Constants ─────────────────────────────────────────────────────────────────

TASTE_MODEL_PATH  = "taste_model.json"
SESSION_LOG_PATH  = "session_log.jsonl"
FEATURE_KEYS_FULL = ["energy", "valence", "danceability", "tempo", "acousticness", "instrumentalness"]
ALGO_NAMES        = ["centroid", "boundary", "frontier", "artist_graph"]
MAX_LOVED_VECTORS = 200   # rolling window of loved track feature dicts
SEEN_EXPIRY_DAYS  = 90    # non-disliked seen tracks re-eligible after this many days


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
        # seen_ids: dict[track_id -> ISO timestamp] — expires after SEEN_EXPIRY_DAYS
        self.seen_ids: dict       = {}
        self.disliked_ids: set    = set()
        self.saved_ids: set       = set()
        self.clusters: list       = []
        self.top_artist_ids: list = []
        self.built_at: str        = ""
        self.session_count: int   = 0

        # Bandit state: {algo_name: {tries, rewards}}
        # rewards accumulates (loved*2 + liked) / shown  (range 0–2 per session).
        # Thompson Sampling derives Beta(alpha, beta) from these at pick time;
        # no storage format change needed.
        self.bandit: dict = {a: {"tries": 0, "rewards": 0.0} for a in ALGO_NAMES}

        # Feature weights — uniform until GradientBoosting has enough data (~20+ ratings)
        self.feature_weights: dict = {k: 1.0 for k in FEATURE_KEYS_FULL}

        # Rolling window of loved-track audio feature dicts (capped at MAX_LOVED_VECTORS)
        self.loved_vectors: list = []

    # ── Persistence ───────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "known_ids":       list(self.known_ids),
            "seen_ids":        self.seen_ids,          # dict[str, str]
            "disliked_ids":    list(self.disliked_ids),
            "saved_ids":       list(self.saved_ids),
            "clusters":        self.clusters,
            "top_artist_ids":  self.top_artist_ids,
            "built_at":        self.built_at,
            "session_count":   self.session_count,
            "bandit":          self.bandit,
            "feature_weights": self.feature_weights,
            "loved_vectors":   self.loved_vectors,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TasteModel":
        m = cls()
        m.known_ids       = set(d.get("known_ids", []))
        m.disliked_ids    = set(d.get("disliked_ids", []))
        m.saved_ids       = set(d.get("saved_ids", []))
        m.clusters        = d.get("clusters", [])
        m.top_artist_ids  = d.get("top_artist_ids", [])
        m.built_at        = d.get("built_at", "")
        m.session_count   = d.get("session_count", 0)
        m.bandit          = d.get("bandit", {a: {"tries": 0, "rewards": 0.0} for a in ALGO_NAMES})
        m.feature_weights = d.get("feature_weights", {k: 1.0 for k in FEATURE_KEYS_FULL})
        m.loved_vectors   = d.get("loved_vectors", [])

        # Migrate seen_ids: old format was a list of IDs; new format is dict[id -> timestamp].
        # Old tracks get a very old timestamp so they immediately re-qualify.
        raw_seen = d.get("seen_ids", [])
        if isinstance(raw_seen, list):
            m.seen_ids = {tid: "2000-01-01T00:00:00Z" for tid in raw_seen}
        else:
            m.seen_ids = raw_seen

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
        """
        A track is excluded if it is:
          - in the user's known library (permanent)
          - explicitly disliked (permanent)
          - in seen_ids AND the seen timestamp is within SEEN_EXPIRY_DAYS
        Tracks that were seen but whose timestamp has expired re-enter the pool.
        """
        if track_id in self.known_ids or track_id in self.disliked_ids:
            return True
        if track_id in self.seen_ids:
            try:
                seen_at = datetime.datetime.strptime(
                    self.seen_ids[track_id], "%Y-%m-%dT%H:%M:%SZ"
                )
                if (datetime.datetime.utcnow() - seen_at).days < SEEN_EXPIRY_DAYS:
                    return True
            except Exception:
                return True  # unparseable timestamp → exclude conservatively
        return False

    def mark_seen(self, track_ids: list[str]):
        """Record track IDs as seen with the current UTC timestamp."""
        now_str = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        for tid in track_ids:
            self.seen_ids[tid] = now_str

    # ── Thompson Sampling bandit ───────────────────────────────────────────────

    def bandit_pick(self) -> str:
        """
        Thompson Sampling: for each algorithm arm, derive Beta distribution
        parameters from accumulated reward history, draw a sample, and return
        the arm with the highest draw.

        Storage stays as {tries, rewards} — alpha/beta are computed on the fly:
          avg_reward_norm = (rewards / tries) / 2.0   (normalises 0–2 to 0–1)
          alpha = avg_reward_norm * tries + 1          (weighted successes + prior)
          beta  = (1 – avg_reward_norm) * tries + 1   (weighted failures  + prior)

        Arms with tries=0 use Beta(1,1) = Uniform, ensuring all arms are tried.
        """
        if not HAS_ML:
            # Fallback: round-robin without numpy
            untried = [a for a in ALGO_NAMES if self.bandit[a]["tries"] == 0]
            if untried:
                return untried[0]
            return max(ALGO_NAMES, key=lambda a: self.bandit[a]["rewards"] /
                       max(self.bandit[a]["tries"], 1))

        best_algo, best_sample = ALGO_NAMES[0], -1.0
        for algo in ALGO_NAMES:
            s     = self.bandit[algo]
            tries = s["tries"]
            if tries == 0:
                alpha, beta = 1.0, 1.0
            else:
                avg_norm = (s["rewards"] / tries) / 2.0   # → [0, 1]
                alpha    = avg_norm * tries + 1.0
                beta     = (1.0 - avg_norm) * tries + 1.0
            sample = float(np.random.beta(alpha, beta))
            if sample > best_sample:
                best_sample, best_algo = sample, algo
        return best_algo

    def bandit_update(self, algo: str, loved: int, liked: int, shown: int):
        """Update bandit arm reward after a session. reward ∈ [0, 2] per track."""
        if shown == 0 or algo not in self.bandit:
            return
        reward = (loved * 2 + liked * 1) / shown
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

    # ── Clustering (GMM) ──────────────────────────────────────────────────────

    def build_clusters(self, features: list[dict], track_ids: list[str], n_clusters: int = 4):
        """
        Fit a Gaussian Mixture Model over the 6 audio features.
        Falls back from full → diag covariance if the dataset is too small.
        Centers and spreads are stored in original (unscaled) feature space.
        """
        if not HAS_ML or not features:
            return
        X = np.array([[f.get(k, 0) for k in FEATURE_KEYS_FULL] for f in features])
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n        = min(n_clusters, len(X))

        # Try full covariance; fall back to diagonal if it fails (too few samples)
        labels = None
        for cov_type in ("full", "diag"):
            try:
                gm     = GaussianMixture(n_components=n, covariance_type=cov_type,
                                         random_state=42, n_init=10)
                labels = gm.fit_predict(X_scaled)
                break
            except Exception:
                continue
        if labels is None:
            return

        self.clusters = []
        for i in range(n):
            mask       = labels == i
            subset_raw = X[mask]
            if len(subset_raw) == 0:
                continue
            center = {k: float(subset_raw[:, j].mean()) for j, k in enumerate(FEATURE_KEYS_FULL)}
            spread = {k: float(subset_raw[:, j].std())  for j, k in enumerate(FEATURE_KEYS_FULL)}
            # Two seed tracks closest to centroid, used as API seeds
            dists = [
                (sum((features[ti].get(k, 0) - center[k]) ** 2 for k in FEATURE_KEYS_FULL),
                 track_ids[ti])
                for ti, label in enumerate(labels) if label == i and ti < len(track_ids)
            ]
            dists.sort(key=lambda x: x[0])
            seed_ids = [tid for _, tid in dists[:2]]
            self.clusters.append({
                "center": center, "spread": spread,
                "size": int(mask.sum()), "seed_track_ids": seed_ids,
            })
        self.clusters.sort(key=lambda c: -c["center"].get("energy", 0))

    def update_clusters_from_feedback(self, loved_features: list[dict],
                                       disliked_features: list[dict],
                                       ema_alpha: float = 0.10):
        """
        EMA pull: shift the nearest cluster centroid 10% toward the mean of
        loved tracks; nudge 5% away from disliked tracks.
        """
        if not self.clusters or not loved_features:
            return

        loved_X    = np.array([[f.get(k, 0) for k in FEATURE_KEYS_FULL] for f in loved_features])
        loved_mean = loved_X.mean(axis=0)

        for cl in self.clusters:
            center_vec = np.array([cl["center"].get(k, 0) for k in FEATURE_KEYS_FULL])

            # Only pull the cluster that is closest to the loved mean
            dists_to_clusters = [
                np.linalg.norm(loved_mean - np.array([c["center"].get(k, 0) for k in FEATURE_KEYS_FULL]))
                for c in self.clusters
            ]
            if self.clusters.index(cl) != int(np.argmin(dists_to_clusters)):
                continue

            new_center = (1 - ema_alpha) * center_vec + ema_alpha * loved_mean

            if disliked_features:
                disliked_X    = np.array([[f.get(k, 0) for k in FEATURE_KEYS_FULL]
                                          for f in disliked_features])
                disliked_mean = disliked_X.mean(axis=0)
                new_center   += 0.05 * (new_center - disliked_mean)

            for j, k in enumerate(FEATURE_KEYS_FULL):
                if k == "tempo":
                    new_center[j] = max(50.0, min(200.0, new_center[j]))
                else:
                    new_center[j] = max(0.0,  min(1.0,  new_center[j]))

            cl["center"] = {k: float(new_center[j]) for j, k in enumerate(FEATURE_KEYS_FULL)}

    # ── Loved-track vector memory ──────────────────────────────────────────────

    def add_loved_vectors(self, feature_dicts: list[dict]):
        """Append new loved-track feature dicts, keeping a rolling window of MAX_LOVED_VECTORS."""
        self.loved_vectors = (self.loved_vectors + feature_dicts)[-MAX_LOVED_VECTORS:]

    def score_by_loved_knn(self, features: dict, k: int = 10) -> float:
        """
        0–100 score: average similarity to the k nearest loved track vectors.
        Returns 50.0 (neutral) when fewer than k loved vectors are available.
        """
        if not self.loved_vectors:
            return 50.0
        from ..utils.audio import _similarity_score
        sims = sorted(
            [_similarity_score(lv, features) for lv in self.loved_vectors],
            reverse=True,
        )
        top_k = sims[:min(k, len(sims))]
        return sum(top_k) / len(top_k)

    # ── Feature weight learning (GradientBoosting + temporal decay) ────────────

    def learn_feature_weights(self, session_log_path: str = SESSION_LOG_PATH):
        """
        Fit GradientBoostingClassifier on (audio features → loved/liked=1 vs
        skipped/disliked=0) from the full session log.

        Two forms of weighting are applied before fitting:
          1. Class balancing  — via sklearn compute_sample_weight('balanced')
          2. Temporal decay   — 0.95^(days_ago/30) so recent sessions count more
                                (half-life ≈ 14 months)

        Feature importances are normalised so their mean = 1.0 (same scale as
        the previous uniform weights).

        Requires HAS_ML and ~20+ rated tracks with both positive and negative
        examples.
        """
        if not HAS_ML:
            return

        now = datetime.datetime.utcnow()
        X_rows, y_rows, time_weights = [], [], []

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

                # Temporal decay weight for this session
                try:
                    ts       = datetime.datetime.strptime(session["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
                    days_ago = max(0, (now - ts).days)
                except Exception:
                    days_ago = 0
                time_w = 0.95 ** (days_ago / 30.0)

                for t in session.get("tracks", []):
                    feats  = t.get("audio_features")
                    rating = t.get("rating", "skipped")
                    if not feats:
                        continue
                    row   = [feats.get(k, 0.0) for k in FEATURE_KEYS_FULL]
                    label = 1 if rating in ("loved", "liked") else 0
                    X_rows.append(row)
                    y_rows.append(label)
                    time_weights.append(time_w)

        if len(X_rows) < 20 or sum(y_rows) < 5 or sum(1 - y for y in y_rows) < 5:
            console.print("[dim]Not enough rated tracks yet to learn feature weights (need ~20+).[/dim]")
            return

        X = np.array(X_rows)
        y = np.array(y_rows)

        scaler = StandardScaler()
        X_s    = scaler.fit_transform(X)

        # Combine class-balancing weights with temporal decay
        class_w    = compute_sample_weight("balanced", y)
        sample_w   = np.array(time_weights) * class_w
        # Normalise so weights sum to len(y) (keeps effective sample size stable)
        sample_w  *= len(y) / sample_w.sum()

        gb = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                         random_state=42)
        gb.fit(X_s, y, sample_weight=sample_w)

        importances      = gb.feature_importances_
        importances_norm = importances / importances.mean()   # mean = 1.0

        self.feature_weights = {
            k: round(float(importances_norm[j]), 4)
            for j, k in enumerate(FEATURE_KEYS_FULL)
        }
        console.print(f"[green]Feature weights updated from {len(X_rows)} rated tracks.[/green]")
        console.print("  " + "  ".join(f"{k}={v:.2f}" for k, v in self.feature_weights.items()))

    # ── Target features for discovery ─────────────────────────────────────────

    def get_targets(self, mode: str, cluster_idx: int,
                    session_seed: int | None = None) -> dict:
        """
        Return a target feature dict for the given discovery mode.

        boundary / frontier use a per-session random seed so the direction of
        exploration changes every session (previously the direction was fixed
        by a deterministic hash, meaning the same quadrant was explored forever).
        """
        if not self.clusters:
            return {}
        cl     = self.clusters[cluster_idx % len(self.clusters)]
        center = cl["center"]
        spread = cl["spread"]
        if mode == "centroid":
            return dict(center)

        rng        = random.Random(session_seed)   # None → truly random each call
        target     = dict(center)
        multiplier = 1.5 if mode == "boundary" else 3.0

        for k in FEATURE_KEYS_FULL:
            s         = spread.get(k, 0.1)
            direction = 1 if rng.random() > 0.5 else -1
            if k == "tempo":
                target[k] = max(50.0, min(200.0, target[k] + direction * s * multiplier))
            else:
                target[k] = max(0.0,  min(1.0,   target[k] + direction * s * multiplier))
        return target
