# spotify-manager

A personal Spotify toolkit with both a CLI and a local web UI for playlist cleanup, liked-song organisation, and self-improving music discovery.

---

## What it does

| Task | Command | Description |
|------|---------|-------------|
| 1 · Duplicate cleaner | `--task 1` | Finds and removes duplicate tracks across every playlist |
| 2 · Never-played finder | `--task 2` | Flags tracks not in recent play history, added >N days ago |
| 3 · Playlist size audit | `--task 3` | Surfaces oversized playlists with split suggestions |
| 4 · Liked songs organiser | `--task 4` | Clusters liked songs into genre and/or mood playlists |
| 5 · One-shot discovery | `--task 5` | Finds new tracks similar to a song or a playlist's energy |
| 6 · Taste engine | `--task 6` | Self-improving discovery that learns from your ratings over time |

---

## Interfaces

- **Web UI** — `python -m spotify_manager --web`
  Tasks 1–5 are available in the browser. Tasks 1–3 operate on a selected playlist from a dropdown. Spotify-heavy actions now run through a single background worker so only one large scan runs at a time, and the page auto-refreshes with in-progress status while that job finishes.
- **CLI** — `python -m spotify_manager ...`
  All six tasks are available here, including the full interactive Taste Engine session.

Task 6 is still primarily a CLI workflow because it depends on per-track rating input during a discovery session.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install spotipy scikit-learn numpy rich
```

### 2. Create a Spotify app

1. Go to [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard)
2. Click **Create App**
3. Set the redirect URI to `http://127.0.0.1:8888/callback`
4. Copy your **Client ID** and **Client Secret**

### 3. Configure credentials

Create a `.env` file in the project root:

```env
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8888/callback
```

### 4. Run

```bash
# Launch the local web UI
python -m spotify_manager --web

# Dry-run first — see what would happen, no changes made
python -m spotify_manager --dry-run

# Run the default task set (tasks 1–4: duplicates, never-played, size audit, organise liked)
python -m spotify_manager

# Run specific tasks
python -m spotify_manager --task 1,2
python -m spotify_manager --task 5
python -m spotify_manager --task 6

# Run everything
python -m spotify_manager --task 1,2,3,4,5,6
```

On first run, a browser window will open for Spotify OAuth. The token is cached in `.spotify_token_cache` for future runs. If you authenticated before saved-track editing was added and liked-song cleanup now fails, delete `.spotify_token_cache` once and re-authenticate so Spotify grants the `user-library-modify` scope.

When running `--web`, the app starts a local server at `http://127.0.0.1:8000` by default. You can change that with `--host` and `--port`.

The app now also ships with a shared Spotify client wrapper that:

- spaces requests out slightly instead of firing them back-to-back
- caches stable reads like profile data, playlist pages, liked-song pages, artist lookups, and audio features for longer
- remembers Spotify `429` cooldown windows and reuses cached data where it safely can
- serializes web jobs so the browser UI does not kick off overlapping Spotify-heavy scans

In the browser UI:

- Task 1 scans one selected playlist for duplicate songs
- Task 2 reviews one selected playlist for older tracks not seen in your recent listening history
- Task 3 audits one selected playlist against a size threshold and can preview a smart split
- Task 4 organizes your liked songs
- Task 5 runs one-shot discovery
- When a heavy job is running, the page shows the active job state and temporarily locks the other Spotify write/scan actions until it completes

---

## Task reference

### Task 1 · Duplicate cleaner

Scans playlists you own, and optionally your Liked Songs, to find the same song appearing more than once even when Spotify IDs differ across versions. It prefers ISRC when available, otherwise falls back to normalized song name + primary artist + duration bucket. Prompts before removing and tells you exactly which positions will be deleted.

### Task 2 · Never-played finder

> **Limitation:** Spotify's API only exposes your last 50 played tracks. "Never played" here means: not in your recent 50 *and* added more than N days ago (you choose N at runtime). Treat the output as a review list, not a definitive answer.

Saves results to `never_played_report.json`.

### Task 3 · Playlist size audit

Flags playlists over a configurable track threshold (default 80). Shows how many sub-playlists a split would produce. Optionally runs a k-means audio feature analysis on a specific oversized playlist to suggest how to split it.

### Task 4 · Liked songs organiser

Two grouping strategies — choose one or both:

**By mood/vibe** — k-means clustering on Spotify audio features (energy, valence, danceability, tempo, acousticness). You choose how many clusters. Resulting playlists are labelled by vibe:

| Cluster | Vibe |
|---------|------|
| 🔥 High Energy · Happy | Hype, workout, party |
| ⚡ High Energy · Dark | Intense, focus, dark electronic |
| 🌅 Chill · Upbeat | Feel-good background |
| 🌙 Low Energy · Introspective | Late night, ambient, sad |

**By genre** — fetches genre tags from each track's primary artist and buckets them:

| Bucket | Catches |
|--------|---------|
| 🎤 Hip-Hop / R&B | hip hop, rap, trap, r&b, soul, neo soul |
| 🎸 Rock / Alt | rock, alternative, indie, punk, emo, metal |
| 🎹 Electronic / Dance | edm, house, techno, synthwave, ambient, dubstep |
| 🎷 Jazz / Blues / Soul | jazz, blues, funk, gospel, swing |
| 🎻 Classical / Soundtrack | classical, orchestra, film score |
| 🌍 World / Latin | latin, afrobeats, k-pop, reggaeton, dancehall |
| 🎵 Pop | pop, synth pop, dream pop |
| 🪕 Folk / Country | folk, country, bluegrass, americana |

### Task 5 · One-shot discovery

Targeted single-session recommendations. Three modes:

- **By song** — search for a track, seed Spotify's recommendations API using its exact audio fingerprint as target parameters (not just "fans also liked")
- **By playlist** — computes the average audio feature profile of one of your playlists, uses that as the recommendation target
- **Both** — blends the two (40% song, 60% playlist) so results feel familiar but fit the playlist's energy context

Results show a match % score and can optionally be saved to a new playlist.

### Task 6 · Taste engine

A persistent, self-improving discovery system. The core idea: every track in your library is indexed as "known" and will never be recommended. Tracks surfaced in past sessions expire from the "seen" set after 90 days, so the pool stays novel without permanently shrinking. The system gets more interesting over time, not less.

**Discovery modes:**

| Mode | What it does |
|------|-------------|
| Centroid | Finds tracks closest to your taste cluster centroids — safe, high match rate |
| Boundary | Pushes 1–2 standard deviations outside your clusters — adjacent, genuinely new; exploration direction is randomised per session |
| Frontier | Pushes 2–3 standard deviations out — real adventure, lower hit rate; direction randomised per session |
| Artist graph | 2-hop traversal: your top artists → related → their related; scores top tracks by weighted audio similarity + novelty bonus for low-popularity tracks |
| Artist pool | Same artist graph traversal but re-ranks candidates by weighted similarity locally, bypassing the Spotify recommendations API entirely — surfaces more obscure tracks |

**Rating system** (during a session):

| Key | Meaning |
|-----|---------|
| `l` | Loved — counts double in learning; audio features stored for KNN scoring |
| `k` | Liked |
| `s` | Skip (default) |
| `d` | Disliked — excluded from all future sessions |
| `q` | Quit session |

Each track in the results table shows two scores: **Sim %** (weighted audio similarity to your taste cluster target) and **KNN %** (average similarity to the audio features of tracks you've previously loved).

**What the engine learns from your ratings:**

1. **Session log** (`session_log.jsonl`) — every shown track is logged with its audio features, rating, and which algorithm surfaced it. Append-only. Never deleted.

2. **Algorithm bandit (Thompson Sampling)** — tracks the love+like rate per discovery mode using a Beta distribution. Balances exploiting the best-performing algorithm with exploring under-tried ones probabilistically. View the leaderboard with action 4 in the task menu.

3. **Cluster drift (EMA)** — after each session, cluster centroids shift 10% toward the mean audio features of tracks you loved. Your taste model slowly becomes a reflection of what you actually respond to, not just your historical saves. Clusters are modelled with a Gaussian Mixture Model (full covariance) for better fit on non-spherical taste distributions.

4. **Feature weight learning** — once you have ~20+ rated tracks, a gradient-boosted classifier is fitted on (audio features → loved/liked vs skipped/disliked), with recent ratings weighted more heavily via exponential temporal decay. The resulting feature importance scores replace the uniform weights in the similarity metric. If high valence turns out to be strongly predictive for you, the match scores and target feature selection will reflect that.

5. **Loved-vector memory** — the audio features of every track you love are stored in a rolling window (up to 200 vectors). Each session, new candidates are scored against this window (KNN %) in addition to the cluster-target similarity score, giving you a second signal that's grounded in your actual moment-to-moment reactions rather than the aggregate cluster.

**Seen-track expiry:**

Tracks are marked "seen" with a timestamp rather than permanently excluded. After 90 days a track becomes eligible again — useful for artists with a large back-catalogue that you want to rediscover. Disliked tracks are excluded permanently regardless.

**Persistent files:**

| File | Purpose |
|------|---------|
| `taste_model.json` | Cluster centroids, bandit state, feature weights, loved vectors, exclusion sets |
| `session_log.jsonl` | Full append-only history of every shown/rated track |
| `.spotify_token_cache` | OAuth token (auto-refreshed) |

---

## Project structure

```
spotify-manager/
├── spotify_manager/          # Package
│   ├── __main__.py           # Entry point: python -m spotify_manager
│   ├── auth.py               # OAuth + Spotify client setup
│   ├── cli.py                # Argument parsing and task dispatch
│   ├── web.py                # Local browser UI
│   ├── utils/
│   │   ├── display.py        # Rich/plain-text console helpers
│   │   ├── audio.py          # Audio feature helpers + k-means clustering
│   │   ├── duplicates.py     # Semantic duplicate matching helpers
│   │   ├── spotify.py        # Pagination and library helpers
│   │   └── spotify_client.py # Shared Spotify throttling, cooldowns, and caching
│   ├── models/
│   │   └── taste_model.py    # TasteModel class + session log functions
│   └── tasks/
│       ├── duplicates.py     # Task 1
│       ├── never_played.py   # Task 2
│       ├── size_audit.py     # Task 3
│       ├── organise.py       # Task 4
│       ├── discovery.py      # Task 5
│       └── taste_engine.py   # Task 6
├── requirements.txt
├── .env                      # Credentials (not committed)
├── .env.example              # Template — safe to commit
├── .gitignore
├── CONTRIBUTING.md
├── taste_model.json          # Generated — your persistent taste model
├── session_log.jsonl         # Generated — full session history
├── never_played_report.json  # Generated by task 2
├── discovery_report.json     # Generated by task 5
└── README.md
```

---

## Notes

**On "never played" accuracy** — Spotify does not expose full play history via the API. The heuristic (not in last 50 plays + added > N days ago) is intentionally conservative. Use the JSON report as a starting point for manual review.

**On Spotify's recommendations API** — the centroid/boundary/frontier modes in task 6 call Spotify's recommendations endpoint, which surfaces tracks with meaningful play counts; truly obscure music is less likely to appear here. The artist-graph and artist-pool modes bypass this by fetching directly from artist catalogues and re-ranking locally — they will surface more niche tracks. A popularity cap (`max_popularity = 75`) is applied across all recommendation calls to bias results toward less mainstream picks.

**On feature weights** — the gradient-boosted classifier needs both positive (loved/liked) and negative (skipped/disliked) examples to learn anything meaningful. If you skip everything, the weights won't update. Rate honestly for best results. Temporal decay means recent ratings influence the model more than older ones, so your tastes can shift over time without manual resets.

**On the `.env` file** — never commit it. The `.env.example` template is safe to commit; `.env` is gitignored.

**On missing audio features** — Spotify can omit audio features for some tracks. The clustering and split-preview paths now keep track IDs aligned with returned feature rows so missing entries do not scramble grouping results.

---

## Requirements

- Python 3.10+
- `spotipy >= 2.23`
- `scikit-learn >= 1.3` (tasks 4, 6 feature learning)
- `numpy >= 1.24` (tasks 4, 6)
- `rich >= 13.0` (optional but strongly recommended — prettier output)
- Spotify Premium (required for some API endpoints)
