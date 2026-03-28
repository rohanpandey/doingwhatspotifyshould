# spotify-manager

A personal Spotify toolkit that cleans up your library, organises your liked songs, and builds a self-improving music discovery engine — replacing the "you've already heard all of this" problem with Spotify Radio.

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
3. Set the redirect URI to `http://localhost:8888/callback`
4. Copy your **Client ID** and **Client Secret**

### 3. Configure credentials

Create a `.env` file in the project root:

```env
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
SPOTIFY_REDIRECT_URI=http://localhost:8888/callback
```

### 4. Run

```bash
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

On first run, a browser window will open for Spotify OAuth. The token is cached in `.spotify_token_cache` for future runs.

---

## Task reference

### Task 1 · Duplicate cleaner

Scans all playlists you own and finds tracks appearing more than once (matched by Spotify track ID). Prompts before removing — tells you exactly which positions will be deleted.

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

A persistent, self-improving discovery system. The core idea: every track in your library is indexed as "known" and will never be recommended. Every track shown in any session is added to a "seen" set and won't appear again. The system gets more novel over time, not less.

**Discovery modes:**

| Mode | What it does |
|------|-------------|
| Centroid | Finds tracks closest to your taste cluster centroids — safe, high match rate |
| Boundary | Pushes 1–2 standard deviations outside your clusters — adjacent, genuinely new |
| Frontier | Pushes 2–3 standard deviations out — real adventure, lower hit rate |
| Artist graph | 2-hop traversal: your top artists → related → their related; scores top tracks by audio similarity |

**Rating system** (during a session):

| Key | Meaning |
|-----|---------|
| `l` | Loved — counts double in learning |
| `k` | Liked |
| `s` | Skip (default) |
| `d` | Disliked — permanently excluded from future sessions |
| `q` | Quit session |

**What the engine learns from your ratings:**

1. **Session log** (`session_log.jsonl`) — every shown track is logged with its audio features, rating, and which algorithm surfaced it. Append-only. Never deleted.

2. **Algorithm bandit (UCB1)** — tracks love+like rate per discovery mode. Balances exploiting the best-performing algorithm with exploring under-tried ones. View the leaderboard with action 4 in the task menu.

3. **Cluster drift (EMA)** — after each session, cluster centroids shift 10% toward the mean audio features of tracks you loved. Your taste model slowly becomes a reflection of what you actually respond to, not just your historical saves.

4. **Feature weight learning** — once you have ~20+ rated tracks, a logistic regression is fitted on (audio features → loved/liked vs skipped/disliked). The resulting feature importance scores replace the uniform weights in the similarity metric. If high valence turns out to be strongly predictive for you, the match scores and target feature selection will reflect that.

**Persistent files:**

| File | Purpose |
|------|---------|
| `taste_model.json` | Cluster centroids, bandit state, feature weights, exclusion sets |
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
│   ├── utils/
│   │   ├── display.py        # Rich/plain-text console helpers
│   │   ├── audio.py          # Audio feature helpers + k-means clustering
│   │   └── spotify.py        # Pagination and library helpers
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

**On Spotify's recommendations API** — the API used in tasks 5 and 6 (centroid/boundary/frontier modes) only surfaces tracks with meaningful play counts, so truly obscure music is less likely to appear. The artist graph mode partially sidesteps this by fetching directly from artist catalogs rather than going through the recommendations endpoint.

**On feature weights** — the logistic regression needs both positive (loved/liked) and negative (skipped/disliked) examples to learn anything meaningful. If you skip everything, the weights won't update. Rate honestly for best results.

**On the `.env` file** — never commit it. The `.env.example` template is safe to commit; `.env` is gitignored.

---

## Requirements

- Python 3.10+
- `spotipy >= 2.23`
- `scikit-learn >= 1.3` (tasks 4, 6 feature learning)
- `numpy >= 1.24` (tasks 4, 6)
- `rich >= 13.0` (optional but strongly recommended — prettier output)
- Spotify Premium (required for some API endpoints)

