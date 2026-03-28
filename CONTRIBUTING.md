# Contributing

## Getting started

```bash
git clone <repo>
cd spotify-manager
pip install -r requirements.txt
cp .env.example .env   # fill in your credentials
```

## Running the tool locally

```bash
# Dry-run (no changes made, safe to run any time)
python -m spotify_manager --dry-run

# Run specific tasks
python -m spotify_manager --task 1,2
python -m spotify_manager --task 6
```

## Tests

There are no automated tests yet. This is a known gap.

When tests are added, they will live in a `tests/` directory and can be run with:

```bash
pytest
```

Planned coverage:

**Pure helpers** (`spotify_manager/utils/`)
- `_similarity_score` and `_weighted_similarity` — distance calculations against a feature vector
- `_assign_genre_bucket` — genre string → bucket label mapping
- `_cluster_tracks` — k-means on a list of feature dicts (tasks 3 & 4)

**`TasteModel`** (`spotify_manager/models/taste_model.py`)
- Serialisation / deserialisation round-trips, including migration from the old `seen_ids` list format to the new `dict[str, timestamp]` format
- `is_excluded()` — tracks below the 90-day expiry threshold should be excluded; tracks beyond it should not; disliked tracks should always be excluded
- `mark_seen()` — confirms a UTC timestamp is recorded and that re-calling within the same session overwrites cleanly
- `bandit_pick()` — Thompson Sampling: with one heavily rewarded mode and several cold ones, the rewarded mode should be picked with high probability after many draws
- `score_by_loved_knn()` — returns 0 when the loved-vector store is empty; returns a value in [0, 100] with a known feature vector; score should be higher for a vector identical to a loved one
- `learn_feature_weights()` — requires ≥20 rated tracks with both positive and negative labels; output weights should be normalised to mean=1.0; weights should vary when one feature is strongly predictive
- `get_targets()` with `session_seed` — two calls with different seeds should produce different boundary/frontier directions; same seed should produce the same direction

**Task logic** (mocked Spotify client)
- Task 1 duplicate detection and removal confirmation prompt
- Task 2 "never played" heuristic (track added >N days ago and not in recent 50)
- Task 4 genre bucketing end-to-end with a stub artist-feature payload

If you're adding a feature, please include at least a unit test for any new pure functions.

## Code style

- Python 3.10+, standard library + the deps in `requirements.txt`
- `rich` is optional — all output paths must have a plain-text fallback
- Keep the `--dry-run` flag honoured in every task that makes writes

## Submitting changes

1. Fork the repo and create a feature branch
2. Make your changes
3. Open a pull request with a clear description of what changed and why
