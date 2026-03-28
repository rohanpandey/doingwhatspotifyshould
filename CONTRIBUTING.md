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
- Unit tests for pure helpers (`_similarity_score`, `_assign_genre_bucket`, `_cluster_tracks`, etc.)
- `TasteModel` serialisation / deserialisation round-trips
- Mocked Spotify API responses for task logic

If you're adding a feature, please include at least a unit test for any new pure functions.

## Code style

- Python 3.10+, standard library + the deps in `requirements.txt`
- `rich` is optional — all output paths must have a plain-text fallback
- Keep the `--dry-run` flag honoured in every task that makes writes

## Submitting changes

1. Fork the repo and create a feature branch
2. Make your changes
3. Open a pull request with a clear description of what changed and why
