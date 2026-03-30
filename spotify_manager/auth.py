"""
auth.py — Spotify OAuth setup.
"""

import os
import sys
from pathlib import Path

import spotipy
from spotipy.oauth2 import SpotifyOAuth

from .utils.display import console
from .utils.spotify_client import wrap_spotify_client

SCOPES = " ".join([
    "playlist-read-private",
    "playlist-read-collaborative",
    "playlist-modify-private",
    "playlist-modify-public",
    "user-library-read",
    "user-library-modify",
    "user-read-recently-played",
    "user-top-read",
])


def get_spotify() -> spotipy.Spotify:
    """Load credentials from .env (if present) and return an authenticated client."""
    # Look for .env in the project root (two levels up from this file inside the package)
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    client_id     = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    redirect_uri  = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")

    if not client_id or not client_secret:
        console.print("[bold red]Missing SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET.[/bold red]")
        console.print("Set them in your .env file or as environment variables.")
        console.print("See: https://developer.spotify.com/dashboard")
        sys.exit(1)

    auth = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=SCOPES,
        cache_path=".spotify_token_cache",
        open_browser=True,
    )
    client = spotipy.Spotify(
        auth_manager=auth,
        requests_timeout=15,
        status_retries=0,
    )
    return wrap_spotify_client(client)
