"""
taste_engine.py — Task 6: self-improving music discovery with persistent taste model.

Discovery pipeline
------------------
1. bandit_pick()     → Thompson Sampling suggests the best mode; user can override
2. session_seed      → single int used throughout the session so all random choices
                       (boundary/frontier direction, etc.) are reproducible if needed
3. Candidate source  → Spotify recommendations API  OR  artist-catalog pool
                       (user chooses; artist pool sidesteps mainstream bias)
4. Exclusion filter  → known / disliked (permanent) + seen within SEEN_EXPIRY_DAYS
5. Scoring           → _weighted_similarity with learned feature weights
                       + score_by_loved_knn from TasteModel's rolling vector memory
6. Model update      → EMA cluster drift, loved-vector update,
                       GradientBoosting re-fit with temporal decay, bandit update
"""

import datetime
import time

import spotipy

from ..utils.display import HAS_RICH, console, confirm, ask
from ..utils.spotify import paginate, get_all_playlists, get_playlist_tracks
from ..utils.audio import _batch_audio_features, _batch_audio_features_with_ids, _similarity_score, _weighted_similarity
from ..models.taste_model import (
    TasteModel, log_session, load_session_log,
    FEATURE_KEYS_FULL, MAX_LOVED_VECTORS,
)


# ── Analytics ─────────────────────────────────────────────────────────────────

def _print_algo_analytics(model: TasteModel):
    """Print per-algorithm performance (Thompson Sampling state) + session stats."""
    console.rule("[bold]Algorithm performance[/bold]")

    rows = model.bandit_summary()
    if HAS_RICH:
        from rich.table import Table
        table = Table(show_lines=True)
        table.add_column("Algorithm",  style="yellow")
        table.add_column("Sessions",   justify="right")
        table.add_column("Avg reward", justify="right", style="green")
        table.add_column("Status",     style="dim")
        for r in rows:
            avg    = f"{r['avg_reward']:.3f}" if r["avg_reward"] is not None else "—"
            status = "untried" if r["tries"] == 0 else ("best" if rows[0]["algo"] == r["algo"] else "")
            table.add_row(r["algo"], str(r["tries"]), avg, status)
        console.print(table)
    else:
        for r in rows:
            avg = f"{r['avg_reward']:.3f}" if r["avg_reward"] is not None else "—"
            print(f"  {r['algo']}: {r['tries']} sessions, avg reward {avg}")

    console.print("\n[bold]Feature weights (learned from your ratings):[/bold]")
    if all(v == 1.0 for v in model.feature_weights.values()):
        console.print("  [dim]Uniform — not enough ratings yet to learn weights.[/dim]")
    else:
        bar_w = 16
        for k, v in sorted(model.feature_weights.items(), key=lambda x: -x[1]):
            filled = min(bar_w, int(round(v / 2.0 * bar_w)))
            bar    = "█" * filled + "░" * (bar_w - filled)
            console.print(f"  {k:<20} {bar}  {v:.2f}")

    console.print(f"\n[bold]KNN memory:[/bold] {len(model.loved_vectors)} loved vectors stored "
                  f"(max {MAX_LOVED_VECTORS})")

    sessions = load_session_log()
    if sessions:
        total_tracks = sum(len(s.get("tracks", [])) for s in sessions)
        loved  = sum(1 for s in sessions for t in s.get("tracks", []) if t.get("rating") == "loved")
        liked  = sum(1 for s in sessions for t in s.get("tracks", []) if t.get("rating") == "liked")
        skip   = sum(1 for s in sessions for t in s.get("tracks", []) if t.get("rating") == "skipped")
        dislik = sum(1 for s in sessions for t in s.get("tracks", []) if t.get("rating") == "disliked")
        console.print(f"\n[bold]All-time stats[/bold]  ({len(sessions)} sessions, {total_tracks} tracks rated)")
        console.print(f"  Loved {loved} · Liked {liked} · Skipped {skip} · Disliked {dislik}")
        if total_tracks:
            console.print(f"  Love rate: {(loved+liked)/total_tracks*100:.1f}%")


# ── Model build / display ──────────────────────────────────────────────────────

def _print_model_summary(model: TasteModel):
    console.print(f"\n[bold]Taste Model[/bold]  [dim](built {model.built_at or 'never'})[/dim]")
    console.print(f"  Known (excluded):     {len(model.known_ids):,}")
    console.print(f"  Seen in window:       {len(model.seen_ids):,}  "
                  f"[dim](expire after {model.SEEN_EXPIRY_DAYS if hasattr(model, 'SEEN_EXPIRY_DAYS') else 90}d)[/dim]")
    console.print(f"  Disliked (excluded):  {len(model.disliked_ids):,}")
    console.print(f"  Saved from sessions:  {len(model.saved_ids):,}")
    console.print(f"  Sessions run:         {model.session_count}")
    console.print(f"  Loved vectors stored: {len(model.loved_vectors)}")
    console.print(f"\n  [bold]Clusters:[/bold]")
    for i, cl in enumerate(model.clusters):
        c, s = cl["center"], cl["spread"]
        console.print(
            f"  {i+1}.  energy {c.get('energy',0):.2f}±{s.get('energy',0):.2f}  "
            f"valence {c.get('valence',0):.2f}±{s.get('valence',0):.2f}  "
            f"bpm {c.get('tempo',0):.0f}  "
            f"({cl.get('size',0)} tracks)"
        )


def _build_taste_model(sp: spotipy.Spotify, model: TasteModel):
    """Scan the full library, index known tracks, and fit GMM taste clusters."""
    console.print("\n[bold]Step 1 · Indexing your library (liked songs + all playlists)…[/bold]")

    liked_raw   = paginate(sp.current_user_saved_tracks)
    liked_items = [
        it for it in liked_raw
        if it and it.get("track") and it["track"].get("id") and not it["track"].get("is_local")
    ]
    liked_ids       = [it["track"]["id"] for it in liked_items]
    model.known_ids = set(liked_ids)

    playlists = get_all_playlists(sp)
    for pl in playlists:
        for item in get_playlist_tracks(sp, pl["id"], pl.get("snapshot_id")):
            model.known_ids.add(item["track"]["id"])

    console.print(f"  {len(model.known_ids):,} tracks indexed as 'known' (will never be recommended)")

    console.print("\n[bold]Step 2 · Fetching audio features for liked songs…[/bold]")
    feature_rows = _batch_audio_features_with_ids(sp, liked_ids)
    features = [features for _, features in feature_rows]
    feature_track_ids = [track_id for track_id, _ in feature_rows]
    console.print(f"  Features fetched for {len(features):,} tracks")

    n_clusters = int(ask("How many taste clusters to model? (3–5 recommended)", default="4"))
    model.build_clusters(features, feature_track_ids, n_clusters=n_clusters)
    console.print(f"  Fitted {len(model.clusters)} clusters (GMM, full covariance)")

    try:
        top                  = sp.current_user_top_artists(limit=15, time_range="medium_term")
        model.top_artist_ids = [a["id"] for a in top.get("items", [])]
    except Exception:
        model.top_artist_ids = []

    model.built_at = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    _print_model_summary(model)


# ── Discovery helpers ──────────────────────────────────────────────────────────

def _discover_audio(sp: spotipy.Spotify, model: TasteModel, mode: str,
                    cluster_idx: int, n: int = 100,
                    session_seed: int | None = None) -> list:
    """
    Call Spotify recommendations API with audio targets from the taste model.

    Improvements vs original:
    - Tolerances are scaled by learned feature weights (tighter on important
      features, looser on less-predictive ones)
    - max_popularity=75 cap to reduce mainstream bias
    - session_seed is forwarded to get_targets so boundary/frontier directions
      are random-per-session, not hash-fixed
    """
    cluster = model.clusters[cluster_idx % len(model.clusters)]
    targets = model.get_targets(mode, cluster_idx, session_seed=session_seed)
    seeds   = cluster.get("seed_track_ids", [])[:2]
    spread  = cluster["spread"]

    tol_mult = {"centroid": 0.8, "boundary": 1.5, "frontier": 2.5}.get(mode, 1.0)

    kwargs: dict = {"seed_tracks": seeds or None, "limit": min(n, 100)}

    # Popularity cap — avoids top-40 mainstream results
    kwargs["max_popularity"] = 75

    BASE_TOL = {"energy": 0.12, "valence": 0.12, "danceability": 0.12,
                "tempo": 18.0, "acousticness": 0.18, "instrumentalness": 0.18}

    for k in FEATURE_KEYS_FULL:
        val = targets.get(k)
        if val is None:
            continue
        # Feature-weight-adjusted tolerance:
        # higher weight → feature is more predictive → tighter window
        base_tol = spread.get(k, BASE_TOL.get(k, 0.15))
        weight   = model.feature_weights.get(k, 1.0)
        tol      = (base_tol / max(weight, 0.25)) * tol_mult   # clamp weight floor

        kwargs[f"target_{k}"] = val
        if k == "tempo":
            kwargs[f"min_{k}"] = max(40.0, val - tol)
            kwargs[f"max_{k}"] = min(220.0, val + tol)
        else:
            kwargs[f"min_{k}"] = max(0.0, round(val - tol, 3))
            kwargs[f"max_{k}"] = min(1.0, round(val + tol, 3))

    if not kwargs.get("seed_tracks"):
        if model.top_artist_ids:
            kwargs.pop("seed_tracks", None)
            kwargs["seed_artists"] = model.top_artist_ids[:2]
        else:
            console.print("[yellow]No seeds available — rebuild model first.[/yellow]")
            return []

    try:
        return sp.recommendations(**kwargs).get("tracks", [])
    except Exception as e:
        console.print(f"[red]Recommendations API error: {e}[/red]")
        return []


def _discover_artist_graph(sp: spotipy.Spotify, model: TasteModel) -> list:
    """
    2-hop artist graph: top artists → related → their related.

    Scoring uses _weighted_similarity (learned feature weights) and adds a
    novelty bonus of up to 15 points for low-popularity tracks, pushing
    obscure finds ahead of mainstream ones with identical audio similarity.
    """
    if not model.top_artist_ids:
        console.print("[yellow]No top artists in model. Rebuild model first.[/yellow]")
        return []

    known_artist_ids = set(model.top_artist_ids)

    console.print("  Traversing artist graph — hop 1…")
    hop1: set = set()
    for aid in model.top_artist_ids[:6]:
        try:
            related = sp.artist_related_artists(aid)
            hop1.update(a["id"] for a in related.get("artists", []))
            time.sleep(0.08)
        except Exception:
            pass
    hop1 -= known_artist_ids

    console.print("  Traversing artist graph — hop 2…")
    hop2: set = set()
    for aid in list(hop1)[:12]:
        try:
            related = sp.artist_related_artists(aid)
            hop2.update(a["id"] for a in related.get("artists", []))
            time.sleep(0.08)
        except Exception:
            pass
    novel = hop2 - hop1 - known_artist_ids
    if not novel:
        novel = hop1

    console.print(f"  Fetching top tracks from {min(len(novel), 20)} novel artists…")
    candidate_tracks: list = []
    for aid in list(novel)[:20]:
        try:
            top_tracks = sp.artist_top_tracks(aid, country="US")
            candidate_tracks.extend(top_tracks.get("tracks", [])[:3])
            time.sleep(0.06)
        except Exception:
            pass

    if not candidate_tracks or not model.clusters:
        return candidate_tracks

    ids        = [t["id"] for t in candidate_tracks]
    feats_list = _batch_audio_features(sp, ids)
    feats_map  = {f["id"]: f for f in feats_list if f}
    centroids  = [cl["center"] for cl in model.clusters]

    def score(track) -> float:
        tid        = track["id"]
        f          = feats_map.get(tid, {})
        if not f:
            return 0.0
        audio_sim  = max(_weighted_similarity(c, f, model.feature_weights) for c in centroids)
        popularity = track.get("popularity", 50) / 100.0
        # Novelty bonus: up to +15 for very obscure tracks (popularity ≈ 0)
        novelty    = (1.0 - popularity) * 15.0
        return audio_sim + novelty

    candidate_tracks.sort(key=score, reverse=True)
    return candidate_tracks


def _discover_audio_via_artist_pool(sp: spotipy.Spotify, model: TasteModel,
                                     mode: str, cluster_idx: int,
                                     n: int, session_seed: int | None) -> list:
    """
    Bypass the Spotify recommendations API entirely.

    1. Generates a candidate pool via the 2-hop artist graph (same as artist_graph
       mode but used as the source for any mode).
    2. Scores each candidate against the mode's target position in taste space
       using weighted similarity + a novelty bonus.

    This sidesteps Spotify's recommendations API mainstream bias at the cost of
    a few extra API calls for artist traversal.
    """
    console.print("  [dim]Artist-pool mode: fetching candidates from artist catalog…[/dim]")
    candidates_raw = _discover_artist_graph(sp, model)
    if not candidates_raw:
        console.print("[yellow]Artist pool empty — falling back to Spotify API.[/yellow]")
        return _discover_audio(sp, model, mode, cluster_idx, n, session_seed)

    targets  = model.get_targets(mode, cluster_idx, session_seed=session_seed)
    ids      = [t["id"] for t in candidates_raw]
    feats_map = {f["id"]: f for f in _batch_audio_features(sp, ids) if f}

    scored = []
    for t in candidates_raw:
        f = feats_map.get(t["id"])
        if not f:
            continue
        audio_sim  = _weighted_similarity(targets, f, model.feature_weights)
        popularity = t.get("popularity", 50) / 100.0
        novelty    = (1.0 - popularity) * 15.0
        scored.append((t, audio_sim + novelty))

    scored.sort(key=lambda x: -x[1])
    return [t for t, _ in scored[:n]]


def _feedback_session(candidates: list, model: TasteModel) -> tuple[set, set, set, set]:
    """
    Interactive per-track rating loop.
    Commands: l=loved, k=like, s=skip (default), d=dislike, q=quit
    Returns (loved, liked, skipped, disliked) sets of track IDs.
    """
    loved, liked, skipped, disliked = set(), set(), set(), set()
    console.print(f"\n[bold]Rate these tracks[/bold] "
                  f"[dim](l=loved · k=like · s=skip · d=dislike · q=quit)[/dim]\n")

    for i, track in enumerate(candidates, 1):
        name   = track.get("name", "?")
        artist = track["artists"][0]["name"] if track.get("artists") else "?"
        url    = track.get("external_urls", {}).get("spotify", "")

        console.print(f"  [dim]{i:>2}/{len(candidates)}[/dim]  [bold]{name}[/bold]  —  {artist}")
        if url:
            console.print(f"         [dim]{url}[/dim]")

        choice = ask("  Rate", default="s").lower().strip()

        if choice == "q":
            break
        elif choice == "l":
            loved.add(track["id"])
            console.print("  [green]♥♥ loved[/green]")
        elif choice == "k":
            liked.add(track["id"])
            console.print("  [green]♥ liked[/green]")
        elif choice == "d":
            disliked.add(track["id"])
            console.print("  [red]✗ disliked — excluded from future sessions[/red]")
        else:
            skipped.add(track["id"])
            console.print("  [dim]→ skipped[/dim]")

    return loved, liked, skipped, disliked


# ── Task 6 ────────────────────────────────────────────────────────────────────

def task_taste_engine(sp: spotipy.Spotify, dry_run: bool):
    console.rule("[bold cyan]Task 6 · Taste Engine[/bold cyan]")

    model = TasteModel.load()
    built = bool(model.clusters)

    if HAS_RICH:
        console.print("[bold]What would you like to do?[/bold]")
        console.print("  [cyan]1[/cyan] · (Re)build taste model from your library")
        console.print("  [cyan]2[/cyan] · Discover new music")
        console.print("  [cyan]3[/cyan] · View model summary")
        console.print("  [cyan]4[/cyan] · View algorithm performance & learned feature weights")
        action = ask("Choose", default="2" if built else "1")
    else:
        action = ask("Action (1=build, 2=discover, 3=summary, 4=analytics)",
                     default="2" if built else "1")

    if action == "1":
        _build_taste_model(sp, model)
        model.save()
        return

    if action == "4":
        _print_algo_analytics(model)
        return

    if action == "3":
        if not built:
            console.print("[yellow]No model built yet — run action 1 first.[/yellow]")
        else:
            _print_model_summary(model)
        return

    # ── Discovery ─────────────────────────────────────────────────────────────
    if not built:
        console.print("[yellow]Building taste model first (required for discovery)…[/yellow]")
        _build_taste_model(sp, model)
        model.save()

    # Single seed for this session — keeps boundary/frontier direction consistent
    # across the session while being different every time you run
    session_seed = int(time.time())

    # Thompson Sampling suggests the best mode based on past performance
    suggested     = model.bandit_pick()
    mode_map      = {"1": "centroid", "2": "boundary", "3": "frontier", "4": "artist_graph"}
    mode_map_inv  = {v: k for k, v in mode_map.items()}
    suggested_num = mode_map_inv.get(suggested, "2")

    console.print("\n[bold]Discovery strategy:[/bold]")
    console.print("  [cyan]1[/cyan] · Centroid     — very similar to what you love (safe, ~85% match)")
    console.print("  [cyan]2[/cyan] · Boundary     — adjacent to your taste, genuinely unfamiliar")
    console.print("  [cyan]3[/cyan] · Frontier     — outside your clusters, direction unknown (adventure)")
    console.print("  [cyan]4[/cyan] · Artist graph — 2-hop related artists, novelty-boosted scoring")
    console.print(f"  [dim]Thompson Sampling suggests: {suggested}[/dim]")
    strategy = ask("Choose", default=suggested_num)
    mode     = mode_map.get(strategy, suggested)

    cluster_idx = 0
    if mode != "artist_graph" and len(model.clusters) > 1:
        console.print("\n[bold]Which taste cluster to explore from?[/bold]")
        for i, cl in enumerate(model.clusters):
            c = cl["center"]
            console.print(
                f"  [cyan]{i+1}[/cyan] · energy={c.get('energy',0):.2f}  "
                f"valence={c.get('valence',0):.2f}  "
                f"bpm={c.get('tempo',0):.0f}  "
                f"({cl.get('size',0)} tracks)"
            )
        cluster_idx = int(ask("Cluster number", default="1")) - 1

    n_results = int(ask("How many tracks to fetch?", default="20"))

    # ── Candidate source ───────────────────────────────────────────────────────
    use_artist_pool = False
    if mode != "artist_graph" and model.top_artist_ids:
        console.print(
            "\n[dim]Artist-pool mode fetches candidates directly from artist catalogs\n"
            "instead of Spotify's recommendations API — finds more obscure tracks.[/dim]"
        )
        pool_ans        = ask("Use artist catalog as candidate pool? (y/n)", default="n")
        use_artist_pool = pool_ans.lower().startswith("y")

    # ── Fetch candidates ───────────────────────────────────────────────────────
    console.print(f"\nRunning [bold]{mode}[/bold] discovery…")
    if mode == "artist_graph":
        candidates_raw = _discover_artist_graph(sp, model)
    elif use_artist_pool:
        candidates_raw = _discover_audio_via_artist_pool(
            sp, model, mode, cluster_idx, n=max(n_results * 4, 100),
            session_seed=session_seed,
        )
    else:
        candidates_raw = _discover_audio(
            sp, model, mode, cluster_idx, n=max(n_results * 4, 100),
            session_seed=session_seed,
        )

    # ── Exclusion filter ───────────────────────────────────────────────────────
    seen_this_response: set = set()
    candidates = []
    for t in candidates_raw:
        tid = t.get("id")
        if tid and not model.is_excluded(tid) and tid not in seen_this_response:
            candidates.append(t)
            seen_this_response.add(tid)

    if not candidates:
        console.print(
            "[yellow]No new tracks found after exclusion filtering.\n"
            "Try a different mode, or rebuild the model if your library has grown.[/yellow]"
        )
        return

    display = candidates[:n_results]

    # ── Scoring ────────────────────────────────────────────────────────────────
    rec_feats = {f["id"]: f for f in _batch_audio_features(sp, [t["id"] for t in display]) if f}
    target    = model.clusters[cluster_idx]["center"] if model.clusters else {}

    console.print(
        f"\n[bold]{len(display)} new tracks[/bold] "
        f"(excluded {len(candidates_raw) - len(candidates)} already known/seen):\n"
    )
    if HAS_RICH:
        from rich.table import Table
        table = Table(show_lines=True)
        table.add_column("#",        width=3,  justify="right")
        table.add_column("Track",    style="white")
        table.add_column("Artist",   style="dim")
        table.add_column("Energy",   justify="right", width=7)
        table.add_column("Valence",  justify="right", width=7)
        table.add_column("BPM",      justify="right", width=5)
        table.add_column("Sim %",    justify="right", style="cyan",  width=7)
        table.add_column("KNN %",    justify="right", style="green", width=7)
        for i, t in enumerate(display, 1):
            f       = rec_feats.get(t["id"], {})
            sim_pct = _weighted_similarity(target, f, model.feature_weights) if target else 0.0
            knn_pct = model.score_by_loved_knn(f) if f else 50.0
            table.add_row(
                str(i), t["name"][:36],
                t["artists"][0]["name"] if t.get("artists") else "?",
                f"{f.get('energy',0):.2f}", f"{f.get('valence',0):.2f}",
                f"{f.get('tempo',0):.0f}",
                f"{sim_pct:.0f}%",
                f"{knn_pct:.0f}%",
            )
        console.print(table)
        console.print("[dim]Sim % = weighted audio similarity to cluster target  "
                      "· KNN % = similarity to your loved-track memory[/dim]")
    else:
        for i, t in enumerate(display, 1):
            f       = rec_feats.get(t["id"], {})
            sim_pct = _weighted_similarity(target, f, model.feature_weights) if target else 0.0
            knn_pct = model.score_by_loved_knn(f) if f else 50.0
            print(
                f"  {i:>2}. {t['name']} — "
                f"{t['artists'][0]['name'] if t.get('artists') else '?'}  "
                f"(sim {sim_pct:.0f}%  knn {knn_pct:.0f}%)"
            )

    # ── Feedback ──────────────────────────────────────────────────────────────
    loved, liked_set, skipped, disliked = _feedback_session(display, model)

    # ── Build rated track records ──────────────────────────────────────────────
    rated_records = []
    for t in display:
        tid    = t["id"]
        rating = (
            "loved"    if tid in loved     else
            "liked"    if tid in liked_set else
            "disliked" if tid in disliked  else
            "skipped"
        )
        rated_records.append({
            "track_id":       tid,
            "name":           t.get("name", ""),
            "artist":         t["artists"][0]["name"] if t.get("artists") else "",
            "rating":         rating,
            "audio_features": {k: rec_feats.get(tid, {}).get(k) for k in FEATURE_KEYS_FULL},
        })

    log_session(mode, cluster_idx, rated_records)

    # ── Update model ───────────────────────────────────────────────────────────
    # 1. Mark tracks as seen (with timestamp so they expire after SEEN_EXPIRY_DAYS)
    model.mark_seen([t["id"] for t in display])
    model.disliked_ids |= disliked
    model.saved_ids    |= loved | liked_set
    model.session_count += 1

    # 2. Bandit update (Thompson Sampling posterior will shift on next bandit_pick)
    model.bandit_update(mode, loved=len(loved), liked=len(liked_set), shown=len(display))

    # 3. EMA cluster drift toward loved tracks
    loved_feats    = [rec_feats[tid] for tid in loved    if tid in rec_feats]
    disliked_feats = [rec_feats[tid] for tid in disliked if tid in rec_feats]
    if loved_feats:
        model.update_clusters_from_feedback(loved_feats, disliked_feats)
        console.print("[dim]Cluster centroids updated toward loved tracks.[/dim]")

    # 4. Append loved vectors to rolling window
    if loved_feats:
        model.add_loved_vectors(loved_feats)
        console.print(f"[dim]KNN memory updated ({len(model.loved_vectors)} vectors).[/dim]")

    # 5. Re-fit feature weights (GradientBoosting + temporal decay)
    model.learn_feature_weights()

    model.save()

    console.print(
        f"\n[bold]Session summary:[/bold]  "
        f"♥♥ {len(loved)} loved · ♥ {len(liked_set)} liked · "
        f"✗ {len(disliked)} disliked · → {len(skipped)} skipped\n"
        f"Seen pool: {len(model.seen_ids):,} tracks  "
        f"(re-eligible after 90 days)"
    )

    # ── Save loved/liked to playlist ──────────────────────────────────────────
    to_save = loved | liked_set
    if to_save and not dry_run:
        if confirm(f"Save {len(to_save)} ♥ track(s) to a new playlist?", dry_run=False):
            pl_name = ask("Playlist name",
                          default=f"Taste Engine · Session {model.session_count}")
            me      = sp.current_user()
            pl      = sp.user_playlist_create(
                me["id"], pl_name, public=False,
                description=f"Discovered via Taste Engine ({mode} mode, "
                            f"session {model.session_count})"
            )
            sp.playlist_add_items(pl["id"], [f"spotify:track:{tid}" for tid in to_save])
            console.print(f"[green]✓ '{pl_name}' created with {len(to_save)} tracks.[/green]")
