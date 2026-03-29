"""
cli.py — Argument parsing and main entry point.
"""

import argparse

from .auth import get_spotify
from .utils.display import HAS_RICH, console
from .tasks.duplicates import task_duplicates
from .tasks.never_played import task_never_played
from .tasks.size_audit import task_size_audit
from .tasks.organise import task_organize_liked
from .tasks.discovery import task_discovery
from .tasks.taste_engine import task_taste_engine
from .web import run_web_app


def main():
    parser = argparse.ArgumentParser(
        description="Spotify Playlist Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m spotify_manager --web        # launch the local browser UI\n"
            "  python -m spotify_manager              # run tasks 1–4\n"
            "  python -m spotify_manager --dry-run    # preview only, no changes\n"
            "  python -m spotify_manager --task 6     # taste engine only\n"
            "  python -m spotify_manager --task 1,2,3,4,5,6  # run everything"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would happen without making any changes",
    )
    parser.add_argument(
        "--task", default="1,2,3,4",
        help="Comma-separated task numbers to run (default: 1,2,3,4)",
    )
    parser.add_argument(
        "--web", action="store_true",
        help="Launch the local browser UI instead of the terminal workflow",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Host to bind the web UI to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port to bind the web UI to (default: 8000)",
    )
    args = parser.parse_args()

    if args.web:
        run_web_app(host=args.host, port=args.port)
        return

    tasks_to_run = {int(t.strip()) for t in args.task.split(",") if t.strip().isdigit()}
    dry_run      = args.dry_run

    if HAS_RICH:
        from rich.panel import Panel
        console.print(Panel(
            "[bold]Spotify Playlist Manager[/bold]\n"
            f"Tasks: {sorted(tasks_to_run)} | Dry-run: {dry_run}",
            style="blue",
        ))
    else:
        print("=== Spotify Playlist Manager ===")
        print(f"Tasks: {sorted(tasks_to_run)} | Dry-run: {dry_run}\n")

    if dry_run:
        console.print("[yellow]DRY-RUN MODE — no changes will be made[/yellow]\n")

    sp = get_spotify()

    if 1 in tasks_to_run:
        task_duplicates(sp, dry_run)
    if 2 in tasks_to_run:
        task_never_played(sp, dry_run)
    if 3 in tasks_to_run:
        task_size_audit(sp, dry_run)
    if 4 in tasks_to_run:
        task_organize_liked(sp, dry_run)
    if 5 in tasks_to_run:
        task_discovery(sp, dry_run)
    if 6 in tasks_to_run:
        task_taste_engine(sp, dry_run)

    console.rule("[bold green]Done[/bold green]")
