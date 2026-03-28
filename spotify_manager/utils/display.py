"""
display.py — Rich/plain-text console helpers.

Provides a `console` singleton, `HAS_RICH` flag, and the `confirm`/`ask`
wrappers that every task uses for interactive prompts.
"""

try:
    from rich.console import Console
    from rich.prompt import Confirm, Prompt
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

    class Console:  # type: ignore[no-redef]
        def print(self, *a, **kw): print(*a)
        def rule(self, t=""): print(f"\n{'─'*60} {t} {'─'*60}\n")

    Confirm = None  # type: ignore[assignment]
    Prompt = None   # type: ignore[assignment]

console = Console()


def confirm(prompt: str, dry_run: bool) -> bool:
    if dry_run:
        return False
    if HAS_RICH and Confirm:
        return Confirm.ask(prompt)
    ans = input(f"{prompt} [y/N]: ").strip().lower()
    return ans == "y"


def ask(prompt: str, default: str = "") -> str:
    if HAS_RICH and Prompt:
        return Prompt.ask(prompt, default=default)
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default
