"""Terminal look of the CafChem MedRAg CLI (colors in the spirit of MoDrAg: cyan / magenta / orange) and live status reporting.

Pipeline steps and agent tools report through here, so both entry points show what is happening and how much came out of each
step. When output is not a terminal (a log file, a pipe) everything falls back to plain `[run] step name` lines, which is what
the log watchers grep for."""
from __future__ import annotations

import time
from contextlib import contextmanager

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console(highlight=False)
RUN_ID = ""

CYAN, MAGENTA, ORANGE, GREEN, YELLOW, RED, DIM = "bold cyan", "bold magenta", "dark_orange", "green", "yellow", "red", "grey62"

# "MedRAg" in figlet-style letters
_BANNER = r"""
 __  __             _  ____      _
|  \/  |  ___    __| ||  _ \    / \     __ _
| |\/| | / _ \  / _` || |_) |  / _ \   / _` |
| |  | ||  __/ | (_| ||  _ <  / ___ \ | (_| |
|_|  |_| \___|  \__,_||_| \_\/_/   \_\ \__, |
                                       |___/"""


def banner(run_id: str = "", backend: str = "", model: str = "") -> None:
    """The start-up banner: CafChem MedRAg in a double-line box, with the tagline in small grey type."""
    if not console.is_terminal:
        print(f"CafChem MedRAg | run {run_id} | {backend}:{model}")
        return
    art = Text("\n").join(Text(line, style=CYAN if i % 2 == 0 else MAGENTA)
                          for i, line in enumerate(_BANNER.strip("\n").splitlines()))
    parts = [Text("CafChem", style=ORANGE), art, Text("medicinal chemistry agent", style=f"italic {DIM}"),
             Text("generate -> filter -> dock -> site check -> MLIP validate -> report", style=DIM)]
    if run_id:
        parts.append(Text(f"run {run_id}  |  {backend}:{model}", style=DIM))
    console.print(Panel(Group(*parts), box=box.DOUBLE, border_style=CYAN, padding=(0, 2), expand=False))


def set_run(run_id: str) -> None:
    global RUN_ID
    RUN_ID = run_id


def step_start(name: str, detail: str = "") -> float:
    """Announce a pipeline step; returns the start time for step_done."""
    if console.is_terminal:
        console.print(f"[{CYAN}]▶ {name}[/] [{DIM}]{detail}[/]")
    else:
        print(f"[{RUN_ID}] step {name}", flush=True)
    return time.time()


def step_done(name: str, summary: str, t0: float | None = None) -> None:
    secs = f" ({time.time() - t0:.0f}s)" if t0 else ""
    if console.is_terminal:
        console.print(f"[{GREEN}]✓[/] [bold]{name}[/]  {summary}[{DIM}]{secs}[/]")
    else:
        print(f"[{RUN_ID}]   {name}: {summary}{secs}", flush=True)


def note(text: str, style: str = DIM) -> None:
    """One indented line of detail inside a step (e.g. one generator's yield)."""
    if console.is_terminal:
        console.print(f"    [{style}]{text}[/]")
    else:
        print(f"[{RUN_ID}]     {text}", flush=True)


def warn(text: str) -> None:
    if console.is_terminal:
        console.print(f"[{YELLOW}]![/] {text}")
    else:
        print(f"[{RUN_ID}] WARNING {text}", flush=True)


def tool_call(text: str) -> None:
    if console.is_terminal:
        console.print(f"[{ORANGE}]⚙ {text}[/]")
    else:
        print(f"[tool] {text}", flush=True)


def tool_result(text: str) -> None:
    first = text.strip().splitlines()[0] if text.strip() else ""
    if console.is_terminal:
        console.print(f"  [{DIM}]{first[:160]}[/]")
    else:
        print(f"[result] {first[:160]}", flush=True)


class _Plain:
    """Progress stand-in for non-terminals: one line every ~10% instead of a live bar."""

    def __init__(self, label: str, total: int):
        self.label, self.total, self.n, self.next = label, total, 0, 0.1

    def advance(self, detail: str = "", n: int = 1) -> None:
        self.n += n
        if self.total and self.n / self.total >= self.next:
            self.next += 0.1
            print(f"[{RUN_ID}]     {self.label} {self.n}/{self.total} {detail}", flush=True)


class _Live:
    def __init__(self, bar, task):
        self.bar, self.task = bar, task

    def advance(self, detail: str = "", n: int = 1) -> None:
        self.bar.update(self.task, advance=n, description=f"{self.bar.tasks[self.task].fields['label']} [{DIM}]{detail[:48]}[/]")


@contextmanager
def progress(label: str, total: int):
    """`with ui.progress("docking", 54) as p: ... p.advance("CCO...")` - a live bar on a terminal, 10% log lines otherwise."""
    if not console.is_terminal or total <= 0:
        yield _Plain(label, total)
        return
    bar = Progress(SpinnerColumn(style=CYAN), TextColumn("{task.description}"), BarColumn(bar_width=28), MofNCompleteColumn(),
                   TimeElapsedColumn(), console=console, transient=True)
    with bar:
        task = bar.add_task(label, total=total, label=label)
        yield _Live(bar, task)


def verdict_line(species: str, verdict: str, strain: float | None, vina: float | None, k: int, n: int) -> None:
    color = {"accept": GREEN, "demote": YELLOW, "reject": RED}.get(verdict, DIM)
    extra = (f"strain {strain:.1f} kcal/mol" if strain is not None else "") + (f"  vina {vina}" if vina is not None else "")
    if console.is_terminal:
        console.print(f"  [{color}]{verdict:7s}[/] ({k}/{n}) {species[:46]}  [{DIM}]{extra}[/]")
    else:
        print(f"[{RUN_ID}]     validated {k}/{n} {verdict} {species[:46]} {extra}", flush=True)
