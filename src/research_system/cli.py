"""`research-system` command line interface.

Four commands: `ingest`, `demo`, `evaluate`, `benchmark`.

Expected failures (`ResearchSystemError`) print as a single actionable line and
exit 1. Anything else keeps its traceback, because it is a bug worth seeing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .config import VALID_STRATEGIES, get_settings
from .errors import ResearchSystemError

app = typer.Typer(
    name="research-system",
    help="Multi-agent research system: Planner -> Retriever -> Generator -> Verifier.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

EXIT_QUIT_WORDS = {"quit", "exit", "q"}


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=verbose)],
    )


def _fail(message: str) -> None:
    console.print(f"[bold red]Error:[/bold red] {message}")
    raise typer.Exit(code=1)


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Multi-agent research system."""
    _setup_logging(verbose)


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------
@app.command()
def ingest(
    source: Path = typer.Argument(..., help="Directory of PDF/Markdown/TXT/HTML files."),
    collection: str | None = typer.Option(
        None, "--collection", help="Qdrant collection name. Defaults to QDRANT_COLLECTION."
    ),
    chunk_size: int | None = typer.Option(
        None, "--chunk-size", help="Tokens per chunk. Defaults to CHUNK_SIZE (512)."
    ),
    chunk_overlap: int | None = typer.Option(
        None,
        "--chunk-overlap",
        help="Token overlap between chunks. Defaults to CHUNK_OVERLAP (50).",
    ),
) -> None:
    """Index a directory of documents into Qdrant.

    Re-running on unchanged files updates the same points instead of creating
    duplicates.
    """
    from .ingestion import ingest_directory

    settings = get_settings()
    try:
        with console.status("[cyan]Loading, chunking and embedding documents..."):
            report = ingest_directory(
                source,
                settings=settings,
                collection=collection,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
    except ResearchSystemError as exc:
        _fail(str(exc))
        return

    table = Table(title="Ingestion complete", show_header=False, box=None)
    table.add_row("Documents loaded", str(report.documents_loaded))
    table.add_row("Chunks created", str(report.chunks_created))
    table.add_row("Points indexed", str(report.points_indexed))
    table.add_row(
        "Collection", report.collection + (" (created)" if report.collection_created else "")
    )
    table.add_row("Total points in collection", str(report.total_points_in_collection))
    console.print(table)

    for warning in report.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    if report.skipped_files:
        console.print(f"[dim]Skipped {len(report.skipped_files)} unsupported file(s):[/dim]")
        for name in report.skipped_files[:10]:
            console.print(f"  [dim]- {name}[/dim]")
        if len(report.skipped_files) > 10:
            console.print(f"  [dim]... and {len(report.skipped_files) - 10} more[/dim]")


# --------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------
def _render_result(result: dict[str, Any]) -> None:
    """Print an answer plus its verification status."""
    if result.get("error"):
        console.print(f"[bold red]Error:[/bold red] {result['error']}")
    for warning in result.get("warnings") or []:
        console.print(f"[yellow]Warning:[/yellow] {warning}")

    response = result.get("response") or ""
    if response:
        console.print(Panel(Markdown(response), title="Answer", border_style="cyan"))

    # Three states, not two. A declined response is truthful, so it verifies --
    # but labelling it VERIFIED reads as "you got an answer", which is wrong.
    if not result.get("answered", True):
        status = "[bold yellow]NO ANSWER[/bold yellow]"
    elif result.get("is_verified", False):
        status = "[bold green]VERIFIED[/bold green]"
    else:
        status = "[bold yellow]UNVERIFIED[/bold yellow]"

    bits = [
        status,
        f"confidence {result.get('confidence', 0.0) * 100:.0f}%",
        f"retries {result.get('retry_count', 0)}",
        f"sources {len(result.get('sources') or [])}",
    ]
    verification = result.get("verification")
    if verification:
        bits.append(f"faithfulness {verification['faithfulness_score'] * 100:.0f}%")
        bits.append(
            f"claims {verification['supported_claims']}/{verification['total_claims']} supported"
        )
    if result.get("elapsed_s") is not None:
        bits.append(f"{result['elapsed_s']:.1f}s")
    console.print("  ".join(bits))

    # Make the retrieved-but-unused evidence visible, so "sources 0" does not
    # look like retrieval never ran.
    if not result.get("answered", True):
        retrieved = result.get("documents_retrieved", 0)
        if retrieved:
            console.print(
                f"[dim]{retrieved} document(s) retrieved, none relevant to this question[/dim]"
            )
        else:
            console.print("[dim]no documents were retrieved[/dim]")

    sources = result.get("sources") or []
    if sources:
        table = Table(title="Sources", show_lines=False)
        table.add_column("#", justify="right", style="dim", width=3)
        table.add_column("Source", overflow="fold")
        table.add_column("Score", justify="right", width=8)
        for entry in sources:
            label = entry.get("title") or entry.get("source", "")
            if entry.get("title"):
                label = f"{entry['title']}\n[dim]{entry.get('source', '')}[/dim]"
            table.add_row(str(entry.get("index", "")), str(label), f"{entry.get('score', 0.0):.4f}")
        console.print(table)


@app.command()
def demo(
    user_id: str = typer.Option("demo_user", "--user-id", help="Identity used to scope memory."),
    no_memory: bool = typer.Option(False, "--no-memory", help="Disable conversational memory."),
) -> None:
    """Interactive research session."""
    from .core.pipeline import ResearchPipeline
    from .tracing import configure_tracing

    settings = get_settings()
    configure_tracing(settings)
    use_memory = not no_memory

    console.print(
        Panel.fit(
            "[bold cyan]Multi-Agent Research System[/bold cyan]\n"
            "Planner -> Retriever -> Generator -> Verifier\n\n"
            f"user: [green]{user_id}[/green]   "
            f"memory: [green]{'on' if use_memory else 'off'}[/green]   "
            f"model: [green]{settings.default_llm}[/green]\n"
            "[dim]Type 'quit', 'exit' or 'q' to leave.[/dim]",
            border_style="cyan",
        )
    )

    try:
        pipeline = ResearchPipeline(use_memory=use_memory)
    except ResearchSystemError as exc:
        _fail(str(exc))
        return

    if use_memory and not pipeline.memory.enabled:
        console.print("[yellow]Warning:[/yellow] memory is unavailable; running without it.")

    chat_history: list[dict[str, str]] = []

    while True:
        try:
            question = console.input("\n[bold cyan]>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not question:
            continue
        if question.lower() in EXIT_QUIT_WORDS:
            console.print("[dim]Goodbye.[/dim]")
            break

        try:
            with console.status("[cyan]Researching..."):
                result = pipeline.run(
                    question,
                    user_id=user_id,
                    chat_history=chat_history,
                    use_memory=use_memory,
                )
        except ResearchSystemError as exc:
            # Configuration and provider problems are recoverable: the operator
            # may fix .env and ask again without restarting.
            console.print(f"[bold red]Error:[/bold red] {exc}")
            continue
        except KeyboardInterrupt:
            console.print("[dim]Cancelled.[/dim]")
            continue

        _render_result(result)

        chat_history.append({"role": "user", "content": question})
        if result.get("response"):
            chat_history.append({"role": "assistant", "content": result["response"]})


# --------------------------------------------------------------------------
# evaluate / benchmark
# --------------------------------------------------------------------------
def _print_eval_report(report: dict[str, Any]) -> None:
    summary = report.get("summary", {})
    table = Table(title="Pipeline summary", show_header=False, box=None)
    for label, key, suffix in (
        ("Questions", "total_questions", ""),
        ("Failed", "failed_questions", ""),
        ("Avg latency", "avg_latency_s", "s"),
        ("p95 latency", "p95_latency_s", "s"),
        ("Retry rate", "retry_rate", ""),
        ("Verification pass rate", "verification_pass_rate", ""),
    ):
        table.add_row(label, f"{summary.get(key, 0)}{suffix}")
    console.print(table)

    scores = report.get("ragas_scores", {})
    if "skipped" in scores:
        console.print(f"[yellow]Ragas skipped:[/yellow] {scores['skipped']}")
    else:
        metric_table = Table(title="Ragas scores")
        metric_table.add_column("Metric")
        metric_table.add_column("Score", justify="right")
        for name, value in scores.items():
            if isinstance(value, (int, float)):
                metric_table.add_row(name, f"{value:.4f}")
        console.print(metric_table)

    comparison = report.get("comparison")
    if comparison and comparison.get("metrics"):
        compare_table = Table(title="Multi-agent vs single-agent baseline")
        compare_table.add_column("Metric")
        compare_table.add_column("Multi", justify="right")
        compare_table.add_column("Single", justify="right")
        compare_table.add_column("Delta", justify="right")
        for name, values in comparison["metrics"].items():
            delta = values["delta"]
            colour = "green" if delta > 0 else ("red" if delta < 0 else "white")
            compare_table.add_row(
                name,
                f"{values['multi_agent']:.4f}",
                f"{values['single_agent']:.4f}",
                f"[{colour}]{delta:+.4f}[/{colour}]",
            )
        console.print(compare_table)
        console.print(f"[dim]{comparison['caveat']}[/dim]")


@app.command()
def evaluate(
    dataset: Path = typer.Argument(..., help="JSON array of {question, ground_truth} objects."),
    baseline: bool = typer.Option(
        False, "--baseline", help="Also run the single-agent baseline and report deltas."
    ),
    baseline_strategy: str = typer.Option(
        "hybrid", "--baseline-strategy", help="Retrieval strategy for the baseline."
    ),
    output: Path = typer.Option(
        Path("eval/results/evaluation.json"), "--output", "-o", help="Where to write results JSON."
    ),
) -> None:
    """Score the pipeline on a dataset with Ragas."""
    from .evaluation.evaluate import evaluate_dataset
    from .tracing import configure_tracing

    if baseline_strategy not in VALID_STRATEGIES:
        _fail(f"--baseline-strategy must be one of {', '.join(VALID_STRATEGIES)}.")

    settings = get_settings()
    configure_tracing(settings)

    try:
        report = evaluate_dataset(
            dataset,
            settings=settings,
            include_baseline=baseline,
            baseline_strategy=baseline_strategy,
            output_path=output,
        )
    except ResearchSystemError as exc:
        _fail(str(exc))
        return

    _print_eval_report(report)
    console.print(f"\n[green]Results written to[/green] {output}")


@app.command()
def benchmark(
    dataset: Path = typer.Argument(..., help="JSON array of {question, ground_truth} objects."),
    strategy: str = typer.Option(
        "hybrid", "--strategy", help="Retrieval strategy for the single-agent baseline."
    ),
    output: Path = typer.Option(
        Path("eval/results/benchmark.json"), "--output", "-o", help="Where to write results JSON."
    ),
) -> None:
    """Compare the multi-agent pipeline against the single-agent baseline."""
    from .evaluation.evaluate import evaluate_dataset
    from .tracing import configure_tracing

    if strategy not in VALID_STRATEGIES:
        _fail(f"--strategy must be one of {', '.join(VALID_STRATEGIES)}.")

    settings = get_settings()
    configure_tracing(settings)

    try:
        report = evaluate_dataset(
            dataset,
            settings=settings,
            include_baseline=True,
            baseline_strategy=strategy,
            output_path=output,
        )
    except ResearchSystemError as exc:
        _fail(str(exc))
        return

    _print_eval_report(report)
    console.print(f"\n[green]Results written to[/green] {output}")


@app.command()
def config() -> None:
    """Print the resolved configuration with secrets redacted."""
    settings = get_settings()
    table = Table(title="Configuration", show_header=True)
    table.add_column("Setting")
    table.add_column("Value", overflow="fold")
    for key, value in sorted(settings.redacted().items()):
        table.add_row(key, str(value))
    console.print(table)

    console.print("\n[bold]Backend availability[/bold]")
    for label, ok in (
        ("Document search (Qdrant + embeddings)", settings.vector_available),
        ("Web search (Tavily)", settings.web_available),
        ("Tracing (LangSmith)", settings.tracing_enabled),
    ):
        mark = "[green]ready[/green]" if ok else "[yellow]not configured[/yellow]"
        console.print(f"  {label}: {mark}")


def run() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    run()
