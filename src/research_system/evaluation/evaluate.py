"""Ragas evaluation for the full pipeline and the single-agent baseline.

Records environment metadata (versions, model names, dataset hash) alongside
every score, because a Ragas number is meaningless without the corpus, model
and dataset that produced it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from ..core.deps import Dependencies, build_dependencies
from ..core.pipeline import ResearchPipeline
from ..errors import EvaluationError
from .benchmark import SingleAgentBaseline

logger = logging.getLogger(__name__)

RAGAS_METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

_TRACKED_PACKAGES = (
    "langgraph",
    "langchain-core",
    "litellm",
    "qdrant-client",
    "llama-index-core",
    "openai",
    "ragas",
    "tavily-python",
)


@dataclass
class EvalItem:
    """One dataset row."""

    question: str
    ground_truth: str = ""


@dataclass
class RunRecord:
    """What one question produced."""

    question: str
    ground_truth: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    retry_count: int = 0
    is_verified: bool = False
    faithfulness_score: float | None = None
    num_sources: int = 0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


def load_dataset(path: str | Path) -> list[EvalItem]:
    """Read and validate a JSON evaluation dataset."""
    dataset_path = Path(path).expanduser()
    if not dataset_path.exists():
        raise EvaluationError(f"Dataset not found: {dataset_path}")

    try:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"Dataset {dataset_path} is not valid JSON: {exc}") from exc

    if not isinstance(raw, list):
        raise EvaluationError(
            f"Dataset {dataset_path} must be a JSON array of "
            '{"question": ..., "ground_truth": ...} objects.'
        )

    items: list[EvalItem] = []
    for position, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise EvaluationError(f"Dataset entry {position} is not an object.")
        question = str(entry.get("question") or "").strip()
        if not question:
            raise EvaluationError(f"Dataset entry {position} has an empty 'question'.")
        items.append(
            EvalItem(question=question, ground_truth=str(entry.get("ground_truth") or "").strip())
        )
    return items


def dataset_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def environment_metadata(settings: Settings, dataset_path: str | Path | None) -> dict[str, Any]:
    """Everything needed to reproduce a score."""
    meta: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependency_versions": {name: _package_version(name) for name in _TRACKED_PACKAGES},
        "model": settings.default_llm,
        "embedding_model": settings.embedding_model,
        "qdrant_collection": settings.qdrant_collection,
        "faithfulness_threshold": settings.faithfulness_threshold,
        "max_verification_retries": settings.max_verification_retries,
        "max_retrieval_docs": settings.max_retrieval_docs,
    }
    if dataset_path is not None:
        meta["dataset_path"] = str(dataset_path)
        meta["dataset_sha256"] = dataset_sha256(dataset_path)
    return meta


def _run_one(runner: Callable[[str], dict[str, Any]], item: EvalItem) -> RunRecord:
    started = time.perf_counter()
    try:
        result = runner(item.question)
    except Exception as exc:
        logger.exception("question failed: %s", item.question)
        return RunRecord(
            question=item.question,
            ground_truth=item.ground_truth,
            answer="",
            latency_s=round(time.perf_counter() - started, 3),
            error=f"{type(exc).__name__}: {exc}",
        )

    verification = result.get("verification") or {}
    return RunRecord(
        question=item.question,
        ground_truth=item.ground_truth,
        answer=result.get("response", ""),
        contexts=list(result.get("_contexts") or []),
        latency_s=float(result.get("elapsed_s") or round(time.perf_counter() - started, 3)),
        retry_count=int(result.get("retry_count", 0)),
        is_verified=bool(result.get("is_verified", False)),
        faithfulness_score=verification.get("faithfulness_score"),
        num_sources=len(result.get("sources") or []),
        error=result.get("error"),
        warnings=list(result.get("warnings") or []),
    )


def collect_records(
    items: list[EvalItem], runner: Callable[[str], dict[str, Any]]
) -> list[RunRecord]:
    records = []
    for position, item in enumerate(items, start=1):
        logger.info("[%d/%d] %s", position, len(items), item.question[:80])
        records.append(_run_one(runner, item))
    return records


def summarize(records: list[RunRecord]) -> dict[str, Any]:
    """Aggregate timings and pass rates, safe on an empty dataset."""
    total = len(records)
    if total == 0:
        return {
            "avg_latency_s": 0.0,
            "p95_latency_s": 0.0,
            "retry_rate": 0.0,
            "verification_pass_rate": 0.0,
            "total_questions": 0,
            "failed_questions": 0,
        }

    latencies = sorted(record.latency_s for record in records)
    # Nearest-rank p95; with few samples this is the slowest run, which is honest.
    index = min(len(latencies) - 1, max(0, round(0.95 * len(latencies)) - 1))

    return {
        "avg_latency_s": round(statistics.fmean(latencies), 3),
        "p95_latency_s": round(latencies[index], 3),
        "retry_rate": round(sum(1 for r in records if r.retry_count > 0) / total, 4),
        "verification_pass_rate": round(sum(1 for r in records if r.is_verified) / total, 4),
        "total_questions": total,
        "failed_questions": sum(1 for r in records if r.error),
    }


def run_ragas(records: list[RunRecord]) -> dict[str, Any]:
    """Score records with Ragas.

    Returns a dict of metric -> score, or `{"skipped": reason}`. Rows without an
    answer are excluded; metrics needing a reference are dropped rather than
    silently scored against an empty string.
    """
    scorable = [r for r in records if r.answer.strip() and r.contexts]
    if not scorable:
        return {"skipped": "no question produced both an answer and retrieved context"}

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        return {
            "skipped": f"evaluation extras not installed ({exc}); "
            "install with: pip install 'research-system[eval]'"
        }

    metrics = [faithfulness, answer_relevancy]
    have_references = all(r.ground_truth.strip() for r in scorable)
    if have_references:
        metrics.extend([context_precision, context_recall])
    else:
        logger.warning(
            "some rows lack 'ground_truth'; skipping context_precision and context_recall"
        )

    payload = {
        "question": [r.question for r in scorable],
        "user_input": [r.question for r in scorable],
        "answer": [r.answer for r in scorable],
        "response": [r.answer for r in scorable],
        "contexts": [r.contexts for r in scorable],
        "retrieved_contexts": [r.contexts for r in scorable],
        "ground_truth": [r.ground_truth for r in scorable],
        "reference": [r.ground_truth for r in scorable],
    }

    try:
        dataset = Dataset.from_dict(payload)
        result = evaluate(dataset, metrics=metrics)
    except Exception as exc:
        return {"skipped": f"Ragas run failed: {type(exc).__name__}: {exc}"}

    scores: dict[str, Any] = {"scored_questions": len(scorable)}
    if not have_references:
        scores["omitted_metrics"] = ["context_precision", "context_recall"]

    try:
        raw = result.to_pandas().mean(numeric_only=True).to_dict()
    except Exception:
        raw = dict(result) if hasattr(result, "keys") else {}

    for name in RAGAS_METRIC_NAMES:
        if name in raw:
            try:
                scores[name] = round(float(raw[name]), 4)
            except (TypeError, ValueError):
                continue
    return scores


def evaluate_dataset(
    dataset_path: str | Path,
    *,
    settings: Settings | None = None,
    deps: Dependencies | None = None,
    include_baseline: bool = False,
    baseline_strategy: str = "hybrid",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the pipeline (and optionally the baseline) and score both."""
    settings = settings or get_settings()
    items = load_dataset(dataset_path)
    deps = deps or build_dependencies(settings)

    report: dict[str, Any] = {"environment": environment_metadata(settings, dataset_path)}

    if not items:
        report.update(
            {
                "ragas_scores": {"skipped": "dataset is empty"},
                "timings": [],
                "summary": summarize([]),
            }
        )
        return _maybe_write(report, output_path)

    pipeline = ResearchPipeline(deps)
    records = collect_records(items, lambda q: pipeline.run(q, use_memory=False))

    report["ragas_scores"] = run_ragas(records)
    report["timings"] = [asdict(record) for record in records]
    report["summary"] = summarize(records)

    if include_baseline:
        logger.info("running single-agent baseline on the same %d questions", len(items))
        baseline = SingleAgentBaseline(deps, strategy=baseline_strategy)
        baseline_records = collect_records(items, baseline.run)
        report["baseline"] = {
            "description": (
                "Single agent: no planner decomposition, no verification retry loop. "
                "Same model, temperature, generation rules and retrieval backends."
            ),
            "strategy": baseline_strategy,
            "ragas_scores": run_ragas(baseline_records),
            "summary": summarize(baseline_records),
            "timings": [asdict(record) for record in baseline_records],
        }
        report["comparison"] = _compare(
            report["ragas_scores"], report["baseline"]["ragas_scores"], len(items)
        )

    return _maybe_write(report, output_path)


def _compare(multi: dict[str, Any], baseline: dict[str, Any], sample_size: int) -> dict[str, Any]:
    """Absolute scores and deltas.

    No confidence intervals and no percentage headline: with a handful of
    questions the difference is not statistically meaningful, and the sample
    size is reported so nobody mistakes a delta for a result.
    """
    deltas: dict[str, Any] = {}
    for name in RAGAS_METRIC_NAMES:
        if name in multi and name in baseline:
            deltas[name] = {
                "multi_agent": multi[name],
                "single_agent": baseline[name],
                "delta": round(float(multi[name]) - float(baseline[name]), 4),
            }
    return {
        "sample_size": sample_size,
        "metrics": deltas,
        "caveat": (
            f"Deltas come from {sample_size} question(s) on a single run with no "
            "repeats. No confidence interval is reported and no significance is "
            "claimed. Treat these as directional only."
        ),
    }


def _maybe_write(report: dict[str, Any], output_path: str | Path | None) -> dict[str, Any]:
    if output_path is None:
        return report
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("wrote results to %s", destination)
    return report
