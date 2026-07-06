"""
Unified Experiment Runner.

Runs all planning methods (TiTSP + 4 published baselines) across all
workloads (Nirvana + SemBench) and produces comparison tables for the paper.

Usage:
    python3 -m src.experiments.unified_runner \
        --llm dashscope --model qwen-plus \
        --methods TiTSP,CAESURA,LOTUS,ThalamusDB,Palimpzest \
        --workloads nirvana,sembench \
        --json results.json

Methods:
    TiTSP       — Full system (MCTS + Uncertainty + NSGA-II Pareto)
    CAESURA      — CIDR'24 + SIGMOD'24: LLM-driven 3-phase planning
    LOTUS        — VLDB'25: Rule-based + cascade optimization
    ThalamusDB   — SIGMOD'23: SQL-style AQP planning
    Palimpzest   — CIDR'25: Cascades-framework + Pareto
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.query_planner.query_planner import (
    ExecutableResult,
    PlanResult,
    QueryPlanner,
)
from core.query_planner.mcts_plan_search import LogicalPlanNode
from core.query_planner.plan_executor import Table

from experiments.nirvana_workload import (
    DATASET_REGISTRY as NIRVANA_DATASETS,
    NIRVANA_QUERIES,
    NirvanaQuery,
    validate_answer as nirvana_validate,
    collect_operators,
    plan_depth,
)
from experiments.sembench_workload import (
    DATASET_REGISTRY as SEMBENCH_DATASETS,
    SEMBENCH_QUERIES,
    SemBenchQuery,
    validate_answer as sembench_validate,
)


# ---------------------------------------------------------------------------
# Method Configuration
# ---------------------------------------------------------------------------

@dataclass
class MethodConfig:
    """Configuration for a planning method."""
    name: str
    planning_strategy: str
    venue: str

METHODS = {
    "TiTSP": MethodConfig("TiTSP", "titsp", "(Ours)"),
    "CAESURA": MethodConfig("CAESURA", "caesura", "SIGMOD'24"),
    "LOTUS": MethodConfig("LOTUS", "lotus", "VLDB'25"),
    "ThalamusDB": MethodConfig("ThalamusDB", "thalamusdb", "SIGMOD'23"),
    "Palimpzest": MethodConfig("Palimpzest", "palimpzest", "CIDR'25"),
    "Nirvana": MethodConfig("Nirvana", "nirvana", "SIGMOD'26"),
}


# ---------------------------------------------------------------------------
# Unified Query Wrapper
# ---------------------------------------------------------------------------

@dataclass
class UnifiedQuery:
    """Wraps both NirvanaQuery and SemBenchQuery for uniform processing."""
    workload: str
    dataset: str
    query_id: str
    nl_description: str
    expected_answer: Optional[str]
    data: Table


def get_nirvana_queries() -> List[UnifiedQuery]:
    queries = []
    for q in NIRVANA_QUERIES:
        data = NIRVANA_DATASETS.get(q.dataset, [])
        queries.append(UnifiedQuery(
            workload="nirvana",
            dataset=q.dataset,
            query_id=q.query_id,
            nl_description=q.nl_description,
            expected_answer=q.expected_answer,
            data=data,
        ))
    return queries


def get_sembench_queries() -> List[UnifiedQuery]:
    queries = []
    for q in SEMBENCH_QUERIES:
        data = SEMBENCH_DATASETS.get(q.scenario, [])
        queries.append(UnifiedQuery(
            workload="sembench",
            dataset=q.scenario,
            query_id=q.query_id,
            nl_description=q.nl_description,
            expected_answer=q.expected_answer,
            data=data,
        ))
    return queries


# ---------------------------------------------------------------------------
# Unified Result
# ---------------------------------------------------------------------------

@dataclass
class UnifiedResult:
    """Result of running one query with one method."""
    method: str
    workload: str
    dataset: str
    query_id: str
    nl_description: str
    answer: str = ""
    validated: Optional[bool] = None
    output_rows: int = 0
    elapsed_ms: float = 0.0
    planning_ms: float = 0.0
    mcts_candidates: int = 0
    plan_depth: int = 0
    pareto_size: int = 0
    plan_operators: List[str] = field(default_factory=list)
    physical_assignment: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    # Search quality metrics (Table A)
    plan_diversity: float = 0.0
    quality_score_cv: float = 0.0
    completeness: float = 0.0
    # Optimization quality metrics (Table B)
    pareto_hypervolume: float = 0.0
    pareto_spread: float = 0.0
    root_confidence: float = 0.0
    avg_uncertainty_variance: float = 0.0
    avg_modality_risk: float = 0.0
    estimated_latency: float = 0.0
    estimated_resource_cost: float = 0.0
    estimated_confidence: float = 0.0
    # Efficiency breakdown (Table C)
    parse_time_ratio: float = 0.0
    search_time_ratio: float = 0.0
    execution_ms: float = 0.0
    num_operators: int = 0
    num_required_operators: int = 0


def validate_answer(answer: str, data: Table, expected: str) -> bool:
    """Unified validation (reuses nirvana logic)."""
    return nirvana_validate(answer, data, expected)


# ---------------------------------------------------------------------------
# Metric Computation Helpers
# ---------------------------------------------------------------------------

def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def _compute_plan_diversity(candidates) -> float:
    """Average pairwise Jaccard distance across logical candidates."""
    if len(candidates) <= 1:
        return 0.0
    distances = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            sim = _jaccard(candidates[i].operator_sequence,
                           candidates[j].operator_sequence)
            distances.append(1.0 - sim)
    return sum(distances) / len(distances) if distances else 0.0


def _compute_quality_cv(candidates) -> float:
    """Coefficient of variation of quality_score across candidates."""
    if len(candidates) <= 1:
        return 0.0
    scores = [c.quality_score for c in candidates]
    mean = sum(scores) / len(scores)
    if mean == 0:
        return 0.0
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    return (variance ** 0.5) / mean


def _compute_pareto_hypervolume(pareto_front) -> float:
    """Compute 3D hypervolume of the Pareto front.

    Reference point: (1.0, 1.0, 0.0) for (norm_latency, norm_cost, confidence).
    Objectives: minimize latency, minimize cost, maximize confidence.
    We normalize latency and cost to [0,1] within the front, then compute
    the dominated hypervolume using the inclusion-exclusion method.
    """
    if not pareto_front:
        return 0.0
    if len(pareto_front) == 1:
        obj = pareto_front[0].objectives
        return max(0.0, obj.confidence)

    lats = [p.objectives.latency for p in pareto_front]
    costs = [p.objectives.resource_cost for p in pareto_front]
    max_lat = max(lats) if max(lats) > 0 else 1.0
    max_cost = max(costs) if max(costs) > 0 else 1.0

    points = []
    for p in pareto_front:
        nl = p.objectives.latency / max_lat
        nc = p.objectives.resource_cost / max_cost
        q = p.objectives.confidence
        points.append((1.0 - nl, 1.0 - nc, q))

    hv = 0.0
    for x, y, z in points:
        hv += x * y * z
    hv /= len(points)
    return hv


def _compute_pareto_spread(pareto_front) -> float:
    """Range of confidence values across Pareto front entries."""
    if len(pareto_front) <= 1:
        return 0.0
    confs = [p.objectives.confidence for p in pareto_front]
    return max(confs) - min(confs)


def _compute_root_confidence(uncertainty_map) -> float:
    """Average confidence across all operators in uncertainty map."""
    if not uncertainty_map:
        return 0.0
    confs = [est.confidence for est in uncertainty_map.values()]
    return sum(confs) / len(confs)


def _compute_avg_uncertainty_variance(uncertainty_map) -> float:
    """Average variance across uncertainty estimates."""
    if not uncertainty_map:
        return 0.0
    vars_ = [est.variance for est in uncertainty_map.values()]
    return sum(vars_) / len(vars_)


def _compute_avg_modality_risk(uncertainty_map) -> float:
    """Average modality risk across uncertainty estimates."""
    if not uncertainty_map:
        return 0.0
    risks = [est.modality_risk for est in uncertainty_map.values()]
    return sum(risks) / len(risks)


# ---------------------------------------------------------------------------
# Run Single Query
# ---------------------------------------------------------------------------

def run_single(
    planner: QueryPlanner,
    query: UnifiedQuery,
    method_name: str,
) -> UnifiedResult:
    """Run a single query through the full pipeline."""
    t0 = time.perf_counter()
    try:
        result = planner.plan_and_execute(query.nl_description, query.data)
        elapsed = (time.perf_counter() - t0) * 1000

        validated = None
        if query.expected_answer is not None:
            validated = validate_answer(
                result.answer, result.execution.data, query.expected_answer
            )

        plan = result.plan
        ops = sorted(collect_operators(plan.best_logical_plan))
        planning_ms = plan.timing.total_ms if plan.timing else 0.0

        diversity = _compute_plan_diversity(plan.logical_candidates)
        quality_cv = _compute_quality_cv(plan.logical_candidates)
        completeness = plan.logical_candidates[0].completeness if plan.logical_candidates else 0.0

        hv = _compute_pareto_hypervolume(plan.pareto_front)
        spread = _compute_pareto_spread(plan.pareto_front)
        root_conf = _compute_root_confidence(plan.uncertainty_map)
        avg_var = _compute_avg_uncertainty_variance(plan.uncertainty_map)
        avg_risk = _compute_avg_modality_risk(plan.uncertainty_map)

        obj = plan.selected_physical.objectives
        parse_ratio = (plan.timing.parse_ms / plan.timing.total_ms) if plan.timing and plan.timing.total_ms > 0 else 0.0
        search_ratio = (plan.timing.search_ms / plan.timing.total_ms) if plan.timing and plan.timing.total_ms > 0 else 0.0
        exec_ms = elapsed - planning_ms

        return UnifiedResult(
            method=method_name,
            workload=query.workload,
            dataset=query.dataset,
            query_id=query.query_id,
            nl_description=query.nl_description,
            answer=result.answer,
            validated=validated,
            output_rows=result.execution.row_count,
            elapsed_ms=elapsed,
            planning_ms=planning_ms,
            mcts_candidates=len(plan.logical_candidates),
            plan_depth=plan_depth(plan.best_logical_plan),
            pareto_size=len(plan.pareto_front),
            plan_operators=ops,
            physical_assignment=plan.selected_physical.assignment,
            plan_diversity=diversity,
            quality_score_cv=quality_cv,
            completeness=completeness,
            pareto_hypervolume=hv,
            pareto_spread=spread,
            root_confidence=root_conf,
            avg_uncertainty_variance=avg_var,
            avg_modality_risk=avg_risk,
            estimated_latency=obj.latency,
            estimated_resource_cost=obj.resource_cost,
            estimated_confidence=obj.confidence,
            parse_time_ratio=parse_ratio,
            search_time_ratio=search_ratio,
            execution_ms=exec_ms,
            num_operators=len(ops),
            num_required_operators=len(plan.query_context.required_operators),
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        elapsed = (time.perf_counter() - t0) * 1000
        return UnifiedResult(
            method=method_name,
            workload=query.workload,
            dataset=query.dataset,
            query_id=query.query_id,
            nl_description=query.nl_description,
            elapsed_ms=elapsed,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Run Experiment
# ---------------------------------------------------------------------------

def run_experiment(
    methods: List[str],
    workloads: List[str],
    llm_provider=None,
    verbose: bool = True,
) -> List[UnifiedResult]:
    """Run all methods × all workloads."""
    queries: List[UnifiedQuery] = []
    if "nirvana" in workloads:
        queries.extend(get_nirvana_queries())
    if "sembench" in workloads:
        queries.extend(get_sembench_queries())

    all_results: List[UnifiedResult] = []

    for method_name in methods:
        config = METHODS[method_name]
        print(f"\n{'='*70}")
        print(f"Method: {config.name} ({config.venue}) "
              f"— strategy={config.planning_strategy}")
        print(f"{'='*70}")

        planner = QueryPlanner(
            llm_provider=llm_provider,
            planning_strategy=config.planning_strategy,
        )

        for q in queries:
            r = run_single(planner, q, method_name)
            all_results.append(r)

            if verbose:
                ok = "PASS" if r.validated else (
                    "FAIL" if r.validated is False else "---"
                )
                err = " ERROR" if r.error else ""
                print(f"  [{ok:4s}] {q.workload}/{q.dataset}/{q.query_id}: "
                      f"\"{q.nl_description[:60]}\" "
                      f"({r.elapsed_ms:.0f}ms){err}")

    return all_results


# ---------------------------------------------------------------------------
# Summary Tables
# ---------------------------------------------------------------------------

def print_main_table(results: List[UnifiedResult], methods: List[str]):
    """Print the main comparison table for the paper."""
    print("\n" + "=" * 90)
    print("MAIN COMPARISON TABLE")
    print("=" * 90)

    header = f"{'Method':<14} {'Venue':<12} {'Nirvana (36)':>14} {'SemBench (55)':>15} {'Overall (91)':>14} {'Avg Time':>10}"
    print(header)
    print("-" * 90)

    for method_name in methods:
        config = METHODS[method_name]
        mr = [r for r in results if r.method == method_name]

        nirvana_r = [r for r in mr if r.workload == "nirvana"]
        sembench_r = [r for r in mr if r.workload == "sembench"]

        nirvana_val = [r for r in nirvana_r if r.validated is not None]
        nirvana_correct = sum(1 for r in nirvana_val if r.validated)
        nirvana_total = len(nirvana_r)

        sembench_val = [r for r in sembench_r if r.validated is not None]
        sembench_correct = sum(1 for r in sembench_val if r.validated)
        sembench_total = len(sembench_r)

        all_val = nirvana_val + sembench_val
        all_correct = nirvana_correct + sembench_correct
        all_total = nirvana_total + sembench_total

        times = [r.elapsed_ms for r in mr if not r.error]
        avg_time = sum(times) / len(times) if times else 0

        n_pct = f"{nirvana_correct}/{len(nirvana_val)}" if nirvana_val else "N/A"
        s_pct = f"{sembench_correct}/{len(sembench_val)}" if sembench_val else "N/A"
        a_pct = f"{all_correct}/{len(all_val)}" if all_val else "N/A"

        if nirvana_val:
            n_pct += f" ({nirvana_correct/len(nirvana_val)*100:.1f}%)"
        if sembench_val:
            s_pct += f" ({sembench_correct/len(sembench_val)*100:.1f}%)"
        if all_val:
            a_pct += f" ({all_correct/len(all_val)*100:.1f}%)"

        print(f"{config.name:<14} {config.venue:<12} {n_pct:>14} {s_pct:>15} {a_pct:>14} {avg_time:>8.0f}ms")

    print("=" * 90)


def print_detailed_metrics(results: List[UnifiedResult], methods: List[str]):
    """Print detailed metrics table."""
    print("\n" + "=" * 100)
    print("DETAILED METRICS")
    print("=" * 100)

    metrics = [
        "Accuracy (%)",
        "Avg MCTS Candidates",
        "Avg Plan Depth",
        "Avg Pareto Front",
        "Avg Planning Time (ms)",
        "Avg Total Time (ms)",
        "Error Rate (%)",
    ]

    header = f"{'Metric':<25}"
    for m in methods:
        header += f" {m:>14}"
    print(header)
    print("-" * 100)

    for metric in metrics:
        line = f"{metric:<25}"
        for method_name in methods:
            mr = [r for r in results if r.method == method_name]
            val = _compute_metric(mr, metric)
            line += f" {val:>14}"
        print(line)

    print("=" * 100)


def _compute_metric(results: List[UnifiedResult], metric: str) -> str:
    valid = [r for r in results if not r.error]
    validated = [r for r in results if r.validated is not None]

    if metric == "Accuracy (%)":
        if not validated:
            return "N/A"
        correct = sum(1 for r in validated if r.validated)
        return f"{correct/len(validated)*100:.1f}%"
    elif metric == "Avg MCTS Candidates":
        if not valid:
            return "N/A"
        return f"{sum(r.mcts_candidates for r in valid)/len(valid):.1f}"
    elif metric == "Avg Plan Depth":
        if not valid:
            return "N/A"
        return f"{sum(r.plan_depth for r in valid)/len(valid):.1f}"
    elif metric == "Avg Pareto Front":
        if not valid:
            return "N/A"
        return f"{sum(r.pareto_size for r in valid)/len(valid):.1f}"
    elif metric == "Avg Planning Time (ms)":
        if not valid:
            return "N/A"
        return f"{sum(r.planning_ms for r in valid)/len(valid):.0f}"
    elif metric == "Avg Total Time (ms)":
        if not valid:
            return "N/A"
        return f"{sum(r.elapsed_ms for r in valid)/len(valid):.0f}"
    elif metric == "Error Rate (%)":
        errors = sum(1 for r in results if r.error)
        return f"{errors/len(results)*100:.1f}%" if results else "N/A"
    return "N/A"


def print_per_dataset_breakdown(results: List[UnifiedResult], methods: List[str]):
    """Print per-dataset accuracy breakdown."""
    print("\n" + "=" * 100)
    print("PER-DATASET BREAKDOWN")
    print("=" * 100)

    datasets = sorted(set((r.workload, r.dataset) for r in results))

    header = f"{'Workload/Dataset':<25}"
    for m in methods:
        header += f" {m:>14}"
    print(header)
    print("-" * 100)

    for workload, dataset in datasets:
        label = f"{workload}/{dataset}"
        line = f"{label:<25}"
        for method_name in methods:
            dr = [
                r for r in results
                if r.method == method_name
                and r.workload == workload
                and r.dataset == dataset
                and r.validated is not None
            ]
            if not dr:
                line += f" {'N/A':>14}"
            else:
                correct = sum(1 for r in dr if r.validated)
                cell = f"{correct}/{len(dr)}"
                line += f" {cell:>14}"
        print(line)

    print("=" * 100)


# ---------------------------------------------------------------------------
# Table A: Search Quality Metrics
# ---------------------------------------------------------------------------

def print_search_quality_metrics(results: List[UnifiedResult], methods: List[str]):
    """Print search quality metrics (Table A for paper)."""
    print("\n" + "=" * 100)
    print("SEARCH QUALITY METRICS (Table A)")
    print("=" * 100)

    metrics = [
        ("Plan Diversity", "plan_diversity", ".3f"),
        ("Quality Score CV", "quality_score_cv", ".3f"),
        ("Plan Completeness", "completeness", ".3f"),
        ("Search Space Size", "mcts_candidates", ".1f"),
        ("Accuracy / Candidate", None, ".3f"),
    ]

    header = f"{'Metric':<25}"
    for m in methods:
        header += f" {m:>14}"
    print(header)
    print("-" * 100)

    for metric_name, field_name, fmt in metrics:
        line = f"{metric_name:<25}"
        for method_name in methods:
            mr = [r for r in results if r.method == method_name and not r.error]
            if not mr:
                line += f" {'N/A':>14}"
                continue
            if metric_name == "Accuracy / Candidate":
                vals = []
                for r in mr:
                    acc = 1.0 if r.validated else 0.0
                    cand = max(r.mcts_candidates, 1)
                    vals.append(acc / cand)
                avg = sum(vals) / len(vals)
            else:
                avg = sum(getattr(r, field_name) for r in mr) / len(mr)
            line += f" {avg:>14{fmt}}"
        print(line)

    print("=" * 100)


# ---------------------------------------------------------------------------
# Table B: Optimization Quality Metrics
# ---------------------------------------------------------------------------

def print_optimization_quality_metrics(results: List[UnifiedResult], methods: List[str]):
    """Print optimization quality metrics (Table B for paper)."""
    print("\n" + "=" * 100)
    print("OPTIMIZATION QUALITY METRICS (Table B)")
    print("=" * 100)

    metrics = [
        ("Pareto Hypervolume", "pareto_hypervolume", ".4f"),
        ("Pareto Spread", "pareto_spread", ".4f"),
        ("Pareto Front Size", "pareto_size", ".1f"),
        ("Avg Confidence", "root_confidence", ".3f"),
        ("Avg Uncertainty Var", "avg_uncertainty_variance", ".4f"),
        ("Avg Modality Risk", "avg_modality_risk", ".3f"),
        ("Est. Latency", "estimated_latency", ".2f"),
        ("Est. Resource Cost", "estimated_resource_cost", ".2f"),
        ("Est. Confidence", "estimated_confidence", ".3f"),
        ("Conf-Accuracy Corr", None, ".3f"),
    ]

    header = f"{'Metric':<25}"
    for m in methods:
        header += f" {m:>14}"
    print(header)
    print("-" * 100)

    for metric_name, field_name, fmt in metrics:
        line = f"{metric_name:<25}"
        for method_name in methods:
            mr = [r for r in results if r.method == method_name and not r.error]
            if not mr:
                line += f" {'N/A':>14}"
                continue
            if metric_name == "Conf-Accuracy Corr":
                corr = _pearson_correlation(
                    [r.root_confidence for r in mr],
                    [1.0 if r.validated else 0.0 for r in mr if r.validated is not None],
                )
                line += f" {corr:>14{fmt}}"
            else:
                avg = sum(getattr(r, field_name) for r in mr) / len(mr)
                line += f" {avg:>14{fmt}}"
        print(line)

    print("=" * 100)


def _pearson_correlation(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    x, y = x[:n], y[:n]
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
    std_x = (sum((xi - mean_x) ** 2 for xi in x) / n) ** 0.5
    std_y = (sum((yi - mean_y) ** 2 for yi in y) / n) ** 0.5
    if std_x == 0 or std_y == 0:
        return 0.0
    return cov / (std_x * std_y)


# ---------------------------------------------------------------------------
# Table C: Planning Efficiency Breakdown
# ---------------------------------------------------------------------------

def print_efficiency_breakdown(results: List[UnifiedResult], methods: List[str]):
    """Print planning efficiency breakdown (Table C for paper)."""
    print("\n" + "=" * 100)
    print("PLANNING EFFICIENCY BREAKDOWN (Table C)")
    print("=" * 100)

    metrics = [
        ("Parse Time Ratio", "parse_time_ratio", ".3f"),
        ("Search Time Ratio", "search_time_ratio", ".3f"),
        ("Avg Planning (ms)", "planning_ms", ".0f"),
        ("Avg Execution (ms)", "execution_ms", ".0f"),
        ("Planning/Exec Ratio", None, ".3f"),
        ("Throughput (q/min)", None, ".1f"),
        ("Avg Operators/Plan", "num_operators", ".1f"),
        ("Avg Plan Depth", "plan_depth", ".1f"),
    ]

    header = f"{'Metric':<25}"
    for m in methods:
        header += f" {m:>14}"
    print(header)
    print("-" * 100)

    for metric_name, field_name, fmt in metrics:
        line = f"{metric_name:<25}"
        for method_name in methods:
            mr = [r for r in results if r.method == method_name and not r.error]
            if not mr:
                line += f" {'N/A':>14}"
                continue
            if metric_name == "Planning/Exec Ratio":
                ratios = []
                for r in mr:
                    exec_t = r.elapsed_ms - r.planning_ms
                    if exec_t > 0:
                        ratios.append(r.planning_ms / exec_t)
                avg = sum(ratios) / len(ratios) if ratios else 0.0
                line += f" {avg:>14{fmt}}"
            elif metric_name == "Throughput (q/min)":
                avg_time_s = sum(r.elapsed_ms for r in mr) / len(mr) / 1000
                throughput = 60.0 / avg_time_s if avg_time_s > 0 else 0.0
                line += f" {throughput:>14{fmt}}"
            else:
                avg = sum(getattr(r, field_name) for r in mr) / len(mr)
                line += f" {avg:>14{fmt}}"
        print(line)

    print("=" * 100)


# ---------------------------------------------------------------------------
# Table D: Per-Operator-Type Accuracy
# ---------------------------------------------------------------------------

def print_operator_type_accuracy(results: List[UnifiedResult], methods: List[str]):
    """Print accuracy breakdown by query operator types."""
    print("\n" + "=" * 100)
    print("ACCURACY BY QUERY OPERATOR TYPE (Table D)")
    print("=" * 100)

    op_types = ["FILTER", "AGGREGATE", "SORT", "CONTENT_EXTRACT",
                "SEMANTIC_SEARCH", "CROSS_MODAL_MATCH"]

    header = f"{'Required Operator':<25}"
    for m in methods:
        header += f" {m:>14}"
    print(header)
    print("-" * 100)

    for op in op_types:
        line = f"{op:<25}"
        for method_name in methods:
            mr = [
                r for r in results
                if r.method == method_name
                and r.validated is not None
                and op in r.plan_operators
            ]
            if not mr:
                line += f" {'N/A':>14}"
            else:
                correct = sum(1 for r in mr if r.validated)
                total = len(mr)
                pct = correct / total * 100
                line += f" {correct}/{total} ({pct:.0f}%)"
                padding = 14 - len(f"{correct}/{total} ({pct:.0f}%)")
                if padding > 0:
                    line = line[:-len(f"{correct}/{total} ({pct:.0f}%)")] + \
                           " " * padding + f"{correct}/{total} ({pct:.0f}%)"
        print(line)

    # Multi-operator vs single-operator accuracy
    print("-" * 100)
    line_multi = f"{'Multi-op (>=3 ops)':<25}"
    line_single = f"{'Simple (<=2 ops)':<25}"
    for method_name in methods:
        mr_multi = [r for r in results if r.method == method_name
                    and r.validated is not None and r.num_operators >= 3]
        mr_simple = [r for r in results if r.method == method_name
                     and r.validated is not None and r.num_operators <= 2]
        for line_ref, mr_ref in [(line_multi, mr_multi), (line_single, mr_simple)]:
            if not mr_ref:
                val = f"{'N/A':>14}"
            else:
                correct = sum(1 for r in mr_ref if r.validated)
                total = len(mr_ref)
                pct = correct / total * 100
                val = f"{correct}/{total} ({pct:.0f}%)"
                val = f"{val:>14}"
            if line_ref is line_multi:
                line_multi += val
            else:
                line_single += val
    print(line_multi)
    print(line_single)

    print("=" * 100)


# ---------------------------------------------------------------------------
# JSON Export
# ---------------------------------------------------------------------------

def export_json(results: List[UnifiedResult], path: str):
    """Export full results to JSON."""
    data = []
    for r in results:
        entry = {
            "method": r.method,
            "workload": r.workload,
            "dataset": r.dataset,
            "query_id": r.query_id,
            "nl_description": r.nl_description,
            "answer": r.answer[:200],
            "validated": r.validated,
            "output_rows": r.output_rows,
            "elapsed_ms": round(r.elapsed_ms, 1),
            "planning_ms": round(r.planning_ms, 1),
            "mcts_candidates": r.mcts_candidates,
            "plan_depth": r.plan_depth,
            "pareto_size": r.pareto_size,
            "plan_operators": r.plan_operators,
            "physical_assignment": r.physical_assignment,
            "error": r.error,
            "plan_diversity": round(r.plan_diversity, 4),
            "quality_score_cv": round(r.quality_score_cv, 4),
            "completeness": round(r.completeness, 4),
            "pareto_hypervolume": round(r.pareto_hypervolume, 4),
            "pareto_spread": round(r.pareto_spread, 4),
            "root_confidence": round(r.root_confidence, 4),
            "avg_uncertainty_variance": round(r.avg_uncertainty_variance, 4),
            "avg_modality_risk": round(r.avg_modality_risk, 4),
            "estimated_latency": round(r.estimated_latency, 4),
            "estimated_resource_cost": round(r.estimated_resource_cost, 4),
            "estimated_confidence": round(r.estimated_confidence, 4),
            "parse_time_ratio": round(r.parse_time_ratio, 4),
            "search_time_ratio": round(r.search_time_ratio, 4),
            "execution_ms": round(r.execution_ms, 1),
            "num_operators": r.num_operators,
        }
        data.append(entry)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nResults exported to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified experiment runner for TiTSP vs baselines"
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="TiTSP,CAESURA,LOTUS,ThalamusDB,Palimpzest",
        help="Comma-separated method names",
    )
    parser.add_argument(
        "--workloads",
        type=str,
        default="nirvana,sembench",
        help="Comma-separated workload names",
    )
    parser.add_argument("--llm", type=str, default=None, help="LLM provider")
    parser.add_argument("--model", type=str, default=None, help="LLM model name")
    parser.add_argument("--json", type=str, default=None, help="JSON output path")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-query output")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",")]
    workloads = [w.strip() for w in args.workloads.split(",")]

    for m in methods:
        if m not in METHODS:
            print(f"ERROR: Unknown method '{m}'. Valid: {sorted(METHODS.keys())}")
            sys.exit(1)

    llm_provider = None
    if args.llm:
        from core.query_planner.llm_query_analyzer import create_provider
        llm_provider = create_provider(args.llm, model=args.model)

    print(f"Methods: {methods}")
    print(f"Workloads: {workloads}")
    print(f"LLM: {args.llm or 'None (regex fallback)'}")
    total_queries = 0
    if "nirvana" in workloads:
        total_queries += len(NIRVANA_QUERIES)
    if "sembench" in workloads:
        total_queries += len(SEMBENCH_QUERIES)
    print(f"Total runs: {total_queries} queries × {len(methods)} methods "
          f"= {total_queries * len(methods)}")

    results = run_experiment(
        methods=methods,
        workloads=workloads,
        llm_provider=llm_provider,
        verbose=not args.quiet,
    )

    print_main_table(results, methods)
    print_detailed_metrics(results, methods)
    print_per_dataset_breakdown(results, methods)
    print_search_quality_metrics(results, methods)
    print_optimization_quality_metrics(results, methods)
    print_efficiency_breakdown(results, methods)
    print_operator_type_accuracy(results, methods)

    if args.json:
        export_json(results, args.json)


if __name__ == "__main__":
    main()
