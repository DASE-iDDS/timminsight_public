"""
Extended Experiments: Ablation, Scalability, MCTS Sensitivity,
LLM Model Comparison, and Statistical Significance.

Usage:
    # Run all experiments
    python3 -m src.experiments.extended_experiments \
        --llm dashscope --model qwen-plus --json extended_results.json

    # Run specific experiment
    python3 -m src.experiments.extended_experiments \
        --llm dashscope --model qwen-plus --experiment ablation

    # Run MCTS sensitivity only
    python3 -m src.experiments.extended_experiments \
        --llm dashscope --model qwen-plus --experiment sensitivity

Experiments:
    ablation     — TiTSP vs 3 ablation variants (no-MCTS, no-Uncertainty, no-Pareto)
    scalability  — Latency vs data size (1x, 2x, 4x, 8x, 16x rows)
    sensitivity  — MCTS iterations (10, 50, 100, 200, 300, 500) vs accuracy/time
    llm          — Different LLM models comparison
    significance — Multiple runs for confidence intervals
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.query_planner.query_planner import QueryPlanner, PlanResult
from core.query_planner.mcts_plan_search import MCTSLogicalConfig
from core.query_planner.plan_executor import Table

from experiments.nirvana_workload import (
    DATASET_REGISTRY as NIRVANA_DATASETS,
    NIRVANA_QUERIES,
    validate_answer as nirvana_validate,
    collect_operators,
    plan_depth,
)
from experiments.sembench_workload import (
    DATASET_REGISTRY as SEMBENCH_DATASETS,
    SEMBENCH_QUERIES,
    validate_answer as sembench_validate,
)


# ---------------------------------------------------------------------------
# Query wrapper (same as unified_runner)
# ---------------------------------------------------------------------------

@dataclass
class Query:
    workload: str
    dataset: str
    query_id: str
    nl_description: str
    expected_answer: Optional[str]
    data: Table


def get_all_queries() -> List[Query]:
    queries = []
    for q in NIRVANA_QUERIES:
        data = NIRVANA_DATASETS.get(q.dataset, [])
        queries.append(Query(
            workload="nirvana", dataset=q.dataset, query_id=q.query_id,
            nl_description=q.nl_description,
            expected_answer=q.expected_answer, data=data,
        ))
    for q in SEMBENCH_QUERIES:
        data = SEMBENCH_DATASETS.get(q.scenario, [])
        queries.append(Query(
            workload="sembench", dataset=q.scenario, query_id=q.query_id,
            nl_description=q.nl_description,
            expected_answer=q.expected_answer, data=data,
        ))
    return queries


# ---------------------------------------------------------------------------
# Run single query and collect result
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    method: str
    workload: str
    dataset: str
    query_id: str
    validated: Optional[bool] = None
    elapsed_ms: float = 0.0
    planning_ms: float = 0.0
    execution_ms: float = 0.0
    mcts_candidates: int = 0
    plan_depth: int = 0
    pareto_size: int = 0
    num_operators: int = 0
    plan_quality: float = 0.0       # Q of the selected logical plan
    plan_latency: float = 0.0       # selected plan: cost-model latency objective
    plan_resource: float = 0.0      # selected plan: cost-model resource objective
    plan_confidence: float = 0.0    # selected plan: cost-model confidence objective
    prompt_tokens: int = 0          # measured LLM prompt tokens for this query
    completion_tokens: int = 0      # measured LLM completion tokens
    total_tokens: int = 0           # measured LLM total tokens
    llm_calls: int = 0              # number of LLM calls for this query
    req_ops: int = 0                # number of required operators (query complexity)
    req_coverage: float = 1.0       # fraction of required operators present in plan
    error: Optional[str] = None


def _usage_snapshot(planner: QueryPlanner):
    """Snapshot the provider's token counter, or None if unavailable."""
    provider = planner.llm_provider
    if provider is not None and hasattr(provider, "usage"):
        return provider.usage.snapshot()
    return None


def run_query(
    planner: QueryPlanner,
    query: Query,
    method_name: str,
) -> RunResult:
    # Reset the provider's token counter so we measure only this query.
    provider = planner.llm_provider
    if provider is not None and hasattr(provider, "usage"):
        provider.usage.reset()

    t0 = time.perf_counter()
    try:
        result = planner.plan_and_execute(query.nl_description, query.data)
        elapsed = (time.perf_counter() - t0) * 1000
        plan = result.plan
        planning_ms = plan.timing.total_ms if plan.timing else 0.0
        tok = _usage_snapshot(planner)

        validated = None
        if query.expected_answer is not None:
            validated = nirvana_validate(
                result.answer, result.execution.data, query.expected_answer,
            )

        required = set(plan.query_context.required_operators)
        present = set(plan.best_logical_plan.get_operator_types())
        coverage = len(present & required) / len(required) if required else 1.0

        return RunResult(
            method=method_name,
            workload=query.workload,
            dataset=query.dataset,
            query_id=query.query_id,
            validated=validated,
            elapsed_ms=elapsed,
            planning_ms=planning_ms,
            execution_ms=elapsed - planning_ms,
            mcts_candidates=len(plan.logical_candidates),
            plan_depth=plan_depth(plan.best_logical_plan),
            pareto_size=len(plan.pareto_front),
            num_operators=len(collect_operators(plan.best_logical_plan)),
            plan_quality=plan.best_quality_score,
            plan_latency=plan.selected_physical.objectives.latency,
            plan_resource=plan.selected_physical.objectives.resource_cost,
            plan_confidence=plan.selected_physical.objectives.confidence,
            prompt_tokens=tok.prompt_tokens if tok else 0,
            completion_tokens=tok.completion_tokens if tok else 0,
            total_tokens=tok.total_tokens if tok else 0,
            llm_calls=tok.calls if tok else 0,
            req_ops=len(required),
            req_coverage=coverage,
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        elapsed = (time.perf_counter() - t0) * 1000
        tok = _usage_snapshot(planner)
        return RunResult(
            method=method_name,
            workload=query.workload,
            dataset=query.dataset,
            query_id=query.query_id,
            elapsed_ms=elapsed,
            prompt_tokens=tok.prompt_tokens if tok else 0,
            completion_tokens=tok.completion_tokens if tok else 0,
            total_tokens=tok.total_tokens if tok else 0,
            llm_calls=tok.calls if tok else 0,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def summarize(results: List[RunResult], label: str) -> Dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r.validated)
    valid = [r for r in results if not r.error]
    avg_time = sum(r.elapsed_ms for r in valid) / len(valid) if valid else 0
    avg_plan = sum(r.planning_ms for r in valid) / len(valid) if valid else 0
    avg_exec = sum(r.execution_ms for r in valid) / len(valid) if valid else 0
    avg_cand = sum(r.mcts_candidates for r in valid) / len(valid) if valid else 0
    avg_depth = sum(r.plan_depth for r in valid) / len(valid) if valid else 0
    avg_quality = sum(r.plan_quality for r in valid) / len(valid) if valid else 0
    avg_plan_lat = sum(r.plan_latency for r in valid) / len(valid) if valid else 0
    avg_plan_res = sum(r.plan_resource for r in valid) / len(valid) if valid else 0
    avg_plan_conf = sum(r.plan_confidence for r in valid) / len(valid) if valid else 0
    # Measured runtime efficiency: token usage and throughput.
    avg_ptok = sum(r.prompt_tokens for r in valid) / len(valid) if valid else 0
    avg_ctok = sum(r.completion_tokens for r in valid) / len(valid) if valid else 0
    avg_ttok = sum(r.total_tokens for r in valid) / len(valid) if valid else 0
    avg_calls = sum(r.llm_calls for r in valid) / len(valid) if valid else 0
    total_tokens = sum(r.total_tokens for r in valid)
    total_time_s = sum(r.elapsed_ms for r in valid) / 1000.0
    throughput = (len(valid) / total_time_s) if total_time_s > 0 else 0  # queries/sec
    return {
        "method": label,
        "total": total,
        "passed": passed,
        "accuracy": passed / total if total else 0,
        "avg_time_ms": round(avg_time, 1),          # measured wall-clock latency
        "avg_planning_ms": round(avg_plan, 1),
        "avg_execution_ms": round(avg_exec, 1),
        "throughput_qps": round(throughput, 4),     # measured throughput
        "avg_candidates": round(avg_cand, 1),
        "avg_depth": round(avg_depth, 1),
        "avg_quality": round(avg_quality, 4),
        "avg_plan_latency": round(avg_plan_lat, 3),
        "avg_plan_resource": round(avg_plan_res, 3),
        "avg_plan_confidence": round(avg_plan_conf, 4),
        "avg_prompt_tokens": round(avg_ptok, 1),    # measured token usage
        "avg_completion_tokens": round(avg_ctok, 1),
        "avg_total_tokens": round(avg_ttok, 1),
        "total_tokens": int(total_tokens),
        "avg_llm_calls": round(avg_calls, 1),
        "errors": sum(1 for r in results if r.error),
    }


def print_table(title: str, rows: List[Dict], columns: List[Tuple[str, str, str]]):
    """Print a formatted table.
    columns: list of (key, header, format_str).
    """
    print(f"\n{'='*100}")
    print(title)
    print(f"{'='*100}")
    header = ""
    for key, hdr, fmt in columns:
        header += f"  {hdr:>14}"
    print(header)
    print("-" * 100)
    for row in rows:
        line = ""
        for key, hdr, fmt in columns:
            val = row.get(key, "")
            line += f"  {fmt.format(val):>14}"
        print(line)
    print("=" * 100)


# ===================================================================
# Experiment 1: Ablation Study
# ===================================================================

def run_ablation(llm_provider, queries: List[Query], verbose: bool) -> Dict:
    print("\n" + "#" * 80)
    print("# EXPERIMENT 1: ABLATION STUDY")
    print("#" * 80)

    variants = [
        ("TiTSP (Full)", "titsp"),
        ("w/o MCTS", "titsp-no-mcts"),
        ("w/o Uncertainty", "titsp-no-uncertainty"),
        ("w/o Pareto", "titsp-no-pareto"),
    ]

    all_results = {}
    for label, strategy in variants:
        print(f"\n--- Running: {label} (strategy={strategy}) ---")
        planner = QueryPlanner(
            llm_provider=llm_provider,
            planning_strategy=strategy,
        )
        results = []
        for q in queries:
            r = run_query(planner, q, label)
            results.append(r)
            if verbose:
                ok = "PASS" if r.validated else ("FAIL" if r.validated is False else "---")
                print(f"  [{ok}] {q.workload}/{q.dataset}/{q.query_id} ({r.elapsed_ms:.0f}ms)")
        all_results[label] = results

    # Print summary
    summaries = []
    for label, _ in variants:
        s = summarize(all_results[label], label)
        # Per-workload breakdown
        for wl in ["nirvana", "sembench"]:
            wl_results = [r for r in all_results[label] if r.workload == wl]
            wl_passed = sum(1 for r in wl_results if r.validated)
            s[f"{wl}_acc"] = f"{wl_passed}/{len(wl_results)}"
        summaries.append(s)

    print(f"\n{'='*124}")
    print("ABLATION STUDY RESULTS (cost-model estimates + measured runtime)")
    print(f"{'='*124}")
    print(f"{'Method':<20}{'Nirvana':>9}{'SemBench':>10}{'Acc':>8}"
          f"{'PlanQ':>8}{'P-Lat':>8}{'P-Res':>8}{'P-Conf':>8}"
          f"{'Tokens':>9}{'QPS':>8}{'Time':>10}")
    print("-" * 124)
    for s in summaries:
        print(f"{s['method']:<20}{s.get('nirvana_acc',''):>9}{s.get('sembench_acc',''):>10}"
              f"{s['accuracy']*100:>7.1f}%{s['avg_quality']:>8.3f}{s['avg_plan_latency']:>8.2f}"
              f"{s['avg_plan_resource']:>8.2f}{s['avg_plan_confidence']:>8.3f}"
              f"{s['avg_total_tokens']:>9.0f}{s['throughput_qps']:>8.3f}{s['avg_time_ms']:>8.0f}ms")

    # Per-metric component contribution: how each metric changes when a stage
    # is removed. Each variant should degrade on the metric its stage targets:
    #   no-MCTS -> lower PlanQ; no-Uncertainty -> lower Acc; no-Pareto -> higher P-Lat/P-Res.
    full = summaries[0]
    print(f"\nComponent contribution (Δ vs Full when the stage is removed):")
    print(f"  {'Variant':<20}{'ΔAcc':>10}{'ΔPlanQ':>10}{'ΔP-Lat':>10}"
          f"{'ΔP-Res':>10}{'ΔTokens':>10}")
    for s in summaries[1:]:
        print(f"  {s['method']:<20}"
              f"{(full['accuracy']-s['accuracy'])*100:>+9.1f}%"
              f"{full['avg_quality']-s['avg_quality']:>+10.3f}"
              f"{full['avg_plan_latency']-s['avg_plan_latency']:>+10.2f}"
              f"{full['avg_plan_resource']-s['avg_plan_resource']:>+10.2f}"
              f"{full['avg_total_tokens']-s['avg_total_tokens']:>+10.0f}")

    return {"ablation": summaries}


# ===================================================================
# Experiment 2: Scalability
# ===================================================================

def scale_data(data: Table, factor: int) -> Table:
    """Replicate data rows to simulate larger datasets."""
    if factor <= 1:
        return data
    scaled = []
    for i in range(factor):
        for row in data:
            new_row = dict(row)
            if i > 0:
                for key in new_row:
                    if isinstance(new_row[key], str) and key.lower() in ("title", "name", "id"):
                        new_row[key] = f"{new_row[key]} (copy-{i})"
            scaled.append(new_row)
    return scaled


def run_scalability(llm_provider, queries: List[Query], verbose: bool) -> Dict:
    print("\n" + "#" * 80)
    print("# EXPERIMENT 2: SCALABILITY")
    print("#" * 80)

    scale_factors = [1, 2, 4, 8, 16]
    subset = [q for q in queries if q.workload == "nirvana"][:12]

    all_summaries = []
    for factor in scale_factors:
        label = f"{factor}x ({factor * 6}-{factor * 8} rows)"
        print(f"\n--- Scale factor: {factor}x ---")

        planner = QueryPlanner(llm_provider=llm_provider, planning_strategy="titsp")
        results = []
        for q in subset:
            scaled_q = Query(
                workload=q.workload, dataset=q.dataset, query_id=q.query_id,
                nl_description=q.nl_description,
                expected_answer=q.expected_answer,
                data=scale_data(q.data, factor),
            )
            r = run_query(planner, scaled_q, label)
            results.append(r)
            if verbose:
                rows = len(scaled_q.data)
                print(f"  {q.query_id}: {r.elapsed_ms:.0f}ms (plan={r.planning_ms:.0f}ms, exec={r.execution_ms:.0f}ms, {rows} rows)")

        s = summarize(results, label)
        s["scale_factor"] = factor
        s["avg_rows"] = sum(len(scale_data(q.data, factor)) for q in subset) / len(subset)
        all_summaries.append(s)

    print(f"\n{'='*100}")
    print("SCALABILITY RESULTS")
    print(f"{'='*100}")
    print(f"{'Scale':<25} {'Rows':>8} {'Accuracy':>10} {'PlanTime':>10} {'ExecTime':>10} {'TotalTime':>10}")
    print("-" * 100)
    for s in all_summaries:
        print(f"{s['method']:<25} {s['avg_rows']:>7.0f} "
              f"{s['passed']}/{s['total']} ({s['accuracy']*100:.1f}%) "
              f"{s['avg_planning_ms']:>9.0f}ms {s['avg_execution_ms']:>9.0f}ms {s['avg_time_ms']:>9.0f}ms")

    return {"scalability": all_summaries}


# ===================================================================
# Experiment 3: MCTS Parameter Sensitivity
# ===================================================================

def run_sensitivity(llm_provider, queries: List[Query], verbose: bool) -> Dict:
    print("\n" + "#" * 80)
    print("# EXPERIMENT 3: MCTS PARAMETER SENSITIVITY")
    print("#" * 80)

    iteration_counts = [10, 50, 100, 200, 300, 500]
    all_summaries = []

    for n_iter in iteration_counts:
        label = f"MCTS-{n_iter}"
        print(f"\n--- MCTS iterations: {n_iter} ---")

        config = MCTSLogicalConfig(num_iterations=n_iter)
        planner = QueryPlanner(
            llm_provider=llm_provider,
            planning_strategy="titsp",
            mcts_config=config,
        )
        results = []
        for q in queries:
            r = run_query(planner, q, label)
            results.append(r)
            if verbose:
                ok = "PASS" if r.validated else ("FAIL" if r.validated is False else "---")
                print(f"  [{ok}] {q.workload}/{q.dataset}/{q.query_id} ({r.elapsed_ms:.0f}ms, cand={r.mcts_candidates})")
        s = summarize(results, label)
        s["iterations"] = n_iter
        # Per-workload
        for wl in ["nirvana", "sembench"]:
            wl_results = [r for r in results if r.workload == wl]
            wl_passed = sum(1 for r in wl_results if r.validated)
            s[f"{wl}_acc"] = f"{wl_passed}/{len(wl_results)}"
        all_summaries.append(s)

    print(f"\n{'='*100}")
    print("MCTS SENSITIVITY RESULTS")
    print(f"{'='*100}")
    print(f"{'Config':<15} {'Iters':>6} {'Nirvana':>10} {'SemBench':>10} {'Overall':>12} {'Rate':>8} {'Candidates':>11} {'PlanTime':>10} {'TotalTime':>10}")
    print("-" * 100)
    for s in all_summaries:
        rate = f"{s['accuracy']*100:.1f}%"
        print(f"{s['method']:<15} {s['iterations']:>6} {s.get('nirvana_acc',''):>10} {s.get('sembench_acc',''):>10} "
              f"{s['passed']}/{s['total']:>3} {rate:>8} {s['avg_candidates']:>10.1f} {s['avg_planning_ms']:>9.0f}ms {s['avg_time_ms']:>9.0f}ms")

    return {"sensitivity": all_summaries}


# ===================================================================
# Experiment 4: LLM Model Comparison
# ===================================================================

# Models compared in the LLM-comparison experiment. DashScope is a unified
# gateway hosting Qwen, GLM, Kimi, etc. under one DASHSCOPE_API_KEY, so these
# all route through provider_type="dashscope". Edit this list or override at
# the CLI with --models "name1,name2" (or "name:provider,...").
# NOTE: names must be the EXACT model IDs accepted by the API.
DEFAULT_LLM_MODELS: List[Tuple[str, str]] = [
    ("qwen3.6-plus", "dashscope"),
    ("kimi-k2.6", "dashscope"),
]


def parse_models(spec: str, default_provider: str) -> List[Tuple[str, str]]:
    """Parse a --models spec into [(model_name, provider_type)].

    Accepts comma-separated entries, each either "name" (uses
    default_provider) or "name:provider".
    """
    out: List[Tuple[str, str]] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            name, prov = tok.split(":", 1)
            out.append((name.strip(), prov.strip()))
        else:
            out.append((tok, default_provider))
    return out


def run_llm_comparison(base_provider, queries: List[Query], verbose: bool,
                       models: Optional[List[Tuple[str, str]]] = None) -> Dict:
    print("\n" + "#" * 80)
    print("# EXPERIMENT 4: LLM MODEL COMPARISON")
    print("#" * 80)

    from core.query_planner.llm_query_analyzer import create_provider

    model_list = models if models else DEFAULT_LLM_MODELS

    all_summaries = []
    for model_name, provider_type in model_list:
        label = f"TiTSP ({model_name})"
        print(f"\n--- Model: {model_name} via {provider_type} ---")

        try:
            provider = create_provider(provider_type, model=model_name)
        except Exception as e:
            print(f"  SKIPPED: {e}")
            continue

        planner = QueryPlanner(llm_provider=provider, planning_strategy="titsp")
        results = []
        for q in queries:
            r = run_query(planner, q, label)
            results.append(r)
            if verbose:
                ok = "PASS" if r.validated else ("FAIL" if r.validated is False else "---")
                print(f"  [{ok}] {q.workload}/{q.dataset}/{q.query_id} ({r.elapsed_ms:.0f}ms)")

        s = summarize(results, label)
        s["model"] = model_name
        for wl in ["nirvana", "sembench"]:
            wl_results = [r for r in results if r.workload == wl]
            wl_passed = sum(1 for r in wl_results if r.validated)
            s[f"{wl}_acc"] = f"{wl_passed}/{len(wl_results)}"
        all_summaries.append(s)

    print(f"\n{'='*100}")
    print("LLM MODEL COMPARISON RESULTS")
    print(f"{'='*100}")
    print(f"{'Method':<28}{'Nirvana':>9}{'SemBench':>10}{'Acc':>8}"
          f"{'Tok/q':>9}{'TotTok':>10}{'QPS':>8}{'Time':>10}")
    print("-" * 100)
    for s in all_summaries:
        print(f"{s['method']:<28}{s.get('nirvana_acc',''):>9}{s.get('sembench_acc',''):>10}"
              f"{s['accuracy']*100:>7.1f}%{s['avg_total_tokens']:>9.0f}{s['total_tokens']:>10}"
              f"{s['throughput_qps']:>8.3f}{s['avg_time_ms']:>8.0f}ms")

    return {"llm_comparison": all_summaries}


# ===================================================================
# Experiment 5: Statistical Significance
# ===================================================================

def run_significance(llm_provider, queries: List[Query], verbose: bool, n_runs: int = 3) -> Dict:
    print("\n" + "#" * 80)
    print(f"# EXPERIMENT 5: STATISTICAL SIGNIFICANCE ({n_runs} runs)")
    print("#" * 80)

    methods = [
        ("TiTSP", "titsp"),
        ("CAESURA", "caesura"),
        ("LOTUS", "lotus"),
        ("Palimpzest", "palimpzest"),
        ("Nirvana", "nirvana"),
    ]

    run_results = defaultdict(list)  # method -> list of per-run accuracies

    for run_idx in range(n_runs):
        print(f"\n=== Run {run_idx + 1}/{n_runs} ===")
        for label, strategy in methods:
            print(f"  --- {label} ---")
            planner = QueryPlanner(
                llm_provider=llm_provider,
                planning_strategy=strategy,
            )
            results = []
            for q in queries:
                r = run_query(planner, q, label)
                results.append(r)
            s = summarize(results, label)
            run_results[label].append(s["accuracy"])
            print(f"    Accuracy: {s['passed']}/{s['total']} ({s['accuracy']*100:.1f}%)")

    # Compute statistics
    print(f"\n{'='*100}")
    print("STATISTICAL SIGNIFICANCE RESULTS")
    print(f"{'='*100}")
    print(f"{'Method':<15} {'Runs':>5} {'Mean Acc':>10} {'Std':>8} {'Min':>8} {'Max':>8} {'95% CI':>16}")
    print("-" * 100)

    stats = []
    for label, _ in methods:
        accs = run_results[label]
        n = len(accs)
        mean = sum(accs) / n
        variance = sum((a - mean) ** 2 for a in accs) / n if n > 1 else 0
        std = variance ** 0.5
        se = std / (n ** 0.5) if n > 1 else 0
        t_val = 2.776 if n == 5 else (4.303 if n == 3 else 1.96)
        ci_low = mean - t_val * se
        ci_high = mean + t_val * se
        ci_str = f"[{ci_low*100:.1f}%, {ci_high*100:.1f}%]"
        print(f"{label:<15} {n:>5} {mean*100:>9.1f}% {std*100:>7.1f}% {min(accs)*100:>7.1f}% {max(accs)*100:>7.1f}% {ci_str:>16}")
        stats.append({
            "method": label,
            "n_runs": n,
            "mean_accuracy": round(mean, 4),
            "std": round(std, 4),
            "min": round(min(accs), 4),
            "max": round(max(accs), 4),
            "ci_low": round(ci_low, 4),
            "ci_high": round(ci_high, 4),
            "raw_accuracies": [round(a, 4) for a in accs],
        })

    # Paired comparison: TiTSP vs each baseline
    print(f"\nPaired Comparison (TiTSP vs baselines):")
    titsp_accs = run_results["TiTSP"]
    for label, _ in methods[1:]:
        bl_accs = run_results[label]
        diffs = [t - b for t, b in zip(titsp_accs, bl_accs)]
        n = len(diffs)
        mean_diff = sum(diffs) / n
        var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1) if n > 1 else 0
        se_diff = (var_diff / n) ** 0.5 if n > 1 else 0
        t_stat = mean_diff / se_diff if se_diff > 0 else float("inf")
        # Simplified p-value approximation for paired t-test
        df = n - 1
        if df >= 1 and se_diff > 0:
            p_approx = min(1.0, 2.0 * math.exp(-0.717 * abs(t_stat) - 0.416 * t_stat * t_stat / df))
        else:
            p_approx = 0.0 if mean_diff > 0 else 1.0
        sig = "***" if p_approx < 0.001 else ("**" if p_approx < 0.01 else ("*" if p_approx < 0.05 else "n.s."))
        print(f"  vs {label:<12}: mean Δ = {mean_diff*100:>+.1f}%, t = {t_stat:>6.2f}, p ≈ {p_approx:.4f} {sig}")

    return {"significance": stats}


# ===================================================================
# Helpers for the additional experiments
# ===================================================================

# Approximate DashScope list prices, CNY per 1M tokens, as (input, output).
# NOTE: placeholders — VERIFY against current pricing before publishing.
PRICE_TABLE_CNY: Dict[str, Tuple[float, float]] = {
    "qwen3.6-plus": (0.8, 8.0),
    "kimi-k2.6": (4.0, 16.0),
}
DEFAULT_PRICE_CNY: Tuple[float, float] = (1.0, 4.0)


def _cost_cny(model: str, prompt_tokens: float, completion_tokens: float) -> float:
    pin, pout = PRICE_TABLE_CNY.get(model, DEFAULT_PRICE_CNY)
    return (prompt_tokens / 1e6) * pin + (completion_tokens / 1e6) * pout


def _pareto_2d(points: List[Tuple[float, float]]) -> set:
    """Indices that are Pareto-optimal for (x: lower better, y: higher better)."""
    front = set()
    for i, (xi, yi) in enumerate(points):
        dominated = False
        for j, (xj, yj) in enumerate(points):
            if j != i and xj <= xi and yj >= yi and (xj < xi or yj > yi):
                dominated = True
                break
        if not dominated:
            front.add(i)
    return front


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    denom = (vx * vy) ** 0.5
    return cov / denom if denom > 0 else 0.0


def _ece(pairs: List[Tuple[float, Optional[bool]]], n_bins: int = 10):
    """Expected Calibration Error from (confidence, correct) pairs."""
    data = [(c, 1.0 if ok else 0.0) for c, ok in pairs if ok is not None]
    if not data:
        return 0.0, []
    n = len(data)
    ece = 0.0
    bins = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        members = [(c, y) for c, y in data
                   if (lo < c <= hi) or (b == 0 and c <= hi)]
        if not members:
            bins.append({"bin": f"({lo:.1f},{hi:.1f}]", "count": 0,
                         "avg_conf": 0.0, "accuracy": 0.0})
            continue
        avg_conf = sum(c for c, _ in members) / len(members)
        acc = sum(y for _, y in members) / len(members)
        ece += (len(members) / n) * abs(avg_conf - acc)
        bins.append({"bin": f"({lo:.1f},{hi:.1f}]", "count": len(members),
                     "avg_conf": round(avg_conf, 3), "accuracy": round(acc, 3)})
    return ece, bins


def _perturb_query(text: str, seed: int) -> str:
    """Inject light char-level typos (adjacent swaps) deterministically."""
    chars = list(text)
    rng = random.Random(seed)
    n_swaps = max(1, len(chars) // 25)
    for _ in range(n_swaps):
        if len(chars) < 3:
            break
        i = rng.randint(0, len(chars) - 2)
        if chars[i].isalpha() and chars[i + 1].isalpha():
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def _by_query_key(r: RunResult) -> Tuple[str, str, str]:
    return (r.workload, r.dataset, r.query_id)


# ===================================================================
# Experiment 6: Cost–Quality Pareto (also same-backbone fair comparison)
# ===================================================================

COMPARISON_METHODS = [
    ("TiTSP", "titsp"),
    ("CAESURA", "caesura"),
    ("LOTUS", "lotus"),
    ("ThalamusDB", "thalamusdb"),
    ("Palimpzest", "palimpzest"),
    ("Nirvana", "nirvana"),
]


def run_cost_quality(llm_provider, queries: List[Query], verbose: bool,
                     model_name: str = "") -> Dict:
    print("\n" + "#" * 80)
    print("# EXPERIMENT 6: COST-QUALITY PARETO (same backbone for all methods)")
    print("#" * 80)

    summaries = []
    for label, strat in COMPARISON_METHODS:
        print(f"\n--- {label} (strategy={strat}) ---")
        planner = QueryPlanner(llm_provider=llm_provider, planning_strategy=strat)
        results = [run_query(planner, q, label) for q in queries]
        s = summarize(results, label)
        for wl in ["nirvana", "sembench"]:
            wlr = [r for r in results if r.workload == wl]
            wlp = sum(1 for r in wlr if r.validated)
            s[f"{wl}_acc"] = f"{wlp}/{len(wlr)}"
        s["cost_cny"] = round(
            _cost_cny(model_name, s["avg_prompt_tokens"], s["avg_completion_tokens"]), 6)
        summaries.append(s)

    pts = [(s["avg_total_tokens"], s["accuracy"]) for s in summaries]
    front = _pareto_2d(pts)

    print(f"\n{'='*100}")
    print("COST-QUALITY RESULTS  (* = Pareto-optimal on tokens vs accuracy)")
    print(f"{'='*100}")
    print(f"{'Method':<14}{'Acc':>8}{'Tok/q':>9}{'¥/q':>10}{'QPS':>8}{'Time':>10}{'':>4}")
    print("-" * 100)
    for i, s in enumerate(summaries):
        star = " *" if i in front else ""
        print(f"{s['method']:<14}{s['accuracy']*100:>7.1f}%{s['avg_total_tokens']:>9.0f}"
              f"{s['cost_cny']:>10.4f}{s['throughput_qps']:>8.3f}{s['avg_time_ms']:>8.0f}ms{star:>4}")

    # Headline vs the strongest *baseline* (highest-accuracy non-TiTSP).
    titsp = summaries[0]
    baselines = summaries[1:]
    if baselines:
        strongest = max(baselines, key=lambda s: s["accuracy"])
        d_acc = (titsp["accuracy"] - strongest["accuracy"]) * 100
        tok_ratio = (strongest["avg_total_tokens"] / titsp["avg_total_tokens"]
                     if titsp["avg_total_tokens"] else 0)
        print(f"\nTiTSP vs strongest baseline ({strongest['method']}): "
              f"Δacc = {d_acc:+.1f}%, token ratio = {tok_ratio:.2f}x")

    return {"cost_quality": summaries,
            "pareto_methods": [summaries[i]["method"] for i in sorted(front)]}


# ===================================================================
# Experiment 7: Uncertainty Calibration + Operator-dropping
# ===================================================================

def run_calibration(llm_provider, queries: List[Query], verbose: bool) -> Dict:
    print("\n" + "#" * 80)
    print("# EXPERIMENT 7: UNCERTAINTY CALIBRATION + OPERATOR-DROPPING")
    print("#" * 80)

    full_planner = QueryPlanner(llm_provider=llm_provider, planning_strategy="titsp")
    full = [run_query(full_planner, q, "TiTSP") for q in queries]

    pairs = [(r.plan_confidence, r.validated) for r in full]
    ece, bins = _ece(pairs)
    cs = [r.plan_confidence for r in full if r.validated is not None]
    ys = [1.0 if r.validated else 0.0 for r in full if r.validated is not None]
    corr = _pearson(cs, ys)

    print(f"\nCalibration of predicted plan confidence vs actual correctness:")
    print(f"  ECE = {ece:.4f}   point-biserial corr = {corr:+.4f}   (n={len(cs)})")
    print(f"  {'confidence bin':<16}{'count':>8}{'avg_conf':>10}{'accuracy':>10}")
    for b in bins:
        if b["count"]:
            print(f"  {b['bin']:<16}{b['count']:>8}{b['avg_conf']:>10.3f}{b['accuracy']:>10.3f}")

    nou_planner = QueryPlanner(llm_provider=llm_provider,
                               planning_strategy="titsp-no-uncertainty")
    nou = [run_query(nou_planner, q, "w/o U") for q in queries]

    def cov_stats(rs):
        cov = [r.req_coverage for r in rs if not r.error]
        drops = sum(1 for c in cov if c < 1.0)
        return (sum(cov) / len(cov) if cov else 0.0, drops, len(cov))

    f_cov, f_drop, f_n = cov_stats(full)
    n_cov, n_drop, n_n = cov_stats(nou)
    print(f"\nRequired-operator coverage (uncertainty-aware selection effect):")
    print(f"  {'variant':<16}{'avg_coverage':>14}{'dropped_queries':>18}")
    print(f"  {'TiTSP (Full)':<16}{f_cov:>14.3f}{f_drop:>13}/{f_n}")
    print(f"  {'w/o Uncertainty':<16}{n_cov:>14.3f}{n_drop:>13}/{n_n}")

    return {"calibration": {
        "ece": round(ece, 4), "correlation": round(corr, 4),
        "bins": bins,
        "coverage_full": round(f_cov, 4), "dropped_full": f_drop,
        "coverage_no_uncertainty": round(n_cov, 4), "dropped_no_uncertainty": n_drop,
    }}


# ===================================================================
# Experiment 8: Query-complexity stratification (MCTS value vs complexity)
# ===================================================================

def run_complexity(llm_provider, queries: List[Query], verbose: bool) -> Dict:
    print("\n" + "#" * 80)
    print("# EXPERIMENT 8: ACCURACY BY QUERY COMPLEXITY (TiTSP vs w/o MCTS)")
    print("#" * 80)

    variants = [("TiTSP", "titsp"), ("w/o MCTS", "titsp-no-mcts")]
    runs = {}
    for label, strat in variants:
        planner = QueryPlanner(llm_provider=llm_provider, planning_strategy=strat)
        runs[label] = {_by_query_key(r): r for r in
                       (run_query(planner, q, label) for q in queries)}

    complexity = {k: r.req_ops for k, r in runs["TiTSP"].items()}
    bins_def = [("simple (<=2)", lambda c: c <= 2),
                ("medium (3)", lambda c: c == 3),
                ("complex (4)", lambda c: c == 4),
                ("very complex (>=5)", lambda c: c >= 5)]

    print(f"\n{'complexity bin':<20}{'n':>5}{'TiTSP':>10}{'w/oMCTS':>10}{'gap':>9}")
    print("-" * 60)
    rows = []
    for name, pred in bins_def:
        keys = [k for k, c in complexity.items() if pred(c)]
        if not keys:
            continue
        def acc(label):
            rs = [runs[label][k] for k in keys if k in runs[label]]
            ev = [r for r in rs if r.validated is not None]
            return (sum(1 for r in ev if r.validated) / len(ev)) if ev else 0.0
        a_full, a_nomcts = acc("TiTSP"), acc("w/o MCTS")
        gap = (a_full - a_nomcts) * 100
        print(f"{name:<20}{len(keys):>5}{a_full*100:>9.1f}%{a_nomcts*100:>9.1f}%{gap:>+8.1f}%")
        rows.append({"bin": name, "n": len(keys),
                     "titsp_acc": round(a_full, 4), "no_mcts_acc": round(a_nomcts, 4),
                     "gap_pct": round(gap, 2)})

    return {"complexity": rows}


# ===================================================================
# Experiment 9: Preference-steering Pareto navigation (Stage-4 / Tchebycheff)
# ===================================================================

def run_preference(llm_provider, queries: List[Query], verbose: bool) -> Dict:
    print("\n" + "#" * 80)
    print("# EXPERIMENT 9: PREFERENCE-STEERING PARETO NAVIGATION (planning-only)")
    print("#" * 80)

    from core.query_planner.mcts_plan_search import UserPreference

    prefs = [
        ("latency-first",   UserPreference(0.80, 0.10, 0.10)),
        ("balanced",        UserPreference(0.34, 0.33, 0.33)),
        ("confidence-first", UserPreference(0.10, 0.10, 0.80)),
    ]

    print(f"\n{'preference':<18}{'n':>5}{'sel.Lat':>10}{'sel.Res':>10}"
          f"{'sel.Conf':>10}{'paretoSz':>10}")
    print("-" * 64)
    rows = []
    for label, pref in prefs:
        planner = QueryPlanner(llm_provider=llm_provider,
                               planning_strategy="titsp", preference=pref)
        lat = res = conf = fsize = 0.0
        n = 0
        for q in queries:
            try:
                pr = planner.plan(q.nl_description)  # planning only (no execution)
            except Exception:
                continue
            o = pr.selected_physical.objectives
            lat += o.latency; res += o.resource_cost; conf += o.confidence
            fsize += len(pr.pareto_front); n += 1
        if n == 0:
            continue
        row = {"preference": label, "n": n,
               "sel_latency": round(lat / n, 3), "sel_resource": round(res / n, 3),
               "sel_confidence": round(conf / n, 4), "avg_pareto_size": round(fsize / n, 2)}
        rows.append(row)
        print(f"{label:<18}{n:>5}{row['sel_latency']:>10.3f}{row['sel_resource']:>10.3f}"
              f"{row['sel_confidence']:>10.4f}{row['avg_pareto_size']:>10.2f}")

    print("\n(Expect: latency-first -> lowest sel.Lat; confidence-first -> highest sel.Conf.)")
    return {"preference": rows}


# ===================================================================
# Experiment 10: Per-stage error attribution
# ===================================================================

def run_error_analysis(llm_provider, queries: List[Query], verbose: bool) -> Dict:
    print("\n" + "#" * 80)
    print("# EXPERIMENT 10: PER-STAGE ERROR ATTRIBUTION (heuristic)")
    print("#" * 80)

    planner = QueryPlanner(llm_provider=llm_provider, planning_strategy="titsp")
    cats = {"parse/planning": 0, "execution": 0, "answer-gen": 0, "exception": 0}
    n_eval = 0
    n_fail = 0
    for q in queries:
        if q.expected_answer is None:
            continue
        n_eval += 1
        try:
            result = planner.plan_and_execute(q.nl_description, q.data)
        except Exception:
            cats["exception"] += 1
            n_fail += 1
            continue
        ok = nirvana_validate(result.answer, result.execution.data, q.expected_answer)
        if ok:
            continue
        n_fail += 1
        plan = result.plan
        required = set(plan.query_context.required_operators)
        present = set(plan.best_logical_plan.get_operator_types())
        if required and len(present & required) < len(required):
            cats["parse/planning"] += 1
        elif not result.execution.data:
            cats["execution"] += 1
        else:
            cats["answer-gen"] += 1

    print(f"\nEvaluated {n_eval} queries, {n_fail} failures. Attribution:")
    for cat, cnt in cats.items():
        pct = (cnt / n_fail * 100) if n_fail else 0.0
        print(f"  {cat:<16}: {cnt:>3}  ({pct:>5.1f}% of failures)")

    return {"error_analysis": {"evaluated": n_eval, "failures": n_fail, "categories": cats}}


# ===================================================================
# Experiment 11: Plan optimality gap (MCTS vs bounded exhaustive search)
# ===================================================================

def _exhaustive_best_q(ocg, ctx, search, max_ops: int, cap: int):
    """Bounded exhaustive enumeration of OCG-valid plans; returns
    (best_Q, n_complete) or (None, explored) if the space exceeds `cap`."""
    import uuid
    from core.query_planner.mcts_plan_search import (
        LogicalPlanNode, PartialPlan, OpenPosition,
    )

    root = LogicalPlanNode(node_id="r", operator_type=ctx.root_operator_type)
    p0 = PartialPlan(root=root, ocg=ocg)
    spec = ocg.get_spec(ctx.root_operator_type)
    if spec and not spec.is_leaf:
        for i in range(spec.min_children):
            p0.open_positions.append(OpenPosition(root, i, 1))

    frontier = [p0]
    best_q = None
    n_complete = 0
    explored = 0
    while frontier:
        if explored > cap:
            return None, explored
        p = frontier.pop()
        explored += 1
        if p.is_terminal():
            q, cand = search._evaluate(p.root)
            if cand is not None and (best_q is None or q > best_q):
                best_q = q
            n_complete += 1
            continue
        if p.node_count > max_ops:
            continue
        pos = p.open_positions[0]
        for child_type, _ in ocg.get_valid_children(pos.parent_node.operator_type):
            np_ = p.clone()
            npos = np_.open_positions[0]
            new_node = LogicalPlanNode(node_id=str(uuid.uuid4())[:8],
                                       operator_type=child_type,
                                       parent=npos.parent_node)
            npos.parent_node.children.append(new_node)
            np_.open_positions.pop(0)
            np_.node_count += 1
            cspec = ocg.get_spec(child_type)
            if cspec and not cspec.is_leaf:
                for i in range(cspec.min_children):
                    np_.open_positions.append(
                        OpenPosition(new_node, i, npos.depth + 1))
            frontier.append(np_)
    return best_q, n_complete


def run_optimality(llm_provider, queries: List[Query], verbose: bool,
                   max_ops: int = 6, cap: int = 4000) -> Dict:
    print("\n" + "#" * 80)
    print("# EXPERIMENT 11: PLAN OPTIMALITY GAP (MCTS vs exhaustive)")
    print("#" * 80)

    from core.query_planner.operator_compatibility_graph import (
        OperatorCompatibilityGraph, QueryContext,
    )
    from core.query_planner.mcts_plan_search import (
        MCTSLogicalPlanSearch, MCTSLogicalConfig,
    )
    from core.query_planner.default_impl_registry import build_default_impl_registry

    reg = build_default_impl_registry()
    gaps = []
    skipped = 0
    for q in queries:
        try:
            ctx = (QueryContext.from_query_llm(q.nl_description, llm_provider)
                   if llm_provider is not None
                   else QueryContext.from_query(q.nl_description))
        except Exception:
            ctx = QueryContext.from_query(q.nl_description)
        ocg = OperatorCompatibilityGraph()
        search = MCTSLogicalPlanSearch(
            ocg, ctx, MCTSLogicalConfig(random_seed=0), reg)
        cands = search.search()
        if not cands:
            continue
        q_mcts = cands[0].quality_score
        q_star, n_plans = _exhaustive_best_q(ocg, ctx, search, max_ops, cap)
        if q_star is None:
            skipped += 1
            continue
        gap = (q_star - q_mcts) / q_star * 100 if q_star > 0 else 0.0
        gaps.append(gap)

    if gaps:
        avg_gap = sum(gaps) / len(gaps)
        max_gap = max(gaps)
        optimal = sum(1 for g in gaps if g <= 1e-6)
        print(f"\nEvaluated {len(gaps)} queries ({skipped} skipped: space > {cap}).")
        print(f"  avg optimality gap = {avg_gap:.3f}%   max = {max_gap:.3f}%")
        print(f"  MCTS found the exact optimum on {optimal}/{len(gaps)} queries.")
    else:
        avg_gap = max_gap = 0.0
        optimal = 0
        print(f"\nNo queries within the enumeration cap (all skipped).")

    return {"optimality": {
        "n": len(gaps), "skipped": skipped,
        "avg_gap_pct": round(avg_gap, 4), "max_gap_pct": round(max_gap, 4),
        "exact_optimum": optimal,
    }}


# ===================================================================
# Experiment 12: Robustness (input noise + bilingual)
# ===================================================================

def run_robustness(llm_provider, queries: List[Query], verbose: bool,
                   bilingual_n: int = 20) -> Dict:
    print("\n" + "#" * 80)
    print("# EXPERIMENT 12: ROBUSTNESS (input noise + bilingual)")
    print("#" * 80)

    planner = QueryPlanner(llm_provider=llm_provider, planning_strategy="titsp")

    clean = [run_query(planner, q, "clean") for q in queries]
    perturbed = []
    for i, q in enumerate(queries):
        pq = Query(q.workload, q.dataset, q.query_id,
                   _perturb_query(q.nl_description, i), q.expected_answer, q.data)
        perturbed.append(run_query(planner, pq, "perturbed"))

    s_clean = summarize(clean, "clean")
    s_pert = summarize(perturbed, "perturbed")
    print(f"\nInput-noise robustness (char-level typos):")
    print(f"  clean      acc = {s_clean['accuracy']*100:.1f}%")
    print(f"  perturbed  acc = {s_pert['accuracy']*100:.1f}%  "
          f"(Δ = {(s_pert['accuracy']-s_clean['accuracy'])*100:+.1f}%)")

    out = {"clean": s_clean, "perturbed": s_pert}

    if llm_provider is not None and bilingual_n > 0:
        subset = queries[:bilingual_n]
        en = [run_query(planner, q, "en") for q in subset]
        zh = []
        for q in subset:
            try:
                tr = llm_provider.complete([
                    {"role": "system", "content":
                     "Translate the user's data-analysis query into Chinese. "
                     "Output only the translation."},
                    {"role": "user", "content": q.nl_description},
                ]).strip()
            except Exception:
                tr = q.nl_description
            zq = Query(q.workload, q.dataset, q.query_id, tr,
                       q.expected_answer, q.data)
            zh.append(run_query(planner, zq, "zh"))
        s_en = summarize(en, "en")
        s_zh = summarize(zh, "zh")
        print(f"\nBilingual (first {len(subset)} queries):")
        print(f"  EN acc = {s_en['accuracy']*100:.1f}%   "
              f"ZH acc = {s_zh['accuracy']*100:.1f}%  "
              f"(Δ = {(s_zh['accuracy']-s_en['accuracy'])*100:+.1f}%)")
        out["bilingual"] = {"en": s_en, "zh": s_zh}

    return {"robustness": out}


# ===================================================================
# Experiment 13: Cross-domain generalization (per-dataset consistency)
# ===================================================================

def run_generalization(llm_provider, queries: List[Query], verbose: bool) -> Dict:
    print("\n" + "#" * 80)
    print("# EXPERIMENT 13: CROSS-DOMAIN GENERALIZATION (per-dataset, no tuning)")
    print("#" * 80)

    planner = QueryPlanner(llm_provider=llm_provider, planning_strategy="titsp")
    rs = [run_query(planner, q, "TiTSP") for q in queries]

    domains: Dict[str, List[RunResult]] = {}
    for r in rs:
        domains.setdefault(f"{r.workload}/{r.dataset}", []).append(r)

    print(f"\n{'domain':<22}{'n':>5}{'Acc':>9}{'Tok/q':>9}{'Time':>10}")
    print("-" * 56)
    rows = []
    for key in sorted(domains):
        s = summarize(domains[key], key)
        rows.append({"domain": key, "n": s["total"], "accuracy": s["accuracy"],
                     "avg_total_tokens": s["avg_total_tokens"],
                     "avg_time_ms": s["avg_time_ms"]})
        print(f"{key:<22}{s['total']:>5}{s['accuracy']*100:>8.1f}%"
              f"{s['avg_total_tokens']:>9.0f}{s['avg_time_ms']:>8.0f}ms")

    accs = [r["accuracy"] for r in rows]
    mean = sum(accs) / len(accs) if accs else 0.0
    std = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5 if accs else 0.0
    print(f"\nCross-domain accuracy: mean = {mean*100:.1f}%, std = {std*100:.1f}% "
          f"(lower std = more consistent / better generalization)")

    return {"generalization": {"per_domain": rows,
                               "acc_mean": round(mean, 4), "acc_std": round(std, 4)}}


# ===================================================================
# Experiment 14: Parameter sensitivity (magic constants)
# ===================================================================

def run_param_sensitivity(llm_provider, queries: List[Query], verbose: bool) -> Dict:
    print("\n" + "#" * 80)
    print("# EXPERIMENT 14: PARAMETER SENSITIVITY (Q-weights & Tchebycheff weights)")
    print("#" * 80)

    from core.query_planner.mcts_plan_search import MCTSLogicalConfig, UserPreference

    # --- MCTS quality weights Q(P)=a*comp + b/(1+cost) + g*conf + d*ocg ---
    qweights = [
        ("default .30/.25/.25/.20", (0.30, 0.25, 0.25, 0.20)),
        ("comp-heavy .55/.15/.15/.15", (0.55, 0.15, 0.15, 0.15)),
        ("cost-heavy .15/.55/.15/.15", (0.15, 0.55, 0.15, 0.15)),
        ("conf-heavy .15/.15/.55/.15", (0.15, 0.15, 0.55, 0.15)),
        ("uniform .25/.25/.25/.25", (0.25, 0.25, 0.25, 0.25)),
    ]
    print(f"\n[A] MCTS quality weights (a,b,g,d)")
    print(f"  {'config':<28}{'Acc':>8}{'PlanQ':>8}{'Cand':>7}{'Tok':>7}")
    qrows = []
    for label, (a, b, g, d) in qweights:
        cfg = MCTSLogicalConfig(completeness_weight=a, cost_weight=b,
                                confidence_weight=g, ocg_weight=d, random_seed=0)
        planner = QueryPlanner(llm_provider=llm_provider, planning_strategy="titsp",
                               mcts_config=cfg)
        rs = [run_query(planner, q, label) for q in queries]
        s = summarize(rs, label); s["weights"] = [a, b, g, d]
        qrows.append(s)
        print(f"  {label:<28}{s['accuracy']*100:>7.1f}%{s['avg_quality']:>8.3f}"
              f"{s['avg_candidates']:>7.0f}{s['avg_total_tokens']:>7.0f}")

    # --- Tchebycheff preference weights w=(latency, cost, confidence) ---
    wpref = [
        ("default .4/.3/.3", (0.4, 0.3, 0.3)),
        ("latency .8/.1/.1", (0.8, 0.1, 0.1)),
        ("cost .1/.8/.1", (0.1, 0.8, 0.1)),
        ("conf .1/.1/.8", (0.1, 0.1, 0.8)),
        ("balanced .34/.33/.33", (0.34, 0.33, 0.33)),
    ]
    print(f"\n[B] Tchebycheff preference weights (latency, cost, conf)")
    print(f"  {'config':<24}{'Acc':>8}{'P-Lat':>8}{'P-Res':>8}{'P-Conf':>8}")
    wrows = []
    for label, (lw, cw, fw) in wpref:
        pref = UserPreference(latency_weight=lw, cost_weight=cw, confidence_weight=fw)
        planner = QueryPlanner(llm_provider=llm_provider, planning_strategy="titsp",
                               preference=pref,
                               mcts_config=MCTSLogicalConfig(random_seed=0))
        rs = [run_query(planner, q, label) for q in queries]
        s = summarize(rs, label); s["weights"] = [lw, cw, fw]
        wrows.append(s)
        print(f"  {label:<24}{s['accuracy']*100:>7.1f}%{s['avg_plan_latency']:>8.2f}"
              f"{s['avg_plan_resource']:>8.2f}{s['avg_plan_confidence']:>8.3f}")

    # accuracy spread = robustness to the constants
    qa = [s["accuracy"] for s in qrows]; wa = [s["accuracy"] for s in wrows]
    print(f"\n  accuracy spread: Q-weights {max(qa)*100-min(qa)*100:.1f} pts, "
          f"w-weights {max(wa)*100-min(wa)*100:.1f} pts")

    return {"param_sensitivity": {"q_weights": qrows, "tcheb_weights": wrows}}


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Extended Experiments")
    parser.add_argument("--llm", default=None, help="LLM provider")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--models", default=None,
                        help="Models for the 'llm' experiment, comma-separated, "
                             "e.g. 'qwen-plus,kimi-k2' or 'name:provider,...'")
    parser.add_argument("--experiment", default="all",
                        choices=["all", "ablation", "scalability", "sensitivity", "llm",
                                 "significance", "costquality", "calibration", "complexity",
                                 "preference", "errors", "optimality", "robustness",
                                 "generalization", "param-sensitivity"],
                        help="Which experiment to run")
    parser.add_argument("--json", default=None, help="Export results to JSON")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--significance-runs", type=int, default=3,
                        help="Number of runs for significance test")
    args = parser.parse_args()

    llm_provider = None
    if args.llm:
        from core.query_planner.llm_query_analyzer import create_provider
        llm_provider = create_provider(args.llm, model=args.model)

    queries = get_all_queries()
    verbose = not args.quiet

    print(f"Extended Experiments")
    print(f"Experiment: {args.experiment}")
    print(f"LLM: {args.llm} / {args.model}")
    print(f"Total queries: {len(queries)}")

    all_experiment_results = {}

    if args.experiment in ("all", "ablation"):
        r = run_ablation(llm_provider, queries, verbose)
        all_experiment_results.update(r)

    if args.experiment in ("all", "scalability"):
        r = run_scalability(llm_provider, queries, verbose)
        all_experiment_results.update(r)

    if args.experiment in ("all", "sensitivity"):
        r = run_sensitivity(llm_provider, queries, verbose)
        all_experiment_results.update(r)

    if args.experiment in ("all", "llm"):
        models = (parse_models(args.models, args.llm or "dashscope")
                  if args.models else None)
        r = run_llm_comparison(llm_provider, queries, verbose, models=models)
        all_experiment_results.update(r)

    if args.experiment in ("all", "significance"):
        r = run_significance(llm_provider, queries, verbose,
                             n_runs=args.significance_runs)
        all_experiment_results.update(r)

    if args.experiment in ("all", "costquality"):
        r = run_cost_quality(llm_provider, queries, verbose,
                             model_name=args.model or "")
        all_experiment_results.update(r)

    if args.experiment in ("all", "calibration"):
        r = run_calibration(llm_provider, queries, verbose)
        all_experiment_results.update(r)

    if args.experiment in ("all", "complexity"):
        r = run_complexity(llm_provider, queries, verbose)
        all_experiment_results.update(r)

    if args.experiment in ("all", "preference"):
        r = run_preference(llm_provider, queries, verbose)
        all_experiment_results.update(r)

    if args.experiment in ("all", "errors"):
        r = run_error_analysis(llm_provider, queries, verbose)
        all_experiment_results.update(r)

    if args.experiment in ("all", "optimality"):
        r = run_optimality(llm_provider, queries, verbose)
        all_experiment_results.update(r)

    if args.experiment in ("all", "robustness"):
        r = run_robustness(llm_provider, queries, verbose)
        all_experiment_results.update(r)

    if args.experiment in ("all", "generalization"):
        r = run_generalization(llm_provider, queries, verbose)
        all_experiment_results.update(r)

    if args.experiment in ("all", "param-sensitivity"):
        r = run_param_sensitivity(llm_provider, queries, verbose)
        all_experiment_results.update(r)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_experiment_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults exported to {args.json}")


if __name__ == "__main__":
    main()
