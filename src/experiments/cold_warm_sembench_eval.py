"""Cold-vs-warm history calibration on SemBench (paper Eq. 3 / Eq. 9).

Companion to cold_warm_history_eval.py (Nirvana). Uses the sembench_workload
harness -- the same SemBench queries + validate_answer that the
extended_experiments ablations use -- so the calibration ablation is consistent
with the rest of the ablation suite. Correctness is the harness's binary
validate_answer (substring / numeric match), which is the correctness signal for
the reliability / ECE analysis. Warm history is built leave-one-out, identical
to the Nirvana driver.

Caveat: sembench_workload's scenarios (movies / wildlife / ecommerce / medical /
mmqa) and its substring-match scoring are the repo's adapted SemBench, not the
graded Cars/EComm/MMQA/Movie pipeline behind the main Fig. 5 numbers. This driver
is for the calibration ablation only.

Run from planning/ with DASHSCOPE_API_KEY set:
  python3.12 -m src.experiments.cold_warm_sembench_eval --scenario movies --out cw_sb_movies.json
"""

import sys
import json
import signal
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.query_planner.query_planner import QueryPlanner
from core.query_planner.llm_query_analyzer import create_provider
from core.query_planner.execution_history import ExecutionHistoryManager
from experiments.sembench_workload import (
    SEMBENCH_QUERIES, DATASET_REGISTRY, validate_answer,
)
# reuse the tested calibration/report helpers from the Nirvana driver
from experiments.cold_warm_history_eval import (
    reliability, plan_confidence, plan_signature,
)


class _QueryTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _QueryTimeout()


signal.signal(signal.SIGALRM, _alarm_handler)
QUERY_TIMEOUT_S = 200


def run_one(planner, query, rows):
    signal.alarm(QUERY_TIMEOUT_S)
    try:
        return planner.plan_and_execute(query, rows)
    finally:
        signal.alarm(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True,
                    help="movies | wildlife | ecommerce | medical | mmqa")
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--llm", default="dashscope")
    ap.add_argument("--correct-threshold", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    provider = create_provider(args.llm, model=args.model)
    queries = [q for q in SEMBENCH_QUERIES
               if q.scenario == args.scenario and q.expected_answer]
    rows = DATASET_REGISTRY.get(args.scenario, [])
    if not queries:
        print(f"no scoreable queries for scenario={args.scenario}", flush=True)
        return
    print(f"SemBench {args.scenario}: {len(queries)} queries, {len(rows)} rows, model={args.model}", flush=True)

    def score(res, expected):
        return 1.0 if validate_answer(res.answer, res.execution.data, expected) else 0.0

    corpus = ExecutionHistoryManager()

    # ---- Pass A: COLD (priors only), harvest history ----
    print("=== PASS A: COLD (history_manager=None) ===", flush=True)
    cold = {}
    for q in queries:
        try:
            planner = QueryPlanner(llm_provider=provider, planning_strategy="titsp",
                                   history_manager=None)
            res = run_one(planner, q.nl_description, rows)
            s = score(res, q.expected_answer)
            corpus.record_from_execution(res.plan.best_logical_plan,
                                         res.execution.operator_results, query_id=q.query_id)
            cold[q.query_id] = {"score": s, "conf": plan_confidence(res), "sig": plan_signature(res)}
            print(f"  {q.query_id}: cold={s}", flush=True)
        except Exception as e:
            cold[q.query_id] = {"score": 0.0, "conf": None, "sig": None, "error": str(e)[:80]}
            print(f"  {q.query_id}: COLD ERROR {str(e)[:80]}", flush=True)
    print(f"  corpus after cold pass: {corpus.stats()}", flush=True)

    # ---- Pass B: WARM (leave-one-out history) ----
    print("=== PASS B: WARM (leave-one-out history) ===", flush=True)
    warm = {}
    for q in queries:
        try:
            hist = corpus.snapshot_excluding(q.query_id)
            planner = QueryPlanner(llm_provider=provider, planning_strategy="titsp",
                                   history_manager=hist)
            res = run_one(planner, q.nl_description, rows)
            s = score(res, q.expected_answer)
            warm[q.query_id] = {"score": s, "conf": plan_confidence(res), "sig": plan_signature(res),
                                "hist": hist.stats()}
            print(f"  {q.query_id}: warm={s}", flush=True)
        except Exception as e:
            warm[q.query_id] = {"score": 0.0, "conf": None, "sig": None, "error": str(e)[:80]}
            print(f"  {q.query_id}: WARM ERROR {str(e)[:80]}", flush=True)

    # ---- report (mirrors the Nirvana driver) ----
    common = [q for q in cold if q in warm]
    thr = args.correct_threshold
    cs = [cold[q]["score"] for q in common]
    ws = [warm[q]["score"] for q in common]
    changed = sum(1 for q in common if cold[q]["sig"] != warm[q]["sig"])
    cold_ece, cold_brier, cold_bins = reliability(
        [(cold[q]["conf"], cold[q]["score"] >= thr) for q in common])
    warm_ece, warm_brier, warm_bins = reliability(
        [(warm[q]["conf"], warm[q]["score"] >= thr) for q in common])
    acc_c = sum(1 for x in cs if x >= thr) / len(cs) if cs else 0.0
    acc_w = sum(1 for x in ws if x >= thr) / len(ws) if ws else 0.0
    d_ece = None if (cold_ece is None or warm_ece is None) else round(warm_ece - cold_ece, 4)

    print("\n=== COLD vs WARM (leave-one-out) ===", flush=True)
    print(f"  n={len(common)}", flush=True)
    print(f"  PRIMARY  ECE (lower=better): cold={cold_ece}  warm={warm_ece}  Δ={d_ece}", flush=True)
    print(f"           Brier             : cold={cold_brier}  warm={warm_brier}", flush=True)
    print(f"  SECOND   accuracy@{thr}    : cold={acc_c:.3f}  warm={acc_w:.3f}  Δ={acc_w-acc_c:+.3f}", flush=True)
    print(f"  plans changed by warm start: {changed}/{len(common)}", flush=True)

    if args.out:
        json.dump({
            "family": "sembench", "scenario": args.scenario, "model": args.model,
            "protocol": "leave-one-out", "n": len(common),
            "cold_ece": cold_ece, "warm_ece": warm_ece, "delta_ece": d_ece,
            "cold_brier": cold_brier, "warm_brier": warm_brier,
            "cold_bins": cold_bins, "warm_bins": warm_bins,
            "acc_cold": acc_c, "acc_warm": acc_w, "plans_changed": changed,
            "per_query": {q: {"cold": cold[q], "warm": warm[q]} for q in common},
        }, open(args.out, "w"), indent=2, default=str)
        print(f"  saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
