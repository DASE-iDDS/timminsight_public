"""Cold-vs-warm history calibration experiment (paper Eq. 3 / Eq. 9).

Measures whether turning ON history-based calibration (warm start) changes
end-to-end accuracy relative to the hand-set cold-start priors, on real
Nirvana data scored with the same LLM-judge / relative-error protocol as the
main results.

Protocol (leak-free leave-one-out):
  Pass A (COLD): every query is planned with history_manager=None (pure priors
    w0 / c0). Its answer is scored, AND its per-operator / per-edge outcomes are
    harvested into a shared corpus, tagged by query id.
  Pass B (WARM): every query is re-planned with a history manager built from the
    corpus with that query's OWN records removed (snapshot_excluding), so a query
    is never calibrated on its own execution. Its answer is re-scored.
  Report mean(cold) vs mean(warm), the per-query delta, how many selected plans
    changed, and a coarse ECE of the plan confidence against answer correctness.

Only the full TiTSP planner is exercised; baselines are irrelevant here because
calibration is a TiTSP-internal mechanism. Vision is disabled (Nirvana gold is
text-derived), so the two steam cover-image queries are scored from text in BOTH
passes; the cold-vs-warm delta is therefore still a fair paired comparison.

Run from the planning/ directory with DASHSCOPE_API_KEY in the environment:
  python3.12 -m src.experiments.cold_warm_history_eval --domain steam --out cw_steam.json
"""

import os
import sys
import json
import re
import time
import signal
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.query_planner.query_planner import QueryPlanner
from core.query_planner.llm_query_analyzer import create_provider
from core.query_planner.execution_history import ExecutionHistoryManager


class _QueryTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _QueryTimeout()


signal.signal(signal.SIGALRM, _alarm_handler)
QUERY_TIMEOUT_S = 200

NIR = "data/nirvana_repo/nirvana-main"
DATA = {
    "imdb":   f"{NIR}/testdata/movie_data.csv",
    "steam":  f"{NIR}/testdata/steam_games.csv",
    "estate": f"{NIR}/testdata/multimodal_real_estate.parquet",
}
GOLD_DIR = lambda d: f"{NIR}/workloads/{d}_output"

JUDGE_PROMPT = """Here are a golden analysis result obtained by a golden data processing plan and a result derived from an alternative data processing plan.
Evaluate the two data analysis results and return a rating between 0 and 10, where 0 means the two results are completely different and 10 means they are exactly the same.

Ground truth:
{ground_truth}

Result from the alternative plan:
{result}

You should carefully consider all values (and their semantics) in both analysis results. The rating score is enclosed within <score></score> tags, i.e., <score>Rating Score</score>."""


# --- scoring core (identical protocol to results_canonical/score_vs_gold.py) ---

def _is_img_col(c):
    cl = str(c).lower()
    return "image" in cl or cl in ("poster", "rating", "url")


def load_rows(domain, sample=0):
    path = DATA[domain]
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    if sample and sample > 0:
        df = df.head(sample)
    return df.to_dict("records")


def load_gold(path):
    clean = path.replace(".csv", ".clean.csv")
    if os.path.exists(clean):
        return pd.read_csv(clean)
    df = pd.read_csv(path, usecols=lambda c: not _is_img_col(c))
    df.to_csv(clean, index=False)
    return df


def strip_img(rows):
    if not isinstance(rows, list):
        return rows
    return [{k: v for k, v in r.items()
             if not _is_img_col(k) and not isinstance(v, (bytes, bytearray))}
            for r in rows]


def serialize(obj):
    if isinstance(obj, pd.DataFrame):
        return "No data in the output." if obj.empty else obj.to_json(orient="records", lines=True).strip()
    if isinstance(obj, list):
        return "No data in the output." if not obj else "\n".join(
            json.dumps(r, default=str, ensure_ascii=False) for r in obj[:200])
    if isinstance(obj, dict):
        return json.dumps(obj, default=str, ensure_ascii=False, indent=1)
    return str(obj)


def agg_gold_num(gold_df):
    try:
        if len(gold_df) == 1:
            return float(str(gold_df.iloc[0, 0]).replace(",", ""))
    except Exception:
        pass
    return None


def num_gold_for(query, gold_df):
    gnum = agg_gold_num(gold_df)
    if gnum is None:
        return None
    ql = str(query).lower()
    if re.search(r"\b(count|average|avg|mean|sum|total|how many|number of|"
                 r"maximum|minimum|lowest|highest|compute|ratio)\b", ql) and not re.search(
                 r"\b(genre|character|name|title|director|actor|who|main character)\b", ql):
        return gnum
    return None


def _all_nums(s):
    out = []
    for n in re.findall(r"\d{1,3}(?:[ ,]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?", str(s)):
        try:
            out.append(float(n.replace(",", "").replace(" ", "")))
        except Exception:
            pass
    return out


def num_score(gold_val, answer, table):
    if abs(gold_val) < 1.0:
        txt = f"{answer} {serialize(table) if table is not None else ''}".lower()
        cites_price = bool(re.search(r"(rp|usd|idr|\$|€|£)\s*[1-9]", txt))
        conveys_zero = bool(re.search(r"\bfree\b|\bzero\b|\b0\b|\bno\b[^.]{0,25}\b(paid|priced|match|result|game|found)", txt))
        return 1.0 if (conveys_zero and not cites_price) else 0.0
    nums = _all_nums(answer) or (_all_nums(serialize(table)) if table is not None else [])
    if not nums:
        return 0.0
    best = min(nums, key=lambda v: abs(v - gold_val))
    rel = abs(best - gold_val) / (abs(gold_val) + 1e-9)
    return round(1.0 / (1.0 + rel), 3) if rel < 0.5 else 0.0


def judge(provider, gold, result):
    prompt = JUDGE_PROMPT.format(ground_truth=serialize(gold)[:8000], result=serialize(result)[:8000])
    for _ in range(4):
        try:
            txt = provider.complete([{"role": "user", "content": prompt}])
            m = re.search(r"<score>(.*?)</score>", txt, re.DOTALL)
            if m:
                return max(0.0, min(1.0, float(re.findall(r"[\d.]+", m.group(1))[0]) / 10.0))
        except Exception:
            time.sleep(2)
    return 0.0


def load_nl(domain):
    nl = {}
    for mf in ("gold_manifest.json", "gold_manifest_s100.json", "imdb_gold_s100.json"):
        p = f"{NIR}/{mf}"
        if not os.path.exists(p):
            continue
        try:
            man = json.load(open(p))
        except Exception:
            continue
        for m in man:
            gold_ok = os.path.exists(f"{NIR}/workloads/{domain}_output/q{m.get('q')}_out_wolo_wopo.csv")
            if m.get("domain") == domain and ("error" not in m or gold_ok):
                nl[m["q"]] = re.sub(r"^Q\d+:\s*", "", m.get("nl", "")).strip()
    return nl


# --- cold-vs-warm driver ------------------------------------------------------

def run_one(planner, query, rows):
    """Plan+execute one query; return (score_inputs, plan handles) or raise."""
    signal.alarm(QUERY_TIMEOUT_S)
    try:
        res = planner.plan_and_execute(query, rows)
    finally:
        signal.alarm(0)
    return res


def score_res(res, query, gold):
    out = getattr(getattr(res, "execution", None), "data", None)
    answer = getattr(res, "answer", "")
    result_repr = {
        "final_answer": answer,
        "result_table": (strip_img(out[:200]) if isinstance(out, list) else None),
    }
    gnum = num_gold_for(query, gold)
    if gnum is not None:
        return num_score(gnum, answer, out)
    return None, result_repr  # judge deferred (needs provider)


def plan_confidence(res):
    try:
        return float(res.plan.selected_physical.objectives.confidence)
    except Exception:
        return None


def plan_signature(res):
    try:
        return tuple(n.operator_type for n in res.plan.best_logical_plan.get_all_operators())
    except Exception:
        return None


def reliability(pairs, n_bins=10):
    """Return (ece, brier, bins) from (confidence, correct_bool) pairs.

    ECE is the standard binned expected calibration error (lower is better);
    brier is the mean squared error between confidence and correctness; bins is
    the reliability diagram, one entry per confidence bin with its average
    predicted confidence and empirical accuracy. This is the PRIMARY output of
    the cold-vs-warm experiment: warm-start (Eq. 3 / Eq. 9) is expected to make
    the plan confidence track correctness more closely, i.e. lower ECE.
    """
    data = [(c, 1.0 if ok else 0.0) for c, ok in pairs if c is not None and ok is not None]
    if not data:
        return None, None, []
    n = len(data)
    e = 0.0
    bins = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        members = [(c, y) for c, y in data if (lo < c <= hi) or (b == 0 and c <= hi)]
        if not members:
            bins.append({"range": [round(lo, 1), round(hi, 1)], "count": 0,
                         "avg_conf": None, "accuracy": None})
            continue
        avg_conf = sum(c for c, _ in members) / len(members)
        acc = sum(y for _, y in members) / len(members)
        e += (len(members) / n) * abs(avg_conf - acc)
        bins.append({"range": [round(lo, 1), round(hi, 1)], "count": len(members),
                     "avg_conf": round(avg_conf, 3), "accuracy": round(acc, 3)})
    brier = sum((c - y) ** 2 for c, y in data) / n
    return round(e, 4), round(brier, 4), bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="steam")
    ap.add_argument("--queries", nargs="*", type=int, default=None)
    ap.add_argument("--model", default="glm-5.2", help="SYSTEM backbone")
    ap.add_argument("--llm", default="dashscope")
    ap.add_argument("--judge-model", default="glm-5.2")
    ap.add_argument("--judge-llm", default="dashscope")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--correct-threshold", type=float, default=0.5,
                    help="score >= threshold counts as correct (binary accuracy / ECE)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    provider = create_provider(args.llm, model=args.model)
    judge_provider = create_provider(args.judge_llm, model=args.judge_model)
    rows = load_rows(args.domain, args.sample)
    nl = load_nl(args.domain)
    qids = args.queries or sorted(nl.keys())

    def scored(res, q, gold):
        s = score_res(res, q, gold)
        if isinstance(s, tuple):          # judge path
            _, repr_ = s
            return judge(judge_provider, gold, repr_)
        return s                           # numeric path

    corpus = ExecutionHistoryManager()
    rowset = []

    # ---- Pass A: COLD (priors only), harvest history ----
    print("=== PASS A: COLD (history_manager=None) ===", flush=True)
    cold = {}
    for q in qids:
        gcsv = f"{GOLD_DIR(args.domain)}/q{q}_out_wolo_wopo.csv"
        if q not in nl or not os.path.exists(gcsv):
            continue
        gold = load_gold(gcsv)
        query = nl[q]
        try:
            planner = QueryPlanner(llm_provider=provider, planning_strategy="titsp",
                                   history_manager=None)
            res = run_one(planner, query, rows)
            s = scored(res, query, gold)
            corpus.record_from_execution(res.plan.best_logical_plan,
                                         res.execution.operator_results, query_id=f"q{q}")
            cold[q] = {"score": s, "conf": plan_confidence(res), "sig": plan_signature(res)}
            print(f"  q{q}: cold={s:.2f}", flush=True)
        except Exception as e:
            cold[q] = {"score": 0.0, "conf": None, "sig": None, "error": str(e)[:80]}
            print(f"  q{q}: COLD ERROR {str(e)[:80]}", flush=True)
    print(f"  corpus after cold pass: {corpus.stats()}", flush=True)

    # ---- Pass B: WARM (leave-one-out history) ----
    print("=== PASS B: WARM (leave-one-out history) ===", flush=True)
    warm = {}
    for q in qids:
        gcsv = f"{GOLD_DIR(args.domain)}/q{q}_out_wolo_wopo.csv"
        if q not in nl or not os.path.exists(gcsv):
            continue
        gold = load_gold(gcsv)
        query = nl[q]
        try:
            hist = corpus.snapshot_excluding(f"q{q}")
            planner = QueryPlanner(llm_provider=provider, planning_strategy="titsp",
                                   history_manager=hist)
            res = run_one(planner, query, rows)
            s = scored(res, query, gold)
            warm[q] = {"score": s, "conf": plan_confidence(res), "sig": plan_signature(res),
                       "hist": hist.stats()}
            print(f"  q{q}: warm={s:.2f}", flush=True)
        except Exception as e:
            warm[q] = {"score": 0.0, "conf": None, "sig": None, "error": str(e)[:80]}
            print(f"  q{q}: WARM ERROR {str(e)[:80]}", flush=True)

    # ---- report ----
    common = [q for q in qids if q in cold and q in warm]
    cs = [cold[q]["score"] for q in common]
    ws = [warm[q]["score"] for q in common]
    thr = args.correct_threshold
    changed = sum(1 for q in common if cold[q]["sig"] != warm[q]["sig"])
    cold_pairs = [(cold[q]["conf"], cold[q]["score"] >= thr) for q in common]
    warm_pairs = [(warm[q]["conf"], warm[q]["score"] >= thr) for q in common]
    cold_ece, cold_brier, cold_bins = reliability(cold_pairs)
    warm_ece, warm_brier, warm_bins = reliability(warm_pairs)

    mean_c = sum(cs) / len(cs) if cs else 0.0
    mean_w = sum(ws) / len(ws) if ws else 0.0
    acc_c = sum(1 for x in cs if x >= thr) / len(cs) if cs else 0.0
    acc_w = sum(1 for x in ws if x >= thr) / len(ws) if ws else 0.0
    d_ece = None if (cold_ece is None or warm_ece is None) else round(warm_ece - cold_ece, 4)

    print("\n=== COLD vs WARM (leave-one-out) ===", flush=True)
    print(f"  n={len(common)}", flush=True)
    print(f"  PRIMARY  ECE (plan-conf vs correctness, lower=better): "
          f"cold={cold_ece}  warm={warm_ece}  Δ={d_ece}", flush=True)
    print(f"           Brier (lower=better)                        : "
          f"cold={cold_brier}  warm={warm_brier}", flush=True)
    print(f"  SECOND   accuracy@{thr}                              : "
          f"cold={acc_c:.3f}  warm={acc_w:.3f}  Δ={acc_w-acc_c:+.3f}", flush=True)
    print(f"           mean graded score                          : "
          f"cold={mean_c:.3f}  warm={mean_w:.3f}  Δ={mean_w-mean_c:+.3f}", flush=True)
    print(f"  plans changed by warm start: {changed}/{len(common)}", flush=True)
    print("  reliability diagram (bin -> count : avg_conf -> empirical_acc):", flush=True)
    for cb, wb in zip(cold_bins, warm_bins):
        if cb["count"] == 0 and wb["count"] == 0:
            continue
        print(f"    {cb['range']}:  cold n={cb['count']} {cb['avg_conf']}->{cb['accuracy']}"
              f"   |   warm n={wb['count']} {wb['avg_conf']}->{wb['accuracy']}", flush=True)

    if args.out:
        json.dump({
            "domain": args.domain, "model": args.model, "protocol": "leave-one-out",
            "n": len(common),
            "cold_ece": cold_ece, "warm_ece": warm_ece, "delta_ece": d_ece,
            "cold_brier": cold_brier, "warm_brier": warm_brier,
            "cold_bins": cold_bins, "warm_bins": warm_bins,
            "acc_cold": acc_c, "acc_warm": acc_w,
            "mean_cold": mean_c, "mean_warm": mean_w,
            "delta_score": mean_w - mean_c, "plans_changed": changed,
            "per_query": {q: {"cold": cold[q], "warm": warm[q]} for q in common},
        }, open(args.out, "w"), indent=2, default=str)
        print(f"  saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
