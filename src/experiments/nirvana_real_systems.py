"""Real Nirvana SYSTEM comparison via the actual pipelines (paper Section 6.2).

For each query, every system runs its REAL planner and executor
(QueryPlanner(planning_strategy=X).plan_and_execute on the real data, with
vision wired for image predicates), and the final answer is scored against an
INDEPENDENT ground truth (deterministic rules, or assistant-labeled vision).
This replaces the earlier prompt-proxy comparison, which only measured LLM
paraphrase noise and was discarded.

Estate, deterministic-gold queries first (no vision oracle needed).

Usage:
  export DASHSCOPE_API_KEY=...
  python3 -m src.experiments.nirvana_real_systems --n 40 --model qwen3.6-plus
"""

import re
import json
import random
import argparse

import pyarrow.parquet as pq

from ..core.query_planner.query_planner import QueryPlanner
from ..core.query_planner.llm_query_analyzer import create_provider

TESTDATA = "data/nirvana_repo/nirvana-main/testdata"
ESTATE_PARQUET = "data/nirvana/estate/estate_1041.parquet"
SYSTEMS = ["titsp", "caesura", "lotus", "thalamusdb", "palimpzest", "nirvana"]


def _bed(d):
    m = re.search(r"(\d+)\s*bedroom", str(d.get("Title", "")), re.I)
    return bool(m) and 3 < int(m.group(1)) < 6


# Deterministic-gold queries per dataset. Each gold rule computes the row set
# from real column values with no system or model involved.
DATASETS = {
    "estate": {
        "load": lambda n: _load_estate(n),
        "keep": ["Title", "Location", "Details"],
        "queries": [
            {"id": "es_loc_ajah", "nl": "Whether the house is located in Ajah, Lagos.",
             "gold": lambda d: "ajah" in str(d.get("Location", "")).lower()},
            {"id": "es_loc_lekki", "nl": "Whether the house is located in Lekki, Lagos.",
             "gold": lambda d: "lekki" in str(d.get("Location", "")).lower()},
            {"id": "es_bed", "nl": "Find estates with more than 3 and less than 6 bedrooms.",
             "gold": _bed},
            # multi-operator (conjunctive): bedrooms 4-5 AND located in Lekki
            {"id": "es_bed_lekki", "nl": "Find estates with more than 3 and less than 6 bedrooms located in Lekki.",
             "gold": lambda d: _bed(d) and "lekki" in str(d.get("Location", "")).lower()},
        ],
    },
    "imdb": {
        "load": lambda n: _load_csv(f"{TESTDATA}/movie_data.csv", n),
        "keep": ["Title", "Genre1", "Genre2", "Genre3", "IMDB_rating", "Plot"],
        "queries": [
            {"id": "im_crime", "nl": "Find movies that belong to the crime genre.",
             "gold": lambda d: any(str(d.get(c, "")).strip().lower() == "crime"
                                   for c in ("Genre1", "Genre2", "Genre3"))},
            {"id": "im_action", "nl": "Find movies that belong to the action genre.",
             "gold": lambda d: any(str(d.get(c, "")).strip().lower() == "action"
                                   for c in ("Genre1", "Genre2", "Genre3"))},
            {"id": "im_rating85", "nl": "Find movies whose IMDB rating is higher than 8.5.",
             "gold": lambda d: (lambda v: v is not None and v > 8.5)(_num(d.get("IMDB_rating")))},
            # multi-operator (conjunctive) query: tests whether the planner applies
            # BOTH filters. A planner that drops one filter scores poorly.
            {"id": "im_crime_hi", "nl": "Find crime movies whose IMDB rating is higher than 8.",
             "gold": lambda d: any(str(d.get(c, "")).strip().lower() == "crime"
                                   for c in ("Genre1", "Genre2", "Genre3"))
                               and (lambda v: v is not None and v > 8)(_num(d.get("IMDB_rating")))},
        ],
    },
    "steam": {
        "load": lambda n: _load_csv(f"{TESTDATA}/steam_games.csv", n),
        "keep": ["title", "platforms", "genre", "rating", "language"],
        "queries": [
            {"id": "st_action", "nl": "Find games in the action genre.",
             "gold": lambda d: "action" in str(d.get("genre", "")).lower()},
            {"id": "st_adventure", "nl": "Find games in the adventure genre.",
             "gold": lambda d: "adventure" in str(d.get("genre", "")).lower()},
            {"id": "st_pegi18", "nl": "Find games whose PEGI age rating is 18 (adults only).",
             "gold": lambda d: "18" in str(d.get("rating", ""))},
            # multi-operator (conjunctive): action genre AND PEGI 18
            {"id": "st_action_18", "nl": "Find action games whose PEGI age rating is 18 (adults only).",
             "gold": lambda d: "action" in str(d.get("genre", "")).lower() and "18" in str(d.get("rating", ""))},
        ],
    },
}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _load_estate(n):
    full = pq.ParquetFile(ESTATE_PARQUET).read().to_pandas()
    idx = list(range(len(full))); random.Random(20260612).shuffle(idx); idx = sorted(idx[: n])
    return full.iloc[idx].reset_index(drop=True)


def _load_csv(path, n):
    import pandas as pd
    full = pd.read_csv(path, low_memory=False)
    idx = list(range(len(full))); random.Random(20260612).shuffle(idx); idx = sorted(idx[: n])
    return full.iloc[idx].reset_index(drop=True)


def f1(got, gold):
    tp = len(got & gold); fp = len(got - gold); fn = len(gold - got)
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--model", default="qwen3.6-plus")
    ap.add_argument("--datasets", default="estate,imdb,steam")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    txt = create_provider("dashscope", model=args.model)
    planners = {s: QueryPlanner(llm_provider=txt, planning_strategy=s) for s in SYSTEMS}
    sysF1 = {s: [] for s in SYSTEMS}
    results = []
    print(f"Real Nirvana SYSTEM comparison via actual pipelines (independent gold). "
          f"datasets={args.datasets} n={args.n} model={args.model}\n")

    for ds in args.datasets.split(","):
        cfg = DATASETS[ds]
        df = cfg["load"](args.n)
        data = [{"_id": i, **{k: (str(r[k])[:150] if k == "Details" else r[k]) for k in cfg["keep"]}}
                for i, (_, r) in enumerate(df.iterrows())]
        print(f"--- {ds} (rows={len(data)}) ---")
        for q in cfg["queries"]:
            gold = {d["_id"] for d in data if q["gold"](d)}
            rec = {"dataset": ds, "id": q["id"], "gold_n": len(gold), "arms": {}}
            for s in SYSTEMS:
                try:
                    res = planners[s].plan_and_execute(q["nl"], data)
                    got = {r.get("_id") for r in res.execution.data if isinstance(r, dict) and "_id" in r}
                    sc = f1(got, gold)
                except Exception as e:
                    sc = 0.0; rec["arms"][s] = {"f1": 0.0, "err": type(e).__name__}
                else:
                    rec["arms"][s] = {"got_n": len(got), "f1": round(sc, 3)}
                sysF1[s].append(sc)
            results.append(rec)
            cols = "  ".join(f"{s[:5]}={rec['arms'][s]['f1']:.2f}" for s in SYSTEMS)
            print(f"  [{q['id']:12s} gold={len(gold):2d}] {cols}")

    print("\n=== mean F1 vs independent gold (real Nirvana, actual pipelines, all queries) ===")
    for s in SYSTEMS:
        tag = "  <- TiMMInsight (ours)" if s == "titsp" else " [baseline]"
        print(f"  {s:11s}: {sum(sysF1[s])/len(sysF1[s]):.3f}{tag}")
    if args.json:
        json.dump({"meanF1": {s: sum(sysF1[s])/len(sysF1[s]) for s in SYSTEMS}, "results": results},
                  open(args.json, "w"), indent=2)
        print(f"saved -> {args.json}")


if __name__ == "__main__":
    main()
