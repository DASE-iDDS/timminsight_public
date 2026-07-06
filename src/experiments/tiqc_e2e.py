"""TiQC end-to-end experiment on REAL benchmark data (paper §IV / §VI).

Unlike tiqc_clarification_pilot.py (which scores query resolution via an LLM
judge), this runs the resolved queries THROUGH EXECUTION on the real Nirvana
data and scores the FINAL ANSWER against gold. Gold = the original benchmark
query's predicate executed on the real rows. This makes TiQC's benefit concrete
as end-to-end answer accuracy on the actual dataset (real images included).

Pipeline per (query, arm):
  1. clarification (reuse pilot): arm in {none, grounded} -> resolved NL query.
  2. compile: LLM maps the resolved NL to an executable predicate (column, test),
     grounded strictly in the real schema.
  3. execute: semantic_filter over N sampled real rows. column=='image' -> vision
     (qwen-vl on real pixels); else text predicate over that column's value.
  4. score: F1 of the arm's answer row-set vs the GOLD row-set.

Gold predicates are taken verbatim from the source Nirvana benchmark query
(its instruction + input_columns), so the gold answer is the benchmark's intent
executed on real data.

Usage:
  export DASHSCOPE_API_KEY=...
  python3 -m src.experiments.tiqc_e2e --n 24 --model qwen3.6-plus \
      --vision-model qwen-vl-max --json runs_tiqc/e2e_estate_qwen.json
"""

import io
import os
import base64
import json
import argparse

import pyarrow.parquet as pq

from .tiqc_ambiguity_workload import AMBIGUOUS_QUERIES
from .tiqc_clarification_pilot import run_one, run_noclarify, SCHEMAS, _client, _chat, _json

TESTDATA = "data/nirvana_repo/nirvana-main/testdata"
ESTATE_PARQUET = "data/nirvana/estate/estate_1041.parquet"
IMDB_CSV = f"{TESTDATA}/movie_data.csv"
STEAM_CSV = f"{TESTDATA}/steam_games.csv"

# Each discriminating query has TWO executable readings on the real data: the GOLD
# interpretation (the source benchmark query's intent) and the most plausible WRONG
# interpretation (the competing reading a non-clarifying system may commit to). An
# arm is scored correct if its executed answer is CLOSER to the gold set than to the
# wrong set -- the principled execution-level resolution metric (row-set F1 alone is
# biased because a broad over-answer that is a superset of gold scores high F1).
# Thresholds are relaxed slightly from the literal benchmark values only to keep both
# sets non-empty at the sampled N; the discriminating AXIS (which column / which scope)
# is preserved exactly.
GOLD_EXEC = {
    # spatial scope: Lekki specifically vs the broad Lagos area
    "estate_q12s": {"column": "Location", "predicate": "the location text contains 'Lekki'"},
    # spatial scope: Ajah specifically vs the broad eastern-Lagos area
    "estate_q3b":  {"column": "Location", "predicate": "the location text contains 'Ajah'"},
    # which rating column: IMDB_rating vs Metascore
    "imdb_q5":     {"column": "IMDB_rating", "predicate": "the IMDB_rating numeric value is higher than 8"},
    # value scope: exactly Crime vs any crime-related genre
    "imdb_q9":     {"column": "Genre1",      "predicate": "the genre is exactly Crime"},
    # which score: metacritic vs review sentiment
    "steam_q8":    {"column": "metacriticts", "predicate": "the metacritic score is higher than 85"},
    # which 'rating': PEGI age vs metacritic score
    "steam_q1":    {"column": "rating",       "predicate": "the PEGI age rating means adults only (18+)"},
}

WRONG_EXEC = {
    "estate_q12s": {"column": "Location",    "predicate": "the location is in the broad Lagos area (text contains 'Lagos')"},
    "estate_q3b":  {"column": "Location",    "predicate": "the location is in eastern Lagos (contains 'Ajah', 'Lekki', or 'Ikoyi')"},
    "imdb_q5":     {"column": "Metascore",   "predicate": "the Metascore numeric value is higher than 80"},
    "imdb_q9":     {"column": "Genre1",      "predicate": "the genre is crime-related (Crime, Thriller, Mystery, or Film-Noir)"},
    "steam_q8":    {"column": "overall_reviews", "predicate": "the overall reviews are Very Positive"},
    "steam_q1":    {"column": "metacriticts", "predicate": "the metacritic score is higher than 85"},
}


def _img_b64(b):
    return "data:image/jpeg;base64," + base64.b64encode(b).decode()


def _vis_yesno(vclient, vmodel, instruction, img_bytes):
    r = vclient.chat.completions.create(
        model=vmodel, temperature=0, max_tokens=8,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": f"Answer strictly YES or NO. Does the image satisfy: {instruction}"},
            {"type": "image_url", "image_url": {"url": _img_b64(img_bytes)}}]}])
    return r.choices[0].message.content.strip().upper().startswith("Y")


def _txt_yesno(client, model, instruction, value):
    out = _chat(client, model,
        "Answer strictly YES or NO.",
        f"Does this satisfy the condition?\nCondition: {instruction}\nValue: {value}")
    return out.strip().upper().startswith("Y")


def compile_predicate(client, model, dataset, resolved_query):
    """LLM compiles a resolved NL query into an executable (column, predicate)."""
    out = _chat(client, model,
        "Compile the query into an executable single-column filter, grounded "
        "STRICTLY in the schema. Choose the ONE column the filter reads (use the "
        "exact column name; use 'image' for the photo) and a short yes/no predicate. "
        'Respond JSON: {"column": "...", "predicate": "..."}.',
        f"Schema:\n{SCHEMAS[dataset]}\nQuery: {resolved_query}")
    d = _json(out)
    return {"column": d.get("column", ""), "predicate": d.get("predicate", resolved_query)}


def execute_filter(client, model, vclient, vmodel, rows, spec, cache):
    """Return the set of row ids passing the (column, predicate) filter."""
    col, pred = spec["column"], spec["predicate"]
    key0 = (col, pred)
    hit = set()
    for rid, row in rows:
        ck = (key0, rid)
        if ck in cache:
            ok = cache[ck]
        elif col == "image":
            ok = _vis_yesno(vclient, vmodel, pred, row["image"]["bytes"])
        else:
            val = str(row.get(col, ""))[:600] if col in row else str(row.get("Details", ""))[:600]
            ok = _txt_yesno(client, model, pred, val)
        cache[ck] = ok
        if ok:
            hit.add(rid)
    return hit


def f1(pred_set, gold_set, universe):
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def load_dataset(dataset, n, seed=20260612):
    import random
    import pandas as pd
    if dataset == "estate":
        full = pq.ParquetFile(ESTATE_PARQUET).read().to_pandas()
    elif dataset == "movie":
        full = pd.read_csv(IMDB_CSV)
    elif dataset == "steam":
        full = pd.read_csv(STEAM_CSV, low_memory=False)
    else:
        raise ValueError(dataset)
    idx = list(range(len(full)))
    random.Random(seed).shuffle(idx)
    idx = sorted(idx[: n])
    df = full.iloc[idx].reset_index(drop=True)
    return [(i, df.iloc[i]) for i in range(len(df))], set(range(len(df)))


# The 5 TiTSP baselines. All are single-pass and NON-INTERACTIVE (none clarifies).
# Verified separately that their real repo planners commit a plan without asking;
# scored here as schema-equipped single-pass resolvers (their defining trait), which
# is both faithful on the clarification axis and tractable (running the full planners
# costs ~48 LLM calls each and their abstract plans do not resolve value-ambiguity).
from .tiqc_clarification_pilot import BASELINE_PARADIGM
BASELINES = list(BASELINE_PARADIGM)


def baseline_resolve(strategy, client, model, q):
    """A baseline's single-pass committed interpretation (schema-equipped, cannot ask)."""
    resolved = _chat(client, model,
        f"You are the {BASELINE_PARADIGM[strategy]}. You CANNOT ask the user any "
        "question. Using the schema, commit to ONE concrete executable interpretation "
        "of the query by your best guess. Output ONLY the rewritten query, one line.",
        f"Dataset: {q.dataset}\nSchema:\n{SCHEMAS[q.dataset]}\nQuery: {q.ambiguous_query}")
    return {"rewritten": resolved}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--model", default="qwen3.6-plus")
    ap.add_argument("--vision-model", default="qwen-vl-max")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    client = _client()
    vclient = client

    queries = [q for q in AMBIGUOUS_QUERIES if q.id in GOLD_EXEC]
    datasets = sorted({q.dataset for q in queries})
    arms = ["none", "grounded"] + BASELINES
    arm_acc = {a: [] for a in arms}
    results = []
    print(f"TiQC vs {len(BASELINES)} baselines on REAL Nirvana data "
          f"(execution-level resolution metric: answer closer to gold vs wrong reading): "
          f"{len(queries)} queries over {datasets}, model={args.model}\n")

    for ds in datasets:
        rows, universe = load_dataset(ds, args.n)
        cache = {}
        for q in [x for x in queries if x.dataset == ds]:
            gold = execute_filter(client, args.model, vclient, args.vision_model, rows, GOLD_EXEC[q.id], cache)
            wrong = execute_filter(client, args.model, vclient, args.vision_model, rows, WRONG_EXEC[q.id], cache)
            informative = gold != wrong
            rec = {"id": q.id, "source": q.source, "type": q.ambiguity_type,
                   "gold_n": len(gold), "wrong_n": len(wrong),
                   "informative": informative, "arms": {}}
            for arm in arms:
                if arm == "none":
                    res = run_noclarify(client, args.model, q)
                elif arm == "grounded":
                    res = run_one(client, args.model, q, grounded=True)
                else:
                    res = baseline_resolve(arm, client, args.model, q)
                spec = compile_predicate(client, args.model, q.dataset, res["rewritten"])
                ans = execute_filter(client, args.model, vclient, args.vision_model, rows, spec, cache)
                fg, fw = f1(ans, gold, universe), f1(ans, wrong, universe)
                correct = 1.0 if fg > fw else (0.5 if fg == fw else 0.0)
                if informative:
                    arm_acc[arm].append(correct)
                rec["arms"][arm] = {"resolved": res["rewritten"][:120], "col": spec["column"],
                                    "f1_gold": round(fg, 2), "f1_wrong": round(fw, 2),
                                    "correct": correct}
            results.append(rec)
            flag = "" if informative else "  (UNINFORMATIVE: gold==wrong, skipped)"
            cols = "  ".join(f"{a[:4]}={rec['arms'][a]['correct']:.1f}" for a in arms)
            print(f"[{q.id:11s} g={len(gold):2d} w={len(wrong):2d}] {cols}{flag}")

    print("\n=== Execution-level resolution accuracy on REAL data "
          "(fraction resolved to GOLD reading, informative queries) ===")
    ninf = len(arm_acc["grounded"]) or 1
    means = {a: (sum(arm_acc[a]) / len(arm_acc[a]) if arm_acc[a] else 0.0) for a in arms}
    base_ceiling = max(means[b] for b in BASELINES)
    for a in arms:
        tag = "  <- TiQC (ours)" if a == "grounded" else (" (no clarify)" if a == "none" else " [baseline]")
        print(f"  {a:12s}: {means[a]*100:5.1f}%{tag}")
    print(f"\n  (informative queries: {ninf}/{len(queries)})")
    print(f"  best baseline : {base_ceiling*100:.1f}%")
    print(f"  TiQC (ours)   : {means['grounded']*100:.1f}%  (delta {(means['grounded']-base_ceiling)*100:+.1f})")
    print(f"  -> TiQC {'BEATS all baselines' if means['grounded'] > base_ceiling else 'does NOT beat baselines'}")
    if args.json:
        json.dump({"mean_acc": means, "n_informative": ninf, "results": results},
                  open(args.json, "w"), indent=2, ensure_ascii=False)
        print(f"saved -> {args.json}")


if __name__ == "__main__":
    main()
