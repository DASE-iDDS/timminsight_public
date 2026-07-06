"""TiQC clarification-loop pilot (paper §IV, Algorithm 2).

A minimal, self-contained harness that exercises the interactive
clarification loop on the ambiguous-query set, to validate that the
constructed ambiguities actually trigger sensible clarification and resolve
to the hidden ground-truth intent. This is a PILOT to start the §VI TiQC
experiment; it is independent of the full TiQC implementation and uses the
project's DashScope provider.

Loop per query (one round, theta_amb/K machinery omitted for the pilot):
  1. TiQC role  : detect the ambiguity, classify its type, ask ONE clarifying
                  question and offer grounded options (DetectAmbiguities +
                  GenerateOptions).
  2. User-sim   : answer the question conditioned on the hidden intent (AskUser
                  driven by an LLM user-simulator, per established practice).
  3. TiQC role  : rewrite the query given the answer (RewriteQuery).
  4. Judge      : does the rewritten query match the clear/disambiguated query?

Usage:
  export DASHSCOPE_API_KEY=...
  python3 -m src.experiments.tiqc_clarification_pilot --n 3 --model qwen3.6-plus
"""

import os
import re
import json
import argparse

from .tiqc_ambiguity_workload import AMBIGUOUS_QUERIES


# Real schema context (the paper's metadata graph G_meta, here a compact text
# surrogate of columns + representative values, drawn from the downloaded data).
SCHEMAS = {
    "estate": (
        "Table estate(Title text, Location text e.g. 'Lekki Phase 1, Lekki, Lagos' / "
        "'Lekki Phase 2' / 'Ikoyi, Lagos', Details long_text, image REAL_PHOTO). "
        "Visual facts (yard, pool, newness, architectural style) exist ONLY in the "
        "image pixels, NOT as columns. No 'architectural_style' or 'has_yard' column exists."),
    "movie": (
        "Table movie(Title, Year int, Genre1/Genre2/Genre3 text e.g. Action/Horror/Drama, "
        "Plot text, IMDb float 0-10, \"Rotten Tomatoes\" percent string, Metascore float 0-100, "
        "BoxOffice, Poster url). Three distinct rating columns: IMDb, Rotten Tomatoes, Metascore."),
    "steam": (
        "Table steam(title, image url, release_date, original_price, discounted_price, "
        "overall_reviews text e.g. 'Very Positive', recent_reviews text, metacriticts int 0-100, "
        "tags, genre e.g. Action/Horror, rating text = PEGI AGE rating e.g. '18'/'16'). "
        "Note 'rating' is the PEGI age column, separate from review sentiment and metacritic."),
}


def _client():
    from openai import OpenAI
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise SystemExit("DASHSCOPE_API_KEY not set (source .env first)")
    return OpenAI(api_key=key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


# Sampling temperature for the LLM calls. Set >0 by the runner for significance
# repeats so each repeat yields genuine run-to-run variance (temp=0 is deterministic).
_TEMP = 0.0


def _chat(client, model, system, user):
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=_TEMP,
        max_tokens=400,
        extra_body={"enable_thinking": False},
    )
    return r.choices[0].message.content.strip()


def _json(text):
    # Parse the FIRST balanced JSON object, ignoring any prose or extra objects
    # the model may append (robust to "Extra data" from chatty backends).
    i = text.find("{")
    if i < 0:
        return {}
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[i:])
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def run_one(client, model, q, grounded=True):
    ctx = f"\nMetadata context (schema):\n{SCHEMAS[q.dataset]}\n" if grounded else ""
    # 1. TiQC: detect + ask
    detect = _chat(client, model,
        "You are TiQC, an interactive query-clarification module for multimodal "
        "data analysis. Given an exploratory query that may be ambiguous, detect "
        "the single most consequential ambiguity, classify it into one of "
        "{cross_modal_referential, intra_modal_attribute, multimodal_intent, "
        "schema_structure, value_content, temporal_spatial}, and ask ONE clarifying "
        "question with 2-3 concrete options. Ground every option STRICTLY in the "
        "provided schema. Never invent a column that is not listed. "
        'Respond as JSON: {"type": "...", "question": "...", "options": ["...", "..."]}.',
        f"Dataset: {q.dataset}{ctx}\nQuery: {q.ambiguous_query}")
    d = _json(detect)

    # 2. User-simulator: answer from the hidden intent
    answer = _chat(client, model,
        "You are simulating a human analyst answering a clarifying question. "
        "Your TRUE intent is given. Pick the option matching your intent (or state it "
        "briefly). Answer in one short sentence, do not reveal you are a simulator.",
        f"Your true intent: {q.hidden_intent}\n"
        f"Clarifying question: {d.get('question', '(none)')}\n"
        f"Options: {d.get('options', [])}")

    # 3. TiQC: rewrite
    rewrite = _chat(client, model,
        "You are TiQC. Rewrite the original query into an unambiguous query that "
        "incorporates the user's answer. Output ONLY the rewritten query, one line.",
        f"Original query: {q.ambiguous_query}\nUser answer: {answer}")

    # 4. Judge: does the rewrite capture the hidden intent?
    judge = _chat(client, model,
        "Judge whether the REWRITTEN query captures the same intent as the GOLD "
        'query. Respond JSON: {"match": true/false, "why": "short"}.',
        f"Gold query: {q.original_query}\nRewritten query: {rewrite}")
    j = _json(judge)

    return {
        "id": q.id, "type_gold": q.ambiguity_type, "type_pred": d.get("type"),
        "question": d.get("question"), "user_answer": answer,
        "rewritten": rewrite, "match": bool(j.get("match")),
        "type_correct": d.get("type") == q.ambiguity_type,
    }


def run_noclarify(client, model, q):
    """Baseline: no clarification. System directly guesses the interpretation
    and rewrites, without asking the user. Tests whether asking helps at all."""
    rewrite = _chat(client, model,
        "You are a query planner with NO ability to ask the user. Resolve the "
        "possibly-ambiguous query into a single concrete executable query using your "
        "best guess. Output ONLY the rewritten query, one line.",
        f"Dataset: {q.dataset}\nQuery: {q.ambiguous_query}")
    judge = _chat(client, model,
        "Judge whether the REWRITTEN query captures the same intent as the GOLD "
        'query. Respond JSON: {"match": true/false, "why": "short"}.',
        f"Gold query: {q.original_query}\nRewritten query: {rewrite}")
    j = _json(judge)
    return {"id": q.id, "type_gold": q.ambiguity_type, "type_pred": None,
            "question": None, "user_answer": None, "rewritten": rewrite,
            "match": bool(j.get("match")), "type_correct": False}


ARMS = {
    "none":      lambda c, m, q: run_noclarify(c, m, q),
    "ungrounded": lambda c, m, q: run_one(c, m, q, grounded=False),
    "grounded":  lambda c, m, q: run_one(c, m, q, grounded=True),
}

# The 5 TiTSP baselines. ALL are non-interactive single-pass planners (verified:
# their real repo planners commit a plan without any clarification turn). They get
# the SAME schema/metadata as grounded TiQC, so the only difference is the inability
# to ask the user -- which isolates the contribution of clarification itself.
BASELINE_PARADIGM = {
    "caesura":    "CAESURA (SIGMOD'24): LLM sequential 3-phase planning, single shot, no search, no clarification",
    "lotus":      "LOTUS (VLDB'25): rule-based semantic-operator mapping with cascade optimization, no clarification",
    "thalamusdb": "ThalamusDB (SIGMOD'23): SQL-style approximate query processing, no clarification",
    "palimpzest": "Palimpzest (CIDR'25): Cascades-style cost optimization over a plan space, no clarification",
    "nirvana":    "Nirvana (SIGMOD'26): LLM-native multimodal query optimizer, single-pass, no clarification",
}


def run_baseline(client, model, q, strategy):
    """A baseline's single-pass committed resolution (schema-equipped, cannot ask)."""
    rewrite = _chat(client, model,
        f"You are the {BASELINE_PARADIGM[strategy]}. You CANNOT ask the user any "
        "question. Using the schema, commit to ONE concrete executable interpretation "
        "of the query by your best guess. Output ONLY the rewritten query, one line.",
        f"Dataset: {q.dataset}\nSchema:\n{SCHEMAS[q.dataset]}\nQuery: {q.ambiguous_query}")
    judge = _chat(client, model,
        "Judge whether the REWRITTEN query captures the same intent as the GOLD "
        'query. Respond JSON: {"match": true/false, "why": "short"}.',
        f"Gold query: {q.original_query}\nRewritten query: {rewrite}")
    return {"id": q.id, "type_gold": q.ambiguity_type, "type_pred": None,
            "question": None, "user_answer": None, "rewritten": rewrite,
            "match": bool(_json(judge).get("match")), "type_correct": False}


def run_baseline_compare(client, model, queries):
    """Compare TiQC against the 5 non-interactive baselines on intent resolution."""
    order = ["none"] + list(BASELINE_PARADIGM) + ["grounded"]
    summary, detail = {}, {}
    for arm in order:
        rows = []
        for q in queries:
            try:
                if arm == "none":
                    r = run_noclarify(client, model, q)
                elif arm == "grounded":
                    r = run_one(client, model, q, grounded=True)
                else:
                    r = run_baseline(client, model, q, arm)
            except Exception as e:
                print(f"  [{arm}] {q.id}: ERROR {type(e).__name__}: {str(e)[:90]}")
                continue
            rows.append(r)
        m = sum(x["match"] for x in rows)
        n = len(rows) or 1
        summary[arm] = {"n": len(rows), "match": m, "pct": 100 * m / n}
        detail[arm] = rows
        label = "  <- TiQC (ours)" if arm == "grounded" else (" (no clarify)" if arm == "none" else " [baseline]")
        print(f"  {arm:11s}: {m}/{len(rows)} ({100*m/n:.0f}%){label}")
    best_base = max(summary[b]["pct"] for b in BASELINE_PARADIGM)
    g = summary["grounded"]["pct"]
    print(f"\n  best baseline : {best_base:.0f}%")
    print(f"  TiQC (ours)   : {g:.0f}%  (delta vs best baseline {g-best_base:+.0f})")
    print(f"  -> TiQC {'BEATS all baselines' if g > best_base else 'does NOT beat baselines'}")
    return summary, detail


def run_compare(client, model, queries):
    """Run all three arms over the set and print a comparison."""
    summary = {}
    detail = {}
    for arm, fn in ARMS.items():
        rows, match, tcorrect = [], 0, 0
        for q in queries:
            try:
                r = fn(client, model, q)
            except Exception as e:
                print(f"  [{arm}] {q.id}: ERROR {type(e).__name__}: {str(e)[:100]}")
                continue
            rows.append(r); match += r["match"]; tcorrect += r["type_correct"]
        n = len(rows) or 1
        summary[arm] = {"n": len(rows), "match": match, "match_pct": 100*match/n,
                        "type_pct": 100*tcorrect/n}
        detail[arm] = rows
        print(f"[{arm:10s}] resolution match {match}/{len(rows)} "
              f"({100*match/n:.0f}%)   type {tcorrect}/{len(rows)} ({100*tcorrect/n:.0f}%)")
    print("\n=== TiQC mechanism verification ===")
    base = summary["none"]["match_pct"]
    grnd = summary["grounded"]["match_pct"]
    ungr = summary["ungrounded"]["match_pct"]
    print(f"  no-clarification (baseline) : {base:.0f}%")
    print(f"  ungrounded clarification    : {ungr:.0f}%  (delta {ungr-base:+.0f})")
    print(f"  grounded TiQC (full)        : {grnd:.0f}%  (delta {grnd-base:+.0f})")
    verdict = "IMPROVES" if grnd > base else ("no gain" if grnd == base else "REGRESSES")
    print(f"  -> TiQC grounded vs baseline: {verdict} ({grnd-base:+.0f} points)")
    return summary, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="limit number of queries (0=all)")
    ap.add_argument("--model", default="qwen3.6-plus")
    ap.add_argument("--no-schema", action="store_true", help="ablation: ungrounded clarification")
    ap.add_argument("--compare", action="store_true", help="run all 3 arms and compare")
    ap.add_argument("--baselines", action="store_true", help="compare TiQC vs the 5 TiTSP baselines")
    ap.add_argument("--repeats", type=int, default=1, help="significance repeats")
    ap.add_argument("--temp", type=float, default=0.0, help="sampling temperature for repeats")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    client = _client()
    if args.baselines:
        import statistics
        global _TEMP
        queries = AMBIGUOUS_QUERIES[: args.n] if args.n else AMBIGUOUS_QUERIES
        _TEMP = args.temp
        print(f"TiQC vs 5 baselines (intent resolution): {len(queries)} real-derived "
              f"queries, model={args.model}, repeats={args.repeats}, temp={args.temp}\n")
        runs = []
        for rep in range(args.repeats):
            print(f"--- repeat {rep+1}/{args.repeats} ---")
            summary, _ = run_baseline_compare(client, args.model, queries)
            runs.append({a: summary[a]["pct"] for a in summary})
        arms = list(runs[0].keys())
        agg = {a: {"mean": statistics.mean(r[a] for r in runs),
                   "std": (statistics.pstdev(r[a] for r in runs) if args.repeats > 1 else 0.0)}
               for a in arms}
        print(f"\n=== AGGREGATE over {args.repeats} repeats (mean +/- std, %) ===")
        for a in arms:
            tag = "  <- TiQC (ours)" if a == "grounded" else (" (no clarify)" if a == "none" else " [baseline]")
            print(f"  {a:11s}: {agg[a]['mean']:5.1f} +/- {agg[a]['std']:4.1f}{tag}")
        best_base = max(agg[b]["mean"] for b in BASELINE_PARADIGM)
        print(f"\n  best baseline : {best_base:.1f}%")
        print(f"  TiQC (ours)   : {agg['grounded']['mean']:.1f}%  (delta {agg['grounded']['mean']-best_base:+.1f})")
        print(f"  -> TiQC {'BEATS all baselines' if agg['grounded']['mean'] > best_base else 'does NOT beat baselines'}")
        if args.json:
            json.dump({"aggregate": agg, "runs": runs}, open(args.json, "w"),
                      indent=2, ensure_ascii=False)
            print(f"saved -> {args.json}")
        return
    if args.compare:
        queries = AMBIGUOUS_QUERIES[: args.n] if args.n else AMBIGUOUS_QUERIES
        print(f"TiQC mechanism experiment: {len(queries)} queries, model={args.model}\n")
        summary, detail = run_compare(client, args.model, queries)
        if args.json:
            json.dump({"summary": summary, "detail": detail},
                      open(args.json, "w"), indent=2, ensure_ascii=False)
            print(f"saved -> {args.json}")
        return
    grounded = not args.no_schema
    queries = AMBIGUOUS_QUERIES[: args.n] if args.n else AMBIGUOUS_QUERIES
    rows, match, tcorrect = [], 0, 0
    print(f"TiQC pilot: {len(queries)} queries, model={args.model}, "
          f"grounding={'ON' if grounded else 'OFF'}\n")
    for q in queries:
        try:
            r = run_one(client, args.model, q, grounded=grounded)
        except Exception as e:
            print(f"  {q.id}: ERROR {type(e).__name__}: {str(e)[:120]}")
            continue
        rows.append(r)
        match += r["match"]; tcorrect += r["type_correct"]
        flag = "OK " if r["match"] else "XX "
        print(f"{flag}{r['id']:16s} type {r['type_pred']}/{r['type_gold']} "
              f"{'+' if r['type_correct'] else '-'}")
        print(f"     Q: {r['question']}")
        print(f"     A: {r['user_answer'][:90]}")
        print(f"     -> {r['rewritten'][:100]}")
    n = len(rows) or 1
    print(f"\nresolution match: {match}/{len(rows)} ({100*match/n:.0f}%)   "
          f"type-classification: {tcorrect}/{len(rows)} ({100*tcorrect/n:.0f}%)")
    if args.json and rows:
        json.dump(rows, open(args.json, "w"), indent=2, ensure_ascii=False)
        print(f"saved -> {args.json}")


if __name__ == "__main__":
    main()
