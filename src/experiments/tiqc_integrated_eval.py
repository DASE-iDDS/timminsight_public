"""TiQC + TiMetagraph INTEGRATED evaluation (paper §VI).

Unlike the earlier pilot (which inlined the clarification logic against a hardcoded
SCHEMAS string), this runner exercises the actual integrated components:
  - TiMetagraph: metadata_graph_builder.build_metadata_graph(dataset) → the grounding
    schema_context() is DERIVED from a MultimodalMetadataGraph built from REAL data.
  - TiQC:        core.query_planner.query_clarifier.TiQCClarifier drives detect →
                 classify → ask → rewrite, via the project's create_provider.

Arms give BOTH the end-to-end comparison and the ablation in one sweep:
  none        = no clarification (guess & commit)            [ablate TiQC]
  ungrounded  = TiQC clarifier WITHOUT TiMetagraph schema    [ablate TiMetagraph]
  grounded    = TiQC + TiMetagraph (full)                    [ours]
  caesura/lotus/thalamusdb/palimpzest/nirvana = baselines (schema-equipped, cannot ask)

Metrics: resolution match% (LLM-judge vs gold intent) + type-classification acc%.
Real data, all queries, no fabrication (per experiment-integrity-rules).

Run: python3 -m src.experiments.tiqc_integrated_eval --compare --baselines \
        --model glm-5.2 --json runs_tiqc_integrated/glm-5.2.json
"""
import os
import re
import json
import argparse

from .tiqc_ambiguity_workload import AMBIGUOUS_QUERIES
from ..core.query_planner.llm_query_analyzer import create_provider
from ..core.query_planner.query_clarifier import TiQCClarifier, AMBIGUITY_TYPES
from ..core.query_planner.metadata_graph_builder import schema_context

BASELINE_PARADIGM = {
    "caesura":    "CAESURA (SIGMOD'24): LLM sequential 3-phase planning, single shot, no search, no clarification",
    "lotus":      "LOTUS (VLDB'25): rule-based semantic-operator mapping with cascade optimization, no clarification",
    "thalamusdb": "ThalamusDB (SIGMOD'23): SQL-style approximate query processing, no clarification",
    "palimpzest": "Palimpzest (CIDR'25): Cascades-style cost optimization over a plan space, no clarification",
    "nirvana":    "Nirvana (SIGMOD'26): LLM-native multimodal query optimizer, single-pass, no clarification",
}


# FIXED judge (like score_vs_gold): isolates the system-backbone effect. Using the
# system backbone as its own judge deflated deepseek-v4-flash (a harsh self-judge
# rated its own correct rewrites as non-matching). Set once in main().
_JUDGE = None


def _jmatch(prov, gold, rewritten):
    prov = _JUDGE or prov
    txt = prov.complete([
        {"role": "system", "content": "Judge whether the REWRITTEN query captures the same "
         "intent as the GOLD query. Respond JSON: {\"match\": true/false}."},
        {"role": "user", "content": f"Gold query: {gold}\nRewritten query: {rewritten}"}])
    m = re.search(r'"match"\s*:\s*(true|false)', str(txt).lower())
    return m and m.group(1) == "true"


def _user_sim(prov, hidden_intent, question, options):
    return prov.complete([
        {"role": "system", "content": "You simulate a human analyst answering a clarifying "
         "question. Pick the option matching your TRUE intent (or state it). One short sentence."},
        {"role": "user", "content": f"Your true intent: {hidden_intent}\nQuestion: {question}\nOptions: {options}"}])


def _run_tiqc(prov, q, grounded):
    # paper Algorithm 2: multi-round, severity-gated clarification with a user simulator
    # conditioned on the hidden ground-truth intent (K=3, θ_amb=0.5 defaults).
    c = TiQCClarifier(prov, dataset=q.dataset, grounded=grounded)
    sim = c.make_user_simulator(q.hidden_intent)
    res = c.clarify_query(q.ambiguous_query, sim)
    rew = res["clarified_query"]
    first = res["trace"][0]["ambiguity"] if res["trace"] else {}
    return {"id": q.id, "type_gold": q.ambiguity_type,
            "type_pred": first.get("ambiguity_type"),
            "rounds": res["rounds"], "question": first.get("question"),
            "user_answer": res["trace"][0]["answer"] if res["trace"] else None,
            "rewritten": rew,
            "match": bool(_jmatch(prov, q.original_query, rew)),
            "type_correct": first.get("ambiguity_type") == q.ambiguity_type}


def _run_none(prov, q):
    rew = prov.complete([
        {"role": "system", "content": "You are a query planner with NO ability to ask the user. "
         "Resolve the possibly-ambiguous query into ONE concrete executable query (best guess). "
         "Output ONLY the rewritten query."},
        {"role": "user", "content": f"Dataset: {q.dataset}\nQuery: {q.ambiguous_query}"}])
    return {"id": q.id, "type_gold": q.ambiguity_type, "type_pred": None, "question": None,
            "user_answer": None, "rewritten": str(rew).strip(),
            "match": bool(_jmatch(prov, q.original_query, rew)), "type_correct": False}


def _run_baseline(prov, q, strat):
    sc = schema_context(q.dataset)
    rew = prov.complete([
        {"role": "system", "content": f"You are the {BASELINE_PARADIGM[strat]}. You CANNOT ask "
         "the user. Using the schema, commit to ONE concrete executable interpretation (best "
         "guess). Output ONLY the rewritten query."},
        {"role": "user", "content": f"Dataset: {q.dataset}\nSchema:\n{sc}\nQuery: {q.ambiguous_query}"}])
    return {"id": q.id, "type_gold": q.ambiguity_type, "type_pred": None, "question": None,
            "user_answer": None, "rewritten": str(rew).strip(),
            "match": bool(_jmatch(prov, q.original_query, rew)), "type_correct": False}


def run_compare(prov, queries):
    arms = {"none": lambda q: _run_none(prov, q),
            "ungrounded": lambda q: _run_tiqc(prov, q, False),
            "grounded": lambda q: _run_tiqc(prov, q, True)}
    summary, detail = {}, {}
    for arm, fn in arms.items():
        rows = []
        for q in queries:
            try:
                rows.append(fn(q))
            except Exception as e:
                print(f"  [{arm}] {q.id}: ERR {type(e).__name__}: {str(e)[:80]}")
        n = len(rows) or 1
        summary[arm] = {"n": len(rows), "match": sum(x["match"] for x in rows),
                        "match_pct": 100 * sum(x["match"] for x in rows) / n,
                        "type_pct": 100 * sum(x["type_correct"] for x in rows) / n}
        detail[arm] = rows
        print(f"[{arm:10s}] match {summary[arm]['match']}/{len(rows)} "
              f"({summary[arm]['match_pct']:.0f}%)  type {summary[arm]['type_pct']:.0f}%")
    b, u, g = (summary[a]["match_pct"] for a in ("none", "ungrounded", "grounded"))
    print(f"  none {b:.0f}% -> ungrounded {u:.0f}% (+{u-b:.0f}) -> grounded {g:.0f}% (+{g-b:.0f})")
    return summary, detail


def run_baselines(prov, queries):
    order = ["none"] + list(BASELINE_PARADIGM) + ["grounded"]
    summary, detail = {}, {}
    for arm in order:
        rows = []
        for q in queries:
            try:
                rows.append(_run_none(prov, q) if arm == "none" else
                            _run_tiqc(prov, q, True) if arm == "grounded" else
                            _run_baseline(prov, q, arm))
            except Exception as e:
                print(f"  [{arm}] {q.id}: ERR {type(e).__name__}: {str(e)[:80]}")
        n = len(rows) or 1
        summary[arm] = {"n": len(rows), "match": sum(x["match"] for x in rows),
                        "pct": 100 * sum(x["match"] for x in rows) / n}
        detail[arm] = rows
        print(f"  {arm:11s}: {summary[arm]['match']}/{len(rows)} ({summary[arm]['pct']:.0f}%)")
    best = max(summary[b]["pct"] for b in BASELINE_PARADIGM)
    g = summary["grounded"]["pct"]
    print(f"  best baseline {best:.0f}% | TiQC {g:.0f}% (delta {g-best:+.0f}) "
          f"-> {'BEATS' if g > best else 'ties/loses'} baselines")
    return summary, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--judge-model", default="glm-5.2", help="FIXED judge across backbones")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    prov = create_provider("dashscope", model=a.model)
    global _JUDGE
    _JUDGE = create_provider("dashscope", model=a.judge_model)   # fixed judge for all backbones
    qs = list(AMBIGUOUS_QUERIES)[: a.n] if a.n else list(AMBIGUOUS_QUERIES)
    out = {"model": a.model, "n_queries": len(qs)}
    if a.compare or not a.baselines:
        print(f"=== COMPARE (ablation: none/ungrounded/grounded) — {a.model} ===")
        s, d = run_compare(prov, qs); out["compare"] = {"summary": s, "detail": d}
    if a.baselines:
        print(f"=== BASELINES (E2E: TiQC vs 5 baselines) — {a.model} ===")
        s, d = run_baselines(prov, qs); out["baselines"] = {"summary": s, "detail": d}
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump(out, open(a.json, "w"), ensure_ascii=False, indent=1)
        print("saved", a.json)


if __name__ == "__main__":
    main()
