"""TiMetagraph STANDALONE evaluation (paper §VI, component: MultimodalMetadataGraph).

The resolution/e2e experiments only exercise TiMetagraph INDIRECTLY, as the grounding
for TiQC (the `grounded` arm). This runner evaluates the metadata graph ON ITS OWN,
on its proper task: schema grounding — turning a NL query into a concrete plan that
references REAL columns and REAL values, and routing image-only attributes to the
image modality.

Two experiments, both on REAL Nirvana data, all 36 queries, no fabricated gold:

  T0  Graph construction (descriptive)
      Build a MultimodalMetadataGraph per dataset from the real rows; report
      modalities / entities / cross-modal value nodes / discovered relations / build
      time. Shows the graph is genuinely constructed from data.

  T1  Schema-grounded planning (E2E + ablation)
      For each real query the backbone emits a structured plan
        {"columns": [...], "filters": [{"column":..., "value":...}], "needs_image": bool}
      scored PROGRAMMATICALLY against real gold (no LLM judge):
        - col_validity   : referenced columns that exist in the REAL schema
                           (gold = real schema)              [hallucination avoidance]
        - value_validity : categorical filter literals that are REAL values of that
                           column (gold = real value set)    [value grounding]
        - image_routing  : needs_image vs the genuinely image-only queries
                           (steam q4/q12) — reported as a small case set, not a headline
      Arms isolate the graph's contribution:
        none  = no schema                               [ablate all grounding]
        flat  = bare column-name list                   [ablate values + modality routing]
        graph = TiMetagraph schema_context (cols + real representative values + routing)

      flat->graph delta == exactly what the graph adds over a plain column list:
      representative real values + image/text modality routing.

Run:
  python3 -m src.experiments.timetagraph_eval --construct         # T0
  python3 -m src.experiments.timetagraph_eval --model glm-5.2 \
      --json runs_tiqc_integrated/timetagraph_glm-5.2.json        # T1
"""
import os
import re
import csv
import json
import time
import argparse
from collections import Counter

from ..core.query_planner.llm_query_analyzer import create_provider
from ..core.query_planner.metadata_graph_builder import (
    build_metadata_graph, schema_context, DATASET_FILES, _load_rows,
    IMAGE_COLS, TEXT_COLS,
)
from ..core.query_planner.entity_relationship_discoverer import EntityRelationshipDiscoverer

csv.field_size_limit(10 ** 9)
NIR = "data/nirvana_repo/nirvana-main"
DOMAINS = ["imdb", "estate", "steam"]

# Genuinely image-only queries in Nirvana: the two steam cover-image queries whose
# gold IS vision-derived (graphic style is in no text column). Same set score_vs_gold
# routes to qwen3-vl. Everything else is answerable from table/text (Nirvana's own
# reference gold was generated from text columns).
IMAGE_ONLY = {("steam", 4), ("steam", 12)}


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# ---------------------------------------------------------------- real gold loaders
def load_queries():
    """36 real Nirvana queries (nl text) from the gold manifests, keyed (domain, q)."""
    q = {}
    for mf in ("gold_manifest.json", "gold_manifest_s100.json"):
        p = f"{NIR}/{mf}"
        if not os.path.exists(p):
            continue
        try:
            man = json.load(open(p))
        except Exception:
            continue
        for m in man:
            d, qi = m.get("domain"), m.get("q")
            if d not in DOMAINS:
                continue
            nl = re.sub(r"^Q\d+:\s*", "", str(m.get("nl", ""))).strip()
            if nl and (d, int(qi)) not in q:
                q[(d, int(qi))] = nl
    return q


def raw_schema(dataset):
    """The real column set of a dataset (gold for col_validity)."""
    path = DATASET_FILES.get(dataset)
    if path and path.endswith(".parquet"):
        import pandas as pd
        return [str(c) for c in pd.read_parquet(path).columns]
    with open(path, newline="") as f:
        return next(csv.reader(f))


def categorical_value_gold(dataset):
    """Full real value set of each LOW-cardinality categorical column (gold for
    value_validity). Uses ALL rows, not the 100-row sample the graph exposes, so a
    valid-but-rare literal is never wrongly marked invalid."""
    rows = _load_rows(dataset, sample=0)  # full file
    if not rows:
        return {}
    cols = list(rows[0].keys())
    img = set(IMAGE_COLS.get(dataset, []))
    txt = set(TEXT_COLS.get(dataset, []))
    gold = {}
    for c in cols:
        if c in img or c in txt:
            continue
        vals = [str(r.get(c, "")).strip() for r in rows if str(r.get(c, "")).strip()]
        uniq = set(vals)
        # same low-cardinality gate the graph builder uses to expose representative
        # values -> exactly the columns where graph grounding can help
        if 1 < len(uniq) <= 40:
            gold[c] = {_norm(v) for v in uniq}
    return gold


# ---------------------------------------------------------------- T1 planning arms
PLAN_SYS = (
    "You are a query planner. Given a natural-language query over a table, output the "
    "concrete execution plan as STRICT JSON with keys: "
    '"columns" (list of column names you will READ/PROJECT), '
    '"filters" (list of {"column":name,"value":literal} you will filter on; [] if none), '
    '"needs_image" (true iff answering requires looking at the image/picture pixels, '
    "false if it is answerable from table columns or text). "
    "Output ONLY the JSON object, nothing else."
)


def _schema_block(dataset, arm):
    if arm == "none":
        return "(schema not provided — infer columns from the query)"
    if arm == "flat":
        return f"Table {dataset} columns: {', '.join(raw_schema(dataset))}."
    return schema_context(dataset)  # graph


def _extract_json(txt):
    txt = str(txt)
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    s = m.group(0)
    for cand in (s, s.replace("'", '"')):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


def plan_query(prov, dataset, nl, arm):
    out = prov.complete([
        {"role": "system", "content": PLAN_SYS},
        {"role": "user", "content": f"Schema: {_schema_block(dataset, arm)}\nQuery: {nl}"}])
    return _extract_json(out)


def score_plan(plan, dataset, is_image_only, schema_norm, val_gold):
    """Programmatic scoring against real schema + real value sets."""
    if not isinstance(plan, dict):
        return None  # parse failure — counted separately, never fabricated
    cols = plan.get("columns") or []
    filts = [f for f in (plan.get("filters") or []) if isinstance(f, dict)]
    ref_cols = [c for c in cols if c] + [f.get("column") for f in filts if f.get("column")]
    # col_validity
    cv_n = len(ref_cols)
    cv_ok = sum(1 for c in ref_cols if _norm(c) in schema_norm)
    # value_validity: only categorical columns the graph exposes real values for
    # (low-cardinality -> a finite real gold set exists). Two strictnesses:
    #   exact = predicted literal IS a real stored value (normalized equality)
    #   loose = predicted literal is a substring-compatible real value
    # The graph's value-grounding benefit shows most in EXACT (matching the precise
    # stored encoding, e.g. "PG-13" not "PG13", the real genre spelling, ...).
    vv_n = vv_exact = vv_loose = 0
    vv_audit = []
    for f in filts:
        c, v = f.get("column"), f.get("value")
        if not c or v is None:
            continue
        key = next((k for k in val_gold if _norm(k) == _norm(c)), None)
        if key is None:
            continue  # numeric/high-card/unknown column -> no finite gold, skip
        vv_n += 1
        nv = _norm(v)
        ex = bool(nv) and nv in val_gold[key]
        lo = ex or (len(nv) > 2 and any(nv in g or g in nv for g in val_gold[key]))
        vv_exact += int(ex); vv_loose += int(lo)
        vv_audit.append({"col": c, "val": v, "exact": ex, "loose": lo})
    return {
        "cv_n": cv_n, "cv_ok": cv_ok,
        "vv_n": vv_n, "vv_exact": vv_exact, "vv_loose": vv_loose, "vv_audit": vv_audit,
        "needs_image": bool(plan.get("needs_image")),
        "image_correct": bool(plan.get("needs_image")) == is_image_only,
    }


def run_t1(prov, model):
    queries = load_queries()
    schema_norm = {d: {_norm(c) for c in raw_schema(d)} for d in DOMAINS}
    val_gold = {d: categorical_value_gold(d) for d in DOMAINS}
    arms = ["none", "flat", "graph"]
    detail = {a: [] for a in arms}
    agg = {a: dict(cv_n=0, cv_ok=0, vv_n=0, vv_exact=0, vv_loose=0, img_ok=0,
                   n=0, parse_fail=0) for a in arms}
    for (d, qi), nl in sorted(queries.items()):
        is_img = (d, qi) in IMAGE_ONLY
        for arm in arms:
            try:
                plan = plan_query(prov, d, nl, arm)
            except Exception as e:
                print(f"  [{arm}] {d} q{qi}: ERR {type(e).__name__}: {str(e)[:70]}")
                plan = None
            sc = score_plan(plan, d, is_img, schema_norm[d], val_gold[d])
            a = agg[arm]
            a["n"] += 1
            if sc is None:
                a["parse_fail"] += 1
            else:
                a["cv_n"] += sc["cv_n"]; a["cv_ok"] += sc["cv_ok"]
                a["vv_n"] += sc["vv_n"]
                a["vv_exact"] += sc["vv_exact"]; a["vv_loose"] += sc["vv_loose"]
                a["img_ok"] += int(sc["image_correct"])
            detail[arm].append({"dataset": d, "q": qi, "nl": nl, "is_image_only": is_img,
                                "plan": plan, "score": sc})
    summary = {}
    for arm in arms:
        a = agg[arm]
        summary[arm] = {
            "n": a["n"], "parse_fail": a["parse_fail"],
            "col_validity": round(100 * a["cv_ok"] / a["cv_n"], 1) if a["cv_n"] else None,
            "value_validity_exact": round(100 * a["vv_exact"] / a["vv_n"], 1) if a["vv_n"] else None,
            "value_validity_loose": round(100 * a["vv_loose"] / a["vv_n"], 1) if a["vv_n"] else None,
            "value_n": a["vv_n"],
            "image_routing_acc": round(100 * a["img_ok"] / a["n"], 1) if a["n"] else None,
        }
    print(f"\n=== T1 schema-grounded planning — {model} ===")
    print(f"{'arm':6s} {'col_valid%':>10s} {'val_exact%':>10s} {'val_loose%':>10s} "
          f"{'val_n':>6s} {'img_acc%':>8s} {'parsefail':>9s}")
    for arm in arms:
        s = summary[arm]
        print(f"{arm:6s} {str(s['col_validity']):>10s} {str(s['value_validity_exact']):>10s} "
              f"{str(s['value_validity_loose']):>10s} {str(s['value_n']):>6s} "
              f"{str(s['image_routing_acc']):>8s} {s['parse_fail']:>9d}")
    fe, ge = summary["flat"]["value_validity_exact"], summary["graph"]["value_validity_exact"]
    if fe is not None and ge is not None:
        print(f"  value grounding (exact): flat {fe}% -> graph {ge}% (delta {ge-fe:+.1f})")
    return {"summary": summary, "detail": detail}


# ---------------------------------------------------------------- T0 construction
def run_t0():
    print("=== T0 TiMetagraph construction (real data) ===")
    rows_out = {}
    disc = EntityRelationshipDiscoverer()
    for d in DOMAINS:  # the 3 real Nirvana datasets T1 evaluates on
        if d not in DATASET_FILES or not os.path.exists(DATASET_FILES[d]):
            continue
        t = time.perf_counter()
        g = build_metadata_graph(d, sample=100)
        try:
            rels = disc.discover_entity_relations(g)
        except Exception as e:
            rels = []
            print(f"  [{d}] relation discovery: {type(e).__name__}: {str(e)[:60]}")
        dt = time.perf_counter() - t
        # semantically clean counts (not raw graph-node totals, which conflate
        # modality nodes with entity-mention nodes)
        types = sorted({m.modality_type.value for m in g.modalities.values()})
        rows_out[d] = {
            "n_modalities": len(g.modalities),
            "modality_types": types,
            "cross_modal_value_nodes": len(g.entities),
            "discovered_relations": len(rels),
            "build_ms": round(dt * 1000, 1),
        }
        print(f"  {d:7s} modalities={len(g.modalities)} types={types} "
              f"value_nodes={len(g.entities)} relations={len(rels)} "
              f"build={rows_out[d]['build_ms']}ms")
    return rows_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--construct", action="store_true", help="run T0 only")
    ap.add_argument("--n", type=int, default=0, help="limit queries (smoke test)")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out = {"model": a.model}
    if a.construct:
        out["t0_construction"] = run_t0()
        if a.json:
            json.dump(out, open(a.json, "w"), ensure_ascii=False, indent=1)
            print("saved", a.json)
        return
    prov = create_provider("dashscope", model=a.model)
    if a.n:  # smoke: monkey-limit query set
        global load_queries
        _orig = load_queries
        load_queries = lambda: dict(list(_orig().items())[: a.n])
    out["t0_construction"] = run_t0()
    out["t1_schema_grounding"] = run_t1(prov, a.model)
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump(out, open(a.json, "w"), ensure_ascii=False, indent=1)
        print("saved", a.json)


if __name__ == "__main__":
    main()
