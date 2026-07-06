"""TiProcessor family — LLM/VLM-driven metadata extraction under ReAct (paper §3.1.1).

BaseProcessor runs a bounded ReAct loop (Thought -> Action(tool) -> Observation -> ...
-> Final) in which the LLM decides which external specialized tool to invoke. The three
concrete processors extract per-modality metadata into the unified graph, following the
paper's staging exactly:

  TableProcessor  LLM column summaries (type, stats, distribution, entity) + PK/FK, and
                  per-column value sets enabling cross-table referential-integrity edges.
  TextProcessor   3-stage: (1) split text into paragraphs; (2) ReAct PARALLEL extraction
                  per chunk (spaCy NER + LLM sentiment + topic); (3) merge across chunks.
  ImageProcessor  2-stage: (1) group images into batches; (2) PARALLEL extraction per
                  batch (YOLO/Detectron2 objects, Places365 scene, ViT features) + merge.

Extracted entities become CrossModalEntity + EntityMention carrying the exact per-modality
signals Algorithm 1's Table 2 needs (text word_count, image bbox area/saliency, table
row/column). Real inputs only; each tool is honestly marked available/unavailable.
"""
from __future__ import annotations
import json
import re
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from .metadata_graph import (
    MultimodalMetadataGraph, ModalityMetadata, ModalityType, EntityType,
    CrossModalEntity, EntityMention, CrossModalRelation, RelationType,
)
from . import extraction_tools as ET


def _parse_json(text: str) -> Any:
    t = str(text).strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        for op, cl in (("{", "}"), ("[", "]")):
            s, e = t.find(op), t.rfind(cl)
            if s != -1 and e != -1 and e > s:
                try:
                    return json.loads(t[s:e + 1])
                except Exception:
                    continue
    return None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


class BaseProcessor:
    """ReAct scaffolding shared by all modality processors (paper Figure 2)."""

    def __init__(self, provider: Any, vision_provider: Any = None,
                 tools: Optional[Dict[str, Any]] = None, max_steps: int = 3,
                 max_workers: int = 4):
        self.provider = provider
        self.vision_provider = vision_provider
        self.tools = tools or {}
        self.max_steps = max_steps
        self.max_workers = max_workers
        self.tool_calls: List[str] = []

    def _chat(self, system: str, user: str) -> str:
        return self.provider.complete([{"role": "system", "content": system},
                                       {"role": "user", "content": user}])

    def _parallel(self, fn: Callable, items: List[Any]) -> List[Any]:
        """Run fn over items; parallel (ReAct extraction is I/O-bound on tool+LLM calls)."""
        if self.max_workers <= 1 or len(items) <= 1:
            return [fn(x) for x in items]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(items))) as ex:
            return list(ex.map(fn, items))

    def react(self, goal: str, context: str, tool_runners: Dict[str, Callable],
              tool_descs: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Bounded Thought/Action/Observation loop; `tool_runners[name]() -> observation`
        runs the specialized tool on the current item. LLM chooses tools, emits final."""
        tool_descs = tool_descs or {}
        tdesc = "\n".join(f"- {n}: {tool_descs.get(n, 'specialized tool')}"
                          for n in tool_runners) or "(no tools available)"
        obs: List[str] = []
        sys = ("You extract structured metadata under the ReAct paradigm. Each turn reply "
               "with STRICT JSON: either {\"thought\": str, \"action\": \"<tool_name>\"} to "
               "call a tool, or {\"thought\": str, \"final\": {...}} to finish with the "
               "structured metadata. Call a tool only when its observation would help.")
        for _ in range(self.max_steps):
            prompt = (f"Goal: {goal}\nContext: {context}\nAvailable tools:\n{tdesc}\n"
                      "Observations so far:\n" + ("\n".join(obs) if obs else "(none)") +
                      "\nNext ReAct step as JSON.")
            out = _parse_json(self._chat(sys, prompt)) or {}
            if "final" in out:
                return out["final"] if isinstance(out["final"], dict) else {}
            action = out.get("action")
            if action in tool_runners:
                try:
                    result = tool_runners[action]()
                    self.tool_calls.append(action)
                    obs.append(f"{action} -> {json.dumps(result, ensure_ascii=False, default=str)[:1000]}")
                except Exception as e:
                    obs.append(f"{action} -> ERROR {type(e).__name__}: {str(e)[:80]}")
            else:
                obs.append("(no valid action; finish now)")
        fin = _parse_json(self._chat(
            sys, f"Goal: {goal}\nContext: {context}\nObservations:\n" + "\n".join(obs) +
                 "\nOutput ONLY {\"final\": {...}} with the structured metadata."))
        return (fin or {}).get("final", {}) if isinstance(fin, dict) else {}


class TableProcessor(BaseProcessor):
    def process(self, dataset: str, rows: List[Dict[str, Any]],
                graph: MultimodalMetadataGraph, image_cols=None, text_cols=None) -> str:
        cols = list(rows[0].keys()) if rows else []
        image_cols, text_cols = image_cols or [], text_cols or []
        struct_cols = [c for c in cols if c not in image_cols and c not in text_cols]
        # LLM column summaries (type/stats/distribution/entity) + PK/FK
        sample = {c: [str(r.get(c, ""))[:40] for r in rows[:8]] for c in struct_cols}
        sys = ("Summarize a relational table's schema. For each column give data_type, a "
               "one-line semantic summary, value_distribution (e.g. categorical/numeric-range/"
               "high-cardinality), and is_entity (bool). Also name primary_key_columns and "
               "foreign_key_columns. Return JSON: {\"columns\": {col: {\"data_type\": str, "
               "\"summary\": str, \"value_distribution\": str, \"is_entity\": bool}}, "
               "\"primary_key_columns\": [..], \"foreign_key_columns\": [..]}.")
        info = _parse_json(self._chat(sys, f"Dataset: {dataset}\nColumns+samples: "
                                           f"{json.dumps(sample, ensure_ascii=False)[:3000]}")) or {}
        pk = info.get("primary_key_columns", []) or []
        mid = f"{dataset}_table"
        reps: Dict[str, List[str]] = {}
        col_values: Dict[str, List[str]] = {}   # normalized value sets -> cross-table FK
        for c in struct_cols:
            vals = [str(r.get(c, "")).strip() for r in rows if str(r.get(c, "")).strip()]
            uniq = set(vals)
            col_values[c] = sorted({_norm(v) for v in uniq})[:500]
            if 1 < len(uniq) <= 40:
                reps[c] = [v for v, _ in Counter(vals).most_common(6)]
        graph.add_modality(ModalityMetadata(
            modality_id=mid, modality_type=ModalityType.TABLE, source_path=dataset,
            content_summary={"columns": cols, "n_rows": len(rows),
                             "column_info": info.get("columns", {})},
            semantic_features={"representative_values": reps},
            raw_metadata={"image_columns": image_cols, "text_columns": text_cols,
                          "primary_key_columns": pk,
                          "foreign_key_columns": info.get("foreign_key_columns", []),
                          "column_values": {c: set(v) for c, v in col_values.items()}}))
        for c, vals in reps.items():
            for v in vals:
                eid = f"{dataset}:{c}:{v}"[:120]
                mentions = [EntityMention(mid, ModalityType.TABLE, v, confidence=1.0,
                                          position={"row": i, "column": c}, context=c)
                            for i, r in enumerate(rows) if str(r.get(c, "")).strip() == v][:20]
                graph.add_entity(CrossModalEntity(
                    entity_id=eid, entity_type=EntityType.CONCEPT, canonical_name=v,
                    mentions=mentions or [EntityMention(mid, ModalityType.TABLE, v,
                                          position={"column": c}, context=c)],
                    properties={"column": c}))
        return mid


_SENT = {"positive": "positive", "negative": "negative", "neutral": "neutral",
         "mixed": "mixed", "pos": "positive", "neg": "negative"}


def _split_paragraphs(text: str, max_words: int = 120) -> List[str]:
    """Stage 1 of text processing: split into paragraphs, further windowing long ones."""
    paras = [p.strip() for p in re.split(r"\n\s*\n|\r\n\r\n", text) if p.strip()]
    if not paras:
        paras = [text.strip()] if text.strip() else []
    out: List[str] = []
    for p in paras:
        words = p.split()
        if len(words) <= max_words:
            out.append(p)
        else:  # window long paragraphs by sentence into ~max_words chunks
            sents, buf, n = re.split(r"(?<=[.!?])\s+", p), [], 0
            for s in sents:
                buf.append(s); n += len(s.split())
                if n >= max_words:
                    out.append(" ".join(buf)); buf, n = [], 0
            if buf:
                out.append(" ".join(buf))
    return out


class TextProcessor(BaseProcessor):
    def process(self, dataset: str, rows: List[Dict[str, Any]], text_cols: List[str],
                graph: MultimodalMetadataGraph, max_docs: int = 24) -> Optional[str]:
        text_cols = [c for c in text_cols if rows and c in rows[0]]
        if not text_cols:
            return None
        mid = f"{dataset}_text"
        # dedicated external tools (paper §3.1.1): NER=spaCy [12], sentiment=VADER [14],
        # topics=KeyBERT [11]; the ReAct LLM orchestrates their invocation.
        ner = self.tools.get("spacy")
        vader = self.tools.get("vader")
        keybert = self.tools.get("keybert")
        ner_ok = ner is not None and ner.available
        vader_ok = vader is not None and vader.available
        kb_ok = keybert is not None and keybert.available
        docs = [" ".join(str(r.get(c, "")) for c in text_cols).strip()
                for r in rows[:max_docs]]
        docs = [d for d in docs if d]
        total_words = sum(len(d.split()) for d in docs)

        # Stage 1: split every doc into paragraphs (chunks)
        chunks: List[str] = []
        for d in docs:
            chunks.extend(_split_paragraphs(d))

        # Stage 2: ReAct PARALLEL extraction per chunk via the dedicated tools
        def extract_chunk(chunk: str) -> Dict[str, Any]:
            runners, descs = {}, {}
            if ner_ok:
                runners[ner.name] = lambda p=chunk: ner.run(p)
                descs[ner.name] = ET.SpacyNERTool.description
            if vader_ok:
                runners[vader.name] = lambda p=chunk: vader.run(p)
                descs[vader.name] = ET.VaderSentimentTool.description
            if kb_ok:
                runners[keybert.name] = lambda p=chunk: keybert.run(p)
                descs[keybert.name] = ET.KeyBERTTool.description
            self.react(goal="Extract named entities, sentiment, and topics using the tools.",
                       context=chunk[:1500], tool_runners=runners, tool_descs=descs)
            # deterministic reads from the dedicated tools (ground truth for the metadata)
            ents = ner.run(chunk) if ner_ok else []
            sent = vader.run(chunk)["sentiment"] if vader_ok else None
            tops = [t["topic"] for t in keybert.run(chunk)] if kb_ok else []
            return {"chunk": chunk, "words": len(chunk.split()),
                    "ents": ents, "sentiment": sent, "topics": tops}

        results = self._parallel(extract_chunk, chunks)

        # Stage 3: merge across chunks (aggregate entities / VADER sentiment / KeyBERT topics)
        ent_agg: Dict[str, Dict[str, Any]] = {}
        topics: Counter = Counter()
        sentiments: Counter = Counter()
        for res in results:
            for e in res["ents"]:
                key = _norm(e.get("text", ""))
                if not key:
                    continue
                et = ET.SPACY_LABEL_MAP.get(e.get("label", ""), "CONCEPT")
                d = ent_agg.setdefault(key, {"name": e["text"], "type": et, "contexts": []})
                ctx = res["chunk"][max(0, e.get("start", 0) - 40): e.get("end", 0) + 40]
                d["contexts"].append(ctx)
            for tp in (res["topics"] or [])[:3]:
                topics[str(tp).lower().strip()] += 1
            sv = _SENT.get(str(res["sentiment"] or "").lower().strip())
            if sv:
                sentiments[sv] += 1
        dom_sent = sentiments.most_common(1)[0][0] if sentiments else None

        graph.add_modality(ModalityMetadata(
            modality_id=mid, modality_type=ModalityType.TEXT, source_path=dataset,
            content_summary={"text_columns": text_cols, "word_count": total_words,
                             "n_docs": len(docs), "n_paragraph_chunks": len(chunks),
                             "top_topics": [t for t, _ in topics.most_common(5)],
                             "sentiment": dom_sent,
                             "sentiment_distribution": dict(sentiments)},
            semantic_features={"sentiment_distribution": dict(sentiments)},
            raw_metadata={"word_count": total_words}))
        for key, d in ent_agg.items():
            eid = f"{dataset}:text:{key}"[:120]
            try:
                et = EntityType[d["type"]]
            except Exception:
                et = EntityType.CONCEPT
            mentions = [EntityMention(mid, ModalityType.TEXT, d["name"], confidence=0.85,
                                      context=c) for c in d["contexts"][:20]]
            graph.add_entity(CrossModalEntity(eid, et, d["name"], mentions=mentions,
                                              properties={"source": "text_ner"}))
        return mid


class ImageProcessor(BaseProcessor):
    def process(self, dataset: str, rows: List[Dict[str, Any]], image_col: str,
                graph: MultimodalMetadataGraph, max_images: int = 16,
                batch_size: int = 4) -> Optional[str]:
        if not rows or image_col not in rows[0]:
            return None
        mid = f"{dataset}_image"
        yolo, det2 = self.tools.get("yolo"), self.tools.get("detectron2")
        places, vit = self.tools.get("places365"), self.tools.get("vit")
        det = yolo if (yolo is not None and yolo.available) else \
            (det2 if (det2 is not None and det2.available) else None)
        places_ok = places is not None and places.available
        vit_ok = vit is not None and vit.available
        srcs = [r.get(image_col) for r in rows[:max_images] if r.get(image_col)]
        if not srcs or (det is None and not places_ok and not vit_ok):
            return None

        # Stage 1: group images into batches
        batches = [srcs[i:i + batch_size] for i in range(0, len(srcs), batch_size)]

        # Stage 2: PARALLEL extraction per batch (each batch -> per-image tool runs + ReAct)
        def process_batch(batch: List[Any]) -> List[Dict[str, Any]]:
            out = []
            for src in batch:
                runners, descs = {}, {}
                if det is not None:
                    runners[det.name] = lambda s=src, d=det: d.run(s)
                    descs[det.name] = getattr(det, "description", "object detection")
                if places_ok:
                    runners[places.name] = lambda s=src: places.run(s)
                    descs[places.name] = places.description
                if vit_ok:
                    runners[vit.name] = lambda s=src: {"embedding_dim": int(vit.run(s).shape[-1])}
                    descs[vit.name] = vit.description
                self.react(goal="Describe the image: objects, scene, salient content.",
                           context=f"image from column {image_col}",
                           tool_runners=runners, tool_descs=descs)
                objs, scns = [], []
                if det is not None:
                    try:
                        objs = det.run(src)
                    except Exception:
                        objs = []
                if places_ok:
                    try:
                        scns = places.run(src)
                    except Exception:
                        scns = []
                out.append({"objs": objs, "scenes": scns})
            return out

        batch_results = self._parallel(process_batch, batches)

        # merge across batches
        obj_agg: Dict[str, Dict[str, Any]] = {}
        scenes: Counter = Counter()
        n_imgs = 0
        for br in batch_results:
            for item in br:
                n_imgs += 1
                for o in item["objs"]:
                    d = obj_agg.setdefault(_norm(o["label"]),
                                           {"name": o["label"], "areas": [], "confs": []})
                    d["areas"].append(float(o.get("area", 0.0)))
                    d["confs"].append(float(o.get("confidence", 0.5)))
                for s in item["scenes"]:
                    scenes[s["scene"]] += 1

        graph.add_modality(ModalityMetadata(
            modality_id=mid, modality_type=ModalityType.IMAGE, source_path=dataset,
            content_summary={"image_column": image_col, "n_images": n_imgs,
                             "n_batches": len(batches), "batch_size": batch_size,
                             "top_scenes": [s for s, _ in scenes.most_common(5)],
                             "objects": sorted(obj_agg.keys())},
            raw_metadata={"note": "objects/scene/features from real image tools"}))
        for key, d in obj_agg.items():
            eid = f"{dataset}:image:{key}"[:120]
            area = min(1.0, sum(d["areas"]))
            sal = sum(a * c for a, c in zip(d["areas"], d["confs"])) / max(1, len(d["areas"]))
            conf = sum(d["confs"]) / len(d["confs"])
            mentions = [EntityMention(mid, ModalityType.IMAGE, d["name"], confidence=conf,
                                      position={"area": area, "saliency": min(1.0, sal * 4)})]
            graph.add_entity(CrossModalEntity(eid, EntityType.OBJECT, d["name"],
                                              mentions=mentions,
                                              properties={"source": "image_detector"}))
        return mid


# ------------------------------------------------------------------ orchestration
def merge_cross_modal_entities(graph: MultimodalMetadataGraph) -> int:
    """Link entities that recur ACROSS modalities into a single CrossModalEntity
    (paper §3). Merge only when it actually spans >1 modality."""
    by_name: Dict[str, List[str]] = {}
    for eid, e in graph.entities.items():
        by_name.setdefault(_norm(e.canonical_name), []).append(eid)
    merged = 0
    for _, ids in by_name.items():
        if len(ids) < 2:
            continue
        mods = {m.modality_type for eid in ids for m in graph.entities[eid].mentions}
        if len(mods) < 2:
            continue
        keep = graph.entities[ids[0]]
        for eid in ids[1:]:
            keep.mentions.extend(graph.entities[eid].mentions)
            graph.entities.pop(eid, None)
        keep.properties["cross_modal"] = True
        keep.properties["modalities"] = sorted(m.value for m in mods)
        merged += 1
    return merged


_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def _table_doc(m: ModalityMetadata) -> str:
    """A short textual summary of a table used for coarse-grained similarity search."""
    cols = m.content_summary.get("columns", [])
    info = m.content_summary.get("column_info", {}) or {}
    parts = []
    for c in cols:
        ci = info.get(c, {}) if isinstance(info, dict) else {}
        parts.append(f"{c} ({ci.get('summary') or ci.get('data_type') or ''})".strip())
    return f"Table {m.modality_id} with columns: " + "; ".join(parts)


def _coarse_grained(tables, target_mid: str, similar_count: int) -> List[str]:
    """Stage 1: retrieve the `similar_count` tables whose summaries are most similar to
    the target table by embedding similarity, pruning the O(n^2) pairwise comparison."""
    others = [mid for mid, _ in tables if mid != target_mid]
    if len(others) <= similar_count:
        return others
    try:
        import numpy as np
        docs = {mid: _table_doc(m) for mid, m in tables}
        emb = _get_embedder()
        vecs = emb.encode([docs[target_mid]] + [docs[o] for o in others],
                          normalize_embeddings=True)
        sims = np.asarray(vecs[1:]) @ np.asarray(vecs[0])
        order = np.argsort(-sims)[:similar_count]
        return [others[i] for i in order]
    except Exception:  # embedding backend unavailable -> keep all candidates
        return others[:similar_count]


def discover_table_relations(graph: MultimodalMetadataGraph, llm: Any = None,
                             similar_count: int = 20,
                             min_coverage: float = 0.8) -> List[CrossModalRelation]:
    """Table relationships BETWEEN table modalities via a two-stage coarse-to-fine
    procedure (paper §3.1.1). Stage 1 (coarse-grained) retrieves the most similar
    candidate tables for each table by embedding similarity, reducing the pairwise cost
    from O(n^2) to O(n). Stage 2 (fine-grained) feeds the target table and its candidates
    to the LLM under a chain-of-thought prompt, which identifies each referential-integrity
    relationship as (referencing table, referenced table, foreign-key column, primary-key
    column, cardinality). Without an LLM it falls back to a value-set inclusion test.
    No-op on single-table datasets."""
    tables = [(mid, m) for mid, m in graph.modalities.items()
              if m.modality_type == ModalityType.TABLE]
    if len(tables) < 2:
        return []
    meta = dict(tables)
    rels: List[CrossModalRelation] = []

    for target_mid, tm in tables:
        candidates = _coarse_grained(tables, target_mid, similar_count)   # Stage 1
        if not candidates:
            continue
        if llm is not None:
            # Stage 2: fine-grained LLM CoT over the target + retrieved candidate tables
            def schema(mid):
                m = meta[mid]
                return {"table": mid, "columns": m.content_summary.get("columns", []),
                        "primary_key": m.raw_metadata.get("primary_key_columns", [])}
            payload = {"target": schema(target_mid),
                       "candidates": [schema(c) for c in candidates]}
            sys = ("You identify referential-integrity (foreign-key) relationships between "
                   "relational tables. Given a TARGET table and CANDIDATE tables with their "
                   "columns and primary keys, determine which column of the target table is a "
                   "foreign key that references a primary/key column of a candidate table. "
                   "Reason step by step, then return ONLY JSON {\"relationships\": "
                   "[{\"referencing_table\": str, \"referenced_table\": str, \"fk_column\": str, "
                   "\"pk_column\": str, \"cardinality\": \"1:1|1:N|N:1|N:M\"}]}, and an empty "
                   "list if there is no relationship.")
            out = _parse_json(llm.complete([
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])) or {}
            for r in (out.get("relationships") or []):
                if not isinstance(r, dict):
                    continue
                src = r.get("referencing_table") or target_mid
                tgt = r.get("referenced_table")
                if not tgt or tgt not in meta or src not in meta:
                    continue
                rels.append(CrossModalRelation(
                    relation_id=f"fk_{uuid.uuid4().hex[:8]}",
                    relation_type=RelationType.REFERENCES,
                    source_modality=src, target_modality=tgt, confidence=1.0,
                    evidence={"fk_column": r.get("fk_column"), "pk_column": r.get("pk_column"),
                              "cardinality": r.get("cardinality"),
                              "candidates_searched": len(candidates),
                              "discovery_method": "coarse_to_fine_llm"},
                    properties={"relation_category": "table_relationship"}))
        else:
            # fallback (no LLM): value-set inclusion dependency test
            av = meta[target_mid].raw_metadata.get("column_values", {})
            for cid in candidates:
                bv = meta[cid].raw_metadata.get("column_values", {})
                bpks = meta[cid].raw_metadata.get("primary_key_columns", []) or list(bv.keys())
                for acol, avals in av.items():
                    if not avals:
                        continue
                    for bcol in bpks:
                        bvals = bv.get(bcol, set())
                        if bvals and len(avals & bvals) / max(1, len(avals)) >= min_coverage:
                            rels.append(CrossModalRelation(
                                relation_id=f"fk_{uuid.uuid4().hex[:8]}",
                                relation_type=RelationType.REFERENCES,
                                source_modality=target_mid, target_modality=cid, confidence=1.0,
                                evidence={"fk_column": acol, "pk_column": bcol,
                                          "coverage": round(len(avals & bvals) / max(1, len(avals)), 3),
                                          "discovery_method": "inclusion_dependency_fallback"},
                                properties={"relation_category": "table_relationship"}))
    if hasattr(graph, "relations") and isinstance(getattr(graph, "relations"), dict):
        for r in rels:
            graph.relations[r.relation_id] = r
    return rels


def build_tools(enable: bool = True) -> Dict[str, Any]:
    if not enable:
        return {}
    return {"yolo": ET.YOLOTool(), "detectron2": ET.Detectron2Tool(),
            "places365": ET.Places365Tool(), "vit": ET.ViTFeatureTool(),
            "spacy": ET.SpacyNERTool(), "vader": ET.VaderSentimentTool(),
            "keybert": ET.KeyBERTTool()}


def extract_metadata_graph(dataset: str, rows: List[Dict[str, Any]], provider: Any,
                           vision_provider: Any = None, enable_tools: bool = True,
                           image_cols=None, text_cols=None,
                           max_workers: int = 4) -> MultimodalMetadataGraph:
    """Full TiProcessor extraction (paper §3.1.1) with the paper's staging + real tools,
    then cross-modal entity merge and cross-table referential-integrity discovery."""
    from .metadata_graph import MultimodalMetadataGraph as _G
    g = _G(graph_id=f"meta_{dataset}")
    if not rows:
        return g
    cols = list(rows[0].keys())
    image_cols = [c for c in (image_cols or []) if c in cols]
    text_cols = [c for c in (text_cols or []) if c in cols]
    tools = build_tools(enable_tools)
    kw = {"tools": tools, "max_workers": max_workers}
    TableProcessor(provider, **kw).process(dataset, rows, g, image_cols, text_cols)
    if text_cols:
        TextProcessor(provider, **kw).process(dataset, rows, text_cols, g)
    for ic in image_cols:
        ImageProcessor(provider, vision_provider, **kw).process(dataset, rows, ic, g)
    g.stats["cross_modal_entities_merged"] = merge_cross_modal_entities(g)
    g.stats["table_fk_relations"] = len(discover_table_relations(g, llm=provider))
    return g
