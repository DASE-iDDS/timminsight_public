"""Cross-modal relationship discovery (paper §3.1.2, Algorithm 1), integrated.

This replaces the previous Jaccard-of-modality-sets heuristic with the paper's
Algorithm 1 (DiscoverCrossModalRelations):

  Stage 1 — recall-oriented statistical pre-filter. For every entity pair, compute the
    geometric-mean normalized co-occurrence J_w (Eq. 1) over the per-modality normalized
    frequency scores f(e,m) of Table 2, and keep only pairs with J_w >= θ_cooc. This
    bounds the number of LLM calls to |{(e1,e2): J_w >= θ_cooc}| << C(n,2).
  Stage 2 — semantic typing. Invoke the LLM ONCE, in batch, over the surviving pairs
    (BatchInferRelationTypes), guided by the five-stage CoT of the paper; returns a
    relationship type and a confidence for each pair.
  Stage 3 — precision-oriented semantic filter. Discard pairs whose inferred confidence
    is below θ_rel; each surviving pair becomes an edge with weight J_w·conf and the
    inferred relationship type.

Table 2 (normalized frequency score f(e,m) ∈ [0,1], per modality):
  Text  : min(1, log(1+n)/log(1+L/α)) · c        (n mentions, L words, c avg conf, α=100)
  Image : min(1, 0.5·s + 0.4·A + 0.1·c)          (s saliency, A area proportion, c conf)
  Table : 0.6·(r/R) + 0.4·w_c                     (r rows seen, R total rows, w_c col-weight)

An `llm` provider is optional: with it, Stage 2 is the paper's LLM batch inference; with
None it degrades to a statistical-only typing (used by construction-stat probes), which
is documented rather than presented as the paper's method.
"""
import json
import logging
import math
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .metadata_graph import (
    CrossModalEntity, EntityType, ModalityType,
    MultimodalMetadataGraph, CrossModalRelation, RelationType,
)

logger = logging.getLogger(__name__)

_ALPHA = 100.0  # Table 2 text normalizer α

# LLM relation-type vocabulary -> RelationType enum (Stage 2 output space)
_REL_VOCAB = {
    "represents": RelationType.REPRESENTS, "describes": RelationType.REPRESENTS,
    "contains": RelationType.CONTAINS, "part_of": RelationType.BELONGS_TO,
    "belongs_to": RelationType.BELONGS_TO, "references": RelationType.REFERENCES,
    "similar_to": RelationType.SIMILAR_TO, "derived_from": RelationType.DERIVED_FROM,
    "temporal_related": RelationType.TEMPORAL_RELATED,
    "spatial_related": RelationType.SPATIAL_RELATED,
    "associated_with": RelationType.ASSOCIATED_WITH,
}


@dataclass
class EntityRelationConfig:
    """Algorithm 1 thresholds."""
    theta_cooc: float = 0.3   # θ_cooc  Stage-1 recall prefilter
    theta_rel: float = 0.5    # θ_rel   Stage-3 precision filter
    max_pairs_to_llm: int = 200  # safety cap on Stage-2 batch size
    # legacy aliases kept so existing callers/config don't break
    cooccurrence_threshold: float = 0.3
    semantic_similarity_threshold: float = 0.4


class EntityRelationshipDiscoverer:
    """Algorithm 1: DiscoverCrossModalRelations(E, L)."""

    def __init__(self, config: Optional[EntityRelationConfig] = None):
        self.config = config or EntityRelationConfig()
        self.discovered_relations: List[CrossModalRelation] = []

    # ---------- Table 2: normalized per-modality frequency score f(e, m) ----------
    def _freq_score(self, entity: CrossModalEntity, modality_id: str,
                    graph: MultimodalMetadataGraph) -> float:
        ms = [m for m in entity.mentions if m.modality_id == modality_id]
        if not ms:
            return 0.0
        meta = graph.modalities.get(modality_id)
        mtype = meta.modality_type if meta else (ms[0].modality_type)
        c = sum(m.confidence for m in ms) / len(ms)          # avg confidence

        if mtype == ModalityType.TEXT:
            n = len(ms)
            L = 0
            if meta:
                L = int(meta.content_summary.get("word_count")
                        or meta.raw_metadata.get("word_count") or 0)
            if L <= 0:  # fall back to summed mention/context length in words
                L = sum(len(str(m.context or m.mention_text).split()) for m in ms) or n
            denom = math.log(1.0 + max(L, 1) / _ALPHA)
            base = (math.log(1.0 + n) / denom) if denom > 1e-9 else 1.0
            return min(1.0, base) * c

        if mtype == ModalityType.IMAGE:
            def g(m, k, d):
                return float((m.position or {}).get(k, d))
            s = sum(g(m, "saliency", 0.5) for m in ms) / len(ms)   # avg visual saliency
            A = min(1.0, sum(g(m, "area", 0.0) for m in ms))       # total area proportion
            return min(1.0, 0.5 * s + 0.4 * A + 0.1 * c)

        # TABLE (default): 0.6·(r/R) + 0.4·w_c
        R = int(meta.content_summary.get("n_rows", 0)) if meta else 0
        rows = {(m.position or {}).get("row") for m in ms if (m.position or {}).get("row") is not None}
        r = len(rows) if rows else len(ms)
        r_ratio = (r / R) if R > 0 else min(1.0, r / max(1, len(ms)))
        pk_cols = set(meta.raw_metadata.get("primary_key_columns", [])) if meta else set()
        cols = {(m.position or {}).get("column") or m.context for m in ms}
        if cols:
            w_c = sum(1.0 if c0 in pk_cols else 0.5 for c0 in cols) / len(cols)
        else:
            w_c = 0.5
        return min(1.0, 0.6 * r_ratio + 0.4 * w_c)

    # ---------- Eq. 1: geometric-mean normalized co-occurrence J_w ----------
    def _jw(self, e1: CrossModalEntity, e2: CrossModalEntity,
            graph: MultimodalMetadataGraph) -> float:
        M1 = {m.modality_id for m in e1.mentions}
        M2 = {m.modality_id for m in e2.mentions}
        if not M1 or not M2:
            return 0.0
        inter, union = M1 & M2, M1 | M2
        num = sum(math.sqrt(max(0.0, self._freq_score(e1, m, graph) * self._freq_score(e2, m, graph)))
                  for m in inter)
        den = sum(max(self._freq_score(e1, m, graph), self._freq_score(e2, m, graph)) for m in union)
        return (num / den) if den > 1e-9 else 0.0

    # ---------- Stage 2: BatchInferRelationTypes(pairs, L) ----------
    def _batch_infer_types(self, pairs: List[Tuple[CrossModalEntity, CrossModalEntity, float]],
                           llm: Any) -> List[Tuple[RelationType, float]]:
        if not pairs:
            return []
        if llm is None:  # documented statistical fallback (no LLM available)
            return [(RelationType.ASSOCIATED_WITH, min(1.0, j)) for _, _, j in pairs]
        sys = (
            "You infer cross-modal relationships between entity pairs. For EACH pair follow "
            "a five-step reasoning: (1) identify each entity's semantic type and salient "
            "attributes; (2) analyze the contextual relationship across their modalities; "
            "(3) examine their co-occurrence pattern; (4) determine the MOST SPECIFIC "
            "relationship type from {represents, contains, belongs_to, references, "
            "similar_to, derived_from, temporal_related, spatial_related, associated_with}; "
            "(5) give a confidence in [0,1] (0 if the pair is unrelated / spurious). Return "
            "ONLY a JSON list [{\"id\": int, \"type\": str, \"confidence\": float}, ...] "
            "covering every id."
        )
        by_id: Dict[int, Tuple[RelationType, float]] = {}
        CHUNK = 25   # keep each LLM response small enough to parse in full
        for start in range(0, len(pairs), CHUNK):
            chunk = pairs[start:start + CHUNK]
            items = [{"id": start + k, "e1": e1.canonical_name, "e1_type": e1.entity_type.value,
                      "e2": e2.canonical_name, "e2_type": e2.entity_type.value,
                      "e1_modalities": sorted({m.modality_type.value for m in e1.mentions}),
                      "e2_modalities": sorted({m.modality_type.value for m in e2.mentions})}
                     for k, (e1, e2, _) in enumerate(chunk)]
            raw = llm.complete([{"role": "system", "content": sys},
                                {"role": "user", "content": json.dumps(items, ensure_ascii=False)}])
            parsed = _parse_json(raw) or []
            if isinstance(parsed, list):
                for o in parsed:
                    if not isinstance(o, dict):
                        continue
                    rt = _REL_VOCAB.get(str(o.get("type", "")).lower().strip(),
                                        RelationType.ASSOCIATED_WITH)
                    try:
                        conf = max(0.0, min(1.0, float(o.get("confidence", 0.0))))
                    except Exception:
                        conf = 0.0
                    try:
                        by_id[int(o.get("id", -1))] = (rt, conf)
                    except Exception:
                        continue
        # any pair the LLM omitted -> low-confidence associated_with (dropped by θ_rel)
        return [by_id.get(i, (RelationType.ASSOCIATED_WITH, 0.0)) for i in range(len(pairs))]

    # ---------- Algorithm 1 top level ----------
    def discover_entity_relations(self, metadata_graph: MultimodalMetadataGraph,
                                  llm: Any = None) -> List[CrossModalRelation]:
        entities = list(metadata_graph.entities.values())
        cfg = self.config
        # Stage 1: statistical recall pre-filter
        pairs: List[Tuple[CrossModalEntity, CrossModalEntity, float]] = []
        for i, e1 in enumerate(entities):
            M1 = {m.modality_id for m in e1.mentions}
            if not M1:
                continue
            for e2 in entities[i + 1:]:
                if not {m.modality_id for m in e2.mentions}:
                    continue
                j = self._jw(e1, e2, metadata_graph)
                if j >= cfg.theta_cooc:
                    pairs.append((e1, e2, j))
        pairs.sort(key=lambda p: p[2], reverse=True)
        if len(pairs) > cfg.max_pairs_to_llm:
            logger.info("Stage-1 kept %d pairs; capping LLM batch to %d",
                        len(pairs), cfg.max_pairs_to_llm)
            pairs = pairs[: cfg.max_pairs_to_llm]

        # Stage 2: LLM batch semantic typing
        typed = self._batch_infer_types(pairs, llm)

        # Stage 3: precision filter + edge construction
        relations: List[CrossModalRelation] = []
        for (e1, e2, j), (rtype, conf) in zip(pairs, typed):
            if conf < cfg.theta_rel:
                continue
            relations.append(CrossModalRelation(
                relation_id=f"rel_{uuid.uuid4().hex[:8]}",
                relation_type=rtype,
                source_modality=e1.entity_id, target_modality=e2.entity_id,
                confidence=round(j * conf, 4),        # edge weight = J_w · conf
                evidence={"J_w": round(j, 4), "type_confidence": round(conf, 4),
                          "discovery_method": "algorithm1_two_stage",
                          "e1": e1.canonical_name, "e2": e2.canonical_name},
                properties={"relation_category": "cross_modal_relationship"}))
        self.discovered_relations = relations
        logger.info("Algorithm 1: %d entities -> %d Stage-1 pairs -> %d edges (θ_cooc=%.2f, θ_rel=%.2f)",
                    len(entities), len(pairs), len(relations), cfg.theta_cooc, cfg.theta_rel)
        return relations

    def get_entity_relationship_summary(self) -> Dict[str, Any]:
        rels = self.discovered_relations
        if not rels:
            return {"total_relations": 0}
        by_type: Dict[str, int] = defaultdict(int)
        for r in rels:
            by_type[r.relation_type.value] += 1
        return {"total_relations": len(rels), "relations_by_type": dict(by_type),
                "average_weight": round(sum(r.confidence for r in rels) / len(rels), 4),
                "config": {"theta_cooc": self.config.theta_cooc, "theta_rel": self.config.theta_rel}}


def _parse_json(text: str) -> Any:
    t = str(text).strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        s, e = t.find("["), t.rfind("]")
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(t[s:e + 1])
            except Exception:
                pass
    return None
