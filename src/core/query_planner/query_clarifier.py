"""TiQC — interactive query clarification (paper §4, Algorithm 2), integrated.

This aligns the component with the paper. The clarifier now implements the full
multi-round Algorithm 2 (ClarifyQuery):

    q ← q0
    for t ← 1 to K:
        A ← { a ∈ DetectAmbiguities(q, G_meta, L) | sev(a) ≥ θ_amb }
        if A = ∅: break
        a* ← argmax_{a∈A} sev(a)
        O ← GenerateOptions(a*, G_meta, L)
        u ← AskUser(a*, O)
        q ← RewriteQuery(q, a*, u, L)
    return q* ← q

Key paper elements now present (were missing in the single-round version):
  - DetectAmbiguities returns a RANKED LIST of ambiguities, each classified into one
    of the 6 taxonomy types and scored with a severity sev(a) ∈ [0,1] under a fixed
    rubric (an ambiguity is severe to the extent its competing interpretations diverge
    in the downstream plan — different operators, different modalities).
  - The K-round loop with the θ_amb severity gate, argmax-severity selection, and one
    question per round (K=3, θ_amb=0.5 by default; K=0 skips clarification).
  - AskUser: a human in live use, or an LLM user-simulator conditioned on a hidden
    ground-truth intent for reproducible batch evaluation (make_user_simulator).
  - Metadata grounding from TiMetagraph: options are drawn STRICTLY from the real
    columns / representative values / modality routing (grounded arm = TiQC +
    TiMetagraph; ungrounded ablates the grounding).

Backward-compatible entry points analyze()/rewrite() are retained; clarify_query() is
the Algorithm-2 core, and clarify_for_planner() runs it autonomously (no human).
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

AMBIGUITY_TYPES = [
    "cross_modal_referential", "intra_modal_attribute", "multimodal_intent",
    "schema_structure", "value_content", "temporal_spatial",
]

# Fixed severity rubric embedded in the detection prompt (paper §4): severity rises as
# the competing readings diverge in the downstream plan (different operators / modalities).
SEVERITY_RUBRIC = (
    "Score severity sev in [0,1]: 0.0-0.3 the readings resolve to the same execution "
    "plan (cosmetic); 0.4-0.6 they change predicates or projected columns but stay in "
    "one modality; 0.7-1.0 they require DIFFERENT operators or touch DIFFERENT "
    "modalities (e.g. a value in a table column vs a fact read from an image). Higher "
    "severity = more consequential for planning."
)


@dataclass
class Ambiguity:
    """A detected ambiguity a, classified and scored (paper §4)."""
    ambiguity_type: str            # one of AMBIGUITY_TYPES
    severity: float                # sev(a) ∈ [0,1]
    description: str
    question: str
    options: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"ambiguity_type": self.ambiguity_type, "severity": self.severity,
                "description": self.description, "question": self.question,
                "options": self.options}


def _parse_json(text: str) -> Any:
    t = str(text).strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        for op, cl in (("[", "]"), ("{", "}")):
            s, e = t.find(op), t.rfind(cl)
            if s != -1 and e != -1 and e > s:
                try:
                    return json.loads(t[s:e + 1])
                except Exception:
                    continue
    return None


class TiQCClarifier:
    """Grounded interactive query clarifier implementing Algorithm 2."""

    def __init__(self, provider: Any, dataset: Optional[str] = None,
                 metadata_graph: Any = None, grounded: bool = True,
                 theta_amb: float = 0.5, round_budget: int = 3):
        self.provider = provider
        self.dataset = dataset
        self.metadata_graph = metadata_graph
        self.grounded = grounded
        self.theta_amb = theta_amb            # θ_amb severity threshold
        self.round_budget = round_budget      # K

    # ---- grounding sourced from the TiMetagraph structures the paper names (§4):
    #      referential/value -> CrossModalEntity mentions; schema -> ModalityMetadata
    #      summaries; temporal/spatial -> the temporal/spatial dimensions ----
    def _graph(self, dataset: Optional[str]):
        if self.metadata_graph is not None:
            return self.metadata_graph
        if not self.grounded:
            return None
        try:
            from .metadata_graph_builder import build_metadata_graph
            self.metadata_graph = build_metadata_graph(dataset or self.dataset)
        except Exception:
            self.metadata_graph = None
        return self.metadata_graph

    def _grounding(self, dataset: Optional[str]) -> str:
        if not self.grounded:
            return ""
        g = self._graph(dataset)
        if g is None:  # fall back to the flat schema string if no graph is available
            try:
                from .metadata_graph_builder import schema_context
                return schema_context(dataset or self.dataset)
            except Exception:
                return ""
        from .metadata_graph import ModalityType
        parts: List[str] = []
        # (a) ModalityMetadata summaries -> schema_structure grounding
        for _, m in g.modalities.items():
            if m.modality_type == ModalityType.TABLE:
                cols = m.content_summary.get("columns", [])
                if cols:
                    parts.append(f"Table columns: {', '.join(map(str, cols))}.")
                img = m.raw_metadata.get("image_columns") or []
                txt = m.raw_metadata.get("text_columns") or []
                if img:
                    parts.append(f"Image column(s) {img} — visual attributes are resolvable "
                                 "ONLY from the pixels, not from any text column.")
                if txt:
                    parts.append(f"Long free-text column(s): {txt}.")
            if m.modality_type == ModalityType.TEXT and m.content_summary.get("top_topics"):
                parts.append(f"Text topics: {m.content_summary.get('top_topics')}.")
        # (b) CrossModalEntity mentions -> cross_modal_referential / value_content candidates
        by_col: Dict[str, List[str]] = {}
        cross_modal: List[str] = []
        for e in list(g.entities.values()):
            col = e.properties.get("column") or e.properties.get("source", "value")
            by_col.setdefault(col, []).append(e.canonical_name)
            if e.properties.get("cross_modal"):
                cross_modal.append(f"{e.canonical_name} ({'/'.join(e.properties.get('modalities', []))})")
        for col, names in list(by_col.items())[:12]:
            uniq = list(dict.fromkeys(names))[:6]
            parts.append(f"Candidate values for '{col}': {', '.join(map(str, uniq))}.")
        if cross_modal:
            parts.append("Cross-modal entities (appear in multiple modalities): "
                         + ", ".join(cross_modal[:8]) + ".")
        # (c) temporal / spatial dimensions
        for _, m in g.modalities.items():
            ti = getattr(m, "temporal_info", None)
            if ti and getattr(ti, "time_expressions", None):
                parts.append(f"Temporal expressions: {list(ti.time_expressions)[:5]}.")
            si = getattr(m, "spatial_info", None)
            if si and getattr(si, "locations", None):
                parts.append(f"Spatial locations: {list(si.locations)[:5]}.")
        return "\n".join(parts)

    # kept for backward compatibility (older callers)
    def _schema(self, dataset: Optional[str]) -> str:
        return self._grounding(dataset)

    def _chat(self, system: str, user: str) -> str:
        return self.provider.complete([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])

    # ---- Algorithm 2, line 4: DetectAmbiguities (4-step: context, CoT+severity,
    #      grounded options, ranking). Returns the FULL ranked list. ----
    def detect_ambiguities(self, query: str, dataset: Optional[str] = None) -> List[Ambiguity]:
        ctx = self._schema(dataset)
        ctx_block = (f"\nMetadata context (schema/representative values/modality routing, "
                     f"from TiMetagraph):\n{ctx}\n") if ctx else "\n(No metadata context provided.)\n"
        sys = (
            "You are TiQC, the interactive query-clarification module of a multimodal "
            "data-analytics system. Given a possibly-ambiguous natural-language query and "
            "the dataset's metadata context, DETECT every consequential ambiguity. For "
            "each, (1) classify it into exactly one of the six types "
            f"{{{', '.join(AMBIGUITY_TYPES)}}}, (2) assign a severity. {SEVERITY_RUBRIC} "
            "(3) Write ONE clarifying question and 2-3 concrete options, drawing options "
            "STRICTLY from the provided metadata context — never invent a column or value "
            "that is not present. Return a JSON list, ranked by severity descending: "
            "[{\"ambiguity_type\": str, \"severity\": float, \"description\": str, "
            "\"question\": str, \"options\": [str, ...]}, ...]. Return [] if the query is "
            "already unambiguous given the metadata."
        )
        raw = self._parse_json(self._chat(
            sys, f"Dataset: {dataset or self.dataset}{ctx_block}\nQuery: {query}"))
        out: List[Ambiguity] = []
        if isinstance(raw, dict):
            raw = raw.get("ambiguities") or [raw]
        if not isinstance(raw, list):
            return out
        for a in raw:
            if not isinstance(a, dict):
                continue
            at = a.get("ambiguity_type")
            try:
                sev = float(a.get("severity", 0.0) or 0.0)
            except Exception:
                sev = 0.0
            out.append(Ambiguity(
                ambiguity_type=at if at in AMBIGUITY_TYPES else (at or "value_content"),
                severity=max(0.0, min(1.0, sev)),
                description=str(a.get("description", "")),
                question=str(a.get("question", "")),
                options=[str(o) for o in (a.get("options") or [])]))
        out.sort(key=lambda x: x.severity, reverse=True)
        return out

    # ---- Algorithm 2, line 7: GenerateOptions (grounded), used if a* has none ----
    def generate_options(self, query: str, amb: Ambiguity,
                         dataset: Optional[str] = None) -> List[str]:
        if amb.options:
            return amb.options
        ctx = self._schema(dataset)
        sys = ("You are TiQC. Produce 2-3 concrete, mutually-exclusive clarification "
               "options for the given ambiguity, each grounded STRICTLY in the schema. "
               "Return a JSON list of strings.")
        raw = self._parse_json(self._chat(
            sys, f"Schema:\n{ctx}\nQuery: {query}\nAmbiguity: {amb.description}\n"
                 f"Question: {amb.question}"))
        return [str(o) for o in raw] if isinstance(raw, list) else []

    # ---- Algorithm 2, line 9: RewriteQuery ----
    def rewrite_query(self, query: str, amb: Ambiguity, answer: str,
                      dataset: Optional[str] = None) -> str:
        ctx = self._schema(dataset)
        ctx_block = f"\nSchema:\n{ctx}\n" if ctx else "\n"
        sys = ("You are TiQC. Rewrite the query into a SINGLE unambiguous, executable "
               "query that resolves the stated ambiguity per the user's answer. Keep it "
               "grounded in the schema. Output ONLY the rewritten query text.")
        r = self._chat(sys, f"Dataset: {dataset or self.dataset}{ctx_block}"
                            f"Original query: {query}\nAmbiguity: {amb.description}\n"
                            f"User answer: {answer}\nRewritten query:")
        return str(r).strip().strip('"').strip()

    # ---- Algorithm 2 core: ClarifyQuery(q0, G_meta, L; θ_amb; K) ----
    def clarify_query(self, query0: str, ask_user: Callable[[Ambiguity, List[str]], str],
                      dataset: Optional[str] = None,
                      round_budget: Optional[int] = None,
                      theta_amb: Optional[float] = None) -> Dict[str, Any]:
        """Multi-round clarification. `ask_user(a*, options) -> answer` is the human in
        live use or an LLM user-simulator in batch eval. Returns the clarified query plus
        a per-round trace. K=0 returns q0 unchanged (the no-clarification configuration)."""
        K = self.round_budget if round_budget is None else round_budget
        tau = self.theta_amb if theta_amb is None else theta_amb
        q = query0
        trace: List[Dict[str, Any]] = []
        for t in range(1, K + 1):
            ambs = [a for a in self.detect_ambiguities(q, dataset) if a.severity >= tau]
            if not ambs:                                   # A = ∅ -> break
                break
            a_star = max(ambs, key=lambda a: a.severity)   # argmax severity
            options = self.generate_options(q, a_star, dataset)
            answer = ask_user(a_star, options)             # human or simulator
            q_new = self.rewrite_query(q, a_star, answer, dataset)
            trace.append({"round": t, "ambiguity": a_star.to_dict(),
                          "options": options, "answer": answer,
                          "query_before": q, "query_after": q_new})
            q = q_new or q
        return {"clarified_query": q, "rounds": len(trace), "trace": trace,
                "original_query": query0}

    # ---- autonomous use inside the planner (no human): a rational-user simulator that
    #      always picks the most schema-grounded reading (option 0, which the detector
    #      ranks best-first) ----
    def clarify_for_planner(self, query: str, dataset: Optional[str] = None) -> Dict[str, Any]:
        def auto(amb: Ambiguity, options: List[str]) -> str:
            return f"Resolve as: {options[0]}" if options else "Use the most likely reading."
        res = self.clarify_query(query, auto, dataset)
        return {"refined_query": res["clarified_query"],
                "needs_clarification": res["rounds"] > 0,
                "rounds": res["rounds"], "trace": res["trace"]}

    # ---- AskUser simulator for reproducible batch evaluation (paper §4): an LLM
    #      conditioned on the hidden ground-truth intent picks the matching option ----
    def make_user_simulator(self, hidden_intent: str) -> Callable[[Ambiguity, List[str]], str]:
        def ask(amb: Ambiguity, options: List[str]) -> str:
            sys = ("You simulate a human analyst answering a clarifying question. Choose "
                   "the option matching your TRUE intent (or state it in one short "
                   "sentence). Answer concisely.")
            return self._chat(sys, f"Your true intent: {hidden_intent}\n"
                                   f"Question: {amb.question}\nOptions: {options}")
        return ask

    # ---- backward-compatible single-shot helpers (used by older eval paths) ----
    def analyze(self, query: str, dataset: Optional[str] = None) -> Dict[str, Any]:
        ambs = self.detect_ambiguities(query, dataset)
        if not ambs:
            return {"needs_clarification": False, "ambiguity_type": None,
                    "question": "", "options": [], "confidence": 1.0, "severity": 0.0}
        a = ambs[0]
        return {"needs_clarification": a.severity >= self.theta_amb,
                "ambiguity_type": a.ambiguity_type, "question": a.question,
                "options": a.options, "confidence": a.severity, "severity": a.severity}

    def rewrite(self, query: str, answer: str, dataset: Optional[str] = None) -> str:
        # thin wrapper matching the old signature (no explicit ambiguity object)
        amb = Ambiguity("value_content", 1.0, "clarified by user", "", [])
        return self.rewrite_query(query, amb, answer, dataset)

    def _parse_json(self, text: str) -> Any:
        return _parse_json(text)
