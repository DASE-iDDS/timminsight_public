"""
Operator Compatibility Graph (OCG) for Multimodal Query Planning.

Defines a directed graph G = (V, E, w) where V = 16 logical operator types,
E = valid parent→child transitions, and w: E → [0,1] compatibility scores.
Constrains the MCTS logical plan search space so only structurally valid
plans are explored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Modality Types
# ---------------------------------------------------------------------------

class ModalityType(Enum):
    TABULAR = "tabular"
    IMAGE = "image"
    TEXT = "text"
    EMBEDDING = "embedding"
    MIXED = "mixed"
    ANY = "any"


# ---------------------------------------------------------------------------
# Operator Specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OperatorSpec:
    """Structural specification for a logical operator type."""
    operator_type: str
    min_children: int
    max_children: int
    input_modalities: FrozenSet[ModalityType]
    output_modality: ModalityType
    is_leaf: bool
    is_terminal: bool
    category: str  # "access", "transform", "cross_modal", "output"


# ---------------------------------------------------------------------------
# OCG Edge
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OCGEdge:
    """A directed edge in the OCG: parent_type → child_type."""
    parent_type: str
    child_type: str
    compatibility: float
    modality_constraint: Optional[ModalityType] = None


# ---------------------------------------------------------------------------
# Query Context
# ---------------------------------------------------------------------------

@dataclass
class QueryContext:
    """Parsed query requirements that guide MCTS search."""
    query_text: str
    required_operators: Set[str]
    optional_operators: Set[str]
    root_operator_type: str
    target_modalities: Set[ModalityType]
    estimated_data_size: int = 1000

    @classmethod
    def from_query(cls, query_text: str) -> "QueryContext":
        """Derive query context via keyword pattern matching."""
        text = query_text.lower()
        required: Set[str] = {"SCAN"}
        optional: Set[str] = set()
        root = "SCAN"
        modalities: Set[ModalityType] = {ModalityType.TABULAR}

        # Visualization detection
        if re.search(r"\b(plot|chart|graph|visuali[sz]e|draw|show)\b", text):
            root = "VISUALIZE"
            required.add("VISUALIZE")

        # Export detection
        if re.search(r"\b(export|save|download|write)\b", text):
            if root == "SCAN":
                root = "EXPORT"
            required.add("EXPORT")

        # Aggregation detection
        if re.search(
            r"\b(count|sum|average|mean|total|group|aggregate|for each|per|by)\b",
            text,
        ):
            required.add("AGGREGATE")

        # Semantic search detection
        if re.search(
            r"\b(depicting|showing|containing|about|similar|like|resembl|match)\b",
            text,
        ):
            required.add("SEMANTIC_SEARCH")
            modalities.add(ModalityType.EMBEDDING)

        # Content extraction detection
        if re.search(r"\b(detect|extract|identify|recogni[sz]e|analyz)\b", text):
            required.add("CONTENT_EXTRACT")
            modalities.add(ModalityType.IMAGE)

        # Cross-modal detection
        if re.search(r"\b(cross.?modal|image.?text|caption|describe)\b", text):
            required.add("CROSS_MODAL_MATCH")
            modalities.add(ModalityType.IMAGE)
            modalities.add(ModalityType.TEXT)

        # Feature transform detection
        if re.search(r"\b(transform|embed|pca|tsne|umap|vectori[sz]e)\b", text):
            required.add("FEATURE_TRANSFORM")
            modalities.add(ModalityType.EMBEDDING)

        # Similarity join detection
        if re.search(r"\b(similar|nearest|closest|knn)\b", text):
            optional.add("SIMILARITY_JOIN")

        # Join detection
        if re.search(r"\b(join|combine|link|connect|merge)\b", text):
            required.add("JOIN")

        # Filter detection
        if re.search(
            r"\b(filter|where|condition|threshold|greater|less|between|over|above|under|below|higher|lower|before|after)\b",
            text,
        ):
            required.add("FILTER")
        else:
            optional.add("FILTER")

        # Sort detection
        if re.search(r"\b(sort|order|rank|top|bottom|highest|lowest)\b", text):
            required.add("SORT")
        else:
            optional.add("SORT")

        # Top-N / limit detection
        if re.search(r"\b(top\s+\d+|bottom\s+\d+|limit|first\s+\d+)\b", text):
            required.add("LIMIT")
            required.add("SORT")
        else:
            optional.add("LIMIT")

        optional.add("PROJECT")
        optional.add("DISTINCT")

        optional -= required

        # Image modality detection
        if re.search(r"\b(image|photo|picture|painting|artwork|visual)\b", text):
            modalities.add(ModalityType.IMAGE)

        # Infer root operator from required set (highest in execution hierarchy)
        if root == "SCAN" and len(required) > 1:
            root_priority = [
                "VISUALIZE", "EXPORT", "AGGREGATE", "SORT", "DISTINCT",
                "LIMIT", "FILTER", "PROJECT", "CONTENT_EXTRACT",
                "CROSS_MODAL_MATCH", "SEMANTIC_SEARCH", "FEATURE_TRANSFORM",
                "JOIN", "SIMILARITY_JOIN", "UNION",
            ]
            for candidate in root_priority:
                if candidate in required:
                    root = candidate
                    break

        return cls(
            query_text=query_text,
            required_operators=required,
            optional_operators=optional,
            root_operator_type=root,
            target_modalities=modalities,
            estimated_data_size=1000,
        )

    @classmethod
    def from_query_llm(
        cls, query_text: str, provider, metadata_context: Optional[str] = None,
    ) -> "QueryContext":
        """Derive query context using an LLM provider for semantic understanding.

        Args:
            query_text: Natural language query.
            provider: An LLMProvider instance (from llm_query_analyzer).
            metadata_context: Optional metadata-graph summary (modality
                bindings, cross-modal entities) used to ground operator
                extraction.

        Falls back to regex-based from_query() if the LLM call fails.
        """
        from .llm_query_analyzer import LLMQueryAnalyzer

        analyzer = LLMQueryAnalyzer(provider, metadata_context=metadata_context)
        try:
            return analyzer.to_query_context(query_text)
        except Exception:
            return cls.from_query(query_text)


# ---------------------------------------------------------------------------
# Operator Compatibility Graph
# ---------------------------------------------------------------------------

class OperatorCompatibilityGraph:
    """Directed graph of valid logical operator transitions.

    G = (V, E, w, S) where:
      V = set of 16 logical operator type nodes
      E = set of directed edges (parent → child)
      w: E → [0, 1] compatibility scoring function
      S: V → OperatorSpec structural constraints
    """

    def __init__(self, history_manager=None, unconstrained: bool = False):
        self._specs: Dict[str, OperatorSpec] = {}
        self._edges: Dict[Tuple[str, str], OCGEdge] = {}
        self._adjacency: Dict[str, List[Tuple[str, float]]] = {}
        self._prior_weights: Dict[Tuple[str, str], float] = {}
        # Ablation: when unconstrained, the OCG imposes no structural constraint —
        # every operator is a valid child of every operator (titsp-no-ocg).
        self._unconstrained = unconstrained
        self._build_default_graph()
        if history_manager is not None:
            self.calibrate_edge_weights(history_manager)

    # ---- Public API ----

    def get_spec(self, operator_type: str) -> Optional[OperatorSpec]:
        return self._specs.get(operator_type)

    @property
    def all_operator_types(self) -> List[str]:
        return list(self._specs.keys())

    def get_valid_children(
        self, parent_type: str
    ) -> List[Tuple[str, float]]:
        """Return [(child_type, compatibility)] sorted by compatibility desc."""
        if self._unconstrained:
            return [(op, 1.0) for op in self._specs if op != parent_type]
        children = self._adjacency.get(parent_type, [])
        return sorted(children, key=lambda x: -x[1])

    def is_valid_edge(self, parent_type: str, child_type: str) -> bool:
        return (parent_type, child_type) in self._edges

    def get_compatibility(self, parent_type: str, child_type: str) -> float:
        edge = self._edges.get((parent_type, child_type))
        return edge.compatibility if edge else 0.0

    def validate_plan(
        self, plan_edges: List[Tuple[str, str]]
    ) -> Tuple[bool, List[str]]:
        """Validate that all edges in a plan are valid OCG transitions.

        Args:
            plan_edges: list of (parent_type, child_type) pairs.

        Returns:
            (is_valid, list_of_error_messages)
        """
        errors: List[str] = []
        for parent, child in plan_edges:
            if parent not in self._specs:
                errors.append(f"Unknown operator type: {parent}")
            elif child not in self._specs:
                errors.append(f"Unknown operator type: {child}")
            elif not self.is_valid_edge(parent, child):
                errors.append(f"Invalid edge: {parent} → {child}")
        return len(errors) == 0, errors

    def compute_plan_ocg_score(
        self, plan_edges: List[Tuple[str, str]]
    ) -> float:
        """Average compatibility score across all edges in a plan."""
        if not plan_edges:
            return 1.0
        total = sum(self.get_compatibility(p, c) for p, c in plan_edges)
        return total / len(plan_edges)

    def get_required_operators(
        self, query_context: QueryContext
    ) -> Set[str]:
        return set(query_context.required_operators)

    def get_optional_operators(
        self, query_context: QueryContext
    ) -> Set[str]:
        return set(query_context.optional_operators)

    # ---- Edge Weight Calibration ----

    def calibrate_edge_weights(
        self, history_manager, prior_weight: float = 2.0, kappa: int = 10
    ):
        """Bayesian update of edge compatibility from historical execution records.

        w(e) = (w₀(e)·λ + s̄(e)·min(k,κ)) / (λ + min(k,κ))
        """
        for (parent, child), prior_w in self._prior_weights.items():
            records = history_manager.get_edge_records(parent, child, limit=20)
            if not records:
                continue
            hist_avg = sum(records) / len(records)
            hist_weight = min(len(records), kappa)
            calibrated = (
                prior_w * prior_weight + hist_avg * hist_weight
            ) / (prior_weight + hist_weight)
            calibrated = max(0.0, min(1.0, calibrated))
            old_edge = self._edges[(parent, child)]
            self._edges[(parent, child)] = OCGEdge(
                parent, child, calibrated, old_edge.modality_constraint
            )
        self._adjacency.clear()
        for (parent, child), edge in self._edges.items():
            self._adjacency.setdefault(parent, []).append(
                (child, edge.compatibility)
            )

    # ---- Graph Construction ----

    def _build_default_graph(self):
        """Register all 16 operator specs and their valid edges."""
        self._register_operators()
        self._register_edges()

    def _register_operators(self):
        ALL = frozenset(ModalityType)
        specs = [
            OperatorSpec("SCAN", 0, 0, frozenset(), ModalityType.ANY,
                         is_leaf=True, is_terminal=False, category="access"),
            OperatorSpec("FILTER", 1, 1, ALL, ModalityType.ANY,
                         is_leaf=False, is_terminal=False, category="transform"),
            OperatorSpec("PROJECT", 1, 1, ALL, ModalityType.ANY,
                         is_leaf=False, is_terminal=False, category="transform"),
            OperatorSpec("AGGREGATE", 1, 1, ALL, ModalityType.TABULAR,
                         is_leaf=False, is_terminal=False, category="transform"),
            OperatorSpec("SORT", 1, 1,
                         frozenset({ModalityType.TABULAR, ModalityType.MIXED, ModalityType.ANY}),
                         ModalityType.ANY,
                         is_leaf=False, is_terminal=False, category="transform"),
            OperatorSpec("LIMIT", 1, 1, ALL, ModalityType.ANY,
                         is_leaf=False, is_terminal=False, category="transform"),
            OperatorSpec("DISTINCT", 1, 1, ALL, ModalityType.ANY,
                         is_leaf=False, is_terminal=False, category="transform"),
            OperatorSpec("JOIN", 2, 2, ALL, ModalityType.MIXED,
                         is_leaf=False, is_terminal=False, category="transform"),
            OperatorSpec("UNION", 2, 4, ALL, ModalityType.ANY,
                         is_leaf=False, is_terminal=False, category="transform"),
            OperatorSpec("SEMANTIC_SEARCH", 1, 1,
                         frozenset({ModalityType.TEXT, ModalityType.IMAGE, ModalityType.ANY}),
                         ModalityType.EMBEDDING,
                         is_leaf=False, is_terminal=False, category="cross_modal"),
            OperatorSpec("CROSS_MODAL_MATCH", 2, 2,
                         frozenset({ModalityType.IMAGE, ModalityType.TEXT, ModalityType.EMBEDDING, ModalityType.ANY}),
                         ModalityType.MIXED,
                         is_leaf=False, is_terminal=False, category="cross_modal"),
            OperatorSpec("CONTENT_EXTRACT", 1, 1,
                         frozenset({ModalityType.IMAGE, ModalityType.TEXT, ModalityType.ANY}),
                         ModalityType.TABULAR,
                         is_leaf=False, is_terminal=False, category="cross_modal"),
            OperatorSpec("SIMILARITY_JOIN", 2, 2,
                         frozenset({ModalityType.EMBEDDING, ModalityType.ANY}),
                         ModalityType.MIXED,
                         is_leaf=False, is_terminal=False, category="cross_modal"),
            OperatorSpec("FEATURE_TRANSFORM", 1, 1,
                         frozenset({ModalityType.IMAGE, ModalityType.TEXT, ModalityType.TABULAR, ModalityType.ANY}),
                         ModalityType.EMBEDDING,
                         is_leaf=False, is_terminal=False, category="cross_modal"),
            OperatorSpec("VISUALIZE", 1, 1, ALL, ModalityType.MIXED,
                         is_leaf=False, is_terminal=True, category="output"),
            OperatorSpec("EXPORT", 1, 1, ALL, ModalityType.TABULAR,
                         is_leaf=False, is_terminal=True, category="output"),
        ]
        for spec in specs:
            self._specs[spec.operator_type] = spec

    def _register_edges(self):
        edge_defs: List[Tuple[str, str, float]] = [
            # VISUALIZE children
            ("VISUALIZE", "AGGREGATE", 1.0),
            ("VISUALIZE", "SORT", 0.9),
            ("VISUALIZE", "PROJECT", 0.85),
            ("VISUALIZE", "FILTER", 0.8),
            ("VISUALIZE", "DISTINCT", 0.8),
            ("VISUALIZE", "CONTENT_EXTRACT", 0.75),
            ("VISUALIZE", "FEATURE_TRANSFORM", 0.7),

            # EXPORT children
            ("EXPORT", "AGGREGATE", 1.0),
            ("EXPORT", "PROJECT", 0.9),
            ("EXPORT", "SORT", 0.9),
            ("EXPORT", "FILTER", 0.85),
            ("EXPORT", "DISTINCT", 0.85),

            # SORT children
            ("SORT", "FILTER", 1.0),
            ("SORT", "AGGREGATE", 0.9),
            ("SORT", "PROJECT", 0.85),
            ("SORT", "SCAN", 0.85),
            ("SORT", "JOIN", 0.85),
            ("SORT", "SEMANTIC_SEARCH", 0.8),

            # LIMIT children
            ("LIMIT", "SORT", 1.0),
            ("LIMIT", "FILTER", 0.8),
            ("LIMIT", "SCAN", 0.7),

            # DISTINCT children
            ("DISTINCT", "FILTER", 1.0),
            ("DISTINCT", "SCAN", 0.9),
            ("DISTINCT", "PROJECT", 0.85),
            ("DISTINCT", "JOIN", 0.8),

            # AGGREGATE children
            ("AGGREGATE", "FILTER", 1.0),
            ("AGGREGATE", "SCAN", 0.9),
            ("AGGREGATE", "SEMANTIC_SEARCH", 0.85),
            ("AGGREGATE", "JOIN", 0.8),
            ("AGGREGATE", "CONTENT_EXTRACT", 0.75),
            ("AGGREGATE", "CROSS_MODAL_MATCH", 0.7),
            ("AGGREGATE", "PROJECT", 0.85),

            # PROJECT children
            ("PROJECT", "FILTER", 1.0),
            ("PROJECT", "SCAN", 1.0),
            ("PROJECT", "JOIN", 0.9),
            ("PROJECT", "AGGREGATE", 0.85),
            ("PROJECT", "SEMANTIC_SEARCH", 0.8),

            # FILTER children
            ("FILTER", "SCAN", 1.0),
            ("FILTER", "SEMANTIC_SEARCH", 0.9),
            ("FILTER", "CONTENT_EXTRACT", 0.85),
            ("FILTER", "JOIN", 0.85),
            ("FILTER", "FEATURE_TRANSFORM", 0.8),
            ("FILTER", "PROJECT", 0.8),

            # JOIN children (requires 2)
            ("JOIN", "SCAN", 1.0),
            ("JOIN", "FILTER", 0.9),
            ("JOIN", "SEMANTIC_SEARCH", 0.85),
            ("JOIN", "PROJECT", 0.8),

            # SIMILARITY_JOIN children (requires 2)
            ("SIMILARITY_JOIN", "SEMANTIC_SEARCH", 1.0),
            ("SIMILARITY_JOIN", "SCAN", 0.9),
            ("SIMILARITY_JOIN", "FEATURE_TRANSFORM", 0.9),
            ("SIMILARITY_JOIN", "FILTER", 0.85),

            # UNION children (requires 2+)
            ("UNION", "SCAN", 0.9),
            ("UNION", "FILTER", 0.85),
            ("UNION", "PROJECT", 0.8),

            # CROSS_MODAL_MATCH children (requires 2)
            ("CROSS_MODAL_MATCH", "CONTENT_EXTRACT", 1.0),
            ("CROSS_MODAL_MATCH", "SEMANTIC_SEARCH", 0.9),
            ("CROSS_MODAL_MATCH", "FEATURE_TRANSFORM", 0.9),
            ("CROSS_MODAL_MATCH", "SCAN", 0.85),

            # CONTENT_EXTRACT children
            ("CONTENT_EXTRACT", "SCAN", 1.0),
            ("CONTENT_EXTRACT", "FILTER", 0.8),

            # SEMANTIC_SEARCH children
            ("SEMANTIC_SEARCH", "SCAN", 1.0),
            ("SEMANTIC_SEARCH", "FILTER", 0.7),

            # FEATURE_TRANSFORM children
            ("FEATURE_TRANSFORM", "SCAN", 1.0),
            ("FEATURE_TRANSFORM", "FILTER", 0.85),
            ("FEATURE_TRANSFORM", "CONTENT_EXTRACT", 0.8),
        ]

        for parent, child, compat in edge_defs:
            edge = OCGEdge(parent, child, compat)
            self._edges[(parent, child)] = edge
            self._adjacency.setdefault(parent, []).append((child, compat))
            self._prior_weights[(parent, child)] = compat
