"""
Uncertainty Propagation for Multimodal Query Plans.

Implements bottom-up confidence propagation through the plan tree,
with Bayesian historical calibration. Each operator is annotated with
an UncertaintyEstimate reflecting both its intrinsic reliability and
the propagated confidence from its children.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class UncertaintyEstimate:
    """Uncertainty annotation for a single operator in the plan tree."""
    confidence: float           # [0, 1] — overall output reliability
    variance: float             # estimated variance of cost/cardinality
    modality_risk: float        # [0, 1] — risk due to multimodal nature
    calibration_source: str     # "prior" | "historical" | "runtime"

    def __post_init__(self):
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.modality_risk = max(0.0, min(1.0, self.modality_risk))


@dataclass
class OperatorUncertaintyProfile:
    """Static uncertainty profile for a physical operator type."""
    base_confidence: float
    confidence_decay_rate: float    # how confidence degrades with cardinality
    modality_sensitivity: float     # how much modality affects confidence
    modality_risk: float            # intrinsic modality risk [0, 1]


class OperatorCategory(Enum):
    DETERMINISTIC = "deterministic"
    STANDARD_DB = "standard_db"
    TEXT_ML = "text_ml"
    IMAGE_ML = "image_ml"
    EMBEDDING = "embedding"
    CROSS_MODAL = "cross_modal"
    SEMANTIC = "semantic"


class PropagationRule(Enum):
    PIPELINE = "pipeline"       # single-input chain: multiplicative
    JOIN = "join"               # multi-input join: min-bounded
    UNION = "union"             # multi-input union: average
    LEAF = "leaf"               # no children


# ---------------------------------------------------------------------------
# Prior Confidence Table
# ---------------------------------------------------------------------------

OPERATOR_PROFILES: Dict[str, OperatorUncertaintyProfile] = {
    # Deterministic operators — exact computation
    "TABLE_SCAN":           OperatorUncertaintyProfile(0.99, 0.001, 0.0, 0.0),
    "INDEX_SCAN":           OperatorUncertaintyProfile(0.99, 0.001, 0.0, 0.0),
    "LINEAR_SCAN":          OperatorUncertaintyProfile(0.98, 0.002, 0.0, 0.0),
    "FILE_SCAN":            OperatorUncertaintyProfile(0.97, 0.003, 0.0, 0.0),
    "PREDICATE_FILTER":     OperatorUncertaintyProfile(0.96, 0.002, 0.0, 0.02),
    "COLUMN_PROJECT":       OperatorUncertaintyProfile(0.99, 0.001, 0.0, 0.0),
    "STREAMING_LIMIT":      OperatorUncertaintyProfile(0.99, 0.001, 0.0, 0.0),
    "TOP_N":                OperatorUncertaintyProfile(0.98, 0.001, 0.0, 0.0),

    # Standard DB operators — deterministic but cardinality estimation varies
    "HASH_JOIN":            OperatorUncertaintyProfile(0.92, 0.005, 0.05, 0.05),
    "NESTED_LOOP_JOIN":     OperatorUncertaintyProfile(0.90, 0.008, 0.05, 0.05),
    "HASH_AGGREGATE":       OperatorUncertaintyProfile(0.93, 0.004, 0.03, 0.03),
    "STREAM_AGGREGATE":     OperatorUncertaintyProfile(0.91, 0.005, 0.03, 0.03),
    "MEMORY_SORT":          OperatorUncertaintyProfile(0.95, 0.003, 0.0, 0.01),
    "EXTERNAL_SORT":        OperatorUncertaintyProfile(0.93, 0.005, 0.0, 0.02),
    "HASH_DISTINCT":        OperatorUncertaintyProfile(0.93, 0.004, 0.0, 0.02),
    "STREAM_DISTINCT":      OperatorUncertaintyProfile(0.91, 0.005, 0.0, 0.02),
    "UNION_ALL":            OperatorUncertaintyProfile(0.95, 0.002, 0.0, 0.01),
    "HASH_INTERSECT":       OperatorUncertaintyProfile(0.93, 0.004, 0.0, 0.02),
    "SORT_INTERSECT":       OperatorUncertaintyProfile(0.92, 0.004, 0.0, 0.02),
    "HASH_EXCEPT":          OperatorUncertaintyProfile(0.93, 0.004, 0.0, 0.02),
    "MATERIALIZE":          OperatorUncertaintyProfile(0.97, 0.002, 0.0, 0.01),
    "CSV_EXPORT":           OperatorUncertaintyProfile(0.98, 0.001, 0.0, 0.0),
    "JSON_EXPORT":          OperatorUncertaintyProfile(0.98, 0.001, 0.0, 0.0),

    # Text ML operators
    "TEXT_ANALYSIS":        OperatorUncertaintyProfile(0.75, 0.015, 0.30, 0.25),

    # Image ML operators
    "YOLO_OBJECT_DETECTOR": OperatorUncertaintyProfile(0.72, 0.018, 0.35, 0.30),
    "DETR_OBJECT_DETECTOR": OperatorUncertaintyProfile(0.75, 0.015, 0.30, 0.28),
    "IMAGE_ANALYSIS":       OperatorUncertaintyProfile(0.68, 0.020, 0.35, 0.32),

    # Embedding operators
    "CLIP_TEXT_ENCODER":    OperatorUncertaintyProfile(0.73, 0.012, 0.25, 0.22),
    "CLIP_IMAGE_ENCODER":   OperatorUncertaintyProfile(0.72, 0.014, 0.28, 0.25),
    "VECTOR_SEARCH":        OperatorUncertaintyProfile(0.70, 0.015, 0.25, 0.22),
    "VECTOR_SCAN":          OperatorUncertaintyProfile(0.70, 0.015, 0.25, 0.22),

    # Cross-modal operators — inherently noisy
    "CROSS_MODAL_JOIN":     OperatorUncertaintyProfile(0.55, 0.025, 0.50, 0.48),
    "CROSS_MODAL_MATCH":    OperatorUncertaintyProfile(0.52, 0.028, 0.55, 0.50),
    "IMAGE_TO_TEXT_CAPTIONER": OperatorUncertaintyProfile(0.58, 0.022, 0.45, 0.42),

    # Semantic operators
    "SIMILARITY_FILTER":    OperatorUncertaintyProfile(0.65, 0.018, 0.30, 0.28),
    "THRESHOLD_FILTER":     OperatorUncertaintyProfile(0.80, 0.010, 0.15, 0.12),
    "INTRA_MODAL_JOIN":     OperatorUncertaintyProfile(0.62, 0.020, 0.35, 0.30),
    "SEMANTIC_JOIN":        OperatorUncertaintyProfile(0.60, 0.022, 0.40, 0.35),
    "RELEVANCE_RANK":       OperatorUncertaintyProfile(0.65, 0.018, 0.30, 0.28),
}

# Operators that incur cross-modal coupling penalty
CROSS_MODAL_OPERATORS = {
    "CROSS_MODAL_JOIN", "CROSS_MODAL_MATCH", "IMAGE_TO_TEXT_CAPTIONER",
    "SEMANTIC_JOIN", "INTRA_MODAL_JOIN", "SIMILARITY_JOIN",
}

# Mapping from operator type to propagation rule.  Both the logical type
# names (JOIN, UNION, ...) and their physical implementations are listed so
# the dataflow rules of Eq. (11) apply at either plan granularity.
JOIN_OPERATORS = {
    "JOIN",
    "HASH_JOIN", "NESTED_LOOP_JOIN", "SEMANTIC_JOIN",
    "INTRA_MODAL_JOIN", "CROSS_MODAL_JOIN", "CROSS_MODAL_MATCH",
    "SIMILARITY_JOIN",
}
UNION_OPERATORS = {
    "UNION",
    "UNION_ALL", "HASH_INTERSECT", "SORT_INTERSECT", "HASH_EXCEPT",
}

DEFAULT_PROFILE = OperatorUncertaintyProfile(0.70, 0.015, 0.20, 0.20)

# Derived priors for *logical* operator types, computed lazily from the
# default physical implementation registry: each logical type's prior is the
# field-wise mean over its implementations' profiles — the same single
# source of truth used for the logical cost/confidence derivation (Eq. 5).
# Logical names that already have an explicit entry (e.g. CROSS_MODAL_MATCH)
# keep it.  The import is deferred to avoid a circular module dependency.
_DERIVED_LOGICAL_PROFILES: Optional[Dict[str, OperatorUncertaintyProfile]] = None


def get_logical_operator_profiles() -> Dict[str, OperatorUncertaintyProfile]:
    global _DERIVED_LOGICAL_PROFILES
    if _DERIVED_LOGICAL_PROFILES is None:
        from .default_impl_registry import build_default_impl_registry

        derived: Dict[str, OperatorUncertaintyProfile] = {}
        for op_type, impls in build_default_impl_registry().items():
            if op_type in OPERATOR_PROFILES:
                continue
            profiles = [
                OPERATOR_PROFILES[i.impl_type]
                for i in impls
                if i.impl_type in OPERATOR_PROFILES
            ]
            if not profiles:
                continue
            n = len(profiles)
            derived[op_type] = OperatorUncertaintyProfile(
                base_confidence=sum(p.base_confidence for p in profiles) / n,
                confidence_decay_rate=sum(p.confidence_decay_rate for p in profiles) / n,
                modality_sensitivity=sum(p.modality_sensitivity for p in profiles) / n,
                modality_risk=sum(p.modality_risk for p in profiles) / n,
            )
        _DERIVED_LOGICAL_PROFILES = derived
    return _DERIVED_LOGICAL_PROFILES


# ---------------------------------------------------------------------------
# Historical Calibrator
# ---------------------------------------------------------------------------

@dataclass
class HistoricalRecord:
    """A simplified historical execution record for calibration."""
    operator_type: str
    parameters_hash: str
    estimated_cardinality: int
    actual_cardinality: int
    estimated_cost: float
    actual_cost: float
    success: bool


class HistoricalCalibrator:
    """Uses historical execution data to calibrate prior confidence via
    Bayesian updating."""

    def __init__(self, history_manager=None):
        self._history_manager = history_manager

    def calibrate(
        self,
        operator_type: str,
        prior_confidence: float,
        parameters: Optional[Dict[str, Any]] = None,
        prior_weight: float = 2.0,
    ) -> Tuple[float, str]:
        """Return (calibrated_confidence, source).

        If no history is available, returns the prior unchanged.
        """
        if self._history_manager is None:
            return prior_confidence, "prior"

        records = self._get_similar_records(operator_type, parameters)
        if not records:
            return prior_confidence, "prior"

        hist_accuracy = self._compute_historical_accuracy(records)
        k = len(records)
        hist_weight = min(k, 10)

        calibrated = (
            prior_confidence * prior_weight + hist_accuracy * hist_weight
        ) / (prior_weight + hist_weight)

        return max(0.0, min(1.0, calibrated)), "historical"

    def _get_similar_records(
        self, operator_type: str, parameters: Optional[Dict[str, Any]]
    ) -> List[HistoricalRecord]:
        if self._history_manager is None:
            return []

        if hasattr(self._history_manager, "get_similar_executions"):
            raw = self._history_manager.get_similar_executions(
                operator_type=operator_type,
                parameters=parameters,
                limit=20,
            )
            return [self._convert_record(r) for r in raw]

        if hasattr(self._history_manager, "get_records_by_operator"):
            raw = self._history_manager.get_records_by_operator(operator_type, limit=20)
            return [self._convert_record(r) for r in raw]

        return []

    @staticmethod
    def _convert_record(raw) -> HistoricalRecord:
        if isinstance(raw, HistoricalRecord):
            return raw
        return HistoricalRecord(
            operator_type=getattr(raw, "operator_type", ""),
            parameters_hash=getattr(raw, "parameters_hash", ""),
            estimated_cardinality=getattr(raw, "estimated_cardinality", 0),
            actual_cardinality=getattr(raw, "actual_cardinality", 0),
            estimated_cost=getattr(raw, "estimated_cost", 0.0),
            actual_cost=getattr(raw, "actual_cost", 0.0),
            success=getattr(raw, "success", True),
        )

    @staticmethod
    def _compute_historical_accuracy(records: List[HistoricalRecord]) -> float:
        if not records:
            return 0.5

        accuracies = []
        for r in records:
            if not r.success:
                accuracies.append(0.0)
                continue
            if r.estimated_cardinality > 0 and r.actual_cardinality > 0:
                ratio = r.actual_cardinality / r.estimated_cardinality
                acc = math.exp(-abs(math.log(max(ratio, 1e-10))))
                accuracies.append(acc)
            else:
                accuracies.append(0.5)

        return sum(accuracies) / len(accuracies) if accuracies else 0.5


# ---------------------------------------------------------------------------
# Uncertainty Prior
# ---------------------------------------------------------------------------

class UncertaintyPrior:
    """Provides prior confidence profiles for logical and physical operator types."""

    def __init__(self, custom_profiles: Optional[Dict[str, OperatorUncertaintyProfile]] = None):
        self._profiles = dict(OPERATOR_PROFILES)
        self._profiles.update(get_logical_operator_profiles())
        if custom_profiles:
            self._profiles.update(custom_profiles)

    def get_profile(self, operator_type: str) -> OperatorUncertaintyProfile:
        return self._profiles.get(operator_type, DEFAULT_PROFILE)

    def get_base_confidence(self, operator_type: str) -> float:
        return self.get_profile(operator_type).base_confidence

    def get_modality_risk(self, operator_type: str) -> float:
        return self.get_profile(operator_type).modality_risk

    @property
    def all_operator_types(self) -> List[str]:
        return list(self._profiles.keys())


# ---------------------------------------------------------------------------
# Annotated Plan
# ---------------------------------------------------------------------------

@dataclass
class AnnotatedPlan:
    """A logical plan annotated with uncertainty estimates per operator."""
    logical_plan: Any  # LogicalPlan from TiMMInsight
    uncertainty_map: Dict[str, UncertaintyEstimate] = field(default_factory=dict)

    @property
    def root_confidence(self) -> float:
        if not self.uncertainty_map:
            return 1.0
        root_id = self._get_root_operator_id()
        if root_id and root_id in self.uncertainty_map:
            return self.uncertainty_map[root_id].confidence
        return 1.0

    def _get_root_operator_id(self) -> Optional[str]:
        if hasattr(self.logical_plan, "root") and self.logical_plan.root:
            return getattr(self.logical_plan.root, "operator_id", None)
        return None

    def get_operator_confidence(self, operator_id: str) -> float:
        if operator_id in self.uncertainty_map:
            return self.uncertainty_map[operator_id].confidence
        return 1.0


# ---------------------------------------------------------------------------
# Propagation Rule Resolver
# ---------------------------------------------------------------------------

def resolve_propagation_rule(operator_type: str, children_count: int) -> PropagationRule:
    if children_count == 0:
        return PropagationRule.LEAF
    if operator_type in JOIN_OPERATORS:
        return PropagationRule.JOIN
    if operator_type in UNION_OPERATORS:
        return PropagationRule.UNION
    return PropagationRule.PIPELINE


def compute_coupling_penalty(operator_type: str) -> float:
    """Cross-modal coupling penalty ρ ∈ [0, 0.2]."""
    if operator_type in CROSS_MODAL_OPERATORS:
        penalties = {
            "CROSS_MODAL_MATCH": 0.20,
            "CROSS_MODAL_JOIN": 0.18,
            "IMAGE_TO_TEXT_CAPTIONER": 0.15,
            "SEMANTIC_JOIN": 0.12,
            "SIMILARITY_JOIN": 0.10,
            "INTRA_MODAL_JOIN": 0.08,
        }
        return penalties.get(operator_type, 0.15)
    return 0.0


def cardinality_factor(cardinality: int) -> float:
    """Data-scale attenuation φ(n): confidence decays with cardinality."""
    if cardinality <= 1:
        return 1.0
    return 1.0 - min(0.3, math.log10(max(cardinality, 1)) / 20.0)


# ---------------------------------------------------------------------------
# Main Propagator
# ---------------------------------------------------------------------------

class UncertaintyPropagator:
    """Propagates uncertainty estimates bottom-up through the plan tree.

    Algorithm:
        For each operator in POST-ORDER traversal:
        1. Look up prior confidence for operator type
        2. Adjust for cardinality
        3. Bayesian calibration with historical data (if available)
        4. Propagate from children using operator-type-specific rules
        5. Compute variance
        6. Annotate operator
    """

    def __init__(
        self,
        prior: Optional[UncertaintyPrior] = None,
        history_manager=None,
        prior_weight: float = 2.0,
    ):
        self._prior = prior or UncertaintyPrior()
        self._calibrator = HistoricalCalibrator(history_manager)
        self._prior_weight = prior_weight

    def propagate(self, logical_plan) -> AnnotatedPlan:
        """Propagate uncertainty through the plan tree.

        Args:
            logical_plan: A plan object with a ``root`` attribute whose
                operators have ``operator_id``, ``operator_type`` (or
                ``physical_type``), ``children``, ``estimated_cardinality``,
                ``estimated_cost``, and ``parameters``.

        Returns:
            AnnotatedPlan with uncertainty_map populated.
        """
        uncertainty_map: Dict[str, UncertaintyEstimate] = {}

        root = getattr(logical_plan, "root", logical_plan)
        self._propagate_recursive(root, uncertainty_map)

        return AnnotatedPlan(
            logical_plan=logical_plan,
            uncertainty_map=uncertainty_map,
        )

    def _propagate_recursive(
        self,
        operator,
        uncertainty_map: Dict[str, UncertaintyEstimate],
    ) -> float:
        """Recursively propagate and return this operator's confidence."""
        children = getattr(operator, "children", []) or []

        # Post-order: process children first
        children_confidences = []
        for child in children:
            child_conf = self._propagate_recursive(child, uncertainty_map)
            children_confidences.append(child_conf)

        # Step 1: Prior confidence
        op_type = self._get_operator_type(operator)
        base_conf = self._prior.get_base_confidence(op_type)

        # Step 2: Cardinality adjustment
        cardinality = getattr(operator, "estimated_cardinality", 1000)
        card_factor = self._cardinality_factor(cardinality)

        # Step 3: Historical Bayesian calibration
        parameters = getattr(operator, "parameters", None)
        calibrated, source = self._calibrator.calibrate(
            op_type, base_conf, parameters, self._prior_weight
        )

        # Step 4: Child confidence propagation
        rule = resolve_propagation_rule(op_type, len(children_confidences))
        confidence = self._apply_propagation_rule(
            rule, calibrated, children_confidences, op_type, card_factor
        )

        # Step 5: Variance estimation
        estimated_cost = getattr(operator, "estimated_cost", 1.0)
        variance = (1.0 - confidence) ** 2 * estimated_cost ** 2

        # Step 6: Annotate
        modality_risk = self._prior.get_modality_risk(op_type)
        op_id = getattr(operator, "operator_id", None) or getattr(operator, "node_id", None) or id(operator)

        estimate = UncertaintyEstimate(
            confidence=confidence,
            variance=variance,
            modality_risk=modality_risk,
            calibration_source=source,
        )
        uncertainty_map[str(op_id)] = estimate

        # Also attach directly to operator if possible
        if hasattr(operator, "uncertainty"):
            operator.uncertainty = estimate

        return confidence

    def _apply_propagation_rule(
        self,
        rule: PropagationRule,
        calibrated: float,
        children_confidences: List[float],
        operator_type: str,
        card_factor: float,
    ) -> float:
        if rule == PropagationRule.LEAF:
            return calibrated * card_factor

        if rule == PropagationRule.PIPELINE:
            product = 1.0
            for c in children_confidences:
                product *= c
            return calibrated * product

        if rule == PropagationRule.JOIN:
            rho = compute_coupling_penalty(operator_type)
            min_child = min(children_confidences) if children_confidences else 1.0
            return calibrated * min_child * (1.0 - rho)

        if rule == PropagationRule.UNION:
            avg = (
                sum(children_confidences) / len(children_confidences)
                if children_confidences
                else 1.0
            )
            return calibrated * avg

        return calibrated

    @staticmethod
    def _cardinality_factor(cardinality: int) -> float:
        """Confidence decays with data scale: large cardinality → less certain estimates."""
        return cardinality_factor(cardinality)

    def estimate_logical_confidence(self, operator_type: str) -> float:
        """Quick confidence estimate for a logical operator type.

        Used by MCTS reward computation at the logical level, where
        physical implementation is not yet chosen.
        """
        return LOGICAL_CONFIDENCE_PRIORS.get(operator_type, 0.70)

    @staticmethod
    def _get_operator_type(operator) -> str:
        for attr in ("physical_type", "operator_type", "type"):
            val = getattr(operator, attr, None)
            if val is not None:
                if isinstance(val, Enum):
                    return val.name
                return str(val)
        return "UNKNOWN"


# ---------------------------------------------------------------------------
# Logical-Level Confidence Priors
# ---------------------------------------------------------------------------

LOGICAL_CONFIDENCE_PRIORS: Dict[str, float] = {
    "SCAN": 0.98,
    "FILTER": 0.93,
    "PROJECT": 0.99,
    "JOIN": 0.88,
    "AGGREGATE": 0.91,
    "SORT": 0.94,
    "LIMIT": 0.98,
    "UNION": 0.94,
    "DISTINCT": 0.92,
    "SEMANTIC_SEARCH": 0.68,
    "CROSS_MODAL_MATCH": 0.53,
    "CONTENT_EXTRACT": 0.72,
    "SIMILARITY_JOIN": 0.62,
    "FEATURE_TRANSFORM": 0.70,
    "VISUALIZE": 0.97,
    "EXPORT": 0.98,
}
