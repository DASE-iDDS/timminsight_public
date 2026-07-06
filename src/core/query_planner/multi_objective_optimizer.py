"""
Multi-Objective Physical Optimizer with Pareto Selection.

Given a fixed logical plan (from MCTS logical search), enumerates physical
implementation combinations, evaluates on 3 objectives (latency, resource_cost,
confidence), extracts the Pareto front via NSGA-II, and selects the final
plan via Weighted Tchebycheff scalarization.

The confidence objective f_conf(pi) = q^pi(root) is obtained by re-running
the bottom-up uncertainty propagation per assignment, substituting each
assigned implementation's intrinsic confidence for the type-level prior.

No MCTS at this level — the search space is small enough for
exhaustive/bounded enumeration.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .mcts_plan_search import (
    LogicalPlanNode,
    PhysicalImplInfo,
    PlanObjectives,
    ResourceConstraints,
    UserPreference,
)
from .uncertainty_propagation import (
    HistoricalCalibrator,
    PropagationRule,
    cardinality_factor,
    compute_coupling_penalty,
    resolve_propagation_rule,
)


# ---------------------------------------------------------------------------
# Physical Optimization Config
# ---------------------------------------------------------------------------

@dataclass
class PhysicalOptConfig:
    """Configuration for physical-level optimization."""
    max_combinations: int = 10000
    random_seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Objective Evaluator
# ---------------------------------------------------------------------------

class ObjectiveEvaluator:
    """Evaluates a physical implementation assignment on 3 objectives.

    Latency and resource cost are additive over the assigned implementations.
    The confidence objective f_conf(pi) = q^pi(root) is computed by re-running
    the bottom-up propagation rules (LEAF/PIPELINE/JOIN/UNION) over the
    logical plan tree, with the type-level prior c0(o) replaced by the
    assigned implementation's intrinsic confidence conf_pi(o); historical
    calibration applies unchanged, keyed by the implementation type. This
    keeps the two stages consistent — the logical stage scores by the
    *expected* confidence over I(o), the physical stage resolves that
    expectation with the actual assignment — and counts every confidence
    factor exactly once.

    When no logical plan is supplied (legacy callers), falls back to the
    flat product of impl confidence and the per-type propagated confidence.
    """

    def __init__(
        self,
        impl_registry: Dict[str, List[PhysicalImplInfo]],
        uncertainty_map: Optional[Dict[str, float]] = None,
        logical_plan: Optional[LogicalPlanNode] = None,
        history_manager=None,
    ):
        self._impl_registry = impl_registry
        self._uncertainty_map = uncertainty_map or {}
        self._logical_plan = logical_plan
        self._calibrator = HistoricalCalibrator(history_manager)

    def evaluate(self, assignment: Dict[str, str]) -> PlanObjectives:
        total_latency = 0.0
        total_cost = 0.0

        for op_type, impl_type in assignment.items():
            impl = self._find_impl(op_type, impl_type)
            if impl is None:
                continue
            total_latency += impl.base_cost
            total_cost += impl.base_cost * max(impl.cpu_cores, 1)

        if self._logical_plan is not None:
            confidence = self._repropagate(self._logical_plan, assignment)
        else:
            confidence = self._flat_confidence(assignment)

        return PlanObjectives(
            latency=total_latency,
            resource_cost=total_cost,
            confidence=confidence,
        )

    def _repropagate(
        self, node: LogicalPlanNode, assignment: Dict[str, str]
    ) -> float:
        """Post-order propagation with implementation-specific priors."""
        child_confs = [
            self._repropagate(child, assignment)
            for child in (node.children or [])
        ]

        op_type = node.operator_type
        impl = self._find_impl(op_type, assignment.get(op_type, ""))
        if impl is not None:
            # Calibrate the impl's intrinsic confidence against the
            # execution history of that implementation.
            calibrated, _ = self._calibrator.calibrate(
                impl.impl_type, impl.base_confidence
            )
        else:
            calibrated = 1.0  # operator without registered impls

        rule = resolve_propagation_rule(op_type, len(child_confs))
        if rule == PropagationRule.LEAF:
            n = getattr(node, "estimated_cardinality", 1000)
            return calibrated * cardinality_factor(n)
        if rule == PropagationRule.PIPELINE:
            product = 1.0
            for c in child_confs:
                product *= c
            return calibrated * product
        if rule == PropagationRule.JOIN:
            rho = compute_coupling_penalty(op_type)
            return calibrated * min(child_confs) * (1.0 - rho)
        # UNION
        return calibrated * (sum(child_confs) / len(child_confs))

    def _flat_confidence(self, assignment: Dict[str, str]) -> float:
        """Legacy fallback when no plan tree is available."""
        confidence = 1.0
        for op_type, impl_type in assignment.items():
            impl = self._find_impl(op_type, impl_type)
            if impl is None:
                continue
            confidence *= impl.base_confidence * self._uncertainty_map.get(op_type, 1.0)
        return confidence

    def _find_impl(
        self, op_type: str, impl_type: str
    ) -> Optional[PhysicalImplInfo]:
        for impl in self._impl_registry.get(op_type, []):
            if impl.impl_type == impl_type:
                return impl
        return None


# ---------------------------------------------------------------------------
# Constraint Checker
# ---------------------------------------------------------------------------

class ConstraintChecker:
    """Validates that a physical assignment meets resource constraints."""

    def __init__(self, constraints: ResourceConstraints):
        self._constraints = constraints

    def is_feasible(
        self,
        assignment: Dict[str, str],
        impl_registry: Dict[str, List[PhysicalImplInfo]],
    ) -> bool:
        total_gpu = 0
        total_cpu = 0
        total_mem = 0

        for op_type, impl_type in assignment.items():
            for impl in impl_registry.get(op_type, []):
                if impl.impl_type == impl_type:
                    total_gpu += impl.gpu_memory_mb
                    total_cpu += impl.cpu_cores
                    total_mem += impl.memory_mb
                    break

        if total_gpu > self._constraints.max_gpu_memory_mb:
            return False
        if total_cpu > self._constraints.max_cpu_cores:
            return False
        if total_mem > self._constraints.max_memory_mb:
            return False

        return True


# ---------------------------------------------------------------------------
# Physical Plan Enumerator
# ---------------------------------------------------------------------------

class PhysicalPlanEnumerator:
    """Enumerates physical implementation combinations for a logical plan."""

    def __init__(
        self,
        logical_plan: LogicalPlanNode,
        impl_registry: Dict[str, List[PhysicalImplInfo]],
        config: Optional[PhysicalOptConfig] = None,
    ):
        self._plan = logical_plan
        self._impl_registry = impl_registry
        self._config = config or PhysicalOptConfig()

    def enumerate(self) -> List[Dict[str, str]]:
        operators = self._plan.get_all_operators()
        op_types = []
        impl_options = []

        for op in operators:
            ot = op.operator_type
            impls = self._impl_registry.get(ot, [])
            if not impls:
                continue
            op_types.append(ot)
            impl_options.append([impl.impl_type for impl in impls])

        if not op_types:
            return []

        total = 1
        for opts in impl_options:
            total *= len(opts)

        if total <= self._config.max_combinations:
            return self._exhaustive(op_types, impl_options)
        else:
            return self._sampled(op_types, impl_options)

    def _exhaustive(
        self,
        op_types: List[str],
        impl_options: List[List[str]],
    ) -> List[Dict[str, str]]:
        results = []
        for combo in itertools.product(*impl_options):
            assignment = dict(zip(op_types, combo))
            results.append(assignment)
        return results

    def _sampled(
        self,
        op_types: List[str],
        impl_options: List[List[str]],
    ) -> List[Dict[str, str]]:
        rng = random.Random(self._config.random_seed)
        seen: set = set()
        results: List[Dict[str, str]] = []

        for _ in range(self._config.max_combinations * 2):
            if len(results) >= self._config.max_combinations:
                break
            combo = tuple(rng.choice(opts) for opts in impl_options)
            if combo not in seen:
                seen.add(combo)
                results.append(dict(zip(op_types, combo)))

        return results


# ---------------------------------------------------------------------------
# Plan Candidate for Physical Optimization
# ---------------------------------------------------------------------------

@dataclass
class PlanCandidate:
    """A physical plan candidate with its objectives."""
    assignment: Dict[str, str]
    objectives: PlanObjectives


# ---------------------------------------------------------------------------
# Pareto Candidate
# ---------------------------------------------------------------------------

@dataclass
class ParetoCandidate:
    """A plan candidate annotated with Pareto rank and crowding distance."""
    assignment: Dict[str, str]
    objectives: PlanObjectives
    rank: int = 0
    crowding_distance: float = 0.0
    plan_object: Any = None


# ---------------------------------------------------------------------------
# NSGA-II Pareto Sorting
# ---------------------------------------------------------------------------

class ParetoSorter:
    """Implements fast non-dominated sorting and crowding distance from NSGA-II."""

    @staticmethod
    def fast_non_dominated_sort(candidates: List[PlanCandidate]) -> List[List[int]]:
        """Return list of fronts, each front is a list of indices."""
        n = len(candidates)
        if n == 0:
            return []

        domination_count = [0] * n
        dominated_set: List[List[int]] = [[] for _ in range(n)]
        fronts: List[List[int]] = []
        first_front: List[int] = []

        for i in range(n):
            for j in range(i + 1, n):
                if ParetoSorter._dominates(candidates[i], candidates[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif ParetoSorter._dominates(candidates[j], candidates[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1

        for i in range(n):
            if domination_count[i] == 0:
                first_front.append(i)

        fronts.append(first_front)

        current_front = first_front
        while current_front:
            next_front: List[int] = []
            for i in current_front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            current_front = next_front

        return fronts

    @staticmethod
    def compute_crowding_distance(
        candidates: List[PlanCandidate], front_indices: List[int]
    ) -> Dict[int, float]:
        k = len(front_indices)
        if k <= 2:
            return {idx: float("inf") for idx in front_indices}

        distances: Dict[int, float] = {idx: 0.0 for idx in front_indices}

        objectives_getters = [
            lambda c: c.objectives.latency,
            lambda c: c.objectives.resource_cost,
            lambda c: -c.objectives.confidence,
        ]

        for getter in objectives_getters:
            sorted_indices = sorted(front_indices, key=lambda i: getter(candidates[i]))
            obj_min = getter(candidates[sorted_indices[0]])
            obj_max = getter(candidates[sorted_indices[-1]])
            obj_range = obj_max - obj_min

            distances[sorted_indices[0]] = float("inf")
            distances[sorted_indices[-1]] = float("inf")

            if obj_range > 1e-10:
                for pos in range(1, k - 1):
                    idx = sorted_indices[pos]
                    prev_val = getter(candidates[sorted_indices[pos - 1]])
                    next_val = getter(candidates[sorted_indices[pos + 1]])
                    distances[idx] += (next_val - prev_val) / obj_range

        return distances

    @staticmethod
    def _dominates(a: PlanCandidate, b: PlanCandidate) -> bool:
        a_obj = a.objectives
        b_obj = b.objectives
        at_least_one_better = False

        if a_obj.latency > b_obj.latency:
            return False
        if a_obj.latency < b_obj.latency:
            at_least_one_better = True

        if a_obj.resource_cost > b_obj.resource_cost:
            return False
        if a_obj.resource_cost < b_obj.resource_cost:
            at_least_one_better = True

        if a_obj.confidence < b_obj.confidence:
            return False
        if a_obj.confidence > b_obj.confidence:
            at_least_one_better = True

        return at_least_one_better


# ---------------------------------------------------------------------------
# Plan Selector (Tchebycheff Scalarization)
# ---------------------------------------------------------------------------

class PlanSelector:
    """Selects from the Pareto front using Weighted Tchebycheff scalarization."""

    def __init__(self, preference: UserPreference):
        self._preference = preference

    def select(
        self,
        candidates: List[ParetoCandidate],
    ) -> Tuple[ParetoCandidate, int]:
        if not candidates:
            raise ValueError("Empty candidate list")
        if len(candidates) == 1:
            return candidates[0], 0

        lats = [c.objectives.latency for c in candidates]
        costs = [c.objectives.resource_cost for c in candidates]
        confs = [c.objectives.confidence for c in candidates]

        ideal_lat = min(lats)
        ideal_cost = min(costs)
        ideal_conf = max(confs)

        range_lat = max(max(lats) - ideal_lat, 1e-10)
        range_cost = max(max(costs) - ideal_cost, 1e-10)
        range_conf = max(ideal_conf - min(confs), 1e-10)

        best_idx = 0
        best_score = float("inf")

        for i, c in enumerate(candidates):
            tcheb = max(
                self._preference.latency_weight * abs(c.objectives.latency - ideal_lat) / range_lat,
                self._preference.cost_weight * abs(c.objectives.resource_cost - ideal_cost) / range_cost,
                self._preference.confidence_weight * abs(c.objectives.confidence - ideal_conf) / range_conf,
            )
            if tcheb < best_score:
                best_score = tcheb
                best_idx = i

        return candidates[best_idx], best_idx


# ---------------------------------------------------------------------------
# Multi-Objective Physical Optimizer
# ---------------------------------------------------------------------------

class MultiObjectivePhysicalOptimizer:
    """Orchestrates physical optimization: enumerate → evaluate → Pareto → select.

    Usage:
        optimizer = MultiObjectivePhysicalOptimizer(
            logical_plan, impl_registry, uncertainty_map
        )
        selected, pareto_front = optimizer.optimize()
    """

    def __init__(
        self,
        logical_plan: LogicalPlanNode,
        impl_registry: Dict[str, List[PhysicalImplInfo]],
        uncertainty_map: Optional[Dict[str, float]] = None,
        constraints: Optional[ResourceConstraints] = None,
        preference: Optional[UserPreference] = None,
        config: Optional[PhysicalOptConfig] = None,
        history_manager=None,
    ):
        self._logical_plan = logical_plan
        self._impl_registry = impl_registry
        self._uncertainty_map = uncertainty_map or {}
        self._constraints = constraints or ResourceConstraints()
        self._preference = preference or UserPreference()
        self._config = config or PhysicalOptConfig()
        self._history_manager = history_manager

    def optimize(self) -> Tuple[ParetoCandidate, List[ParetoCandidate]]:
        """Run full physical optimization pipeline.

        Returns:
            (selected_plan, pareto_front)

        Raises:
            ValueError: if no feasible assignments exist.
        """
        # Step 1: Enumerate
        enumerator = PhysicalPlanEnumerator(
            self._logical_plan, self._impl_registry, self._config
        )
        assignments = enumerator.enumerate()

        if not assignments:
            raise ValueError("No physical implementations available for the logical plan.")

        # Step 2: Evaluate + filter (confidence via per-assignment
        # re-propagation over the logical plan tree)
        evaluator = ObjectiveEvaluator(
            self._impl_registry,
            self._uncertainty_map,
            logical_plan=self._logical_plan,
            history_manager=self._history_manager,
        )
        checker = ConstraintChecker(self._constraints)

        candidates: List[PlanCandidate] = []
        for assignment in assignments:
            if not checker.is_feasible(assignment, self._impl_registry):
                continue
            objectives = evaluator.evaluate(assignment)
            candidates.append(PlanCandidate(assignment=assignment, objectives=objectives))

        if not candidates:
            raise ValueError(
                "No feasible physical plans found. Check resource constraints."
            )

        # Deduplicate
        candidates = self._deduplicate(candidates)

        # Step 3: Pareto front
        fronts = ParetoSorter.fast_non_dominated_sort(candidates)
        pareto_indices = fronts[0] if fronts else list(range(len(candidates)))
        crowding = ParetoSorter.compute_crowding_distance(candidates, pareto_indices)

        pareto_front: List[ParetoCandidate] = []
        for idx in pareto_indices:
            c = candidates[idx]
            pareto_front.append(ParetoCandidate(
                assignment=c.assignment,
                objectives=c.objectives,
                rank=0,
                crowding_distance=crowding.get(idx, 0.0),
            ))

        # Step 4: Tchebycheff selection
        selector = PlanSelector(self._preference)
        selected, _ = selector.select(pareto_front)

        return selected, pareto_front

    @staticmethod
    def _deduplicate(candidates: List[PlanCandidate]) -> List[PlanCandidate]:
        seen: set = set()
        unique: List[PlanCandidate] = []
        for c in candidates:
            key = tuple(sorted(c.assignment.items()))
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique
