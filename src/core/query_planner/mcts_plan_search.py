"""
MCTS Logical Plan Search.

Monte Carlo Tree Search over logical plan structures constrained by the
Operator Compatibility Graph (OCG).  Each MCTS iteration builds a logical
plan tree top-down: the root operator is fixed by the query context, and
subsequent operators are selected from OCG-valid children until all branches
terminate at SCAN leaves.

The search discovers diverse valid logical plans and scores them with a
composite quality function Q(P) combining completeness, estimated cost,
estimated confidence, and OCG compatibility.
"""

from __future__ import annotations

import copy
import math
import os
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .operator_compatibility_graph import (
    ModalityType,
    OperatorCompatibilityGraph,
    QueryContext,
)
from .uncertainty_propagation import LOGICAL_CONFIDENCE_PRIORS


# ---------------------------------------------------------------------------
# Data structures kept for backward compatibility (used by physical optimizer)
# ---------------------------------------------------------------------------

@dataclass
class PlanObjectives:
    """Three objectives for plan evaluation."""
    latency: float = 0.0
    resource_cost: float = 0.0
    confidence: float = 0.0


@dataclass
class ResourceConstraints:
    """Hard resource constraints for physical planning."""
    max_gpu_memory_mb: int = 16384
    max_cpu_cores: int = 16
    max_memory_mb: int = 65536
    max_latency_seconds: float = 300.0


@dataclass
class UserPreference:
    """User preference weights for multi-objective selection."""
    latency_weight: float = 0.4
    cost_weight: float = 0.3
    confidence_weight: float = 0.3

    def __post_init__(self):
        total = self.latency_weight + self.cost_weight + self.confidence_weight
        if total > 0:
            self.latency_weight /= total
            self.cost_weight /= total
            self.confidence_weight /= total


@dataclass
class PhysicalImplInfo:
    """Descriptor for a physical implementation option."""
    impl_type: str
    base_cost: float = 1.0
    base_confidence: float = 0.8
    gpu_memory_mb: int = 0
    cpu_cores: int = 1
    memory_mb: int = 256


@dataclass
class LogicalOperatorInfo:
    """Lightweight descriptor for a logical operator in the plan."""
    operator_id: str
    operator_type: str
    estimated_cardinality: int = 1000
    estimated_cost: float = 1.0


# ---------------------------------------------------------------------------
# Logical Plan Node
# ---------------------------------------------------------------------------

@dataclass
class LogicalPlanNode:
    """A node in a logical plan tree built by MCTS."""
    node_id: str
    operator_type: str
    children: List["LogicalPlanNode"] = field(default_factory=list)
    parent: Optional["LogicalPlanNode"] = None
    estimated_cardinality: int = 1000   # output rows, set by annotate_plan_estimates
    estimated_cost: float = 1.0         # per-operator cost c(o), Eq. (5)

    def clone(self) -> "LogicalPlanNode":
        """Deep-copy the subtree rooted at this node."""
        new_node = LogicalPlanNode(
            node_id=self.node_id,
            operator_type=self.operator_type,
            estimated_cardinality=self.estimated_cardinality,
            estimated_cost=self.estimated_cost,
        )
        for child in self.children:
            child_clone = child.clone()
            child_clone.parent = new_node
            new_node.children.append(child_clone)
        return new_node

    def get_all_operators(self) -> List["LogicalPlanNode"]:
        """DFS traversal returning all nodes."""
        result = [self]
        for child in self.children:
            result.extend(child.get_all_operators())
        return result

    def get_edges(self) -> List[Tuple[str, str]]:
        """Return all (parent_type, child_type) edges in this subtree."""
        edges: List[Tuple[str, str]] = []
        for child in self.children:
            edges.append((self.operator_type, child.operator_type))
            edges.extend(child.get_edges())
        return edges

    def get_operator_types(self) -> List[str]:
        """Return operator types in DFS order."""
        return [n.operator_type for n in self.get_all_operators()]

    def depth(self) -> int:
        d = 0
        node = self
        while node.parent is not None:
            d += 1
            node = node.parent
        return d

    def max_depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.max_depth() for c in self.children)


# ---------------------------------------------------------------------------
# Cardinality / cost annotation
# ---------------------------------------------------------------------------

def annotate_plan_estimates(
    root: LogicalPlanNode,
    base_cardinality: int = 1000,
    cost_lookup: Optional[Any] = None,
) -> None:
    """Annotate every node with an estimated output cardinality and cost.

    Cardinalities propagate bottom-up from SCAN leaves (which receive the
    base table cardinality) using standard selectivity priors, so the
    data-scale attenuation phi(n) and the variance estimate differentiate
    operators by the data volume they actually process.  ``cost_lookup``
    maps an operator type to its per-operator cost c(o) (impl-registry
    mean, Eq. 5) for the variance estimate.
    """

    def visit(node: LogicalPlanNode) -> int:
        child_cards = [visit(c) for c in node.children]
        node.estimated_cardinality = _estimate_output_cardinality(
            node.operator_type, child_cards, base_cardinality
        )
        if cost_lookup is not None:
            node.estimated_cost = cost_lookup(node.operator_type)
        return node.estimated_cardinality

    visit(root)


def _estimate_output_cardinality(
    op_type: str, child_cards: List[int], base_cardinality: int
) -> int:
    """Output cardinality from children via standard selectivity priors."""
    if not child_cards:  # leaf (SCAN): reads the base table
        return max(1, base_cardinality)
    inp = max(child_cards)
    if op_type == "FILTER":
        out = inp * 0.33
    elif op_type == "LIMIT":
        out = min(inp, 10)
    elif op_type == "DISTINCT":
        out = inp * 0.5
    elif op_type == "AGGREGATE":
        out = inp * 0.1
    elif op_type == "UNION":
        out = sum(child_cards)
    elif op_type == "JOIN":
        out = inp  # FK-join assumption: bounded by the larger input
    elif op_type in ("SIMILARITY_JOIN", "CROSS_MODAL_MATCH"):
        out = min(child_cards) if len(child_cards) > 1 else inp * 0.5
    elif op_type == "SEMANTIC_SEARCH":
        out = inp * 0.2  # top-k retrieval
    else:  # PROJECT, SORT, CONTENT_EXTRACT, FEATURE_TRANSFORM, VISUALIZE, EXPORT
        out = inp
    return max(1, int(out))


# ---------------------------------------------------------------------------
# Partial Plan (MCTS state)
# ---------------------------------------------------------------------------

@dataclass
class OpenPosition:
    """An unfilled child slot in the partial plan."""
    parent_node: LogicalPlanNode
    child_slot_index: int
    depth: int


class PartialPlan:
    """State in the MCTS search: a partially-built logical plan tree."""

    def __init__(
        self,
        root: Optional[LogicalPlanNode] = None,
        ocg: Optional[OperatorCompatibilityGraph] = None,
    ):
        self.root = root
        self._ocg = ocg or _get_global_ocg()
        self.open_positions: List[OpenPosition] = []
        self.node_count = 1 if root else 0

    def clone(self) -> "PartialPlan":
        new_plan = PartialPlan(ocg=self._ocg)
        if self.root:
            new_plan.root = self.root.clone()
            new_plan.open_positions = new_plan._rebuild_open_positions(new_plan.root)
            new_plan.node_count = self.node_count
        return new_plan

    def is_terminal(self) -> bool:
        return len(self.open_positions) == 0

    def compute_completeness(self, query_context: QueryContext) -> float:
        if not self.root or not query_context.required_operators:
            return 0.0
        present = set(self.root.get_operator_types())
        required = query_context.required_operators
        return len(present & required) / len(required)

    def _rebuild_open_positions(self, root: LogicalPlanNode) -> List[OpenPosition]:
        positions: List[OpenPosition] = []
        self._scan_open_positions(root, positions)
        return positions

    def _scan_open_positions(
        self, node: LogicalPlanNode, positions: List[OpenPosition]
    ):
        spec = self._ocg.get_spec(node.operator_type)
        if spec and not spec.is_leaf:
            needed = spec.min_children - len(node.children)
            for i in range(needed):
                positions.append(
                    OpenPosition(node, len(node.children) + i, node.depth() + 1)
                )
        for child in node.children:
            self._scan_open_positions(child, positions)


# Global OCG instance
_GLOBAL_OCG: Optional[OperatorCompatibilityGraph] = None


def _get_global_ocg() -> OperatorCompatibilityGraph:
    global _GLOBAL_OCG
    if _GLOBAL_OCG is None:
        _GLOBAL_OCG = OperatorCompatibilityGraph()
    return _GLOBAL_OCG


# ---------------------------------------------------------------------------
# MCTS Configuration
# ---------------------------------------------------------------------------

@dataclass
class MCTSLogicalConfig:
    """Configuration for logical-level MCTS.

    Defaults can be overridden via environment variables (MCTS_ITERS, MCTS_UCT,
    MCTS_W_COMPLETE/COST/CONF/OCG, MCTS_SEED) so parameter-sensitivity sweeps can
    drive any existing scenario harness without code changes. Unset -> normal value.
    """
    num_iterations: int = field(default_factory=lambda: int(os.environ.get("MCTS_ITERS", "300")))
    exploration_constant: float = field(default_factory=lambda: float(os.environ.get("MCTS_UCT", "1.414")))
    max_plan_depth: int = 10
    max_plan_operators: int = 12
    completeness_weight: float = field(default_factory=lambda: float(os.environ.get("MCTS_W_COMPLETE", "0.3")))
    cost_weight: float = field(default_factory=lambda: float(os.environ.get("MCTS_W_COST", "0.25")))
    confidence_weight: float = field(default_factory=lambda: float(os.environ.get("MCTS_W_CONF", "0.25")))
    ocg_weight: float = field(default_factory=lambda: float(os.environ.get("MCTS_W_OCG", "0.2")))
    random_seed: Optional[int] = field(
        default_factory=lambda: int(os.environ["MCTS_SEED"]) if os.environ.get("MCTS_SEED") else None)


# ---------------------------------------------------------------------------
# Logical Plan Candidate (result)
# ---------------------------------------------------------------------------

@dataclass
class LogicalPlanCandidate:
    """A complete logical plan discovered by MCTS with its quality score."""
    plan: LogicalPlanNode
    quality_score: float
    completeness: float
    estimated_cost: float
    estimated_confidence: float
    ocg_score: float
    operator_sequence: List[str]


# ---------------------------------------------------------------------------
# MCTS Node (search tree node)
# ---------------------------------------------------------------------------

class MCTSLogicalNode:
    """Node in the MCTS search tree over logical plan structures."""

    def __init__(
        self,
        partial_plan: PartialPlan,
        action_taken: Optional[str] = None,
        parent: Optional["MCTSLogicalNode"] = None,
    ):
        self.partial_plan = partial_plan
        self.action_taken = action_taken
        self.parent = parent
        self.children: Dict[str, "MCTSLogicalNode"] = {}
        self.visit_count: int = 0
        self.total_reward: float = 0.0
        self._feasible_actions: Optional[List[str]] = None

    @property
    def average_reward(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_reward / self.visit_count

    @property
    def is_terminal(self) -> bool:
        return self.partial_plan.is_terminal()

    def get_feasible_actions(self, ocg: OperatorCompatibilityGraph) -> List[str]:
        if self._feasible_actions is not None:
            return self._feasible_actions

        if self.is_terminal:
            self._feasible_actions = []
            return []

        pos = self.partial_plan.open_positions[0]
        parent_type = pos.parent_node.operator_type
        valid = ocg.get_valid_children(parent_type)
        self._feasible_actions = [op_type for op_type, _ in valid]
        return self._feasible_actions

    @property
    def is_fully_expanded(self) -> bool:
        if self._feasible_actions is None:
            return False
        return len(self.children) >= len(self._feasible_actions)

    def get_unexplored_actions(self, ocg: OperatorCompatibilityGraph) -> List[str]:
        feasible = self.get_feasible_actions(ocg)
        return [a for a in feasible if a not in self.children]


# ---------------------------------------------------------------------------
# Logical Cost Model
# ---------------------------------------------------------------------------

LOGICAL_COST_PRIORS: Dict[str, float] = {
    "SCAN": 1.0,
    "FILTER": 0.5,
    "PROJECT": 0.2,
    "JOIN": 4.0,
    "AGGREGATE": 2.0,
    "SORT": 1.5,
    "LIMIT": 0.1,
    "UNION": 1.0,
    "DISTINCT": 1.5,
    "SEMANTIC_SEARCH": 5.0,
    "CROSS_MODAL_MATCH": 6.0,
    "CONTENT_EXTRACT": 8.0,
    "SIMILARITY_JOIN": 6.0,
    "FEATURE_TRANSFORM": 4.0,
    "VISUALIZE": 1.0,
    "EXPORT": 0.5,
}


class LogicalCostModel:
    """Estimates cost of a logical plan.

    When an impl_registry is provided, costs are derived from the average
    base_cost of each operator's physical implementations.  Otherwise
    falls back to the static LOGICAL_COST_PRIORS table.
    """

    def __init__(
        self,
        impl_registry: Optional[Dict[str, List[PhysicalImplInfo]]] = None,
    ):
        if impl_registry:
            self._costs = {
                op: sum(i.base_cost for i in impls) / len(impls)
                for op, impls in impl_registry.items()
                if impls
            }
        else:
            self._costs = LOGICAL_COST_PRIORS

    def operator_cost(self, operator_type: str) -> float:
        """Per-operator cost c(o): impl-registry mean when available (Eq. 5)."""
        return self._costs.get(operator_type, 2.0)

    def estimate_plan_cost(self, plan_root: LogicalPlanNode) -> float:
        total = 0.0
        for node in plan_root.get_all_operators():
            total += self.operator_cost(node.operator_type)
        return total


class LogicalConfidenceEstimator:
    """Estimates confidence at the logical level.

    When an impl_registry is provided, confidences are derived from the
    average base_confidence of each operator's physical implementations.
    Otherwise falls back to the static LOGICAL_CONFIDENCE_PRIORS table.
    """

    def __init__(
        self,
        impl_registry: Optional[Dict[str, List[PhysicalImplInfo]]] = None,
    ):
        if impl_registry:
            self._confs = {
                op: sum(i.base_confidence for i in impls) / len(impls)
                for op, impls in impl_registry.items()
                if impls
            }
        else:
            self._confs = LOGICAL_CONFIDENCE_PRIORS

    def estimate_plan_confidence(self, plan_root: LogicalPlanNode) -> float:
        confidence = 1.0
        for node in plan_root.get_all_operators():
            confidence *= self._confs.get(node.operator_type, 0.70)
        return confidence


# ---------------------------------------------------------------------------
# Rollout Policy
# ---------------------------------------------------------------------------

class LogicalRolloutPolicy:
    """Completes a partial logical plan using OCG-guided heuristics."""

    def __init__(
        self,
        ocg: OperatorCompatibilityGraph,
        query_context: QueryContext,
        max_depth: int = 10,
    ):
        self._ocg = ocg
        self._context = query_context
        self._max_depth = max_depth

    def complete_plan(self, partial: PartialPlan) -> LogicalPlanNode:
        plan = partial.clone()
        remaining_required = self._context.required_operators - set(
            plan.root.get_operator_types() if plan.root else []
        )

        safety = 50
        while not plan.is_terminal() and safety > 0:
            safety -= 1
            pos = plan.open_positions[0]
            parent_type = pos.parent_node.operator_type
            valid = self._ocg.get_valid_children(parent_type)

            if not valid or pos.depth >= self._max_depth:
                chosen_type = "SCAN"
            else:
                chosen_type = self._select_operator(valid, remaining_required, pos.depth)

            new_node = LogicalPlanNode(
                node_id=str(uuid.uuid4())[:8],
                operator_type=chosen_type,
                parent=pos.parent_node,
            )
            pos.parent_node.children.append(new_node)
            plan.open_positions.pop(0)
            plan.node_count += 1
            remaining_required.discard(chosen_type)

            spec = self._ocg.get_spec(chosen_type)
            if spec and not spec.is_leaf:
                for i in range(spec.min_children):
                    plan.open_positions.append(
                        OpenPosition(new_node, i, pos.depth + 1)
                    )

        return plan.root

    def _select_operator(
        self,
        valid: List[Tuple[str, float]],
        remaining_required: Set[str],
        depth: int,
    ) -> str:
        # Prefer non-leaf required operators (SCAN terminates the branch)
        non_leaf_req = [(t, c) for t, c in valid if t in remaining_required and t != "SCAN"]
        if non_leaf_req:
            return non_leaf_req[0][0]

        # Then leaf required operators only when no non-leaf required remain
        leaf_req = [(t, c) for t, c in valid if t in remaining_required and t == "SCAN"]
        other_req = remaining_required - {"SCAN"}
        if leaf_req and not other_req:
            return leaf_req[0][0]

        if depth >= self._max_depth - 2:
            for op_type, _ in valid:
                if op_type == "SCAN":
                    return op_type

        return valid[0][0]


# ---------------------------------------------------------------------------
# MCTS Logical Plan Search
# ---------------------------------------------------------------------------

class MCTSLogicalPlanSearch:
    """MCTS over logical plan structures using OCG constraints.

    Each MCTS iteration builds a logical plan top-down. The tree is
    explored via UCT, expanded by placing operators at open positions,
    and evaluated with a composite quality function.
    """

    def __init__(
        self,
        ocg: OperatorCompatibilityGraph,
        query_context: QueryContext,
        config: Optional[MCTSLogicalConfig] = None,
        impl_registry: Optional[Dict[str, List[PhysicalImplInfo]]] = None,
    ):
        self._ocg = ocg
        self._context = query_context
        self._config = config or MCTSLogicalConfig()
        self._cost_model = LogicalCostModel(impl_registry)
        self._confidence_estimator = LogicalConfidenceEstimator(impl_registry)
        self._rollout_policy = LogicalRolloutPolicy(
            ocg, query_context, self._config.max_plan_depth
        )
        self._rng = random.Random(self._config.random_seed)

        global _GLOBAL_OCG
        _GLOBAL_OCG = ocg

    def search(self) -> List[LogicalPlanCandidate]:
        """Run MCTS and return discovered logical plan candidates."""
        initial_plan = self._create_initial_plan()
        root_node = MCTSLogicalNode(partial_plan=initial_plan)

        discovered: Dict[tuple, LogicalPlanCandidate] = {}

        for _ in range(self._config.num_iterations):
            # Selection
            node = self._select(root_node)

            # Expansion
            if not node.is_terminal:
                node = self._expand(node)

            # Simulation
            if node.partial_plan.root is None:
                continue
            completed_root = self._rollout_policy.complete_plan(node.partial_plan)

            # Evaluation
            reward, candidate = self._evaluate(completed_root)

            # Store unique plans
            if candidate is not None:
                key = tuple(candidate.operator_sequence)
                if key not in discovered or candidate.quality_score > discovered[key].quality_score:
                    discovered[key] = candidate

            # Backpropagation
            self._backpropagate(node, reward)

        # A plan that omits a REQUIRED operator is incorrect (it cannot answer the
        # query), so completeness gates selection: fully-complete plans always rank
        # above incomplete ones, and quality_score only breaks ties among equally
        # complete plans. This keeps required ops (e.g. a cross-modal JOIN) in the
        # chosen plan instead of letting cost/confidence weights drop them.
        req = self._context.required_operators or set()

        def _completeness(c):
            if not req:
                return 1.0
            return len(set(c.operator_sequence) & req) / len(req)

        return sorted(discovered.values(), key=lambda c: (-_completeness(c), -c.quality_score))

    @property
    def iteration_count(self) -> int:
        return self._config.num_iterations

    def _create_initial_plan(self) -> PartialPlan:
        root_type = self._context.root_operator_type
        root_node = LogicalPlanNode(
            node_id=str(uuid.uuid4())[:8],
            operator_type=root_type,
        )
        plan = PartialPlan(root=root_node, ocg=self._ocg)

        spec = self._ocg.get_spec(root_type)
        if spec and not spec.is_leaf:
            for i in range(spec.min_children):
                plan.open_positions.append(OpenPosition(root_node, i, 1))

        return plan

    def _select(self, root: MCTSLogicalNode) -> MCTSLogicalNode:
        node = root
        while not node.is_terminal and node.is_fully_expanded and node.children:
            node = self._uct_select(node)
        return node

    def _uct_select(self, node: MCTSLogicalNode) -> MCTSLogicalNode:
        c = self._config.exploration_constant
        log_parent = math.log(max(node.visit_count, 1))

        best_child = None
        best_ucb = -float("inf")

        for child in node.children.values():
            if child.visit_count == 0:
                return child
            exploit = child.average_reward
            explore = c * math.sqrt(log_parent / child.visit_count)
            ucb = exploit + explore
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        return best_child or node

    def _expand(self, node: MCTSLogicalNode) -> MCTSLogicalNode:
        unexplored = node.get_unexplored_actions(self._ocg)
        if not unexplored:
            return node

        present_types = set(
            node.partial_plan.root.get_operator_types()
            if node.partial_plan.root else []
        )
        remaining_req = self._context.required_operators - present_types

        # Prefer non-leaf required operators first (SCAN is always placed last)
        non_leaf_req = [op for op in unexplored if op in remaining_req and op != "SCAN"]
        leaf_req = [op for op in unexplored if op in remaining_req and op == "SCAN"]

        chosen = None
        if non_leaf_req:
            chosen = non_leaf_req[0]
        elif leaf_req and len(remaining_req) <= 1:
            chosen = leaf_req[0]

        if chosen is None:
            chosen = unexplored[0]

        new_plan = node.partial_plan.clone()
        pos = new_plan.open_positions[0]

        new_node = LogicalPlanNode(
            node_id=str(uuid.uuid4())[:8],
            operator_type=chosen,
            parent=pos.parent_node,
        )
        pos.parent_node.children.append(new_node)
        new_plan.open_positions.pop(0)
        new_plan.node_count += 1

        spec = self._ocg.get_spec(chosen)
        if spec and not spec.is_leaf:
            for i in range(spec.min_children):
                new_plan.open_positions.append(
                    OpenPosition(new_node, i, pos.depth + 1)
                )

        child_node = MCTSLogicalNode(
            partial_plan=new_plan,
            action_taken=chosen,
            parent=node,
        )
        node.children[chosen] = child_node
        return child_node

    def _evaluate(
        self, plan_root: LogicalPlanNode
    ) -> Tuple[float, Optional[LogicalPlanCandidate]]:
        ops = plan_root.get_operator_types()
        edges = plan_root.get_edges()

        if plan_root.max_depth() > self._config.max_plan_depth:
            return -1.0, None
        if len(ops) > self._config.max_plan_operators:
            return -1.0, None

        present = set(ops)
        required = self._context.required_operators
        completeness = len(present & required) / max(len(required), 1)

        cost = self._cost_model.estimate_plan_cost(plan_root)
        confidence = self._confidence_estimator.estimate_plan_confidence(plan_root)
        ocg_score = self._ocg.compute_plan_ocg_score(edges) if edges else 1.0

        cfg = self._config
        quality = (
            cfg.completeness_weight * completeness
            + cfg.cost_weight * (1.0 / (1.0 + cost))
            + cfg.confidence_weight * confidence
            + cfg.ocg_weight * ocg_score
        )

        candidate = LogicalPlanCandidate(
            plan=plan_root.clone(),
            quality_score=quality,
            completeness=completeness,
            estimated_cost=cost,
            estimated_confidence=confidence,
            ocg_score=ocg_score,
            operator_sequence=ops,
        )

        return quality, candidate

    def _backpropagate(self, node: MCTSLogicalNode, reward: float):
        current: Optional[MCTSLogicalNode] = node
        while current is not None:
            current.visit_count += 1
            current.total_reward += reward
            current = current.parent
