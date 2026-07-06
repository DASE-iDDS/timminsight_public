"""
End-to-End Query Planner.

Orchestrates the full planning pipeline:
  NL query → QueryContext → MCTS logical search → Uncertainty propagation
  → Physical optimization → Plan execution → NL answer

Entry points:
  - ``QueryPlanner.plan(query_text) → PlanResult`` (planning only)
  - ``QueryPlanner.plan_and_execute(query_text, data) → ExecutableResult``
    (planning + execution + NL answer)

Supports multiple planning strategies via ``planning_strategy`` parameter:
  - "titsp"       — Full TiTSP pipeline (MCTS + Uncertainty + NSGA-II Pareto)
  - "caesura"     — CAESURA (SIGMOD'24): LLM-driven 3-phase planning
  - "lotus"       — LOTUS (VLDB'25): Rule-based + cascade optimization
  - "thalamusdb"  — ThalamusDB (SIGMOD'23): SQL-style AQP planning
  - "palimpzest"  — Palimpzest (CIDR'25): Cascades-framework + Pareto
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .default_impl_registry import build_default_impl_registry
from .mcts_plan_search import (
    LogicalCostModel,
    LogicalPlanCandidate,
    LogicalPlanNode,
    MCTSLogicalConfig,
    MCTSLogicalPlanSearch,
    PhysicalImplInfo,
    ResourceConstraints,
    UserPreference,
    annotate_plan_estimates,
)
from .multi_objective_optimizer import (
    MultiObjectivePhysicalOptimizer,
    ParetoCandidate,
    PhysicalOptConfig,
)
from .operator_compatibility_graph import (
    OperatorCompatibilityGraph,
    QueryContext,
)
from .uncertainty_propagation import (
    OperatorUncertaintyProfile,
    UncertaintyEstimate,
    UncertaintyPrior,
    UncertaintyPropagator,
)
from .plan_executor import PlanExecutionResult, PlanExecutor, Table
from .answer_generator import AnswerGenerator


VALID_STRATEGIES = {
    "titsp", "caesura", "lotus", "thalamusdb", "palimpzest", "nirvana",
    "titsp-no-mcts", "titsp-no-uncertainty", "titsp-no-pareto", "titsp-no-ocg",
}


@dataclass
class PlanTiming:
    """Per-stage timing breakdown (milliseconds)."""
    parse_ms: float = 0.0
    search_ms: float = 0.0
    uncertainty_ms: float = 0.0
    physical_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class PlanResult:
    """Complete output of the end-to-end planning pipeline."""
    query_context: QueryContext
    logical_candidates: List[LogicalPlanCandidate]
    best_logical_plan: LogicalPlanNode
    uncertainty_map: Dict[str, UncertaintyEstimate]
    selected_physical: ParetoCandidate
    pareto_front: List[ParetoCandidate]
    timing: Optional[PlanTiming] = None
    best_quality_score: float = 0.0


@dataclass
class ExecutableResult:
    """Full end-to-end result: planning + execution + NL answer."""
    plan: PlanResult
    execution: PlanExecutionResult
    answer: str


class QueryPlanner:
    """End-to-end query planner: natural language → physical plan.

    Usage::

        planner = QueryPlanner()
        result = planner.plan("Plot paintings depicting Madonna by century")
        print(result.selected_physical.assignment)
        print(result.best_logical_plan.operator_type)

    With LLM-based NL parsing::

        from llm_query_analyzer import create_provider
        provider = create_provider("dashscope", model="qwen-plus")
        planner = QueryPlanner(llm_provider=provider)
        result = planner.plan("找出所有包含猫的图片")

    With baseline strategy::

        planner = QueryPlanner(llm_provider=provider, planning_strategy="caesura")
        result = planner.plan("Find top-rated movies about space")
    """

    def __init__(
        self,
        llm_provider=None,
        impl_registry: Optional[Dict[str, List[PhysicalImplInfo]]] = None,
        history_manager=None,
        constraints: Optional[ResourceConstraints] = None,
        preference: Optional[UserPreference] = None,
        mcts_config: Optional[MCTSLogicalConfig] = None,
        planning_strategy: str = "titsp",
        uncertainty_topk: int = 10,
        metadata_context: Optional[str] = None,
        custom_uncertainty_profiles: Optional[Dict[str, "OperatorUncertaintyProfile"]] = None,
        clarifier: Any = None,
        clarify_dataset: Optional[str] = None,
    ):
        """Args (metadata-graph integration):
            metadata_context: Summary of the metadata graph (modality
                bindings, cross-modal entities). Injected into the Stage-0
                analyzer prompt so operator extraction is grounded in where
                attributes actually live (table column vs. image vs. text).
            custom_uncertainty_profiles: Per-operator prior overrides for
                Stage-3 propagation, e.g. instantiated from the metadata
                graph's cross-modal entity confidence scores.
        """
        if planning_strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Unknown planning_strategy '{planning_strategy}'. "
                f"Valid: {sorted(VALID_STRATEGIES)}"
            )
        self._llm_provider = llm_provider
        self._impl_registry = impl_registry or build_default_impl_registry()
        self._history_manager = history_manager
        self._constraints = constraints or ResourceConstraints()
        self._preference = preference or UserPreference()
        self._mcts_config = mcts_config or MCTSLogicalConfig()
        self._planning_strategy = planning_strategy
        self._uncertainty_topk = uncertainty_topk
        self._metadata_context = metadata_context
        self._custom_uncertainty_profiles = custom_uncertainty_profiles
        # TiQC: optional pre-Stage-0 query-clarification component (a TiQCClarifier).
        # When set, plan() first rewrites the NL query into an unambiguous form,
        # grounded by TiMetagraph. clarifier=None reproduces the no-TiQC ablation.
        self._clarifier = clarifier
        self._clarify_dataset = clarify_dataset

    @property
    def llm_provider(self):
        """The configured LLM provider (or None). Exposed so experiment
        harnesses can read/reset its token-usage counter."""
        return self._llm_provider

    def plan(
        self, query_text: str, base_cardinality: Optional[int] = None,
    ) -> PlanResult:
        """Execute the planning pipeline using the configured strategy.

        Args:
            query_text: Natural language query.
            base_cardinality: Size of the input data (rows), used to seed
                bottom-up cardinality estimates for the data-scale
                attenuation phi(n). Defaults to 1000 when unknown.
        """
        # TiQC pre-Stage-0: disambiguate the NL query (grounded by TiMetagraph)
        # before operator extraction. A no-op when no clarifier is configured.
        if self._clarifier is not None:
            try:
                res = self._clarifier.clarify_for_planner(query_text, self._clarify_dataset)
                query_text = (res or {}).get("refined_query") or query_text
            except Exception:
                pass  # clarification is best-effort; never block planning

        if self._planning_strategy.startswith("titsp"):
            return self._plan_titsp(query_text, base_cardinality)
        return self._plan_baseline(query_text)

    def _plan_titsp(
        self, query_text: str, base_cardinality: Optional[int] = None,
    ) -> PlanResult:
        """Full TiTSP pipeline with ablation support.

        Ablation variants:
          - titsp:               Full pipeline (MCTS + Uncertainty + NSGA-II)
          - titsp-no-mcts:       Greedy single-plan (no MCTS search)
          - titsp-no-uncertainty: Skip uncertainty propagation
          - titsp-no-pareto:     Max-confidence selection (no NSGA-II)
        """
        strategy = self._planning_strategy
        timing = PlanTiming()

        t0 = time.perf_counter()
        context = self._parse_query(query_text)
        timing.parse_ms = (time.perf_counter() - t0) * 1000

        ocg = OperatorCompatibilityGraph(history_manager=self._history_manager,
                                         unconstrained=(strategy == "titsp-no-ocg"))

        # --- Stage 2: Logical Plan Search ---
        t1 = time.perf_counter()
        if strategy == "titsp-no-mcts":
            logical_candidates = self._greedy_logical_search(ocg, context)
        else:
            mcts = MCTSLogicalPlanSearch(
                ocg=ocg,
                query_context=context,
                config=self._mcts_config,
                impl_registry=self._impl_registry,
            )
            logical_candidates = mcts.search()
        timing.search_ms = (time.perf_counter() - t1) * 1000

        if not logical_candidates:
            raise ValueError("Planning produced no valid logical plans.")

        # --- Stage 3: Uncertainty Propagation + uncertainty-aware selection ---
        t2 = time.perf_counter()
        if strategy == "titsp-no-uncertainty":
            # Ablation: skip propagation; keep the top-Q plan and annotate
            # every operator with a uniform prior confidence.
            best = logical_candidates[0]
            uncertainty_map = {
                node.node_id: UncertaintyEstimate(
                    confidence=0.8, variance=0.0,
                    modality_risk=0.0, calibration_source="uniform",
                )
                for node in best.plan.get_all_operators()
            }
        else:
            # Annotate candidate plans with bottom-up cardinality estimates
            # (seeding phi(n), Eq. 8) and per-operator costs (Eq. 12).
            cost_model = LogicalCostModel(self._impl_registry)
            for cand in logical_candidates[: self._uncertainty_topk]:
                annotate_plan_estimates(
                    cand.plan,
                    base_cardinality=base_cardinality or 1000,
                    cost_lookup=cost_model.operator_cost,
                )
            propagator = UncertaintyPropagator(
                prior=(
                    UncertaintyPrior(self._custom_uncertainty_profiles)
                    if self._custom_uncertainty_profiles else None
                ),
                history_manager=self._history_manager,
            )
            best, uncertainty_map = self._select_by_uncertainty(
                logical_candidates, propagator,
            )
        timing.uncertainty_ms = (time.perf_counter() - t2) * 1000

        # Reduce per-node estimates to a per-operator-type confidence. The
        # physical optimizer re-propagates confidence per assignment over the
        # plan tree; these floats remain as the legacy fallback and for
        # ablation/logging. Each type collapses to its weakest instance.
        uncertainty_floats: Dict[str, float] = {}
        for node in best.plan.get_all_operators():
            est = uncertainty_map.get(node.node_id)
            if est is None:
                continue
            prev = uncertainty_floats.get(node.operator_type)
            uncertainty_floats[node.operator_type] = (
                est.confidence if prev is None else min(prev, est.confidence)
            )

        # --- Stage 4: Physical Optimization ---
        t3 = time.perf_counter()
        if strategy == "titsp-no-pareto":
            selected, pareto_front = self._greedy_physical_select(
                best.plan, uncertainty_floats,
            )
        else:
            optimizer = MultiObjectivePhysicalOptimizer(
                logical_plan=best.plan,
                impl_registry=self._impl_registry,
                uncertainty_map=uncertainty_floats,
                constraints=self._constraints,
                preference=self._preference,
                history_manager=self._history_manager,
            )
            selected, pareto_front = optimizer.optimize()
        timing.physical_ms = (time.perf_counter() - t3) * 1000

        timing.total_ms = (
            timing.parse_ms + timing.search_ms
            + timing.uncertainty_ms + timing.physical_ms
        )

        return PlanResult(
            query_context=context,
            logical_candidates=logical_candidates,
            best_logical_plan=best.plan,
            uncertainty_map=uncertainty_map,
            selected_physical=selected,
            pareto_front=pareto_front,
            timing=timing,
            best_quality_score=best.quality_score,
        )

    def _select_by_uncertainty(self, candidates, propagator):
        """Re-rank MCTS candidates by propagated end-to-end confidence.

        MCTS ranks plans by the surrogate quality ``Q`` whose confidence
        term is a crude impl-derived estimate. Here we propagate calibrated,
        modality-aware uncertainty through each of the top-k candidates and
        swap that crude term for the true propagated root confidence, so the
        most *reliable* plan---not merely the highest-Q one---is selected.
        This is the decision pathway that makes uncertainty propagation
        actually influence the chosen logical plan.

        Returns ``(best_candidate, uncertainty_map_of_best)``.
        """
        gamma = self._mcts_config.confidence_weight
        top_k = candidates[: self._uncertainty_topk]

        best_cand = None
        best_score = -float("inf")
        best_map: Dict[str, UncertaintyEstimate] = {}
        for cand in top_k:
            umap = propagator.propagate(cand.plan).uncertainty_map
            root_est = umap.get(cand.plan.node_id)
            plan_conf = (
                root_est.confidence if root_est is not None
                else cand.estimated_confidence
            )
            # Replace Q's crude confidence term with the propagated one.
            refined = cand.quality_score + gamma * (
                plan_conf - cand.estimated_confidence
            )
            if refined > best_score:
                best_score = refined
                best_cand = cand
                best_map = umap

        if best_cand is None:  # defensive: empty candidate list
            best_cand = candidates[0]
            best_map = propagator.propagate(best_cand.plan).uncertainty_map
        return best_cand, best_map

    def _greedy_logical_search(
        self, ocg: OperatorCompatibilityGraph, context: QueryContext,
    ) -> List[LogicalPlanCandidate]:
        """Greedy single-plan generation (ablation: no MCTS)."""
        import uuid
        required = sorted(context.required_operators)
        if "SCAN" not in required:
            required = ["SCAN"] + required

        nodes = []
        for op in required:
            nodes.append(LogicalPlanNode(
                node_id=str(uuid.uuid4())[:8],
                operator_type=op,
            ))
        for i in range(len(nodes) - 1):
            nodes[i + 1].children.append(nodes[i])
            nodes[i].parent = nodes[i + 1]

        root = nodes[-1]
        seq = root.get_operator_types()
        return [LogicalPlanCandidate(
            plan=root,
            quality_score=0.5,
            completeness=1.0,
            estimated_cost=1.0,
            estimated_confidence=0.8,
            ocg_score=0.5,
            operator_sequence=seq,
        )]

    def _greedy_physical_select(
        self, logical_plan: LogicalPlanNode, uncertainty_floats: dict,
    ) -> tuple:
        """Max-confidence greedy selection (ablation: no Pareto)."""
        from .multi_objective_optimizer import PlanObjectives
        assignment = {}
        total_conf = 1.0
        total_lat = 0.0
        total_cost = 0.0
        for node in logical_plan.get_all_operators():
            impls = self._impl_registry.get(node.operator_type, [])
            if not impls:
                assignment[node.operator_type] = "default"
                continue
            best_impl = max(impls, key=lambda x: x.base_confidence)
            assignment[node.operator_type] = best_impl.impl_type
            total_conf *= best_impl.base_confidence
            total_lat += best_impl.base_cost
            total_cost += best_impl.cpu_cores * best_impl.base_cost
        selected = ParetoCandidate(
            assignment=assignment,
            objectives=PlanObjectives(
                latency=total_lat,
                resource_cost=total_cost,
                confidence=total_conf,
            ),
            rank=0,
            crowding_distance=float("inf"),
        )
        return selected, [selected]

    def _plan_baseline(self, query_text: str) -> PlanResult:
        """Dispatch to a baseline planner."""
        timing = PlanTiming()

        t0 = time.perf_counter()
        context = self._parse_query(query_text)
        timing.parse_ms = (time.perf_counter() - t0) * 1000

        ocg = OperatorCompatibilityGraph(history_manager=self._history_manager)

        t1 = time.perf_counter()
        planner = self._create_baseline_planner(ocg)
        result = planner.plan(query_text, context)
        elapsed = (time.perf_counter() - t1) * 1000
        timing.search_ms = elapsed
        timing.total_ms = timing.parse_ms + elapsed

        result.timing = timing
        return result

    def _create_baseline_planner(self, ocg):
        """Instantiate the appropriate baseline planner."""
        strategy = self._planning_strategy

        if strategy == "caesura":
            from .baselines.caesura_planner import CaesuraPlanner
            return CaesuraPlanner(self._llm_provider, self._impl_registry, ocg)
        elif strategy == "lotus":
            from .baselines.lotus_planner import LotusPlanner
            return LotusPlanner(self._llm_provider, self._impl_registry, ocg)
        elif strategy == "thalamusdb":
            from .baselines.thalamusdb_planner import ThalamusDBPlanner
            return ThalamusDBPlanner(self._llm_provider, self._impl_registry, ocg)
        elif strategy == "palimpzest":
            from .baselines.palimpzest_planner import PalimpzestPlanner
            return PalimpzestPlanner(self._llm_provider, self._impl_registry, ocg)
        elif strategy == "nirvana":
            from .baselines.nirvana_planner import NirvanaPlanner
            return NirvanaPlanner(self._llm_provider, self._impl_registry, ocg)
        else:
            raise ValueError(f"Unknown baseline strategy: {strategy}")

    def plan_and_execute(
        self,
        query_text: str,
        data: Table,
        vision_provider: Any = None,
        image_column: str = "",
        right_source: Table = None,
    ) -> ExecutableResult:
        """Full pipeline: NL → plan → execute → NL answer.

        Args:
            query_text: Natural language query.
            data: Input data rows (list of dicts).
            vision_provider: optional vision-capable LLM provider for image
                operators. When set together with ``image_column``, semantic
                operators decide image predicates from the real pixels.
            image_column: name of the column holding image data (bytes,
                {bytes,path} struct, or url).

        Returns:
            ExecutableResult with plan, execution output, and NL answer.
        """
        plan_result = self.plan(
            query_text,
            base_cardinality=len(data) if data else None,
        )

        # Single-pass execution for all methods (self-consistency is a no-op at
        # temperature 0 — verified — and would only add cost, so it is disabled
        # to keep the cross-method comparison fair).
        executor = PlanExecutor(
            data_source=data,
            llm_provider=self._llm_provider,
            query_text=query_text,
            vision_provider=vision_provider,
            image_column=image_column,
            self_consistency=1,
            right_source=right_source or [],
        )
        execution = executor.execute(
            plan_result.best_logical_plan,
            plan_result.selected_physical.assignment,
        )

        # Semantic result assembly: queries whose answer is a structure the row
        # pipeline does not directly emit (per-item ratings, sentiment review
        # pairs) are assembled here from the REAL executed rows via the LLM. A
        # no-op for queries that do not need it. Shared by all planning methods.
        from .plan_executor import assemble_structured
        refined = assemble_structured(query_text, execution.data, self._llm_provider,
                                      source_data=data, vision=vision_provider, image_col=image_column)
        if refined is not execution.data:
            execution.data = refined
            execution.row_count = len(refined)

        generator = AnswerGenerator(llm_provider=self._llm_provider)
        answer = generator.generate(query_text, execution)

        return ExecutableResult(
            plan=plan_result,
            execution=execution,
            answer=answer,
        )

    def _parse_query(self, query_text: str) -> QueryContext:
        if self._llm_provider is not None:
            return QueryContext.from_query_llm(
                query_text, self._llm_provider,
                metadata_context=self._metadata_context,
            )
        return QueryContext.from_query(query_text)
