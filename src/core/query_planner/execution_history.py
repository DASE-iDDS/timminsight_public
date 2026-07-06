"""Execution history store for warm-start calibration (paper Eq. 3 / Eq. 9).

This is the concrete, standard-library-only history manager that activates the
history-based calibration already wired into the planner:

  * OCG edge-weight calibration (Eq. 3) reads ``get_edge_records``.
  * Operator-confidence calibration (Eq. 9) reads ``get_records_by_operator`` /
    ``get_similar_executions``.

At cold start no manager is attached (``history_manager=None``), so the planner
falls back to the hand-set priors w0 / c0. Attaching a populated manager turns
on warm start: recent per-operator and per-edge outcomes shrink the priors
toward realized behaviour through the Bayesian blend inside the calibrators
(``lambda`` prior pseudo-count, ``kappa`` history cap).

Every record is tagged with the query that produced it, so a leak-free
leave-one-out evaluation can build a warm history that excludes the query
currently under test (see ``snapshot_excluding``).

This class is deliberately independent of the SQLite, plan-signature manager
under ``TiMMInsight/``: that one stores whole-plan rows keyed by an md5 of the
plan and exposes neither per-operator nor per-edge records, so it cannot serve
either calibrator interface.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .uncertainty_propagation import HistoricalRecord


def cardinality_accuracy(estimated: float, actual: float, success: bool = True) -> float:
    """Per-record fidelity a_r (paper Eq. 10), in ``[0, 1]``.

    Equals 1.0 when the estimated and actual cardinalities coincide and decays
    smoothly as they diverge; 0.0 for a failed operator; 0.5 when a ratio cannot
    be formed. Mirrors ``HistoricalCalibrator._compute_historical_accuracy`` so
    the edge proxy and the operator calibration share one definition of accuracy.
    """
    if not success:
        return 0.0
    if estimated > 0 and actual > 0:
        return math.exp(-abs(math.log(actual / estimated)))
    return 0.5


class ExecutionHistoryManager:
    """Accumulates per-operator and per-edge execution outcomes and serves them
    to the planner's calibrators.

    Read interface (consumed by the planner, already present in the codebase):
      * ``get_edge_records(parent, child, limit)`` -> ``List[float]``
        (Eq. 3; called unconditionally by the OCG, so it is mandatory)
      * ``get_records_by_operator(op, limit)`` -> ``List[HistoricalRecord]`` (Eq. 9)
      * ``get_similar_executions(operator_type=, parameters=, limit=)``
        -> ``List[HistoricalRecord]`` (Eq. 9, preferred over the former)

    Write interface (called by the experiment harness or a closed loop):
      * ``record_operator(record, query_id=)``
      * ``record_edge_outcome(parent, child, score, query_id=)``
      * ``record_from_execution(plan_root, operator_results, query_id=)``
    """

    def __init__(self) -> None:
        # newest-last, grouped by operator type
        self._op_records: Dict[str, List[Tuple[HistoricalRecord, Optional[str]]]] = defaultdict(list)
        # newest-last, grouped by (parent_type, child_type)
        self._edge_records: Dict[Tuple[str, str], List[Tuple[float, Optional[str]]]] = defaultdict(list)

    # -- write side ---------------------------------------------------------

    def record_operator(self, record: HistoricalRecord, query_id: Optional[str] = None) -> None:
        self._op_records[record.operator_type].append((record, query_id))

    def record_edge_outcome(self, parent_type: str, child_type: str,
                            score: float, query_id: Optional[str] = None) -> None:
        score = max(0.0, min(1.0, float(score)))
        self._edge_records[(parent_type, child_type)].append((score, query_id))

    def record_from_execution(self, plan_root, operator_results: Dict[str, Any],
                              query_id: Optional[str] = None) -> int:
        """Harvest per-operator and per-edge outcomes from one executed plan.

        Each plan node is joined to its ``ExecutionResult`` by ``node_id``:
        the estimated cardinality comes from the node, the actual from the
        result's ``row_count``. Each parent->child edge is scored by the
        child's cardinality fidelity. Returns the number of operator records
        added.
        """
        nodes = plan_root.get_all_operators()
        n_added = 0
        for node in nodes:
            er = operator_results.get(node.node_id)
            if er is None:
                continue
            success = er.row_count > 0
            rec = HistoricalRecord(
                operator_type=node.operator_type,
                parameters_hash=str(hash((node.operator_type, getattr(er, "impl_type", "")))),
                estimated_cardinality=int(node.estimated_cardinality),
                actual_cardinality=int(er.row_count),
                estimated_cost=float(node.estimated_cost),
                # per-operator runtime cost is not exposed by the executor; the
                # calibrator (Eq. 9) never reads actual_cost, so a placeholder
                # is sound. See the paper's limitations note.
                actual_cost=float(node.estimated_cost),
                success=success,
            )
            self.record_operator(rec, query_id)
            n_added += 1
        for node in nodes:
            for child in node.children:
                cer = operator_results.get(child.node_id)
                if cer is None:
                    continue
                acc = cardinality_accuracy(
                    child.estimated_cardinality, cer.row_count, success=cer.row_count > 0)
                self.record_edge_outcome(node.operator_type, child.operator_type, acc, query_id)
        return n_added

    # -- read side (planner calibrators) ------------------------------------

    def get_edge_records(self, parent_type: str, child_type: str, limit: int = 20) -> List[float]:
        recs = self._edge_records.get((parent_type, child_type), [])
        return [s for (s, _qid) in recs[-limit:]]

    def get_records_by_operator(self, operator_type: str, limit: int = 20) -> List[HistoricalRecord]:
        recs = self._op_records.get(operator_type, [])
        return [r for (r, _qid) in recs[-limit:]]

    def get_similar_executions(self, operator_type: Optional[str] = None,
                               parameters: Optional[Dict[str, Any]] = None,
                               limit: int = 20) -> List[HistoricalRecord]:
        # ``parameters`` is accepted for interface compatibility but ignored,
        # matching the reference behaviour; filtering is purely by operator type.
        if operator_type is None:
            return []
        return self.get_records_by_operator(operator_type, limit)

    # -- leave-one-out support ---------------------------------------------

    def snapshot_excluding(self, query_id: str) -> "ExecutionHistoryManager":
        """Return a new manager with every record tagged ``query_id`` removed,
        so that query can be evaluated warm without its own outcome leaking in.
        """
        clone = ExecutionHistoryManager()
        for op, recs in self._op_records.items():
            kept = [(r, q) for (r, q) in recs if q != query_id]
            if kept:
                clone._op_records[op] = list(kept)
        for edge, recs in self._edge_records.items():
            kept = [(s, q) for (s, q) in recs if q != query_id]
            if kept:
                clone._edge_records[edge] = list(kept)
        return clone

    # -- introspection ------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        return {
            "operator_types": len(self._op_records),
            "operator_records": sum(len(v) for v in self._op_records.values()),
            "edge_types": len(self._edge_records),
            "edge_records": sum(len(v) for v in self._edge_records.values()),
        }
