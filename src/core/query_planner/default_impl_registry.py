"""
Default Physical Implementation Registry.

Provides 2-3 physical implementations per logical operator type with
different latency/confidence/resource tradeoffs. Used as the default
when no custom registry is provided to the QueryPlanner.
"""

from __future__ import annotations

from typing import Dict, List

from .mcts_plan_search import PhysicalImplInfo


def build_default_impl_registry() -> Dict[str, List[PhysicalImplInfo]]:
    """Build the default physical implementation registry for all 16 logical operators."""
    return {
        "SCAN": [
            PhysicalImplInfo("TABLE_SCAN", base_cost=1.0, base_confidence=0.99,
                             gpu_memory_mb=0, cpu_cores=1, memory_mb=256),
            PhysicalImplInfo("INDEX_SCAN", base_cost=0.5, base_confidence=0.99,
                             gpu_memory_mb=0, cpu_cores=1, memory_mb=128),
        ],
        "FILTER": [
            PhysicalImplInfo("PREDICATE_FILTER", base_cost=0.3, base_confidence=0.96,
                             gpu_memory_mb=0, cpu_cores=1, memory_mb=128),
            PhysicalImplInfo("THRESHOLD_FILTER", base_cost=0.5, base_confidence=0.80,
                             gpu_memory_mb=0, cpu_cores=1, memory_mb=128),
        ],
        "PROJECT": [
            PhysicalImplInfo("COLUMN_PROJECT", base_cost=0.2, base_confidence=0.99,
                             gpu_memory_mb=0, cpu_cores=1, memory_mb=64),
        ],
        "AGGREGATE": [
            PhysicalImplInfo("HASH_AGGREGATE", base_cost=2.0, base_confidence=0.93,
                             gpu_memory_mb=0, cpu_cores=2, memory_mb=512),
            PhysicalImplInfo("STREAM_AGGREGATE", base_cost=1.5, base_confidence=0.91,
                             gpu_memory_mb=0, cpu_cores=1, memory_mb=256),
        ],
        "SORT": [
            PhysicalImplInfo("MEMORY_SORT", base_cost=1.0, base_confidence=0.95,
                             gpu_memory_mb=0, cpu_cores=1, memory_mb=512),
            PhysicalImplInfo("EXTERNAL_SORT", base_cost=2.0, base_confidence=0.93,
                             gpu_memory_mb=0, cpu_cores=2, memory_mb=1024),
        ],
        "LIMIT": [
            PhysicalImplInfo("STREAMING_LIMIT", base_cost=0.1, base_confidence=0.99,
                             gpu_memory_mb=0, cpu_cores=1, memory_mb=32),
        ],
        "DISTINCT": [
            PhysicalImplInfo("HASH_DISTINCT", base_cost=1.5, base_confidence=0.93,
                             gpu_memory_mb=0, cpu_cores=1, memory_mb=512),
            PhysicalImplInfo("STREAM_DISTINCT", base_cost=1.2, base_confidence=0.91,
                             gpu_memory_mb=0, cpu_cores=1, memory_mb=256),
        ],
        "JOIN": [
            PhysicalImplInfo("HASH_JOIN", base_cost=3.0, base_confidence=0.92,
                             gpu_memory_mb=0, cpu_cores=2, memory_mb=1024),
            PhysicalImplInfo("NESTED_LOOP_JOIN", base_cost=5.0, base_confidence=0.90,
                             gpu_memory_mb=0, cpu_cores=1, memory_mb=512),
        ],
        "UNION": [
            PhysicalImplInfo("UNION_ALL", base_cost=1.0, base_confidence=0.95,
                             gpu_memory_mb=0, cpu_cores=1, memory_mb=256),
        ],
        "SEMANTIC_SEARCH": [
            PhysicalImplInfo("VECTOR_SEARCH", base_cost=3.0, base_confidence=0.70,
                             gpu_memory_mb=512, cpu_cores=2, memory_mb=1024),
            PhysicalImplInfo("VECTOR_SCAN", base_cost=5.0, base_confidence=0.70,
                             gpu_memory_mb=0, cpu_cores=4, memory_mb=2048),
        ],
        "CROSS_MODAL_MATCH": [
            PhysicalImplInfo("CROSS_MODAL_MATCH", base_cost=6.0, base_confidence=0.52,
                             gpu_memory_mb=2048, cpu_cores=4, memory_mb=4096),
            PhysicalImplInfo("CROSS_MODAL_JOIN", base_cost=8.0, base_confidence=0.55,
                             gpu_memory_mb=1024, cpu_cores=2, memory_mb=2048),
        ],
        "CONTENT_EXTRACT": [
            PhysicalImplInfo("YOLO_OBJECT_DETECTOR", base_cost=6.0, base_confidence=0.72,
                             gpu_memory_mb=2048, cpu_cores=2, memory_mb=2048),
            PhysicalImplInfo("DETR_OBJECT_DETECTOR", base_cost=8.0, base_confidence=0.75,
                             gpu_memory_mb=4096, cpu_cores=4, memory_mb=4096),
            PhysicalImplInfo("IMAGE_ANALYSIS", base_cost=10.0, base_confidence=0.68,
                             gpu_memory_mb=2048, cpu_cores=2, memory_mb=2048),
        ],
        "SIMILARITY_JOIN": [
            PhysicalImplInfo("INTRA_MODAL_JOIN", base_cost=5.0, base_confidence=0.62,
                             gpu_memory_mb=1024, cpu_cores=2, memory_mb=2048),
            PhysicalImplInfo("SEMANTIC_JOIN", base_cost=7.0, base_confidence=0.60,
                             gpu_memory_mb=1024, cpu_cores=4, memory_mb=4096),
        ],
        "FEATURE_TRANSFORM": [
            PhysicalImplInfo("CLIP_IMAGE_ENCODER", base_cost=4.0, base_confidence=0.72,
                             gpu_memory_mb=2048, cpu_cores=2, memory_mb=2048),
            PhysicalImplInfo("CLIP_TEXT_ENCODER", base_cost=3.0, base_confidence=0.73,
                             gpu_memory_mb=1024, cpu_cores=2, memory_mb=1024),
        ],
        "VISUALIZE": [
            PhysicalImplInfo("MATERIALIZE", base_cost=1.0, base_confidence=0.97,
                             gpu_memory_mb=0, cpu_cores=1, memory_mb=256),
        ],
        "EXPORT": [
            PhysicalImplInfo("CSV_EXPORT", base_cost=0.5, base_confidence=0.98,
                             gpu_memory_mb=0, cpu_cores=1, memory_mb=128),
            PhysicalImplInfo("JSON_EXPORT", base_cost=0.5, base_confidence=0.98,
                             gpu_memory_mb=0, cpu_cores=1, memory_mb=128),
        ],
    }
