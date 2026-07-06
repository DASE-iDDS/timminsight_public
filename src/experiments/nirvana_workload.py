"""
Nirvana Workload Adaptation Experiment.

Tests the planning framework against 36 real-world multimodal queries from
the Nirvana benchmark (https://github.com/JunHao-Zhu/nirvana).

Three datasets (estate / steam / imdb), 12 queries each, using 5 semantic
operators: filter, map, reduce, rank, join.

Operator mapping (Nirvana → TiTSP):
  semantic_filter  → FILTER (+ CONTENT_EXTRACT for image/text understanding)
  semantic_map     → CONTENT_EXTRACT (extract structured info from unstructured)
  semantic_reduce  → AGGREGATE
  semantic_rank    → SORT + LIMIT
  semantic_join    → SIMILARITY_JOIN or JOIN
  (SCAN is always added as leaf)
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.query_planner.query_planner import QueryPlanner, PlanResult, ExecutableResult
from core.query_planner.mcts_plan_search import LogicalPlanNode
from core.query_planner.plan_executor import Table


# ---------------------------------------------------------------------------
# Nirvana Query Definitions
# ---------------------------------------------------------------------------

@dataclass
class NirvanaQuery:
    """A single query from the Nirvana benchmark."""
    dataset: str
    query_id: str
    nl_description: str
    nirvana_operators: List[str]
    expected_our_operators: Set[str]
    expected_answer: Optional[str] = None


NIRVANA_QUERIES: List[NirvanaQuery] = [
    # -----------------------------------------------------------------------
    # Estate dataset (multimodal real estate with images)
    # -----------------------------------------------------------------------
    NirvanaQuery(
        dataset="estate", query_id="q1",
        nl_description="Filter houses that have a yard based on their pictures",
        nirvana_operators=["filter"],
        expected_our_operators={"SCAN", "FILTER", "CONTENT_EXTRACT"},
        expected_answer="4",
    ),
    NirvanaQuery(
        dataset="estate", query_id="q2",
        nl_description="Filter houses located in Lekki Lagos from the location field",
        nirvana_operators=["filter"],
        expected_our_operators={"SCAN", "FILTER"},
        expected_answer="2",
    ),
    NirvanaQuery(
        dataset="estate", query_id="q3",
        nl_description="Extract the number of bedrooms from estate title text",
        nirvana_operators=["map"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT", "PROJECT"},
        expected_answer="Bedroom",
    ),
    NirvanaQuery(
        dataset="estate", query_id="q4",
        nl_description="Extract house price from estate details and filter houses with more than 3 bedrooms",
        nirvana_operators=["map", "filter"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT", "FILTER"},
        expected_answer="3",
    ),
    NirvanaQuery(
        dataset="estate", query_id="q5",
        nl_description="Filter houses that appear to be newly built based on their photos",
        nirvana_operators=["filter"],
        expected_our_operators={"SCAN", "FILTER", "CONTENT_EXTRACT"},
        expected_answer="3",
    ),
    NirvanaQuery(
        dataset="estate", query_id="q6",
        nl_description="Extract house type from title and count houses by type",
        nirvana_operators=["map", "reduce"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT", "AGGREGATE"},
        expected_answer="Detached Duplex",
    ),
    NirvanaQuery(
        dataset="estate", query_id="q7",
        nl_description="Filter detached duplex houses and rank by price descending",
        nirvana_operators=["filter", "rank"],
        expected_our_operators={"SCAN", "FILTER", "SORT"},
        expected_answer="65000000",
    ),
    NirvanaQuery(
        dataset="estate", query_id="q8",
        nl_description="Extract price from details, filter houses with pool from photos, compute average price",
        nirvana_operators=["map", "filter", "reduce"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT", "FILTER", "AGGREGATE"},
        expected_answer="133333333",
    ),
    NirvanaQuery(
        dataset="estate", query_id="q9",
        nl_description="Filter houses with garden from images and sort by number of bedrooms",
        nirvana_operators=["filter", "rank"],
        expected_our_operators={"SCAN", "FILTER", "CONTENT_EXTRACT", "SORT"},
        expected_answer="Mansion",
    ),
    NirvanaQuery(
        dataset="estate", query_id="q10",
        nl_description="Extract price from details, filter houses with 3 to 6 bedrooms that are detached duplexes, find lowest price",
        nirvana_operators=["map", "filter", "filter", "filter", "reduce"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT", "FILTER", "AGGREGATE"},
        expected_answer="45000000",
    ),
    NirvanaQuery(
        dataset="estate", query_id="q11",
        nl_description="Extract location district from address and count houses per district",
        nirvana_operators=["map", "reduce"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT", "AGGREGATE"},
        expected_answer="Lekki",
    ),
    NirvanaQuery(
        dataset="estate", query_id="q12",
        nl_description="Filter luxury houses from photos, extract price, rank top 5 most expensive",
        nirvana_operators=["filter", "map", "rank"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT", "FILTER", "SORT", "LIMIT"},
        expected_answer="250000000",
    ),

    # -----------------------------------------------------------------------
    # Steam dataset (game metadata with text reviews)
    # -----------------------------------------------------------------------
    NirvanaQuery(
        dataset="steam", query_id="q1",
        nl_description="Filter games that are free to play based on price field",
        nirvana_operators=["filter"],
        expected_our_operators={"SCAN", "FILTER"},
        expected_answer="2",
    ),
    NirvanaQuery(
        dataset="steam", query_id="q2",
        nl_description="Extract game genre from description text",
        nirvana_operators=["map"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT"},
        expected_answer="RPG",
    ),
    NirvanaQuery(
        dataset="steam", query_id="q3",
        nl_description="Filter games rated for adults only and count them",
        nirvana_operators=["filter", "reduce"],
        expected_our_operators={"SCAN", "FILTER", "AGGREGATE"},
        expected_answer="3",
    ),
    NirvanaQuery(
        dataset="steam", query_id="q4",
        nl_description="Transform review text to sentiment and filter games with positive reviews",
        nirvana_operators=["map", "filter"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT", "FILTER"},
        expected_answer="5",
    ),
    NirvanaQuery(
        dataset="steam", query_id="q5",
        nl_description="Filter games supporting Windows and MacOS platforms",
        nirvana_operators=["filter"],
        expected_our_operators={"SCAN", "FILTER"},
        expected_answer="4",
    ),
    NirvanaQuery(
        dataset="steam", query_id="q6",
        nl_description="Extract developer name from details and count games per developer",
        nirvana_operators=["map", "reduce"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT", "AGGREGATE"},
        expected_answer="developer",
    ),
    NirvanaQuery(
        dataset="steam", query_id="q7",
        nl_description="Filter multiplayer games and rank by user rating descending",
        nirvana_operators=["filter", "rank"],
        expected_our_operators={"SCAN", "FILTER", "SORT"},
        expected_answer="Stardew Valley",
    ),
    NirvanaQuery(
        dataset="steam", query_id="q8",
        nl_description="Extract genre from description, filter RPG games, compute average price",
        nirvana_operators=["map", "filter", "reduce"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT", "FILTER", "AGGREGATE"},
        expected_answer="53.32",
    ),
    NirvanaQuery(
        dataset="steam", query_id="q9",
        nl_description="Filter games with positive reviews that support Linux",
        nirvana_operators=["map", "filter", "filter"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT", "FILTER"},
        expected_answer="3",
    ),
    NirvanaQuery(
        dataset="steam", query_id="q10",
        nl_description="Transform review to sentiment, filter positive reviews for PEGI 18 games on Windows and MacOS, compute average price",
        nirvana_operators=["map", "filter", "filter", "filter", "reduce"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT", "FILTER", "AGGREGATE"},
        expected_answer="PEGI 18",
    ),
    NirvanaQuery(
        dataset="steam", query_id="q11",
        nl_description="Extract release year from date and count games released per year",
        nirvana_operators=["map", "reduce"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT", "AGGREGATE"},
        expected_answer="2022",
    ),
    NirvanaQuery(
        dataset="steam", query_id="q12",
        nl_description="Filter action games, extract playtime from reviews, rank top 10 most played",
        nirvana_operators=["filter", "map", "rank"],
        expected_our_operators={"SCAN", "FILTER", "CONTENT_EXTRACT", "SORT", "LIMIT"},
        expected_answer="action",
    ),

    # -----------------------------------------------------------------------
    # IMDB dataset (movie metadata with plot text)
    # -----------------------------------------------------------------------
    NirvanaQuery(
        dataset="imdb", query_id="q1",
        nl_description="Filter movies with IMDB rating higher than 8",
        nirvana_operators=["filter"],
        expected_our_operators={"SCAN", "FILTER"},
        expected_answer="3",
    ),
    NirvanaQuery(
        dataset="imdb", query_id="q2",
        nl_description="Extract genre from movie plot description",
        nirvana_operators=["map"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT"},
        expected_answer="Drama",
    ),
    NirvanaQuery(
        dataset="imdb", query_id="q3",
        nl_description="Filter drama movies and count them",
        nirvana_operators=["filter", "reduce"],
        expected_our_operators={"SCAN", "FILTER", "AGGREGATE"},
        expected_answer="3",
    ),
    NirvanaQuery(
        dataset="imdb", query_id="q4",
        nl_description="Extract genre from plot and filter comedy movies",
        nirvana_operators=["map", "filter"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT", "FILTER"},
        expected_answer="0",
    ),
    NirvanaQuery(
        dataset="imdb", query_id="q5",
        nl_description="Filter movies released after 2010 with rating above 7",
        nirvana_operators=["filter", "filter"],
        expected_our_operators={"SCAN", "FILTER"},
        expected_answer="2",
    ),
    NirvanaQuery(
        dataset="imdb", query_id="q6",
        nl_description="Extract director from credits and count movies per director",
        nirvana_operators=["map", "reduce"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT", "AGGREGATE"},
        expected_answer="Christopher Nolan",
    ),
    NirvanaQuery(
        dataset="imdb", query_id="q7",
        nl_description="Filter highly rated movies and rank by box office revenue",
        nirvana_operators=["filter", "rank"],
        expected_our_operators={"SCAN", "FILTER", "SORT"},
        expected_answer="Dark Knight",
    ),
    NirvanaQuery(
        dataset="imdb", query_id="q8",
        nl_description="Extract genre from plot, filter sci-fi movies, compute average rating",
        nirvana_operators=["map", "filter", "reduce"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT", "FILTER", "AGGREGATE"},
        expected_answer="8.75",
    ),
    NirvanaQuery(
        dataset="imdb", query_id="q9",
        nl_description="Filter movies with rating between 7 and 9 released after 2000",
        nirvana_operators=["filter", "filter", "filter"],
        expected_our_operators={"SCAN", "FILTER"},
        expected_answer="Inception",
    ),
    NirvanaQuery(
        dataset="imdb", query_id="q10",
        nl_description="Extract genre from plot, filter crime movies with rating between 8.5 and 9, count them",
        nirvana_operators=["map", "filter", "filter", "filter", "reduce"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT", "FILTER", "AGGREGATE"},
        expected_answer="crime",
    ),
    NirvanaQuery(
        dataset="imdb", query_id="q11",
        nl_description="Extract decade from release year and count movies per decade",
        nirvana_operators=["map", "reduce"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT", "AGGREGATE"},
        expected_answer="2010",
    ),
    NirvanaQuery(
        dataset="imdb", query_id="q12",
        nl_description="Extract genre from plot, filter thriller movies, rank top 5 by rating",
        nirvana_operators=["filter", "map", "rank"],
        expected_our_operators={"SCAN", "FILTER", "CONTENT_EXTRACT", "SORT", "LIMIT"},
        expected_answer="0",
    ),
]


# ---------------------------------------------------------------------------
# Sample Datasets (mirroring Nirvana benchmark structure)
# ---------------------------------------------------------------------------

SAMPLE_ESTATE_DATA: Table = [
    {"id": 1, "Title": "3 Bedroom Detached Duplex in Lekki", "Location": "Lekki, Lagos",
     "Details": "Price: 45000000 NGN. 3 bed, 2 bath.", "price": 45000000,
     "bedrooms": 3, "type": "Detached Duplex", "image": "estate_01.jpg",
     "has_yard": True, "has_pool": False, "is_new": True},
    {"id": 2, "Title": "5 Bedroom Semi-Detached in Ikoyi", "Location": "Ikoyi, Lagos",
     "Details": "Price: 85000000 NGN. 5 bed, 4 bath.", "price": 85000000,
     "bedrooms": 5, "type": "Semi-Detached", "image": "estate_02.jpg",
     "has_yard": True, "has_pool": True, "is_new": False},
    {"id": 3, "Title": "2 Bedroom Flat in Ajah", "Location": "Ajah, Lagos",
     "Details": "Price: 15000000 NGN. 2 bed, 1 bath.", "price": 15000000,
     "bedrooms": 2, "type": "Flat", "image": "estate_03.jpg",
     "has_yard": False, "has_pool": False, "is_new": True},
    {"id": 4, "Title": "4 Bedroom Detached Duplex in VGC", "Location": "VGC, Lagos",
     "Details": "Price: 65000000 NGN. 4 bed, 3 bath.", "price": 65000000,
     "bedrooms": 4, "type": "Detached Duplex", "image": "estate_04.jpg",
     "has_yard": True, "has_pool": True, "is_new": False},
    {"id": 5, "Title": "6 Bedroom Mansion in Banana Island", "Location": "Banana Island, Lagos",
     "Details": "Price: 250000000 NGN. 6 bed, 5 bath.", "price": 250000000,
     "bedrooms": 6, "type": "Mansion", "image": "estate_05.jpg",
     "has_yard": True, "has_pool": True, "is_new": True},
    {"id": 6, "Title": "3 Bedroom Terrace in Lekki Phase 1", "Location": "Lekki, Lagos",
     "Details": "Price: 35000000 NGN. 3 bed, 2 bath.", "price": 35000000,
     "bedrooms": 3, "type": "Terrace", "image": "estate_06.jpg",
     "has_yard": False, "has_pool": False, "is_new": False},
]

SAMPLE_STEAM_DATA: Table = [
    {"id": 1, "name": "Elden Ring", "price": 59.99, "rating": 9.2,
     "genre": "RPG", "platforms": "Windows, MacOS", "review": "Amazing open world RPG",
     "review_sentiment": "positive", "pegi": "PEGI 16", "release_date": "2022-02-25",
     "is_free": False, "is_multiplayer": True},
    {"id": 2, "name": "Counter-Strike 2", "price": 0.0, "rating": 8.5,
     "genre": "FPS", "platforms": "Windows, Linux", "review": "Best competitive shooter",
     "review_sentiment": "positive", "pegi": "PEGI 18", "release_date": "2023-09-27",
     "is_free": True, "is_multiplayer": True},
    {"id": 3, "name": "Stardew Valley", "price": 14.99, "rating": 9.5,
     "genre": "Simulation", "platforms": "Windows, MacOS, Linux", "review": "Relaxing farm sim",
     "review_sentiment": "positive", "pegi": "PEGI 7", "release_date": "2016-02-26",
     "is_free": False, "is_multiplayer": True},
    {"id": 4, "name": "Cyberpunk 2077", "price": 59.99, "rating": 7.8,
     "genre": "RPG", "platforms": "Windows", "review": "Buggy at launch but improved",
     "review_sentiment": "mixed", "pegi": "PEGI 18", "release_date": "2020-12-10",
     "is_free": False, "is_multiplayer": False},
    {"id": 5, "name": "Dota 2", "price": 0.0, "rating": 8.0,
     "genre": "MOBA", "platforms": "Windows, MacOS, Linux", "review": "Steep learning curve",
     "review_sentiment": "positive", "pegi": "PEGI 12", "release_date": "2013-07-09",
     "is_free": True, "is_multiplayer": True},
    {"id": 6, "name": "The Witcher 3", "price": 39.99, "rating": 9.8,
     "genre": "RPG", "platforms": "Windows, MacOS", "review": "Masterpiece of storytelling",
     "review_sentiment": "positive", "pegi": "PEGI 18", "release_date": "2015-05-19",
     "is_free": False, "is_multiplayer": False},
]

SAMPLE_IMDB_DATA: Table = [
    {"Title": "The Shawshank Redemption", "Year": 1994, "IMDB_rating": 9.3,
     "Genre": "Drama", "Director": "Frank Darabont",
     "Plot": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
     "BoxOffice": 58300000},
    {"Title": "The Dark Knight", "Year": 2008, "IMDB_rating": 9.0,
     "Genre": "Action", "Director": "Christopher Nolan",
     "Plot": "When the menace known as the Joker wreaks havoc on Gotham, Batman must accept the consequences of his war on crime.",
     "BoxOffice": 533300000},
    {"Title": "Pulp Fiction", "Year": 1994, "IMDB_rating": 8.9,
     "Genre": "Crime", "Director": "Quentin Tarantino",
     "Plot": "The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption.",
     "BoxOffice": 107900000},
    {"Title": "Inception", "Year": 2010, "IMDB_rating": 8.8,
     "Genre": "Sci-Fi", "Director": "Christopher Nolan",
     "Plot": "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea.",
     "BoxOffice": 292600000},
    {"Title": "Parasite", "Year": 2019, "IMDB_rating": 8.5,
     "Genre": "Drama", "Director": "Bong Joon-ho",
     "Plot": "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan.",
     "BoxOffice": 53400000},
    {"Title": "Interstellar", "Year": 2014, "IMDB_rating": 8.7,
     "Genre": "Sci-Fi", "Director": "Christopher Nolan",
     "Plot": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
     "BoxOffice": 188000000},
    {"Title": "The Godfather", "Year": 1972, "IMDB_rating": 9.2,
     "Genre": "Crime", "Director": "Francis Ford Coppola",
     "Plot": "The aging patriarch of an organized crime dynasty in postwar New York transfers control to his reluctant youngest son.",
     "BoxOffice": 246100000},
    {"Title": "Whiplash", "Year": 2014, "IMDB_rating": 8.5,
     "Genre": "Drama", "Director": "Damien Chazelle",
     "Plot": "A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing.",
     "BoxOffice": 13100000},
]

DATASET_REGISTRY: Dict[str, Table] = {
    "estate": SAMPLE_ESTATE_DATA,
    "steam": SAMPLE_STEAM_DATA,
    "imdb": SAMPLE_IMDB_DATA,
}


# ---------------------------------------------------------------------------
# End-to-End Execution + Validation
# ---------------------------------------------------------------------------

@dataclass
class E2EResult:
    """Result of full end-to-end execution of a single query."""
    query: NirvanaQuery
    executable: Optional[ExecutableResult]
    answer: str
    answer_validated: Optional[bool]
    output_rows: int
    elapsed_ms: float
    mcts_candidates: int = 0
    plan_depth: int = 0
    plan_operators: List[str] = field(default_factory=list)
    pareto_size: int = 0
    physical_assignment: Dict[str, str] = field(default_factory=dict)
    exec_trace: List[str] = field(default_factory=list)
    error: Optional[str] = None


def run_single_e2e(
    planner: QueryPlanner,
    q: NirvanaQuery,
    data: Table,
) -> E2EResult:
    """Run a single query through the full NL→plan→execute→answer pipeline."""
    t0 = time.perf_counter()
    try:
        result = planner.plan_and_execute(q.nl_description, data)
        elapsed = (time.perf_counter() - t0) * 1000

        validated = None
        if q.expected_answer is not None:
            validated = validate_answer(result.answer, result.execution.data, q.expected_answer)

        plan = result.plan
        ops = collect_operators(plan.best_logical_plan)
        exec_trace = []
        for nid, er in result.execution.operator_results.items():
            exec_trace.append(f"{er.operator_type}[{er.impl_type}]: {er.row_count} rows")

        return E2EResult(
            query=q,
            executable=result,
            answer=result.answer,
            answer_validated=validated,
            output_rows=result.execution.row_count,
            elapsed_ms=elapsed,
            mcts_candidates=len(plan.logical_candidates),
            plan_depth=plan_depth(plan.best_logical_plan),
            plan_operators=sorted(ops),
            pareto_size=len(plan.pareto_front),
            physical_assignment=plan.selected_physical.assignment,
            exec_trace=exec_trace,
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        elapsed = (time.perf_counter() - t0) * 1000
        return E2EResult(
            query=q,
            executable=None,
            answer="",
            answer_validated=False,
            output_rows=0,
            elapsed_ms=elapsed,
            error=str(exc),
        )


def validate_answer(answer: str, data: Table, expected: str) -> bool:
    """Check if the answer or data contains the expected value."""
    if expected in answer:
        return True
    data_str = json.dumps(data, default=str)
    if expected in data_str:
        return True
    try:
        expected_num = float(expected)
        for row in data:
            for v in row.values():
                if isinstance(v, (int, float)) and abs(v - expected_num) < 1e-6:
                    return True
                if str(v) == expected:
                    return True
        if str(len(data)) == expected:
            return True
    except (ValueError, TypeError):
        pass
    return False


def format_plan_tree(node: LogicalPlanNode, assignment: Dict[str, str], indent: int = 0) -> str:
    """Format a plan tree as a readable string."""
    impl = assignment.get(node.operator_type, "?")
    line = "  " * indent + f"{node.operator_type} [{impl}]"
    lines = [line]
    for child in node.children:
        lines.append(format_plan_tree(child, assignment, indent + 1))
    return "\n".join(lines)


def run_e2e_experiment(
    planner: Optional[QueryPlanner] = None,
    datasets: Optional[List[str]] = None,
    verbose: bool = True,
) -> List[E2EResult]:
    """Run the full end-to-end experiment with pipeline trace.

    Traces all stages:
      Stage 0: NL → QueryContext (LLM or regex)
      Stage 2: MCTS logical plan search → candidate plans
      Stage 3: Uncertainty propagation
      Stage 4: Physical optimization (Pareto + Tchebycheff)
      Stage 5: Plan execution → data output
      Stage 6: Answer generation → NL answer
    """
    if planner is None:
        planner = QueryPlanner()

    queries = NIRVANA_QUERIES
    if datasets:
        queries = [q for q in queries if q.dataset in datasets]

    results: List[E2EResult] = []

    for q in queries:
        data = DATASET_REGISTRY.get(q.dataset, [])
        r = run_single_e2e(planner, q, data)
        results.append(r)

        if verbose:
            ok = "PASS" if r.answer_validated else ("FAIL" if r.answer_validated is False else "---")
            err_flag = " ERROR" if r.error else ""
            print(f"  [{ok:4s}] {q.dataset}/{q.query_id}: \"{q.nl_description}\"")

            if r.executable:
                plan = r.executable.plan

                print(f"         Stage 0 (NL Parse):    required={plan.query_context.required_operators}")
                print(f"         Stage 2 (MCTS):        {r.mcts_candidates} candidates explored")
                print(f"         Stage 3 (Uncertainty): {len(plan.uncertainty_map)} operators annotated")
                print(f"         Stage 4 (Physical):    Pareto front={r.pareto_size}, "
                      f"selected={r.physical_assignment}")

                tree_str = format_plan_tree(
                    plan.best_logical_plan,
                    plan.selected_physical.assignment
                )
                for line in tree_str.split("\n"):
                    print(f"         Logical Plan: {line}")

                print(f"         Stage 5 (Execute):     {r.output_rows} rows output")
                for et in r.exec_trace:
                    print(f"           {et}")

                ans_preview = r.answer.replace("\n", " ")[:100]
                print(f"         Stage 6 (Answer):      {ans_preview}")
                print(f"         Time: {r.elapsed_ms:.1f}ms")

            if r.error:
                print(f"         Error: {r.error}")
            print()

    return results


def print_e2e_summary(results: List[E2EResult]) -> Dict:
    """Print and return end-to-end experiment summary."""
    total = len(results)
    errors = sum(1 for r in results if r.error)
    successful = total - errors

    validated = [r for r in results if r.answer_validated is not None]
    correct = sum(1 for r in validated if r.answer_validated)
    accuracy = correct / len(validated) if validated else 0

    times = [r.elapsed_ms for r in results if not r.error]
    avg_time = sum(times) / len(times) if times else 0

    rows_out = [r.output_rows for r in results if not r.error]
    avg_rows = sum(rows_out) / len(rows_out) if rows_out else 0

    non_empty = sum(1 for r in results if not r.error and r.output_rows > 0)

    datasets_seen = sorted(set(r.query.dataset for r in results))

    mcts_cands = [r.mcts_candidates for r in results if not r.error]
    avg_mcts = sum(mcts_cands) / len(mcts_cands) if mcts_cands else 0

    depths = [r.plan_depth for r in results if not r.error]
    avg_depth = sum(depths) / len(depths) if depths else 0

    paretos = [r.pareto_size for r in results if not r.error]
    avg_pareto = sum(paretos) / len(paretos) if paretos else 0

    multi_op_plans = sum(1 for r in results if not r.error and r.plan_depth > 1)

    print("\n" + "=" * 70)
    print("END-TO-END PIPELINE — SUMMARY")
    print("=" * 70)
    print(f"Total queries:           {total}")
    print(f"Executed successfully:   {successful}/{total}")
    print(f"Non-empty results:       {non_empty}/{successful}")
    print()
    print(f"--- Planning Stage ---")
    print(f"Avg MCTS candidates:     {avg_mcts:.1f}")
    print(f"Avg plan depth:          {avg_depth:.1f}")
    print(f"Multi-operator plans:    {multi_op_plans}/{successful}")
    print(f"Avg Pareto front size:   {avg_pareto:.1f}")
    print()
    print(f"--- Execution Stage ---")
    print(f"Validated queries:       {len(validated)}")
    print(f"Correct answers:         {correct}/{len(validated)} ({accuracy:.0%})")
    print(f"Avg output rows:         {avg_rows:.1f}")
    print(f"Avg execution time:      {avg_time:.1f} ms")

    print(f"\nPer-dataset breakdown:")
    for ds in datasets_seen:
        ds_r = [r for r in results if r.query.dataset == ds and not r.error]
        ds_val = [r for r in ds_r if r.answer_validated is not None]
        ds_correct = sum(1 for r in ds_val if r.answer_validated)
        ds_nonempty = sum(1 for r in ds_r if r.output_rows > 0)
        ds_multi = sum(1 for r in ds_r if r.plan_depth > 1)
        ds_mcts = sum(r.mcts_candidates for r in ds_r) / len(ds_r) if ds_r else 0
        print(f"  {ds:8s}: {len(ds_r)} queries, "
              f"multi-op={ds_multi}/{len(ds_r)}, "
              f"correct={ds_correct}/{len(ds_val) if ds_val else 0}, "
              f"avg_mcts={ds_mcts:.0f}")

    print("=" * 70)

    return {
        "total": total,
        "successful": successful,
        "errors": errors,
        "validated": len(validated),
        "correct": correct,
        "accuracy": accuracy,
        "avg_mcts_candidates": avg_mcts,
        "avg_depth": avg_depth,
        "multi_op_plans": multi_op_plans,
        "avg_pareto": avg_pareto,
        "avg_rows": avg_rows,
        "avg_time_ms": avg_time,
    }


# ---------------------------------------------------------------------------
# Experiment Runner (planning only — original)
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Result of planning a single Nirvana query."""
    query: NirvanaQuery
    plan_result: Optional[PlanResult]
    planned_operators: Set[str]
    expected_operators: Set[str]
    coverage: float
    extra_operators: Set[str]
    missing_operators: Set[str]
    plan_depth: int
    num_pareto: int
    elapsed_ms: float
    error: Optional[str] = None


def collect_operators(node: LogicalPlanNode) -> Set[str]:
    """Recursively collect all operator types in a plan tree."""
    ops = {node.operator_type}
    for child in node.children:
        ops |= collect_operators(child)
    return ops


def plan_depth(node: LogicalPlanNode) -> int:
    """Compute the depth of a plan tree."""
    if not node.children:
        return 1
    return 1 + max(plan_depth(c) for c in node.children)


def run_single_query(planner: QueryPlanner, q: NirvanaQuery) -> QueryResult:
    """Run a single Nirvana query through the planner."""
    t0 = time.perf_counter()
    try:
        result = planner.plan(q.nl_description)
        elapsed = (time.perf_counter() - t0) * 1000

        planned = collect_operators(result.best_logical_plan)
        expected = q.expected_our_operators

        matched = planned & expected
        coverage = len(matched) / len(expected) if expected else 1.0
        extra = planned - expected
        missing = expected - planned

        return QueryResult(
            query=q,
            plan_result=result,
            planned_operators=planned,
            expected_operators=expected,
            coverage=coverage,
            extra_operators=extra,
            missing_operators=missing,
            plan_depth=plan_depth(result.best_logical_plan),
            num_pareto=len(result.pareto_front),
            elapsed_ms=elapsed,
        )
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        return QueryResult(
            query=q,
            plan_result=None,
            planned_operators=set(),
            expected_operators=q.expected_our_operators,
            coverage=0.0,
            extra_operators=set(),
            missing_operators=q.expected_our_operators,
            plan_depth=0,
            num_pareto=0,
            elapsed_ms=elapsed,
            error=str(exc),
        )


def run_experiment(
    planner: Optional[QueryPlanner] = None,
    datasets: Optional[List[str]] = None,
    verbose: bool = True,
) -> List[QueryResult]:
    """Run the full Nirvana workload experiment.

    Args:
        planner: QueryPlanner instance. Uses regex-based NL parsing if no
                 LLM provider is configured.
        datasets: Subset of datasets to run (default: all three).
        verbose: Print per-query results.
    """
    if planner is None:
        planner = QueryPlanner()

    queries = NIRVANA_QUERIES
    if datasets:
        queries = [q for q in queries if q.dataset in datasets]

    results: List[QueryResult] = []

    for q in queries:
        r = run_single_query(planner, q)
        results.append(r)

        if verbose:
            status = "OK" if r.coverage >= 0.8 else ("PARTIAL" if r.coverage > 0 else "FAIL")
            icon = {"OK": "+", "PARTIAL": "~", "FAIL": "!"}[status]
            print(f"  [{icon}] {q.dataset}/{q.query_id}: coverage={r.coverage:.0%} "
                  f"depth={r.plan_depth} pareto={r.num_pareto} "
                  f"time={r.elapsed_ms:.1f}ms", end="")
            if r.missing_operators:
                print(f"  missing={r.missing_operators}", end="")
            if r.extra_operators:
                print(f"  extra={r.extra_operators}", end="")
            if r.error:
                print(f"  ERROR: {r.error}", end="")
            print()

    return results


def print_summary(results: List[QueryResult]) -> Dict:
    """Print and return experiment summary statistics."""
    total = len(results)
    errors = sum(1 for r in results if r.error)
    successful = total - errors

    coverages = [r.coverage for r in results if not r.error]
    avg_coverage = sum(coverages) / len(coverages) if coverages else 0

    full_match = sum(1 for c in coverages if c >= 1.0)
    high_match = sum(1 for c in coverages if c >= 0.8)

    depths = [r.plan_depth for r in results if not r.error]
    avg_depth = sum(depths) / len(depths) if depths else 0

    paretos = [r.num_pareto for r in results if not r.error]
    avg_pareto = sum(paretos) / len(paretos) if paretos else 0

    times = [r.elapsed_ms for r in results if not r.error]
    avg_time = sum(times) / len(times) if times else 0

    all_missing: Dict[str, int] = {}
    all_extra: Dict[str, int] = {}
    for r in results:
        for op in r.missing_operators:
            all_missing[op] = all_missing.get(op, 0) + 1
        for op in r.extra_operators:
            all_extra[op] = all_extra.get(op, 0) + 1

    # Per-dataset breakdown
    datasets_seen = sorted(set(r.query.dataset for r in results))

    print("\n" + "=" * 70)
    print("NIRVANA WORKLOAD ADAPTATION — SUMMARY")
    print("=" * 70)
    print(f"Total queries:        {total}")
    print(f"Successful:           {successful} ({successful/total:.0%})")
    print(f"Errors:               {errors}")
    print(f"Avg operator coverage:{avg_coverage:.1%}")
    print(f"Full match (100%):    {full_match}/{successful}")
    print(f"High match (≥80%):    {high_match}/{successful}")
    print(f"Avg plan depth:       {avg_depth:.1f}")
    print(f"Avg Pareto size:      {avg_pareto:.1f}")
    print(f"Avg planning time:    {avg_time:.1f} ms")

    if all_missing:
        print(f"\nMost missed operators:")
        for op, cnt in sorted(all_missing.items(), key=lambda x: -x[1]):
            print(f"  {op}: {cnt} queries")

    if all_extra:
        print(f"\nMost common extra operators:")
        for op, cnt in sorted(all_extra.items(), key=lambda x: -x[1]):
            print(f"  {op}: {cnt} queries")

    print(f"\nPer-dataset breakdown:")
    for ds in datasets_seen:
        ds_results = [r for r in results if r.query.dataset == ds and not r.error]
        ds_cov = sum(r.coverage for r in ds_results) / len(ds_results) if ds_results else 0
        ds_full = sum(1 for r in ds_results if r.coverage >= 1.0)
        print(f"  {ds:8s}: {len(ds_results)} queries, "
              f"avg coverage={ds_cov:.1%}, full match={ds_full}/{len(ds_results)}")

    print("=" * 70)

    summary = {
        "total": total,
        "successful": successful,
        "errors": errors,
        "avg_coverage": avg_coverage,
        "full_match": full_match,
        "high_match": high_match,
        "avg_depth": avg_depth,
        "avg_pareto": avg_pareto,
        "avg_time_ms": avg_time,
        "missed_operators": all_missing,
        "extra_operators": all_extra,
    }
    return summary


# ---------------------------------------------------------------------------
# Nirvana ↔ TiTSP Operator Mapping Analysis
# ---------------------------------------------------------------------------

OPERATOR_MAPPING = {
    "filter": {"FILTER", "CONTENT_EXTRACT"},
    "map": {"CONTENT_EXTRACT", "FEATURE_TRANSFORM", "PROJECT"},
    "reduce": {"AGGREGATE"},
    "rank": {"SORT", "LIMIT"},
    "join": {"JOIN", "SIMILARITY_JOIN"},
}


def print_operator_mapping():
    """Print the Nirvana→TiTSP operator mapping table."""
    print("\nNirvana → TiTSP Operator Mapping:")
    print("-" * 50)
    for nv_op, our_ops in OPERATOR_MAPPING.items():
        print(f"  semantic_{nv_op:6s} → {', '.join(sorted(our_ops))}")
    print("-" * 50)
    print("  (SCAN is always added as leaf operator)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Nirvana workload adaptation experiment")
    parser.add_argument("--dataset", choices=["estate", "steam", "imdb"],
                        action="append", help="Run only specific dataset(s)")
    parser.add_argument("--llm", choices=["dashscope", "openai", "bedrock"],
                        help="Use LLM provider for NL parsing and execution")
    parser.add_argument("--model", default="qwen-plus",
                        help="LLM model name (default: qwen-plus)")
    parser.add_argument("--e2e", action="store_true",
                        help="Run full end-to-end: NL → plan → execute → answer")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-query output")
    parser.add_argument("--json", action="store_true",
                        help="Output summary as JSON")
    args = parser.parse_args()

    planner_kwargs: Dict[str, Any] = {}
    if args.llm:
        from core.query_planner.llm_query_analyzer import create_provider
        provider = create_provider(args.llm, model=args.model)
        planner_kwargs["llm_provider"] = provider

    planner = QueryPlanner(**planner_kwargs)
    mode = "LLM: " + args.model if args.llm else "regex fallback"

    print_operator_mapping()

    if args.e2e:
        print(f"\nRunning END-TO-END pipeline on {len(NIRVANA_QUERIES)} queries ({mode})...\n")
        e2e_results = run_e2e_experiment(
            planner=planner,
            datasets=args.dataset,
            verbose=not args.quiet,
        )
        e2e_summary = print_e2e_summary(e2e_results)
        if args.json:
            print("\n" + json.dumps(e2e_summary, indent=2))
    else:
        print(f"\nRunning PLANNING-ONLY on {len(NIRVANA_QUERIES)} queries ({mode})...\n")
        results = run_experiment(
            planner=planner,
            datasets=args.dataset,
            verbose=not args.quiet,
        )
        summary = print_summary(results)
        if args.json:
            print("\n" + json.dumps(summary, indent=2))
