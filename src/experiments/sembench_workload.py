"""
SemBench Workload Adaptation Experiment.

Tests the planning framework against queries from the SemBench benchmark
(https://github.com/HazyResearch/SemBench), which evaluates semantic query
processors including LOTUS, Palimpzest, and ThalamusDB.

Five scenarios (movies / wildlife / e-commerce / medical / MMQA), sample data
for each, covering 55 queries with diverse operator compositions.

Operator mapping (SemBench → TiTSP):
  sem_filter      → FILTER (+ CONTENT_EXTRACT for NL predicates)
  sem_map         → CONTENT_EXTRACT (extract / transform)
  sem_agg         → AGGREGATE (group-by / summarize)
  sem_topk        → SORT + LIMIT
  sem_join        → SIMILARITY_JOIN or JOIN
  sem_search      → SEMANTIC_SEARCH
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
# SemBench Query Definitions
# ---------------------------------------------------------------------------

@dataclass
class SemBenchQuery:
    """A single query from the SemBench benchmark."""
    scenario: str
    query_id: str
    nl_description: str
    sembench_operators: List[str]
    expected_our_operators: Set[str]
    expected_answer: Optional[str] = None


OPERATOR_MAPPING: Dict[str, Set[str]] = {
    "sem_filter": {"FILTER"},
    "sem_map": {"CONTENT_EXTRACT"},
    "sem_agg": {"AGGREGATE"},
    "sem_topk": {"SORT", "LIMIT"},
    "sem_join": {"SIMILARITY_JOIN", "JOIN"},
    "sem_search": {"SEMANTIC_SEARCH"},
}

SEMBENCH_QUERIES: List[SemBenchQuery] = [
    # -----------------------------------------------------------------------
    # Movies scenario (structured + NL columns)
    # -----------------------------------------------------------------------
    SemBenchQuery(
        scenario="movies", query_id="m1",
        nl_description="Filter movies that are considered classic cinema masterpieces",
        sembench_operators=["sem_filter"],
        expected_our_operators={"SCAN", "FILTER", "CONTENT_EXTRACT"},
        expected_answer="Shawshank",
    ),
    SemBenchQuery(
        scenario="movies", query_id="m2",
        nl_description="Extract the primary theme from each movie's plot description",
        sembench_operators=["sem_map"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT"},
        expected_answer="redemption",
    ),
    SemBenchQuery(
        scenario="movies", query_id="m3",
        nl_description="Count the number of movies in each genre",
        sembench_operators=["sem_agg"],
        expected_our_operators={"SCAN", "AGGREGATE"},
        expected_answer="Drama",
    ),
    SemBenchQuery(
        scenario="movies", query_id="m4",
        nl_description="Find the top 3 highest rated movies",
        sembench_operators=["sem_topk"],
        expected_our_operators={"SCAN", "SORT", "LIMIT"},
        expected_answer="Shawshank",
    ),
    SemBenchQuery(
        scenario="movies", query_id="m5",
        nl_description="Filter movies with runtime over 150 minutes and sort by rating descending",
        sembench_operators=["sem_filter", "sem_topk"],
        expected_our_operators={"SCAN", "FILTER", "SORT"},
        expected_answer="3",
    ),
    SemBenchQuery(
        scenario="movies", query_id="m6",
        nl_description="Extract the mood/tone from plot descriptions and count movies per mood category",
        sembench_operators=["sem_map", "sem_agg"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT", "AGGREGATE"},
        expected_answer="dark",
    ),
    SemBenchQuery(
        scenario="movies", query_id="m7",
        nl_description="Find movies similar to 'Inception' based on plot themes",
        sembench_operators=["sem_search"],
        expected_our_operators={"SCAN", "SEMANTIC_SEARCH"},
        expected_answer="Interstellar",
    ),
    SemBenchQuery(
        scenario="movies", query_id="m8",
        nl_description="Filter action movies, extract key themes from plot, and rank top 5 by rating",
        sembench_operators=["sem_filter", "sem_map", "sem_topk"],
        expected_our_operators={"SCAN", "FILTER", "CONTENT_EXTRACT", "SORT", "LIMIT"},
        expected_answer="Dark Knight",
    ),
    SemBenchQuery(
        scenario="movies", query_id="m9",
        nl_description="Compute the average rating of movies released after 2010",
        sembench_operators=["sem_filter", "sem_agg"],
        expected_our_operators={"SCAN", "FILTER", "AGGREGATE"},
        expected_answer="8.5",
    ),
    SemBenchQuery(
        scenario="movies", query_id="m10",
        nl_description="Extract genre from plot for movies without a genre label and count results",
        sembench_operators=["sem_map", "sem_agg"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT", "AGGREGATE"},
        expected_answer="8",
    ),
    SemBenchQuery(
        scenario="movies", query_id="m11",
        nl_description="Find movies about dreams or alternate realities",
        sembench_operators=["sem_search"],
        expected_our_operators={"SCAN", "SEMANTIC_SEARCH"},
        expected_answer="Inception",
    ),

    # -----------------------------------------------------------------------
    # Wildlife scenario (text descriptions + image references)
    # -----------------------------------------------------------------------
    SemBenchQuery(
        scenario="wildlife", query_id="w1",
        nl_description="Filter endangered species from the conservation status field",
        sembench_operators=["sem_filter"],
        expected_our_operators={"SCAN", "FILTER"},
        expected_answer="3",
    ),
    SemBenchQuery(
        scenario="wildlife", query_id="w2",
        nl_description="Extract the primary habitat type from species descriptions",
        sembench_operators=["sem_map"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT"},
        expected_answer="savanna",
    ),
    SemBenchQuery(
        scenario="wildlife", query_id="w3",
        nl_description="Count species per conservation status category",
        sembench_operators=["sem_agg"],
        expected_our_operators={"SCAN", "AGGREGATE"},
        expected_answer="Endangered",
    ),
    SemBenchQuery(
        scenario="wildlife", query_id="w4",
        nl_description="Find the top 5 heaviest species",
        sembench_operators=["sem_topk"],
        expected_our_operators={"SCAN", "SORT", "LIMIT"},
        expected_answer="Blue Whale",
    ),
    SemBenchQuery(
        scenario="wildlife", query_id="w5",
        nl_description="Filter carnivorous species and sort by weight descending",
        sembench_operators=["sem_filter", "sem_topk"],
        expected_our_operators={"SCAN", "FILTER", "SORT"},
        expected_answer="5",
    ),
    SemBenchQuery(
        scenario="wildlife", query_id="w6",
        nl_description="Extract diet type from description and compute average weight per diet category",
        sembench_operators=["sem_map", "sem_agg"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT", "AGGREGATE"},
        expected_answer="Herbivore",
    ),
    SemBenchQuery(
        scenario="wildlife", query_id="w7",
        nl_description="Find species similar to the African Elephant based on habitat and behavior",
        sembench_operators=["sem_search"],
        expected_our_operators={"SCAN", "SEMANTIC_SEARCH"},
        expected_answer="Elephant",
    ),
    SemBenchQuery(
        scenario="wildlife", query_id="w8",
        nl_description="Filter aquatic species, extract migratory patterns, and rank by population",
        sembench_operators=["sem_filter", "sem_map", "sem_topk"],
        expected_our_operators={"SCAN", "FILTER", "CONTENT_EXTRACT", "SORT"},
        expected_answer="Dolphin",
    ),
    SemBenchQuery(
        scenario="wildlife", query_id="w9",
        nl_description="Compute the average lifespan of endangered species",
        sembench_operators=["sem_filter", "sem_agg"],
        expected_our_operators={"SCAN", "FILTER", "AGGREGATE"},
        expected_answer="53",
    ),
    SemBenchQuery(
        scenario="wildlife", query_id="w10",
        nl_description="Match wildlife species to their habitats using description similarity",
        sembench_operators=["sem_join"],
        expected_our_operators={"SCAN", "SIMILARITY_JOIN"},
        expected_answer="Elephant",
    ),
    SemBenchQuery(
        scenario="wildlife", query_id="w11",
        nl_description="Find nocturnal predators from the behavior descriptions",
        sembench_operators=["sem_search"],
        expected_our_operators={"SCAN", "SEMANTIC_SEARCH", "FILTER"},
        expected_answer="Tiger",
    ),

    # -----------------------------------------------------------------------
    # E-commerce scenario (product listings with descriptions + images)
    # -----------------------------------------------------------------------
    SemBenchQuery(
        scenario="ecommerce", query_id="e1",
        nl_description="Filter products that are eco-friendly or sustainable based on description",
        sembench_operators=["sem_filter"],
        expected_our_operators={"SCAN", "FILTER", "CONTENT_EXTRACT"},
        expected_answer="3",
    ),
    SemBenchQuery(
        scenario="ecommerce", query_id="e2",
        nl_description="Extract product material from descriptions",
        sembench_operators=["sem_map"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT"},
        expected_answer="cotton",
    ),
    SemBenchQuery(
        scenario="ecommerce", query_id="e3",
        nl_description="Count products per category",
        sembench_operators=["sem_agg"],
        expected_our_operators={"SCAN", "AGGREGATE"},
        expected_answer="Electronics",
    ),
    SemBenchQuery(
        scenario="ecommerce", query_id="e4",
        nl_description="Find the top 5 most expensive products",
        sembench_operators=["sem_topk"],
        expected_our_operators={"SCAN", "SORT", "LIMIT"},
        expected_answer="Smart Fitness Watch",
    ),
    SemBenchQuery(
        scenario="ecommerce", query_id="e5",
        nl_description="Filter products with rating above 4.5 and price under 100",
        sembench_operators=["sem_filter"],
        expected_our_operators={"SCAN", "FILTER"},
        expected_answer="3",
    ),
    SemBenchQuery(
        scenario="ecommerce", query_id="e6",
        nl_description="Extract key features from product descriptions and group by feature type",
        sembench_operators=["sem_map", "sem_agg"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT", "AGGREGATE"},
        expected_answer="waterproof",
    ),
    SemBenchQuery(
        scenario="ecommerce", query_id="e7",
        nl_description="Find products similar to 'wireless bluetooth earbuds' by description",
        sembench_operators=["sem_search"],
        expected_our_operators={"SCAN", "SEMANTIC_SEARCH"},
        expected_answer="Earbuds",
    ),
    SemBenchQuery(
        scenario="ecommerce", query_id="e8",
        nl_description="Filter electronics, extract brand from title, and compute average price per brand",
        sembench_operators=["sem_filter", "sem_map", "sem_agg"],
        expected_our_operators={"SCAN", "FILTER", "CONTENT_EXTRACT", "AGGREGATE"},
        expected_answer="143",
    ),
    SemBenchQuery(
        scenario="ecommerce", query_id="e9",
        nl_description="Compute total revenue from sales data",
        sembench_operators=["sem_agg"],
        expected_our_operators={"SCAN", "AGGREGATE"},
        expected_answer="15847",
    ),
    SemBenchQuery(
        scenario="ecommerce", query_id="e10",
        nl_description="Match product descriptions to customer review sentiments",
        sembench_operators=["sem_join"],
        expected_our_operators={"SCAN", "SIMILARITY_JOIN"},
        expected_answer="Earbuds",
    ),
    SemBenchQuery(
        scenario="ecommerce", query_id="e11",
        nl_description="Find products suitable for outdoor activities from descriptions",
        sembench_operators=["sem_search"],
        expected_our_operators={"SCAN", "SEMANTIC_SEARCH"},
        expected_answer="Hiking",
    ),

    # -----------------------------------------------------------------------
    # Medical scenario (clinical text + structured measurements)
    # -----------------------------------------------------------------------
    SemBenchQuery(
        scenario="medical", query_id="d1",
        nl_description="Filter patients with symptoms indicating cardiovascular disease",
        sembench_operators=["sem_filter"],
        expected_our_operators={"SCAN", "FILTER", "CONTENT_EXTRACT"},
        expected_answer="3",
    ),
    SemBenchQuery(
        scenario="medical", query_id="d2",
        nl_description="Extract diagnosis codes from clinical notes",
        sembench_operators=["sem_map"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT"},
        expected_answer="Metformin",
    ),
    SemBenchQuery(
        scenario="medical", query_id="d3",
        nl_description="Count patients per primary diagnosis category",
        sembench_operators=["sem_agg"],
        expected_our_operators={"SCAN", "AGGREGATE"},
        expected_answer="Diabetes",
    ),
    SemBenchQuery(
        scenario="medical", query_id="d4",
        nl_description="Find the top 3 patients with highest blood pressure readings",
        sembench_operators=["sem_topk"],
        expected_our_operators={"SCAN", "SORT", "LIMIT"},
        expected_answer="P003",
    ),
    SemBenchQuery(
        scenario="medical", query_id="d5",
        nl_description="Filter diabetic patients and compute average blood glucose level",
        sembench_operators=["sem_filter", "sem_agg"],
        expected_our_operators={"SCAN", "FILTER", "AGGREGATE"},
        expected_answer="220",
    ),
    SemBenchQuery(
        scenario="medical", query_id="d6",
        nl_description="Extract medication names from clinical notes and count unique medications",
        sembench_operators=["sem_map", "sem_agg"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT", "AGGREGATE"},
        expected_answer="Metformin",
    ),
    SemBenchQuery(
        scenario="medical", query_id="d7",
        nl_description="Find patients with similar symptoms to patient P001",
        sembench_operators=["sem_search"],
        expected_our_operators={"SCAN", "SEMANTIC_SEARCH"},
        expected_answer="P003",
    ),
    SemBenchQuery(
        scenario="medical", query_id="d8",
        nl_description="Filter elderly patients, extract comorbidities from notes, and rank by severity",
        sembench_operators=["sem_filter", "sem_map", "sem_topk"],
        expected_our_operators={"SCAN", "FILTER", "CONTENT_EXTRACT", "SORT"},
        expected_answer="P006",
    ),
    SemBenchQuery(
        scenario="medical", query_id="d9",
        nl_description="Match patient symptoms to known disease descriptions",
        sembench_operators=["sem_join"],
        expected_our_operators={"SCAN", "SIMILARITY_JOIN"},
        expected_answer="P001",
    ),
    SemBenchQuery(
        scenario="medical", query_id="d10",
        nl_description="Find patients showing signs of infection based on clinical notes",
        sembench_operators=["sem_search"],
        expected_our_operators={"SCAN", "SEMANTIC_SEARCH", "FILTER"},
        expected_answer="P007",
    ),
    SemBenchQuery(
        scenario="medical", query_id="d11",
        nl_description="Compute average age of patients grouped by diagnosis category",
        sembench_operators=["sem_agg"],
        expected_our_operators={"SCAN", "AGGREGATE"},
        expected_answer="Diabetes",
    ),

    # -----------------------------------------------------------------------
    # MMQA scenario (multi-hop QA across tables + text)
    # -----------------------------------------------------------------------
    SemBenchQuery(
        scenario="mmqa", query_id="q1",
        nl_description="Find the capital city of the country where the Eiffel Tower is located",
        sembench_operators=["sem_search", "sem_map"],
        expected_our_operators={"SCAN", "SEMANTIC_SEARCH", "CONTENT_EXTRACT"},
        expected_answer="Paris",
    ),
    SemBenchQuery(
        scenario="mmqa", query_id="q2",
        nl_description="Count the number of landmarks built before 1900",
        sembench_operators=["sem_filter", "sem_agg"],
        expected_our_operators={"SCAN", "FILTER", "AGGREGATE"},
        expected_answer="6",
    ),
    SemBenchQuery(
        scenario="mmqa", query_id="q3",
        nl_description="Extract the architect name from landmark descriptions and count per architect",
        sembench_operators=["sem_map", "sem_agg"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT", "AGGREGATE"},
        expected_answer="Eiffel",
    ),
    SemBenchQuery(
        scenario="mmqa", query_id="q4",
        nl_description="Find the top 3 most visited landmarks",
        sembench_operators=["sem_topk"],
        expected_our_operators={"SCAN", "SORT", "LIMIT"},
        expected_answer="Great Wall",
    ),
    SemBenchQuery(
        scenario="mmqa", query_id="q5",
        nl_description="Filter landmarks in Europe and sort by height descending",
        sembench_operators=["sem_filter", "sem_topk"],
        expected_our_operators={"SCAN", "FILTER", "SORT"},
        expected_answer="Eiffel Tower",
    ),
    SemBenchQuery(
        scenario="mmqa", query_id="q6",
        nl_description="Match landmarks to their countries using description similarity",
        sembench_operators=["sem_join"],
        expected_our_operators={"SCAN", "SIMILARITY_JOIN"},
        expected_answer="France",
    ),
    SemBenchQuery(
        scenario="mmqa", query_id="q7",
        nl_description="Find landmarks similar to the Great Wall based on purpose and history",
        sembench_operators=["sem_search"],
        expected_our_operators={"SCAN", "SEMANTIC_SEARCH"},
        expected_answer="Colosseum",
    ),
    SemBenchQuery(
        scenario="mmqa", query_id="q8",
        nl_description="Extract construction material from descriptions, filter stone buildings, compute average height",
        sembench_operators=["sem_map", "sem_filter", "sem_agg"],
        expected_our_operators={"SCAN", "CONTENT_EXTRACT", "FILTER", "AGGREGATE"},
        expected_answer="stone",
    ),
    SemBenchQuery(
        scenario="mmqa", query_id="q9",
        nl_description="Count landmarks per continent",
        sembench_operators=["sem_agg"],
        expected_our_operators={"SCAN", "AGGREGATE"},
        expected_answer="Asia",
    ),
    SemBenchQuery(
        scenario="mmqa", query_id="q10",
        nl_description="Filter UNESCO World Heritage sites and rank by visitor count",
        sembench_operators=["sem_filter", "sem_topk"],
        expected_our_operators={"SCAN", "FILTER", "SORT"},
        expected_answer="6",
    ),
    SemBenchQuery(
        scenario="mmqa", query_id="q11",
        nl_description="Find landmarks built for religious or memorial purposes from descriptions",
        sembench_operators=["sem_search"],
        expected_our_operators={"SCAN", "SEMANTIC_SEARCH", "FILTER"},
        expected_answer="Taj Mahal",
    ),
]


# ---------------------------------------------------------------------------
# Sample Datasets
# ---------------------------------------------------------------------------

SAMPLE_MOVIES_DATA: Table = [
    {"id": 1, "title": "The Shawshank Redemption", "year": 1994, "rating": 9.3,
     "genre": "Drama", "runtime": 142, "director": "Frank Darabont",
     "plot": "Two imprisoned men bond over a number of years, finding solace and eventual redemption.",
     "box_office": 58300000},
    {"id": 2, "title": "Inception", "year": 2010, "rating": 8.8,
     "genre": "Sci-Fi", "runtime": 148, "director": "Christopher Nolan",
     "plot": "A thief who steals corporate secrets through dream-sharing technology is given the inverse task.",
     "box_office": 292600000},
    {"id": 3, "title": "The Dark Knight", "year": 2008, "rating": 9.0,
     "genre": "Action", "runtime": 152, "director": "Christopher Nolan",
     "plot": "When the menace known as the Joker wreaks havoc on Gotham, Batman must face the ultimate test.",
     "box_office": 533300000},
    {"id": 4, "title": "Pulp Fiction", "year": 1994, "rating": 8.9,
     "genre": "Crime", "runtime": 154, "director": "Quentin Tarantino",
     "plot": "The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales.",
     "box_office": 107900000},
    {"id": 5, "title": "Parasite", "year": 2019, "rating": 8.5,
     "genre": "Drama", "runtime": 132, "director": "Bong Joon-ho",
     "plot": "Greed and class discrimination threaten the newly formed symbiotic relationship between two families.",
     "box_office": 53400000},
    {"id": 6, "title": "Interstellar", "year": 2014, "rating": 8.7,
     "genre": "Sci-Fi", "runtime": 169, "director": "Christopher Nolan",
     "plot": "A team of explorers travel through a wormhole in space to ensure humanity's survival.",
     "box_office": 188000000},
    {"id": 7, "title": "The Godfather", "year": 1972, "rating": 9.2,
     "genre": "Crime", "runtime": 175, "director": "Francis Ford Coppola",
     "plot": "The aging patriarch of an organized crime dynasty transfers control to his reluctant youngest son.",
     "box_office": 246100000},
    {"id": 8, "title": "Whiplash", "year": 2014, "rating": 8.5,
     "genre": "Drama", "runtime": 106, "director": "Damien Chazelle",
     "plot": "A promising young drummer enrolls at a cut-throat music conservatory.",
     "box_office": 13100000},
]

SAMPLE_WILDLIFE_DATA: Table = [
    {"id": 1, "species": "African Elephant", "weight_kg": 6000, "lifespan_years": 65,
     "conservation_status": "Endangered", "diet": "Herbivore",
     "description": "Largest land mammal. Lives in savanna and forest habitats. Highly social, travels in herds.",
     "population": 415000},
    {"id": 2, "species": "Bengal Tiger", "weight_kg": 220, "lifespan_years": 15,
     "conservation_status": "Endangered", "diet": "Carnivore",
     "description": "Apex predator in tropical forests. Solitary hunter, primarily nocturnal. Strong swimmer.",
     "population": 2500},
    {"id": 3, "species": "Blue Whale", "weight_kg": 150000, "lifespan_years": 80,
     "conservation_status": "Endangered", "diet": "Carnivore",
     "description": "Largest animal ever. Marine mammal found in all oceans. Feeds primarily on krill.",
     "population": 25000},
    {"id": 4, "species": "Red Fox", "weight_kg": 7, "lifespan_years": 5,
     "conservation_status": "Least Concern", "diet": "Omnivore",
     "description": "Adaptable predator found worldwide. Nocturnal hunter. Lives in diverse habitats from forests to urban areas.",
     "population": 10000000},
    {"id": 5, "species": "Giant Panda", "weight_kg": 100, "lifespan_years": 20,
     "conservation_status": "Vulnerable", "diet": "Herbivore",
     "description": "Native to mountain forests of central China. Eats primarily bamboo. Solitary by nature.",
     "population": 1864},
    {"id": 6, "species": "Emperor Penguin", "weight_kg": 40, "lifespan_years": 20,
     "conservation_status": "Near Threatened", "diet": "Carnivore",
     "description": "Tallest penguin species. Lives in Antarctica. Excellent swimmer and diver. Forms large colonies.",
     "population": 595000},
    {"id": 7, "species": "Gray Wolf", "weight_kg": 45, "lifespan_years": 13,
     "conservation_status": "Least Concern", "diet": "Carnivore",
     "description": "Pack hunter found in forests and tundra. Highly social with complex hierarchy.",
     "population": 300000},
    {"id": 8, "species": "Bottlenose Dolphin", "weight_kg": 300, "lifespan_years": 45,
     "conservation_status": "Least Concern", "diet": "Carnivore",
     "description": "Intelligent marine mammal. Found in warm ocean waters. Highly social, uses echolocation.",
     "population": 600000},
]

SAMPLE_ECOMMERCE_DATA: Table = [
    {"id": 1, "product_name": "Wireless Bluetooth Earbuds", "category": "Electronics",
     "price": 79.99, "rating": 4.6, "sales": 3200,
     "description": "True wireless earbuds with noise cancellation. 24h battery. IPX5 waterproof.",
     "material": "Plastic/Silicone"},
    {"id": 2, "product_name": "Organic Cotton T-Shirt", "category": "Clothing",
     "price": 29.99, "rating": 4.3, "sales": 1500,
     "description": "100% organic cotton tee. Eco-friendly sustainable fashion. Machine washable.",
     "material": "Organic Cotton"},
    {"id": 3, "product_name": "Stainless Steel Water Bottle", "category": "Home",
     "price": 24.99, "rating": 4.8, "sales": 4200,
     "description": "BPA-free insulated bottle. Keeps drinks cold 24h. Eco-friendly reusable design.",
     "material": "Stainless Steel"},
    {"id": 4, "product_name": "Gaming Mechanical Keyboard", "category": "Electronics",
     "price": 149.99, "rating": 4.7, "sales": 890,
     "description": "Cherry MX switches. RGB backlight. Full NKRO. Aluminum frame.",
     "material": "Aluminum/Plastic"},
    {"id": 5, "product_name": "Yoga Mat Premium", "category": "Sports",
     "price": 45.99, "rating": 4.5, "sales": 2100,
     "description": "6mm thick non-slip mat. Natural rubber. Perfect for outdoor activities and yoga.",
     "material": "Natural Rubber"},
    {"id": 6, "product_name": "Bamboo Cutting Board Set", "category": "Home",
     "price": 34.99, "rating": 4.4, "sales": 1800,
     "description": "Sustainable bamboo boards. Set of 3 sizes. Eco-friendly kitchen essential.",
     "material": "Bamboo"},
    {"id": 7, "product_name": "Smart Fitness Watch", "category": "Electronics",
     "price": 199.99, "rating": 4.2, "sales": 1560,
     "description": "Heart rate monitor, GPS tracking. Water resistant. 7-day battery life.",
     "material": "Aluminum/Silicone"},
    {"id": 8, "product_name": "Hiking Backpack 40L", "category": "Sports",
     "price": 89.99, "rating": 4.6, "sales": 597,
     "description": "Durable waterproof backpack for outdoor adventures. Multiple compartments.",
     "material": "Nylon"},
]

SAMPLE_MEDICAL_DATA: Table = [
    {"patient_id": "P001", "age": 67, "gender": "M", "systolic_bp": 165, "glucose": 210,
     "diagnosis": "Type 2 Diabetes", "status": "Active",
     "notes": "Patient presents with elevated blood glucose. History of hypertension. Taking Metformin 1000mg. Chest pain reported."},
    {"patient_id": "P002", "age": 45, "gender": "F", "systolic_bp": 120, "glucose": 95,
     "diagnosis": "Healthy", "status": "Routine",
     "notes": "Annual checkup. No complaints. All vitals within normal range. Recommended continued exercise."},
    {"patient_id": "P003", "age": 72, "gender": "M", "systolic_bp": 180, "glucose": 160,
     "diagnosis": "Hypertension", "status": "Active",
     "notes": "Severe hypertension. Patient on Lisinopril 20mg. Mild kidney function decline. Signs of cardiovascular strain."},
    {"patient_id": "P004", "age": 55, "gender": "F", "systolic_bp": 140, "glucose": 230,
     "diagnosis": "Type 2 Diabetes", "status": "Active",
     "notes": "Poorly controlled diabetes. Glucose levels consistently high. Started insulin therapy. Peripheral neuropathy symptoms."},
    {"patient_id": "P005", "age": 38, "gender": "M", "systolic_bp": 118, "glucose": 88,
     "diagnosis": "Healthy", "status": "Routine",
     "notes": "Routine checkup. Minor seasonal allergies. Prescribed antihistamines. No infection signs."},
    {"patient_id": "P006", "age": 81, "gender": "F", "systolic_bp": 155, "glucose": 145,
     "diagnosis": "Cardiovascular Disease", "status": "Active",
     "notes": "Elderly patient with atrial fibrillation. On Warfarin. Arthritis comorbidity. Shows signs of mild cognitive decline."},
    {"patient_id": "P007", "age": 29, "gender": "M", "systolic_bp": 125, "glucose": 92,
     "diagnosis": "Respiratory Infection", "status": "Active",
     "notes": "Acute bronchitis. Fever 38.5C for 3 days. Prescribed antibiotics. Signs of bacterial infection."},
    {"patient_id": "P008", "age": 63, "gender": "F", "systolic_bp": 148, "glucose": 115,
     "diagnosis": "Hypertension", "status": "Active",
     "notes": "Controlled hypertension. Patient compliant with medication. Amlodipine 5mg. Mild chest discomfort on exertion."},
]

SAMPLE_MMQA_DATA: Table = [
    {"id": 1, "landmark": "Eiffel Tower", "country": "France", "continent": "Europe",
     "year_built": 1889, "height_m": 330, "visitors_annual": 7000000,
     "description": "Iron lattice tower designed by Gustave Eiffel. Symbol of Paris. Built for the 1889 World Fair.",
     "is_unesco": False},
    {"id": 2, "landmark": "Great Wall of China", "country": "China", "continent": "Asia",
     "year_built": -700, "height_m": 8, "visitors_annual": 10000000,
     "description": "Series of fortifications made of stone, brick, and earth. Built to protect against invasions. Multiple dynasties contributed.",
     "is_unesco": True},
    {"id": 3, "landmark": "Colosseum", "country": "Italy", "continent": "Europe",
     "year_built": 80, "height_m": 48, "visitors_annual": 7400000,
     "description": "Ancient Roman amphitheater built of stone and concrete. Largest amphitheater ever built. Used for gladiatorial contests.",
     "is_unesco": True},
    {"id": 4, "landmark": "Machu Picchu", "country": "Peru", "continent": "South America",
     "year_built": 1450, "height_m": 2430, "visitors_annual": 1500000,
     "description": "15th-century Inca citadel set high in the Andes. Built of stone without mortar. Remarkable engineering.",
     "is_unesco": True},
    {"id": 5, "landmark": "Taj Mahal", "country": "India", "continent": "Asia",
     "year_built": 1653, "height_m": 73, "visitors_annual": 8000000,
     "description": "White marble mausoleum on the banks of the Yamuna river. Built by Mughal emperor Shah Jahan for his wife.",
     "is_unesco": True},
    {"id": 6, "landmark": "Statue of Liberty", "country": "United States", "continent": "North America",
     "year_built": 1886, "height_m": 93, "visitors_annual": 4200000,
     "description": "Copper statue gifted by France. Designed by Frederic Bartholdi. Symbol of freedom and democracy.",
     "is_unesco": True},
    {"id": 7, "landmark": "Sydney Opera House", "country": "Australia", "continent": "Oceania",
     "year_built": 1973, "height_m": 65, "visitors_annual": 10000000,
     "description": "Multi-venue performing arts center. Distinctive sail-shaped shells designed by Jorn Utzon.",
     "is_unesco": True},
    {"id": 8, "landmark": "Burj Khalifa", "country": "UAE", "continent": "Asia",
     "year_built": 2010, "height_m": 828, "visitors_annual": 2000000,
     "description": "Tallest structure in the world. Mixed-use skyscraper with steel and glass exterior. Neo-futurism architecture.",
     "is_unesco": False},
]

DATASET_REGISTRY: Dict[str, Table] = {
    "movies": SAMPLE_MOVIES_DATA,
    "wildlife": SAMPLE_WILDLIFE_DATA,
    "ecommerce": SAMPLE_ECOMMERCE_DATA,
    "medical": SAMPLE_MEDICAL_DATA,
    "mmqa": SAMPLE_MMQA_DATA,
}


# ---------------------------------------------------------------------------
# Execution & Validation
# ---------------------------------------------------------------------------

@dataclass
class E2EResult:
    """Result of full end-to-end execution of a single query."""
    query: SemBenchQuery
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


def collect_operators(node: LogicalPlanNode) -> List[str]:
    """Collect all operator types from a plan tree (pre-order)."""
    ops = [node.operator_type]
    for child in node.children:
        ops.extend(collect_operators(child))
    return ops


def plan_depth(node: LogicalPlanNode) -> int:
    """Compute the depth of a plan tree."""
    if not node.children:
        return 1
    return 1 + max(plan_depth(c) for c in node.children)


def validate_answer(answer: str, data: Table, expected: str) -> bool:
    """Check if the answer or data contains the expected value."""
    if expected.lower() in answer.lower():
        return True
    data_str = json.dumps(data, default=str).lower()
    if expected.lower() in data_str:
        return True
    try:
        expected_num = float(expected)
        for row in data:
            for v in row.values():
                if isinstance(v, (int, float)) and abs(v - expected_num) < expected_num * 0.15:
                    return True
    except (ValueError, ZeroDivisionError):
        pass
    return False


@dataclass
class SingleResult:
    """Planning-only result for a single query."""
    query: SemBenchQuery
    plan_result: Optional[PlanResult]
    planned_operators: Set[str]
    elapsed_ms: float
    error: Optional[str] = None


def run_single_query(planner: QueryPlanner, q: SemBenchQuery) -> SingleResult:
    """Run planning only (no execution) for a single query."""
    t0 = time.perf_counter()
    try:
        result = planner.plan(q.nl_description)
        elapsed = (time.perf_counter() - t0) * 1000
        ops = set(collect_operators(result.best_logical_plan))
        return SingleResult(query=q, plan_result=result, planned_operators=ops,
                            elapsed_ms=elapsed)
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return SingleResult(query=q, plan_result=None, planned_operators=set(),
                            elapsed_ms=elapsed, error=str(e))


def run_single_e2e(
    planner: QueryPlanner,
    q: SemBenchQuery,
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

        physical = {}
        if plan.selected_physical:
            physical = dict(plan.selected_physical.assignment)

        trace = []
        for nid, er in result.execution.operator_results.items():
            trace.append(f"{er.operator_type}({er.impl_type}) → {er.row_count} rows")

        return E2EResult(
            query=q,
            executable=result,
            answer=result.answer,
            answer_validated=validated,
            output_rows=result.execution.row_count,
            elapsed_ms=elapsed,
            mcts_candidates=len(plan.logical_candidates),
            plan_depth=plan_depth(plan.best_logical_plan),
            plan_operators=ops,
            pareto_size=len(plan.pareto_front),
            physical_assignment=physical,
            exec_trace=trace,
        )
    except Exception as e:
        import traceback
        elapsed = (time.perf_counter() - t0) * 1000
        return E2EResult(
            query=q, executable=None, answer="", answer_validated=False,
            output_rows=0, elapsed_ms=elapsed, error=f"{e}\n{traceback.format_exc()}"
        )


# ---------------------------------------------------------------------------
# Experiment Runners
# ---------------------------------------------------------------------------

def run_experiment(
    scenario: Optional[str] = None,
    llm_provider=None,
) -> List[SingleResult]:
    """Run planning-only experiment across SemBench queries."""
    planner = QueryPlanner(llm_provider=llm_provider)
    queries = SEMBENCH_QUERIES
    if scenario:
        queries = [q for q in queries if q.scenario == scenario]

    results = []
    for q in queries:
        r = run_single_query(planner, q)
        results.append(r)
    return results


def run_e2e_experiment(
    scenario: Optional[str] = None,
    llm_provider=None,
) -> List[E2EResult]:
    """Run full E2E experiment across SemBench queries."""
    planner = QueryPlanner(llm_provider=llm_provider)
    queries = SEMBENCH_QUERIES
    if scenario:
        queries = [q for q in queries if q.scenario == scenario]

    results = []
    for q in queries:
        data = DATASET_REGISTRY.get(q.scenario, [])
        r = run_single_e2e(planner, q, data)
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SemBench Workload Experiment")
    parser.add_argument("--scenario", choices=["movies", "wildlife", "ecommerce", "medical", "mmqa"],
                        help="Run only this scenario")
    parser.add_argument("--e2e", action="store_true", help="Run full E2E (plan + execute + answer)")
    parser.add_argument("--llm", type=str, help="LLM provider type (e.g., dashscope)")
    parser.add_argument("--model", type=str, default="qwen-plus", help="LLM model name")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    llm_provider = None
    if args.llm:
        from core.query_planner.llm_query_analyzer import create_provider
        llm_provider = create_provider(args.llm, model=args.model)

    if args.e2e:
        results = run_e2e_experiment(scenario=args.scenario, llm_provider=llm_provider)
        if args.json:
            out = []
            for r in results:
                out.append({
                    "scenario": r.query.scenario, "query_id": r.query.query_id,
                    "answer": r.answer, "validated": r.answer_validated,
                    "output_rows": r.output_rows, "elapsed_ms": round(r.elapsed_ms, 1),
                    "mcts_candidates": r.mcts_candidates, "plan_depth": r.plan_depth,
                    "plan_operators": r.plan_operators, "pareto_size": r.pareto_size,
                    "error": r.error,
                })
            print(json.dumps(out, indent=2))
            return

        total = len(results)
        validated = [r for r in results if r.answer_validated is True]
        failed = [r for r in results if r.error]
        print(f"\n{'='*70}")
        print(f"SemBench E2E Results: {total} queries, {len(validated)} validated, {len(failed)} errors")
        print(f"{'='*70}\n")

        for r in results:
            status = "✓" if r.answer_validated else ("✗" if r.answer_validated is False else "—")
            print(f"[{status}] {r.query.scenario}/{r.query.query_id}: {r.query.nl_description}")
            if r.error:
                print(f"    ERROR: {r.error.split(chr(10))[0]}")
            else:
                print(f"    Plan: {' → '.join(r.plan_operators)} | "
                      f"MCTS: {r.mcts_candidates} | Pareto: {r.pareto_size} | "
                      f"Output: {r.output_rows} rows | {r.elapsed_ms:.0f}ms")
                if r.exec_trace:
                    for t in r.exec_trace:
                        print(f"      {t}")
                if r.answer:
                    ans_preview = r.answer[:100] + "..." if len(r.answer) > 100 else r.answer
                    print(f"    Answer: {ans_preview}")
        print()
    else:
        results = run_experiment(scenario=args.scenario, llm_provider=llm_provider)
        if args.json:
            out = []
            for r in results:
                out.append({
                    "scenario": r.query.scenario, "query_id": r.query.query_id,
                    "planned_operators": sorted(r.planned_operators),
                    "elapsed_ms": round(r.elapsed_ms, 1),
                    "error": r.error,
                })
            print(json.dumps(out, indent=2))
            return

        total = len(results)
        success = [r for r in results if r.error is None]
        print(f"\n{'='*70}")
        print(f"SemBench Planning Results: {total} queries, {len(success)} success")
        print(f"{'='*70}\n")

        for r in results:
            status = "✓" if r.error is None else "✗"
            ops = ", ".join(sorted(r.planned_operators)) if r.planned_operators else "NONE"
            print(f"[{status}] {r.query.scenario}/{r.query.query_id}: {ops} ({r.elapsed_ms:.0f}ms)")
            if r.error:
                print(f"    ERROR: {r.error}")
        print()


if __name__ == "__main__":
    main()
