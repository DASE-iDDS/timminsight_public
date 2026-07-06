"""TiQC Ambiguous-Query Workload (DERIVED FROM REAL BENCHMARK QUERIES).

Evaluation set for the TiQC interactive query-clarification stage (paper §IV).

Construction principle (decided 2026-06-11, corrected 2026-06-12):
NOTHING is invented. Each item is derived from an ACTUAL Nirvana benchmark
query (data/nirvana_repo/nirvana-main/experiments/workloads/{estate,imdb,steam}/qN.py)
by DELETING the single discriminating qualifier (an exact column, threshold,
value, or modality source). The original benchmark query is kept verbatim as
the gold/disambiguated intent, and the deleted qualifier becomes the hidden
ground-truth intent that drives the LLM user-simulator (AskUser, Algorithm 2).
So both the ambiguity and its resolution come from the benchmark, not from us.

Six ambiguity types (paper §IV): cross_modal_referential, intra_modal_attribute,
multimodal_intent, schema_structure, value_content, temporal_spatial.

Fields per item:
  source            : the benchmark query it is derived from, e.g. "nirvana/estate/q1"
  original_query    : the real benchmark instruction, VERBATIM (this is the gold intent)
  ambiguous_query   : original with the discriminating qualifier removed
  hidden_intent     : the removed qualifier (conditions the user-simulator)
  clarify_dimension : what TiQC should ask about
  options           : grounded interpretations the dataset supports
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class AmbiguousQuery:
    id: str
    dataset: str                 # estate | movie(imdb) | steam
    source: str                  # benchmark query of origin
    ambiguity_type: str
    original_query: str          # REAL benchmark instruction, verbatim = gold intent
    ambiguous_query: str         # original minus the discriminating qualifier
    hidden_intent: str           # the removed qualifier
    clarify_dimension: str
    options: List[str]
    tags: List[str] = field(default_factory=list)


AMBIGUOUS_QUERIES: List[AmbiguousQuery] = [
    # ---- 1. cross_modal_referential (image vs text source) ------------------
    AmbiguousQuery(
        id="estate_q1", dataset="estate", source="nirvana/estate/q1",
        ambiguity_type="cross_modal_referential",
        original_query="Observed from the house picture, whether the house has a yard or not.",
        ambiguous_query="Whether the house has a yard or not.",
        hidden_intent="judge the yard from the house picture (input_columns=['image']), not the Details text.",
        clarify_dimension="Is the yard decided from the house image or from the textual Details?",
        options=["from the house image", "from the Details text"], tags=["image"]),
    AmbiguousQuery(
        id="imdb_q3", dataset="movie", source="nirvana/imdb/q3",
        ambiguity_type="cross_modal_referential",
        original_query="Whether the movie poster image is in the dark style.",
        ambiguous_query="Whether the movie is in the dark style.",
        hidden_intent="judge 'dark style' from the Poster image (input_columns=['Poster']), not the Plot text.",
        clarify_dimension="Decide 'dark style' from the poster image or from the plot/genre text?",
        options=["from the poster image", "from the plot/genre text"], tags=["image"]),

    # ---- 2. intra_modal_attribute (multiple bindings in one modality) -------
    AmbiguousQuery(
        id="imdb_q5", dataset="movie", source="nirvana/imdb/q5",
        ambiguity_type="intra_modal_attribute",
        original_query="The IMDB rating is higher than 9.",
        ambiguous_query="The movie is highly rated.",
        hidden_intent="'rating' is the IMDB_rating column, value > 9 (not Rotten Tomatoes or Metascore).",
        clarify_dimension="Which rating column: IMDB_rating, Rotten Tomatoes, or Metascore?",
        options=["IMDB_rating > 9", "Rotten Tomatoes >= 90%", "Metascore >= 90"], tags=["table"]),
    AmbiguousQuery(
        id="steam_q8", dataset="steam", source="nirvana/steam/q8",
        ambiguity_type="intra_modal_attribute",
        original_query="The rating is higher than 90.",
        ambiguous_query="The game is highly rated.",
        hidden_intent="'rating' is the metacritic score (metacriticts) > 90, not review sentiment.",
        clarify_dimension="Which score: metacritic (metacriticts), or overall_reviews sentiment?",
        options=["metacriticts > 90", "overall_reviews == Very Positive"], tags=["table"]),
    AmbiguousQuery(
        id="steam_q2", dataset="steam", source="nirvana/steam/q2",
        ambiguity_type="intra_modal_attribute",
        original_query="Give the video game a binary review (positive or negative) based on the existing review.",
        ambiguous_query="Give the video game a binary review based on its reviews.",
        hidden_intent="use the overall_reviews column (not recent_reviews).",
        clarify_dimension="Base the label on overall_reviews or recent_reviews?",
        options=["overall_reviews", "recent_reviews"], tags=["table"]),

    # ---- 3. multimodal_intent (analytical goal admits multiple readings) ----
    AmbiguousQuery(
        id="estate_q12", dataset="estate", source="nirvana/estate/q12",
        ambiguity_type="multimodal_intent",
        original_query="Compute the average price for the estates with a gym and a swimming pool and located in Lekki, Lagos.",
        ambiguous_query="Compute the average price for the premium estates.",
        hidden_intent="'premium' means has a gym AND a swimming pool AND located in Lekki, Lagos.",
        clarify_dimension="What defines 'premium': which amenities and which location?",
        options=["gym + pool + Lekki location", "pool only", "any high-priced estate"], tags=["table"]),
    AmbiguousQuery(
        id="estate_q6", dataset="estate", source="nirvana/estate/q6",
        ambiguity_type="multimodal_intent",
        original_query="Compute the average price for the estates that seem to have a yard.",
        ambiguous_query="Compute the average price for the nicer estates.",
        hidden_intent="'nicer' means the estate seems to have a yard (judged from the image).",
        clarify_dimension="What makes an estate 'nicer': has a yard (image), a pool, or more bedrooms?",
        options=["has a yard (image)", "has a pool (image)", "more bedrooms"], tags=["image", "table"]),

    # ---- 4. schema_structure (term matches several columns) -----------------
    AmbiguousQuery(
        id="steam_q1", dataset="steam", source="nirvana/steam/q1",
        ambiguity_type="schema_structure",
        original_query="According to the given PEGI rating (in picture), check if the game is only suitable for adults (18 years or older).",
        ambiguous_query="Check if the game is rated for adults.",
        hidden_intent="'rated' is the PEGI age rating column ('rating'), 18+; not metacritic or reviews.",
        clarify_dimension="'rating' could be PEGI age rating, metacritic score, or review sentiment.",
        options=["PEGI age rating ('rating')", "metacriticts", "overall_reviews"], tags=["table", "column-collision"]),
    AmbiguousQuery(
        id="estate_q2", dataset="estate", source="nirvana/estate/q2",
        ambiguity_type="schema_structure",
        original_query="Extract the house price from the detail about the estate.",
        ambiguous_query="Extract the house price.",
        hidden_intent="the price must be extracted from the Details text column (semantic_map on Details).",
        clarify_dimension="Where does the price come from: the Details text, the Title, or a price column?",
        options=["from Details text", "from Title", "a dedicated price column (does not exist)"], tags=["table"]),

    # ---- 5. value_content (literal matches multiple values) -----------------
    AmbiguousQuery(
        id="estate_q3", dataset="estate", source="nirvana/estate/q3",
        ambiguity_type="value_content",
        original_query="Whether the house is located in Ajah, Lagos.",
        ambiguous_query="Whether the house is located in Ajah.",
        hidden_intent="the location is specifically 'Ajah, Lagos'.",
        clarify_dimension="Which Ajah: 'Ajah, Lagos' specifically, or any location containing 'Ajah'?",
        options=["Ajah, Lagos", "any Location containing 'Ajah'"], tags=["value-match"]),
    AmbiguousQuery(
        id="imdb_q9", dataset="movie", source="nirvana/imdb/q9",
        ambiguity_type="value_content",
        original_query="The movie belongs to crime movies.",
        ambiguous_query="The movie is a crime-related movie.",
        hidden_intent="genre (extracted from Plot) equals exactly 'Crime'.",
        clarify_dimension="Match genre exactly 'Crime', or any crime-related genre (thriller, mystery)?",
        options=["genre == Crime", "any crime-related genre"], tags=["value-match"]),

    # ---- 6. temporal_spatial (unspecified time/space scope) -----------------
    AmbiguousQuery(
        id="steam_q11", dataset="steam", source="nirvana/steam/q11",
        ambiguity_type="temporal_spatial",
        original_query="The game supports VR. Find the earliest release date.",
        ambiguous_query="Find the release date of an early VR game.",
        hidden_intent="'early' means the single EARLIEST release_date among VR-supporting games (a min-reduce).",
        clarify_dimension="Does 'early' mean the single earliest date, or all games before some year?",
        options=["the earliest (min) release_date", "all released before a given year"], tags=["temporal"]),
    AmbiguousQuery(
        id="estate_q12s", dataset="estate", source="nirvana/estate/q12",
        ambiguity_type="temporal_spatial",
        original_query="...located in Lekki, Lagos.",
        ambiguous_query="...located in the Lagos area.",
        hidden_intent="the spatial scope is specifically 'Lekki, Lagos', not all of Lagos.",
        clarify_dimension="What spatial scope: Lekki specifically, or anywhere in Lagos?",
        options=["Lekki, Lagos", "anywhere in Lagos"], tags=["spatial"]),

    # ========================= EXPANDED SET (2026-06-12) =====================
    # More derivations from the real 36 Nirvana queries, for statistical power.
    # ---- cross_modal_referential --------------------------------------------
    AmbiguousQuery(
        id="steam_q4", dataset="steam", source="nirvana/steam/q4",
        ambiguity_type="cross_modal_referential",
        original_query="According to the cover image of the video game, summarize its graphic style.",
        ambiguous_query="Summarize the video game's graphic style.",
        hidden_intent="summarize the graphic style from the cover IMAGE (input_columns=['image']), not the text description.",
        clarify_dimension="Read the graphic style from the cover image or from the text description?",
        options=["from the cover image", "from the text description"], tags=["image"]),
    AmbiguousQuery(
        id="estate_q1b", dataset="estate", source="nirvana/estate/q11",
        ambiguity_type="cross_modal_referential",
        original_query="Is there a swimming pool in the estate (from amenities extracted from Details).",
        ambiguous_query="Whether the estate has a swimming pool.",
        hidden_intent="the pool is determined from the Details text amenities (not from the photo).",
        clarify_dimension="Decide the pool from the Details text or from the house image?",
        options=["from Details text amenities", "from the house image"], tags=["text"]),

    # ---- intra_modal_attribute ----------------------------------------------
    AmbiguousQuery(
        id="imdb_q7", dataset="movie", source="nirvana/imdb/q7",
        ambiguity_type="intra_modal_attribute",
        original_query="Find the highest IMDB_rating movie directed by Steven Spielberg.",
        ambiguous_query="Find the best-rated Spielberg movie.",
        hidden_intent="'best-rated' is the maximum IMDB_rating (not Rotten Tomatoes or Metascore).",
        clarify_dimension="Rank by IMDB_rating, Rotten Tomatoes, or Metascore?",
        options=["max IMDB_rating", "max Rotten Tomatoes", "max Metascore"], tags=["table"]),
    AmbiguousQuery(
        id="steam_q2b", dataset="steam", source="nirvana/steam/q10",
        ambiguity_type="intra_modal_attribute",
        original_query="The game receives a positive overall review.",
        ambiguous_query="The game is positively received.",
        hidden_intent="use overall_reviews being positive (not recent_reviews, not metacritic).",
        clarify_dimension="By overall_reviews, recent_reviews, or metacritic?",
        options=["overall_reviews positive", "recent_reviews positive", "metacriticts high"], tags=["table"]),

    # ---- multimodal_intent --------------------------------------------------
    AmbiguousQuery(
        id="estate_q11", dataset="estate", source="nirvana/estate/q11",
        ambiguity_type="multimodal_intent",
        original_query="Compute the lowest price for the estates that has a swimming pool.",
        ambiguous_query="Compute the lowest price for the luxury estates.",
        hidden_intent="'luxury' means the estate has a swimming pool.",
        clarify_dimension="What defines 'luxury': a swimming pool, a gym, or a high price?",
        options=["has a swimming pool", "has a gym", "above a price threshold"], tags=["table"]),
    AmbiguousQuery(
        id="imdb_q8", dataset="movie", source="nirvana/imdb/q8",
        ambiguity_type="multimodal_intent",
        original_query="Count movies that won 2 Oscars with IMDB rating higher than 9.",
        ambiguous_query="Count the acclaimed movies.",
        hidden_intent="'acclaimed' means won 2 Oscars AND IMDB_rating > 9.",
        clarify_dimension="What makes a movie 'acclaimed': Oscar wins, high rating, or both?",
        options=["2 Oscars AND IMDB>9", "high rating only", "any Oscar win"], tags=["table"]),
    AmbiguousQuery(
        id="estate_q10", dataset="estate", source="nirvana/estate/q10",
        ambiguity_type="multimodal_intent",
        original_query="Lowest price detached duplex with more than 3 and less than 6 bedrooms.",
        ambiguous_query="The lowest price for a good family duplex.",
        hidden_intent="'good family duplex' means a detached duplex with 4 or 5 bedrooms (>3 and <6).",
        clarify_dimension="Which constraints define it: type, bedroom range, or both?",
        options=["detached duplex + 4-5 bedrooms", "any duplex", "4-5 bedrooms only"], tags=["table"]),

    # ---- schema_structure ---------------------------------------------------
    AmbiguousQuery(
        id="estate_q4", dataset="estate", source="nirvana/estate/q4",
        ambiguity_type="schema_structure",
        original_query="Extract Amenities of the estate from the estate details.",
        ambiguous_query="Extract the amenities of the estate.",
        hidden_intent="amenities are extracted from the Details text (there is no Amenities column).",
        clarify_dimension="Extract amenities from the Details text, the Title, or a column?",
        options=["from Details text", "from Title", "from an Amenities column (none exists)"], tags=["table"]),
    AmbiguousQuery(
        id="imdb_q1", dataset="movie", source="nirvana/imdb/q1",
        ambiguity_type="schema_structure",
        original_query="According to the movie plot, extract the genre(s) of each movie.",
        ambiguous_query="Get the genre of each movie.",
        hidden_intent="extract genre by reading the Plot text (the semantic_map), not the Genre1-3 columns.",
        clarify_dimension="Read genre from the Plot text or from the Genre1/2/3 columns?",
        options=["infer from Plot text", "read Genre1-3 columns"], tags=["table"]),

    # ---- value_content ------------------------------------------------------
    AmbiguousQuery(
        id="imdb_q2", dataset="movie", source="nirvana/imdb/q2",
        ambiguity_type="value_content",
        original_query="The movie is directed by Christopher Nolan.",
        ambiguous_query="The movie is directed by Nolan.",
        hidden_intent="director is specifically 'Christopher Nolan'.",
        clarify_dimension="Which Nolan: Christopher Nolan, or any director named Nolan?",
        options=["Christopher Nolan", "any 'Nolan'"], tags=["value-match"]),
    AmbiguousQuery(
        id="steam_q3", dataset="steam", source="nirvana/steam/q3",
        ambiguity_type="value_content",
        original_query="Does the video game support VR (in platforms)?",
        ambiguous_query="Is the game a VR game?",
        hidden_intent="VR support is read from the platforms column listing VR.",
        clarify_dimension="VR judged from the platforms column, the tags, or the genre?",
        options=["platforms lists VR", "tags include VR", "genre is VR"], tags=["value-match"]),
    AmbiguousQuery(
        id="imdb_q4", dataset="movie", source="nirvana/imdb/q4",
        ambiguity_type="value_content",
        original_query="The movie has ever won more than 3 Oscars.",
        ambiguous_query="The movie has won several Oscars.",
        hidden_intent="'several' means strictly more than 3 Oscars.",
        clarify_dimension="What count is 'several': more than 3, at least 2, or any?",
        options=["more than 3", "at least 2", "at least 1"], tags=["threshold"]),
    AmbiguousQuery(
        id="estate_q5", dataset="estate", source="nirvana/estate/q5",
        ambiguity_type="value_content",
        original_query="The estate has more than 3 bedrooms but less than 6 bedrooms.",
        ambiguous_query="The estate is spacious.",
        hidden_intent="'spacious' means more than 3 and less than 6 bedrooms (i.e. 4 or 5).",
        clarify_dimension="What bedroom range counts as 'spacious'?",
        options=["4-5 bedrooms (>3 and <6)", ">=4 bedrooms", ">=6 bedrooms"], tags=["threshold"]),

    # ---- temporal_spatial ---------------------------------------------------
    AmbiguousQuery(
        id="steam_q12", dataset="steam", source="nirvana/steam/q12",
        ambiguity_type="temporal_spatial",
        original_query="Find the earliest release date of an adventure game in cartoon graphic style.",
        ambiguous_query="Find an old cartoon adventure game.",
        hidden_intent="'old' means the single EARLIEST release_date among cartoon-style adventure games.",
        clarify_dimension="Does 'old' mean the earliest one, or any released before some year?",
        options=["the earliest (min) date", "released before a given year"], tags=["temporal"]),
    AmbiguousQuery(
        id="estate_q3b", dataset="estate", source="nirvana/estate/q3",
        ambiguity_type="temporal_spatial",
        original_query="Whether the house is located in Ajah, Lagos.",
        ambiguous_query="Whether the house is in the eastern Lagos area.",
        hidden_intent="the spatial scope is specifically 'Ajah, Lagos'.",
        clarify_dimension="Which area exactly: Ajah specifically, or the broader eastern Lagos?",
        options=["Ajah, Lagos", "broader eastern Lagos (Ajah/Lekki/Ikoyi)"], tags=["spatial"]),
]


def by_type():
    out = {}
    for q in AMBIGUOUS_QUERIES:
        out.setdefault(q.ambiguity_type, []).append(q)
    return out


if __name__ == "__main__":
    groups = by_type()
    print(f"TiQC ambiguous-query set (derived from real Nirvana queries): "
          f"{len(AMBIGUOUS_QUERIES)} queries across {len(groups)} types")
    for t, qs in sorted(groups.items()):
        print(f"  {t:24s}: {len(qs)}  sources=[{', '.join(q.source for q in qs)}]")
