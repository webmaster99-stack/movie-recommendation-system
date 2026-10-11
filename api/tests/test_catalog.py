import numpy as np
import pytest

from recsys_api.bundle import Bundle
from recsys_api.catalog import Catalog, display_title, normalise


@pytest.mark.parametrize(
    ("stored", "shown"),
    [
        ("Matrix, The", "The Matrix"),
        ("Bug's Life, A", "A Bug's Life"),
        ("Toy Story", "Toy Story"),
        ("Good, the Bad and the Ugly, The", "The Good, the Bad and the Ugly"),
        (
            "City of Lost Children, The (Cité des enfants perdus, La)",
            "The City of Lost Children (Cité des enfants perdus, La)",
        ),
        ("Monsters, Inc.", "Monsters, Inc."),
    ],
)
def test_display_title_moves_a_trailing_article_to_the_front(stored: str, shown: str) -> None:
    assert display_title(stored) == shown


def test_normalise_ignores_case_accents_and_punctuation() -> None:
    assert normalise("  Amélie: The   Sequel! ") == "amelie the sequel"
    assert normalise("WALL·E") == "wall e"
    assert normalise("?!") == ""


def test_movies_are_found_by_movielens_id(bundle: Bundle) -> None:
    movie = bundle.catalog.get(60)
    assert movie is not None
    assert (movie.item_idx, movie.title, movie.year, movie.genres) == (5, "Untitled", None, ())
    assert bundle.catalog.get(5) is None  # an item_idx is not a movie id


def test_search_puts_titles_starting_with_the_query_first(bundle: Bundle) -> None:
    titles = [m.title for m in bundle.catalog.search("the matrix", limit=10)]
    assert titles == ["The Matrix", "The Matrix Reloaded"]  # by popularity within the group
    # "t" starts both Matrix titles and "Toy Story"; the rest only contain it, and follow
    # in popularity order.
    assert [m.movie_id for m in bundle.catalog.search("t", limit=10)] == [10, 40, 20, 30, 50, 60]
    assert [m.movie_id for m in bundle.catalog.search("amélie", limit=10)] == [30]
    assert [m.movie_id for m in bundle.catalog.search("reloaded matrix", limit=10)] == [20]
    assert bundle.catalog.search("t", limit=2) == bundle.catalog.search("t", limit=10)[:2]
    assert bundle.catalog.search("zzz", limit=10) == []
    assert bundle.catalog.search("!!", limit=10) == []


def test_onboarding_takes_turns_between_genres(bundle: Bundle) -> None:
    # Genres in alphabetical order, each giving its most popular movie not yet listed:
    # Action -> The Matrix (0), Animation -> Toy Story (3), Children -> (taken),
    # Comedy -> Amelie (2), Crime -> Heat (4), Romance -> (taken), Sci-Fi -> Reloaded (1).
    # The movie without genres comes last.
    assert bundle.catalog.onboarding == [0, 3, 2, 4, 1, 5]


def test_catalog_rejects_records_out_of_order() -> None:
    record = {"movie_id": 1, "title": "A", "year": None, "genres": [], "tmdb_id": None}
    records = [record | {"item_idx": 1, "imdb_id": None}]
    with pytest.raises(ValueError, match="item_idx order"):
        Catalog(records, np.zeros(1))
