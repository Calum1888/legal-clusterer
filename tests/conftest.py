"""
Shared fixtures for the test suite.

ASSUMPTION centralised here: a corpus is a dict mapping document id (str) ->
text (str). If your ingestion/pipeline uses a list or document objects instead,
change `corpus` below and every test follows automatically.
"""
import pytest


@pytest.fixture
def corpus():
    """A small synthetic corpus with two obvious themes (no model needed)."""
    return {
        "cat_1": "cats are small domestic felines that purr and chase mice",
        "cat_2": "the kitten played with a ball of yarn all afternoon",
        "cat_3": "feline companions enjoy napping in warm sunny spots",
        "fin_1": "the quarterly revenue report shows strong profit growth",
        "fin_2": "investors reviewed the balance sheet and cash flow statement",
        "fin_3": "the fiscal budget forecasts higher earnings next year",
    }
