import sys

import pytest
from pydantic import ValidationError

sys.path.insert(0, "./")
from retrieval.search_query import SearchQuery


def test_valid_string_query():
    model = SearchQuery(query="What is GPT-4?", num_blocks=1)
    assert model.query == ["What is GPT-4?"]


def test_valid_list_query():
    model = SearchQuery(query=["What is GPT-4?", "What is LLaMA-3?"], num_blocks=1)
    assert model.query == ["What is GPT-4?", "What is LLaMA-3?"]


def test_valid_long_query():
    model = SearchQuery(query="What is GPT-4?" * 10, num_blocks=1)
    assert model.query == ["What is GPT-4?" * 10]


def test_empty_string_query():
    with pytest.raises(ValidationError):
        SearchQuery(query="", num_blocks=1)


def test_empty_list_query():
    with pytest.raises(ValidationError):
        SearchQuery(query=[], num_blocks=1)


def test_list_with_more_than_100_items():
    with pytest.raises(ValidationError):
        SearchQuery(query=["query"] * 101, num_blocks=1)


def test_list_with_exactly_100_items():
    model = SearchQuery(query=["query"] * 100, num_blocks=1)
    assert len(model.query) == 100


def test_missing_query():
    with pytest.raises(ValidationError):
        SearchQuery(num_blocks=1)
