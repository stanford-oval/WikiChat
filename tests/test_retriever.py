import sys

import pytest
from pydantic import ValidationError

sys.path.insert(0, "./")
from retrieval.retriever_server import QueryData


def test_valid_string_query():
    model = QueryData(query="What is GPT-4?", num_blocks=1)
    assert model.query == "What is GPT-4?"


def test_valid_list_query():
    model = QueryData(query=["What is GPT-4?", "What is LLaMA-3?"], num_blocks=1)
    assert model.query == ["What is GPT-4?", "What is LLaMA-3?"]


def test_valid_long_query():
    model = QueryData(query="What is GPT-4?" * 500, num_blocks=1)
    assert model.query == "What is GPT-4?" * 500


def test_empty_string_query():
    with pytest.raises(ValidationError):
        QueryData(query="", num_blocks=1)


def test_empty_list_query():
    with pytest.raises(ValidationError):
        QueryData(query=[], num_blocks=1)


def test_list_with_more_than_100_items():
    with pytest.raises(ValidationError):
        QueryData(query=["query"] * 101, num_blocks=1)


def test_list_with_exactly_100_items():
    model = QueryData(query=["query"] * 100, num_blocks=1)
    assert len(model.query) == 100


def test_missing_query():
    with pytest.raises(ValidationError):
        QueryData(num_blocks=1)


if __name__ == "__main__":
    pytest.main()
