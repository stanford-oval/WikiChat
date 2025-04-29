import sys
from datetime import datetime

import aiohttp
import pytest

sys.path.insert(0, "./")
from preprocessing.block import Block, BlockLanguage, BlockType
from preprocessing.utils import (
    extract_english_translations,
    find_forest_roots_and_members,
    get_wikidata_english_name,
)


# Test extracting single translation
def test_single_translation():
    text = "This is a sample text. (in English: This is a sample text in English.)"
    expected = ["(in English: This is a sample text in English.)"]
    assert extract_english_translations(text) == expected


# Test extracting multiple translations
def test_multiple_translations():
    text = "Texte en français. (in English: Text.) Más texto en español. (in English: More text.)"
    expected = ["(in English: Text.)", "(in English: More text.)"]
    assert extract_english_translations(text) == expected


# Test with nested parentheses
def test_nested_parentheses():
    text = "Text (in English: Text (nested))"
    expected = ["(in English: Text (nested))"]
    assert extract_english_translations(text) == expected


# Test with text but no translations
def test_no_translations():
    text = "This is a sample text without any translations."
    expected = []
    assert extract_english_translations(text) == expected


# Test with empty string
def test_empty_string():
    text = ""
    expected = []
    assert extract_english_translations(text) == expected


# Test translations at the start and end of the string
def test_translations_at_edges():
    text = "(in English: Start) Some text in the middle (in English: End)"
    expected = ["(in English: Start)", "(in English: End)"]
    assert extract_english_translations(text) == expected


def test_find_forest_roots_and_members():
    incoming_edges = {
        "node1": {"node3", "node4"},
        "node2": set(),
        "node3": {"node5"},
        "node4": set(),
        "node5": set(),
    }

    expected_trees = {
        "node1": {"node5", "node1", "node3", "node4"},
        "node2": {"node2"},
    }

    trees = find_forest_roots_and_members(incoming_edges)
    assert len(trees) == 2, "Should identify two separate trees"
    for root, members in expected_trees.items():
        assert root in trees, f"Missing root {root} in result"
        assert trees[root] == members, (
            f"Tree members for root {root} do not match expected"
        )


@pytest.mark.asyncio
async def test_get_wikidata_entry():
    async with aiohttp.ClientSession() as session:
        result = await get_wikidata_english_name(
            "سیدنی (بازیکن فوتبال)", session, "fa", {}
        )
        # print(result)
        assert len(result) == 2
        assert "es" in result[1]
        assert result[1]["es"] == {
            "Sidnei Rechel da Silva Junior": "Sidnei (footballer, born 1989)"
        }


def test_to_json():
    block = Block(
        content="This is a test content.",
        document_title="Test Article",
        section_title="Test Section",
        last_edit_date="2023-10-01",
        url="http://example.com",
        block_metadata={
            "type": BlockType.TEXT,
            "language": BlockLanguage.ENGLISH,
        },
    )
    print(block)

    expected_output = {
        "url": "http://example.com",
        "document_title": "Test Article",
        "section_title": "Test Section",
        "content": "This is a test content.",
        "last_edit_date": "2023-10-01",
        "num_tokens": 0,
        "block_metadata": {
            "type": BlockType.TEXT,
            "language": BlockLanguage.ENGLISH,
        },
    }

    assert block.model_dump() == expected_output

    reloaded_block = Block(**expected_output)
    assert reloaded_block == block


def test_to_json_missing_optional_fields():
    block = Block(
        document_title="Test Article",
        section_title="Test Section",
        content="This is a test content.",
        block_metadata={"type": BlockType.TEXT},
    )

    expected_output = {
        "url": None,
        "document_title": "Test Article",
        "section_title": "Test Section",
        "content": "This is a test content.",
        "last_edit_date": None,
        "num_tokens": 0,
        "block_metadata": {"type": BlockType.TEXT},
    }

    assert block.model_dump() == expected_output


def test_block_valid_data():
    data = {
        "content": "This is a valid content.",
        "document_title": "Valid Document",
        "section_title": "Section",
        "last_edit_date": "2023-01-01",
        "url": "    http://example.com ",
        "num_tokens": 6,
        "block_metadata": {
            "type": BlockType.TEXT,
            "language": BlockLanguage.ENGLISH,
        },
    }
    block = Block(**data)
    assert block.content == data["content"]
    assert block.document_title == data["document_title"]
    assert block.section_title == data["section_title"]
    assert block.full_title == data["document_title"] + " > " + data["section_title"]
    assert block.last_edit_date == datetime(2023, 1, 1, 0, 0)
    assert block.url == data["url"].strip()
    assert block.num_tokens == data["num_tokens"]
    assert block.block_metadata == data["block_metadata"]


def test_block_swapped_date():
    data = {
        "content": "This is a valid content.",
        "document_title": "Valid Document",
        "section_title": "Section",
        "last_edit_date": "2023-22-01",
        "url": "http://example.com",
        "num_tokens": 6,
        "block_metadata": {
            "type": BlockType.TEXT,
            "language": BlockLanguage.ENGLISH,
        },
    }
    with pytest.raises(
        ValueError,
    ):
        Block(**data)


def test_block_empty_content():
    data = {
        "content": "",
        "document_title": "Valid Document",
        "section_title": "Section",
        "last_edit_date": "2023-01-01T12:00:00Z",
        "url": "http://example.com",
        "num_tokens": 6,
        "block_metadata": {
            "type": BlockType.TEXT,
            "language": BlockLanguage.ENGLISH,
        },
    }
    with pytest.raises(ValueError):
        Block(**data)


def test_block_invalid_last_edit_date():
    data = {
        "content": "This is a valid content.",
        "document_title": "Valid Document",
        "section_title": "Section",
        "last_edit_date": "invalid-date",
        "url": "http://example.com",
        "num_tokens": 6,
        "block_metadata": {
            "type": BlockType.TEXT,
            "language": BlockLanguage.ENGLISH,
        },
    }
    with pytest.raises(
        ValueError,
    ):
        Block(**data)


def test_block_invalid_url():
    data = {
        "content": "This is a valid content.",
        "document_title": "Valid Document",
        "section_title": "Section",
        "last_edit_date": "2023-01-01T12:00:00Z",
        "url": "invalid-url",
        "num_tokens": 6,
        "block_metadata": {
            "type": BlockType.TEXT,
            "language": BlockLanguage.ENGLISH,
        },
    }
    with pytest.raises(ValueError):
        Block(**data)


def test_block_empty_document_title():
    data = {
        "content": "This is a valid content.",
        "document_title": "",
        "section_title": "Section",
        "last_edit_date": "2023-01-01T12:00:00Z",
        "url": "http://example.com",
        "num_tokens": 6,
        "block_metadata": {
            "type": BlockType.TEXT,
            "language": BlockLanguage.ENGLISH,
        },
    }
    with pytest.raises(
        ValueError, match="`document_title` cannot be empty or just whitespace"
    ):
        Block(**data)


def test_update_after_creation():
    data = {
        "content": "This is a valid content.",
        "document_title": "Valid Document",
        "section_title": "Valid Section",
        "last_edit_date": "2023-01-01T12:00:00Z",
        "url": "http://example.com",
        "num_tokens": 6,
        "block_metadata": {
            "type": BlockType.TEXT,
            "language": BlockLanguage.ENGLISH,
        },
    }
    block = Block(**data)

    with pytest.raises(
        ValueError,
        match="Invalid date format for `last_edit_date`: '2024'. It should be in the format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ",
    ):
        block.last_edit_date = "2024"


def test_invalid_metadata_name():
    data = {
        "content": "This is a valid content.",
        "document_title": "Valid Document",
        "section_title": "Valid Section",
        "last_edit_date": "2023-01-01T12:00:00Z",
        "url": "http://example.com",
        "num_tokens": 6,
        "block_metadata": {
            "url": "http://example.com",  # metadata cannot be named url
        },
    }
    with pytest.raises(ValueError):
        Block(**data)
