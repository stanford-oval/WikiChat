import sys

import aiohttp
import pytest
from bs4 import BeautifulSoup

sys.path.insert(0, "./")
from wikipedia_preprocessing.preprocess_html_dump import (
    find_h_tags_hierarchy,
    get_adjacent_tags,
)
from wikipedia_preprocessing.utils import (
    extract_english_translations,
    find_forest_roots_and_members,
    get_wikidata_english_name,
)


@pytest.fixture
def sample_html():
    html_content = """
    <html>
    <body>
        <h1>Main Title</h1>
        <p>Some introduction text.</p>
        <h2>Subtitle Level 1</h2>
        <p>More detailed discussion.</p>
        <h3>Subtitle Level 2</h3>
        <h2>Another Subtitle Level 1</h2>
        <div id="target">Content here</div>
    </body>
    </html>
    """
    return BeautifulSoup(html_content, "html.parser")


def test_find_h_tags_with_simple_hierarchy(sample_html):
    tag = sample_html.find(id="target")
    hierarchy = find_h_tags_hierarchy(tag)
    assert len(hierarchy) == 2
    assert [tag.name for tag in hierarchy] == ["h1", "h2"]


def test_find_h_tags_with_no_previous_h_tags(sample_html):
    # Modify the HTML structure to not include any <h> tags before the target
    sample_html.find("h1").decompose()  # Removing h1
    sample_html.find("h2").decompose()  # Removing the first h2
    sample_html.find("h2").decompose()  # Removing the second h2
    sample_html.find("h3").decompose()  # Removing h3
    tag = sample_html.find(id="target")
    hierarchy = find_h_tags_hierarchy(tag)
    assert len(hierarchy) == 0  # No headers found


def test_reverse_order_correctness(sample_html):
    h3_tag = sample_html.find("h3")
    hierarchy = find_h_tags_hierarchy(h3_tag)
    assert len(hierarchy) == 2  # Only h1 and the first h2 should be captured
    assert [tag.name for tag in hierarchy] == ["h1", "h2"]


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
        assert (
            trees[root] == members
        ), f"Tree members for root {root} do not match expected"


# A function to create a simple HTML soup for testing purposes
def create_soup(html_content):
    return BeautifulSoup(html_content, "html.parser")


@pytest.mark.parametrize(
    "html_content,first_tag,second_tag,expected_output",
    [
        (
            "<p>First Paragraph</p><div>First Div</div>",
            "p",
            "div",
            (["First Paragraph"], ["First Div"]),
        ),
        ("<p>First Paragraph</p><p>Second Paragraph</p>", "p", "div", ([], [])),
        (
            "<div>A div</div><p>A Paragraph</p><div>Another div</div><p>Another Paragraph</p>",
            "div",
            "p",
            (["A div", "Another div"], ["A Paragraph", "Another Paragraph"]),
        ),
        (
            "<h1>Header</h1><p>Paragraph after header</p><div>Div after paragraph</div>",
            "h1",
            "p",
            (["Header"], ["Paragraph after header"]),
        ),
    ],
)
def test_get_adjacent_tags(html_content, first_tag, second_tag, expected_output):
    soup = create_soup(html_content)
    first_tags, second_tags = get_adjacent_tags(soup, first_tag, second_tag)
    assert set([tag.get_text() for tag in first_tags]) == set(expected_output[0])
    assert set([tag.get_text() for tag in second_tags]) == set(expected_output[1])


@pytest.mark.asyncio
async def test_get_wikidata_entry():
    async with aiohttp.ClientSession() as session:
        result = await get_wikidata_english_name("سیدنی (بازیکن فوتبال)", session, "fa")
        # print(result)
        assert len(result) == 2
        assert "es" in result[1]
        assert result[1]["es"] == {
            "Sidnei Rechel da Silva Junior": "Sidnei (footballer, born 1989)"
        }
