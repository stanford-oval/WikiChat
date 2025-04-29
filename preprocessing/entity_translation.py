from bs4 import BeautifulSoup
from preprocessing.utils import (
    get_from_translation_map,
    translation_prefix,
)
from urllib.parse import unquote
from bs4.element import NavigableString


def get_entity_translation_to_english(
    entity_name: str,
    context: str,
    global_translation_map,
    frequent_words_to_exclude,
    inverse_redirection_map,
) -> str:
    """
    The output of this function can be safely used to replace entity_name
    Args:
        entity_name: The name of the entity to translate.
        context: Optional; a string within which the presence of the translated name
                 is checked to avoid redundancy. Defaults to an empty string.

    Returns:
        A string containing the original entity name and its English translation,
        separated by a specific prefix, if the translation is found and deemed
        non-redundant. Returns just the entity name if the translation is redundant or not found.
    """
    cached_english = get_from_translation_map(
        entity_name, inverse_redirection_map, global_translation_map
    )
    if cached_english is not None:
        if cached_english.lower() not in frequent_words_to_exclude and any(
            char.isalpha() for char in cached_english
        ):  # exclude frequent words and numbers
            # remove parenthesis in entities like `XYZ (singer)`
            parenthesis_index = cached_english.find("(")
            if parenthesis_index >= 0:
                cached_english = cached_english[:parenthesis_index].strip()
            if (
                len(cached_english) > 0
                and cached_english.lower()
                not in context.lower()  # don't add hint if the `context` already contains it
            ):
                return f"{entity_name} {translation_prefix}{cached_english})"
            else:
                return entity_name
        else:
            # logger.debug(f"Excluded '{cached_english}' because it is too frequent")
            return entity_name
    else:
        # logger.debug(
        #     f"Did not find link entity in Wikidata for {entity_name}",
        # )
        return entity_name


def url_to_entity_name(url):
    ret = unquote(url).split("/")[-1].replace("_", " ")
    if ret.endswith("?action=edit&redlink=1"):
        ret = ret[: -len("?action=edit&redlink=1")]
    return ret


def preprocess_links(
    html_soup: BeautifulSoup,
    document_title: str,
    should_translate: bool,
    global_translation_map: dict,
    frequent_words_to_exclude: set,
    inverse_redirection_map: dict,
) -> None:
    for a_tag in html_soup.find_all("a", href=True):
        if a_tag["href"].endswith("&action=edit"):
            # delete Wikipedia links like "edit this article" etc.
            a_tag.decompose()
            continue
        if (
            should_translate
            and a_tag["href"].startswith("/wiki")
            and not a_tag["href"].endswith((".jpg", ".png", ".svg"))
        ):
            # internal link to a Wikipedia article
            entity_name = url_to_entity_name(a_tag["href"])
            a_tag.replace_with(
                NavigableString(
                    get_entity_translation_to_english(
                        entity_name,
                        context=document_title,
                        global_translation_map=global_translation_map,
                        frequent_words_to_exclude=frequent_words_to_exclude,
                        inverse_redirection_map=inverse_redirection_map,
                    )
                )
            )
        else:
            # external link, or internal link that doesn't need translation
            a_tag.replace_with(NavigableString(a_tag.text))
