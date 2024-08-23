import argparse
import os
import pathlib
import re
import sys
import tarfile
from multiprocessing import Process, SimpleQueue, cpu_count
from urllib.parse import unquote

import orjson
from bs4 import BeautifulSoup, NavigableString
from markdownify import MarkdownConverter
from mwparserfromhtml.parse.plaintext import html_to_plaintext
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, "./")
from pipelines.utils import get_logger
from wikipedia_preprocessing.utils import *
from wikipedia_preprocessing.wikipedia_disambiguation import is_disambiguation

logger = get_logger(__name__)
from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity(transformers_logging.ERROR)

inverse_redirection_map = (
    {}
)  # used to expand translation search. This map includes a map of each root to itself, for simplicity
frequent_words_to_exclude = set()
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3", fast=True)


class Block:
    """
    A paragraph, list, linearized table, or linearized Infobox
    """

    content_string: str
    article_title: str
    full_section_title: str
    block_type: str
    language: str
    last_edit_date: str
    num_tokens: int

    def __init__(
        self,
        content_string: str,
        full_section_title: str,
        block_type: str,
        article_title: str = None,
        language: str = None,
        last_edit_date: str = None,
    ):
        self.content_string = content_string.strip()
        self.article_title = article_title
        self.full_section_title = full_section_title
        self.block_type = block_type
        self.language = language
        self.last_edit_date = last_edit_date
        self.num_tokens = 0

    def to_json(self, _id: int):
        ret = self.__dict__
        ret["id"] = _id
        return ret

    def deduplicate_translations(self) -> None:
        """
        Deduplicates (in English: ...) from each block
        """
        string = self.full_section_title + " | " + self.content_string
        translation_parenthesis = set(extract_english_translations(string))
        for t in translation_parenthesis:
            string = replace_except_first(string, " " + t, "")

        self.full_section_title, self.content_string = tuple(string.split(" | ", 1))


banned_sections = {
    "en": [
        "See also",
        "References",
        "External links",
        "Notes",
        "Sources",
        "Categories",
        "Further reading",
        "Citations",
        "Footnotes",
    ],
    "fa": [
        "همچنین ببینید",
        "پانویس",
        "منابع",
        "پیوند به بیرون",
        "یادداشت‌ها",
        "منابع و پانویس",
        "رده‌ها",
        "مطالعه بیشتر",
        "جستارهای وابسته",
    ],
    "es": [
        "Véase también",
        "Referencias",
        "Enlaces externos",
        "Notas",
        "Fuentes",
        "Categorías",
        "Lecturas adicionales",
        "Notas al pie",
    ],
    "fr": [
        "Voir aussi",
        "Références",
        "Liens externes",
        "Notes",
        "Sources",
        "Catégories",
        "Lecture complémentaire",
        "Notes et références",
    ],
    "it": [
        "Vedi anche",
        "Note",
        "Riferimenti",
        "Collegamenti esterni",
        "Fonti",
        "Categorie",
        "Bibliografia",
        "Altri progetti",
    ],
    "de": [
        "Siehe auch",
        "Einzelnachweise",
        "Weblinks",
        "Anmerkungen",
        "Quellen",
        "Kategorien",
        "Literatur",
        "Fußnoten",
    ],
    "ja": [
        "関連項目",
        "脚注",
        "注釈",
        "出典",
        "参考文献",
        "外部リンク",
        "参照",
        "参照リンク",
    ],
    "ru": [
        "См. также",
        "Примечания",
        "Ссылки",
        "Источники",
        "Литература",
        "Категории",
        "Дополнительные сведения",
        "Примечания",
    ],
    "pt": [
        "Ver também",
        "Referências",
        "Ligações externas",
        "Notas",
        "Fontes",
        "Categorias",
        "Leitura adicional",
        "Notas de rodapé",
    ],
    "zh": ["参见", "参考文献", "外部链接", "注释", "来源", "分类", "延伸阅读", "脚注"],
}
all_banned_sections = set(
    [s for language in banned_sections for s in banned_sections[language]]
)


def compress_markup(markdown_text: str):
    """
    Replaces multiple spaces and tabls with just one space. This does not affect how Markup is displayed
    """
    return re.sub(r"[ \t]+", " ", markdown_text)


def is_banned_section(title_stack: list[str]) -> bool:
    if len(title_stack) == 2 and title_stack[-1] in all_banned_sections:
        return True
    return False


def find_h_tags_hierarchy(tag):
    hierarchy = []
    current_level = float("inf")  # Start with an infinitely deep level
    for sibling in tag.find_all_previous(["h1", "h2", "h3", "h4", "h5", "h6"]):
        # Stop if another table is encountered
        level = int(sibling.name[1])  # Extract the numeric level of the header
        if level < current_level:
            hierarchy.append(sibling)
            current_level = level
    return hierarchy[::-1]  # Reverse to maintain the order from top to bottom


def tag_to_markdown(table, article_title: str) -> tuple[str, str]:
    md = MarkdownConverter().convert_soup(table)
    md = compress_markup(md)
    md = md.strip()
    hierarchy = [h.text for h in find_h_tags_hierarchy(table)]
    full_section_title = " > ".join([article_title] + hierarchy)

    return (full_section_title, md)


def find_table_descriptions(tag) -> tuple[str, str]:
    """
    Finds descriptions (<dl> tags) before and after a given table element within an HTML document.

    Args:
        tag: The BeautifulSoup tag object representing a table in an HTML document.

    Returns:
        A tuple of two strings: The first string contains the text of the description list
        found immediately before the table (if any), and the second string contains the text
        of the description list found immediately after the table (if any). If no description
        list is found in a respective position, an empty string is returned for that position.
    """
    pre_dl = ""
    post_dl = ""
    # Iterate through previous siblings of the tag
    for sibling in tag.previous_siblings:
        # Check if the sibling is a NavigableString and not empty or just whitespace
        if sibling.name in ["dl"] and sibling.text and sibling.text.strip():
            pre_dl = sibling.text.strip()
            break
    for sibling in tag.next_siblings:
        if (
            hasattr(sibling, "name")
            and sibling.name == "dl"
            and sibling.text
            and sibling.text.strip()
        ):
            post_dl = sibling.text.strip()
            break
    return pre_dl, post_dl


def get_tables_and_infoboxes(
    html_soup: BeautifulSoup, article_title: str, extra_tables: list
) -> list[Block]:

    blocks = []
    tables = set(html_soup.select("table.sidebar, table.wikitable") + extra_tables)
    infoboxes = html_soup.find_all(
        "table", class_=lambda x: (x and "infobox" in x)
    )  # french uses infobox_v2, which this pattern also matches

    for block_type, tag_list in zip(["table", "infobox"], [tables, infoboxes]):
        for tag in tag_list:
            try:
                full_section_title, content = tag_to_markdown(tag, article_title)
                if block_type == "table":
                    pretable, post_table = find_table_descriptions(tag)
                    if pretable:
                        content = pretable + "\n" + content
                    if post_table:
                        content = content + "\n" + post_table
                blocks.append(
                    Block(
                        content_string=content,
                        full_section_title=full_section_title,
                        block_type=block_type,
                    )
                )
            except Exception as e:
                logger.debug(
                    "BeautifulSoup encountered an error while parsing article '%s': %s",
                    article_title,
                    str(e),
                )
                continue

    return blocks


def get_passages(
    html_soup: BeautifulSoup,
    article_title: str,
    pack_to_tokens: int,
    exclude_elements={
        "Reference",
        "ExternalLink",
        "Heading",
        "Category",
        "Citation",
        "Media",
        "Navigation",
        "Note",
        "Messagebox",
        "Infobox",
        "Wikitable",
        "Comment",
        "Source",
        "Table",
    },
) -> list[tuple[str, str]]:
    """
    Extract plaintext from the HTML object in a depth-first manner,
    including full path to a section in headings.

    Args:
        article_title: The title of the article (or root section).
        exclude_elements: Set of HTML element types to exclude.

    Returns:
        A tuple of (heading, plaintext) where heading is the full path to the section.
    """
    section_stack = [
        article_title
    ]  # Initialize stack with the article title as the root section.
    blocks = []

    def get_full_heading():
        return " > ".join(section_stack)

    for i, section in enumerate(html_soup.findAll("section")):
        # Construct heading with full path
        if i != 0:  # Skip the first section since it's the article title itself
            current_heading = section.findChild().text
            if len(section_stack) == 1:
                section_stack.append(
                    current_heading
                )  # Direct subsection of the article
            else:
                if len(section_stack) < 1:
                    logger.warning(
                        "Section structure in article '%s' is malformed.", article_title
                    )
                else:
                    section_stack[-1] = (
                        current_heading  # Replace the last section title with the current one
                    )

        # get plaintext for each paragraph in the section
        plaintext = ""
        prev_para_context = "pre-first-para"
        for (
            node_plaintext,
            _,
            element_types,
            para_context,
        ) in html_to_plaintext(section):
            # Check for nested sections to update heading path
            if element_types.count("Section") > 1:
                if node_plaintext not in section_stack:
                    section_stack.append(
                        node_plaintext
                    )  # Nest deeper for new subsections
                else:
                    if len(section_stack) > 0:
                        section_stack.pop()  # Ascend as we exit a subsection
                    else:
                        logger.warning(
                            "Hierarchy of sections for article %s ran into an error.",
                            article_title,
                        )
                break

            if is_banned_section(section_stack) or (
                exclude_elements and exclude_elements.intersection(element_types)
            ):
                continue
            if node_plaintext == "\n" and set(element_types) == {"Section"}:
                if plaintext.strip():
                    blocks.append(
                        Block(
                            content_string=plaintext,
                            full_section_title=get_full_heading(),
                            block_type="text",
                        )
                    )
                plaintext = ""
                prev_para_context = para_context
            elif para_context != prev_para_context:
                if plaintext.strip():
                    blocks.append(
                        Block(
                            content_string=plaintext,
                            full_section_title=get_full_heading(),
                            block_type="text",
                        )
                    )

                plaintext = node_plaintext
                prev_para_context = para_context
            else:
                plaintext += node_plaintext
                prev_para_context = para_context

        if plaintext.strip():
            blocks.append(
                Block(
                    content_string=plaintext,
                    full_section_title=get_full_heading(),
                    block_type="text",
                )
            )

        # Reset or ascend the section stack as necessary
        if i != 0 and len(section_stack) > 2:
            section_stack.pop()  # Ascend when leaving a subsection to its parent

    blocks = pack_blocks(blocks, pack_to_tokens)

    return blocks


def num_tokens(text: str) -> int:
    return len(tokenizer(text)["input_ids"])


def pack_blocks(blocks: list[tuple[str, str]], pack_to_tokens: int) -> list[Block]:
    """
    Passages is the list of tuples where each tuple is (subsection title, passage).

    This function concatenates consecutive passages with the same subsection
    title as long as their combined length does not exceed `pack_to_tokens` tokens.
    """

    if not blocks:
        return []

    packed_blocks = []
    current_block = blocks[0]

    current_block.num_tokens = num_tokens(
        current_block.full_section_title + " " + current_block.content_string
    )
    for next_block in blocks[1:]:
        # Check if the next block has the exact same section title and does not exceed the character limit
        num_tokens_after_merge = current_block.num_tokens + num_tokens(
            "\n" + next_block.content_string
        )
        if (
            next_block.full_section_title == current_block.full_section_title
            and num_tokens_after_merge < pack_to_tokens
        ):
            current_block.content_string += (
                " " + next_block.content_string
            )  # Concatenate blocks with a space in between
            current_block.num_tokens = num_tokens_after_merge
        else:
            # Once a block reaches the limit or a new title is found, append the current state and move on
            packed_blocks.append(current_block)
            current_block = next_block
            current_block.num_tokens = num_tokens(
                current_block.full_section_title + " " + current_block.content_string
            )

    # Adding the last accumulated paragraph
    packed_blocks.append(current_block)

    return packed_blocks


def get_entity_translation_to_english(
    source_language: str, entity_name: str, context: str = ""
) -> str:
    """
    The output of this function can be safely used to replace entity_name
    Args:
        source_language: The language code of the source entity name.
        entity_name: The name of the entity to translate.
        context: Optional; a string within which the presence of the translated name
                 is checked to avoid redundancy. Defaults to an empty string.

    Returns:
        A string containing the original entity name and its English translation,
        separated by a specific prefix, if the translation is found and deemed
        non-redundant. Returns just the entity name if the translation is redundant or not found.
    """
    cached_english = get_from_translation_cache(
        source_language, entity_name, inverse_redirection_map
    )
    if cached_english is not None:
        if cached_english not in frequent_words_to_exclude:
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
            logger.debug("Excluded %s because it is too frequnt", cached_english)
            return entity_name
    else:
        logger.debug(
            "Did not find link entity in Wikidata for %s",
            entity_name,
        )
        return entity_name


def preprocess_links(
    html_soup: BeautifulSoup, article_title: str, should_translate: bool, language: str
) -> None:
    for a_tag in html_soup.find_all("a", href=True):
        if a_tag["href"].endswith("&action=edit"):
            # delete Wikipedia links like "edit this article" etc.
            a_tag.decompose()
            continue
        if should_translate and a_tag["href"].startswith("./"):
            # internal link to a Wikipedia article
            entity_name = url_to_entity_name(a_tag["href"])
            a_tag.replace_with(
                NavigableString(
                    get_entity_translation_to_english(
                        language, entity_name, context=article_title
                    )
                )
            )
        else:
            # external link, or internal link that doesn't need translation
            a_tag.replace_with(NavigableString(a_tag.text))


def get_adjacent_tags(
    soup, tag_that_comes_first: str, tag_that_comes_second: str
) -> tuple:
    tags_coming_first = []
    tags_coming_second = []

    for tag_1 in soup.find_all(tag_that_comes_first):
        next_sibling = tag_1.find_next_sibling()

        if next_sibling and next_sibling.name == tag_that_comes_second:
            tags_coming_second.append(next_sibling)
            tags_coming_first.append(tag_1)

    return tags_coming_first, tags_coming_second


def prepend_dls(html_soup):
    """
    In Wikipedia, often <dl> tags are used incorrectly instead of <p> or even <h3>.
    This function is a heuristic and imperfect way of connecting <dl> tags to their relevant context.
    """
    filtered_dl_tags = set()
    dls, posts = get_adjacent_tags(html_soup, "dl", "ul")
    for dl, post in zip(dls, posts):
        # Check if the <dl> tag is not a descendant of a <table> tag
        if not dl.find_parent("table") and not dl.find_all("table"):
            # Ensure none of the descendants have a class "mwe-math-element"
            if not dl.find_all(class_="mwe-math-element"):
                filtered_dl_tags.add(dl)
                post.insert(0, NavigableString(dl.text + "\n"))

    dls, posts = get_adjacent_tags(html_soup, "dl", "p")
    for dl, post in zip(dls, posts):
        # Check if the <dl> tag is not a descendant of a <table> tag
        if not dl.find_parent("table") and not dl.find_all("table"):
            # Ensure none of the descendants have a class "mwe-math-element"
            if not dl.find_all(class_="mwe-math-element"):
                filtered_dl_tags.add(dl)
                post.insert(0, NavigableString(dl.text + "\n"))

    for tag in filtered_dl_tags:
        tag.decompose()


def process_articles(
    input_queue,
    output_queue,
    pack_to_tokens: int,
    language: str,
    should_translate: bool,
):
    while True:
        article = input_queue.get()
        if article is None:
            break

        article_blocks = []
        html = article["article_body"]["html"]
        article_title = article["name"]

        if should_translate:
            # add English translation to title
            article_title = get_entity_translation_to_english(
                language, article_title, context=article_title
            )  # don't add the translation if the article_title already has or is in English

        html_soup = BeautifulSoup(html, features="lxml")

        # Remove all citations and style tags
        for tag in html_soup.select("sup.reference, style"):
            tag.decompose()

        # Display math equations better
        for tag in html_soup.find_all(
            "math", alttext=lambda value: value and value.startswith("{\displaystyle")
        ):
            tag.replace_with(
                NavigableString(tag["alttext"][len("{\displaystyle") : -1])
            )
        preprocess_links(
            html_soup,
            article_title,
            should_translate=should_translate,
            language=language,
        )

        # <dl> right after <table>
        tables1, dls1 = get_adjacent_tags(
            html_soup, tag_that_comes_first="table", tag_that_comes_second="dl"
        )
        # <table> right after <dl>
        dls2, tables2 = get_adjacent_tags(
            html_soup, tag_that_comes_first="dl", tag_that_comes_second="table"
        )
        article_blocks.extend(
            get_tables_and_infoboxes(html_soup, article_title, tables1 + tables2)
        )

        # sidebars are already processed together with tables
        # https://en.wikipedia.org/wiki/Template:Sidebar
        # https://en.wikipedia.org/wiki/Template:Clade
        for t in html_soup.select(
            "table.sidebar, table.clade, figure, .shortdescription"
        ):
            t.decompose()
        # <dl> tags before or after tables are already indcluded with the table, so remove them here
        for dl in dls1 + dls2:
            dl.decompose()

        prepend_dls(html_soup)

        article_blocks.extend(
            get_passages(
                html_soup=html_soup,
                article_title=article_title,
                pack_to_tokens=pack_to_tokens,
            )
        )

        for block in article_blocks:
            if len(block.content_string) < 3:
                continue  # skip empty blocks

            if block.num_tokens == 0:
                block.num_tokens = num_tokens(
                    block.full_section_title + " " + block.content_string
                )
            block.article_title = article_title
            block.language = language
            block.last_edit_date = article["date_modified"]
            if should_translate:
                block.deduplicate_translations()

            output_queue.put(block)

    output_queue.put(None)  # signal the end


def url_to_entity_name(url):
    ret = unquote(url).split("/")[-1].replace("_", " ")
    if ret.endswith("?action=edit&redlink=1"):
        ret = ret[: -len("?action=edit&redlink=1")]
    return ret


def build_redirection_map(file_path: str) -> dict:
    redirection_incoming_edges: dict[str, set[str]] = (
        {}
    )  # maps an article url with all urls that redirect to it, via one or multiple hops

    for article in tqdm(
        tarfile_loader(file_path),
        desc="Building the Wikipedia redirection graph",
        miniters=1e-6,
        unit_scale=1,
        unit=" Articles",
        smoothing=0,
    ):
        if is_disambiguation(article):
            continue

        url = url_to_entity_name(article["url"])
        if url not in redirection_incoming_edges:
            redirection_incoming_edges[url] = set()
        if "redirects" in article:
            # Add redirects even if we have already seen it. This way, multi-hop redirects will be handled correctly.
            for redirect in article["redirects"]:
                redirected_url = url_to_entity_name(redirect["url"])
                redirection_incoming_edges[url].add(redirected_url)

    # print("before multihop consolidation: ", len(redirection_incoming_edges))
    # the structure of this dictionary describes a forest (i.e. collection of trees), with each item describing the incoming edges of a node
    # we want to find the root of all trees
    redirect_map = find_forest_roots_and_members(redirection_incoming_edges)
    # print("after multihop consolidation: ", len(redirect_map))
    return redirect_map


def tarfile_loader(file_path: str):
    """
    Generator that sequentially loads articles from a tar.gz file containing NDJSON formatted articles.
    Skips over articles with identifiers that have been seen already, ignoring redirects or duplicates.
    """

    tar_file_ = tarfile.open(file_path, mode="r|gz")
    while True:
        ndjson_file = tar_file_.next()
        if ndjson_file is None:
            tar_file_.close()
            break
        else:
            with tar_file_.extractfile(ndjson_file) as file_content:
                for line in file_content:
                    article = orjson.loads(line)
                    yield article

    tar_file_.close()


def articles_without_disambiguation_or_redirections(
    file_path: str,
    num_workers: int,
    queue,
    redirect_map: dict,
    max_articles: int,
):
    # the reason we iterate over and process the Wikipedia dump file again is we don't want to keep everything in memory, especially for large dump files.
    pbar = tqdm(
        desc="Extracting blocks",
        miniters=1e-6,
        unit_scale=1,
        unit=" Blocks",
        smoothing=0,
        total=len(redirect_map),
    )
    counter = 0
    for article in tarfile_loader(file_path):
        if is_disambiguation(article):
            continue
        url = url_to_entity_name(article["url"])
        if url not in redirect_map:
            continue
        queue.put(article)
        pbar.update(1)
        counter += 1
        if counter == max_articles:
            break

    for _ in range(num_workers):
        queue.put(None)  # signal end to all workers


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="A Wikipedia HTML dump, which is a tar.gz file containing multiple .ndjson files",
    )
    arg_parser.add_argument("--output_path", type=str, required=True)
    arg_parser.add_argument("--language", type=str, required=True)
    arg_parser.add_argument(
        "--should_translate",
        action="store_true",
        help="If we should translate named entities to English using Wikidata. Has no effect if `--language` is English",
    )
    arg_parser.add_argument(
        "--translation_cache",
        type=str,
        help="Where to read/write the translation cache.",
    )
    arg_parser.add_argument("--num_workers", type=int, default=max(1, cpu_count() - 4))
    arg_parser.add_argument(
        "--pack_to_tokens",
        type=int,
        default=0,
        help="If consecutive paragraphs in the same subsection are small, we greedily concatenate them together, while keeping the result shorter than this many tokens."
        " This helps reduce the number of vector embeddings when indexing. BAAI/bge-m3 tokenizer is used to determine token boundaries.",
    )
    arg_parser.add_argument(
        "--max_articles",
        type=int,
        default=-1,
        help="Will stop after processing this many articles. -1 means no limit. Used for testing.",
    )
    arg_parser.add_argument(
        "--num_exclude_frequent_words_from_translation",
        type=int,
        default=0,
        help="Will exclude translations for the top N most frequent words used in the English Wikipedia.",
    )

    args = arg_parser.parse_args()
    if args.language == "en":
        args.should_translate = False

    redirection_map = build_redirection_map(args.input_path)

    for node, redirectors in redirection_map.items():
        for node2 in redirectors:
            inverse_redirection_map[node2] = node

    if args.should_translate:
        logger.info(
            "Number of articles, excluding disambiguation and redirection pages: %d",
            len(redirection_map),
        )
        if args.num_exclude_frequent_words_from_translation > 0:
            with open("wikipedia_preprocessing/word_list.txt") as f:
                for line in f:
                    frequent_words_to_exclude.add(line.strip())
                    if (
                        len(frequent_words_to_exclude)
                        >= args.num_exclude_frequent_words_from_translation
                    ):
                        break

        load_translation_cache(args.translation_cache)
        non_cached_titles = []
        for url in redirection_map:
            if (
                get_from_translation_cache(args.language, url, inverse_redirection_map)
                is None
            ):
                non_cached_titles.append(url)

        if len(non_cached_titles) > 0:
            logger.info(
                "Did not find %d articles in the cache, will call the Wikidata API for them",
                len(non_cached_titles),
            )
            asyncio.run(
                batch_get_wikidata_english_name(non_cached_titles, args.language)
            )
            save_translation_cache(args.translation_cache)

    input_queue = SimpleQueue()
    output_queue = SimpleQueue()
    all_worker_processes = []

    for worker_id in range(args.num_workers):
        all_worker_processes.append(
            Process(
                target=process_articles,
                args=(
                    input_queue,
                    output_queue,
                    args.pack_to_tokens,
                    args.language,
                    args.should_translate,
                ),
            )
        )

    # The process that feeds the articles to workers
    reader_process = Process(
        target=articles_without_disambiguation_or_redirections,
        args=(
            args.input_path,
            args.num_workers,
            input_queue,
            redirection_map,
            args.max_articles,
        ),
    )

    for p in all_worker_processes + [reader_process]:
        p.start()

    workers_finished = 0
    all_blocks = []
    text_count, table_count, infobox_count = 0, 0, 0

    # make parent directories
    pathlib.Path(os.path.dirname(args.output_path)).mkdir(parents=True, exist_ok=True)

    # compact just removes the extra space in the json formatting
    counter = 0

    while True:
        block = output_queue.get()
        if block is None:
            workers_finished += 1
            if workers_finished == len(all_worker_processes):
                break
            continue
        if block.block_type == "text":
            text_count += 1
        elif block.block_type == "table":
            table_count += 1
        elif block.block_type == "infobox":
            infobox_count += 1
        else:
            assert False, "Unknown block type:" + str(block["block_type"])
        all_blocks.append(block.to_json(counter))
        counter += 1

    logger.info("Saving the collection to %s", args.output_path)
    orjsonl.save(args.output_path, all_blocks, compression_format="gz")

    # save the collection size
    with open(
        os.path.join(os.path.dirname(args.output_path), "collection_size.txt"), "w"
    ) as f:
        f.write(str(len(all_blocks)))

    # Wait for processes to complete
    for p in all_worker_processes + [reader_process]:
        p.join()

    logger.info("Found {:,d} text blocks (including lists)".format(text_count))
    logger.info("Found {:,d} table blocks".format(table_count))
    logger.info("Found {:,d} infobox blocks".format(infobox_count))
    logger.info(
        "Total number of blocks: {:,d}".format(text_count + table_count + infobox_count)
    )

    # print_histogram([len(block.content_string) for block in all_blocks])
    draw_and_save_histogram_log_bins(
        [block["num_tokens"] for block in all_blocks],
        args.output_path.split(".")[0] + "_histogram.png",
    )
