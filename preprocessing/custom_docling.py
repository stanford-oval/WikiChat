import json
import typing
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple, Union, cast

from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString, PreformattedString
from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.utils.utils import create_file_hash
from docling_core.transforms.chunker.base import BaseChunk
from docling_core.transforms.chunker.hierarchical_chunker import (
    DocChunk,
    DocMeta,
    HierarchicalChunker,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc.document import (
    CodeItem,
    ContentLayer,
    DocItem,
    DoclingDocument,
    DocumentOrigin,
    GroupItem,
    LevelNumber,
    ListItem,
    NodeItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
)
from docling.backend.html_backend import TAGS_FOR_NODE_ITEMS
from docling_core.types.doc.labels import DocItemLabel
from pandas import DataFrame
from pydantic import BaseModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from typing_extensions import override

from preprocessing.block import Block
from preprocessing.entity_translation import preprocess_links
from retrieval.embedding_model_info import get_embedding_model_parameters
from tasks.defaults import DEFAULT_EMBEDDING_MODEL_NAME
from utils.logging import logger

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
        "فهرست",
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
    "zh": [
        "参见",
        "参考文献",
        "外部链接",
        "注释",
        "来源",
        "分类",
        "延伸阅读",
        "脚注",
    ],
    "ar": [
        "انظر أيضًا",
        "المراجع",
        "روابط خارجية",
        "ملاحظات",
        "مصادر",
        "تصنيفات",
        "قراءات إضافية",
    ],
    "pl": [
        "Zobacz też",
        "Przypisy",
        "Odnośniki zewnętrzne",
        "Notatki",
        "Źródła",
        "Kategorie",
        "Dalsza lektura",
    ],
    "nl": [
        "Zie ook",
        "Referenties",
        "Externe links",
        "Notities",
        "Bronnen",
        "Categorieën",
        "Verdere leesstof",
    ],
    "uk": [
        "Див. також",
        "Примітки",
        "Посилання",
        "Джерела",
        "Категорії",
        "Додаткове читання",
    ],
    "he": [
        "ראה גם",
        "הערות",
        "מקורות",
        "קישורים חיצוניים",
        "קטגוריות",
        "למידע נוסף",
    ],
    "id": [
        "Lihat juga",
        "Referensi",
        "Tautan eksternal",
        "Catatan",
        "Sumber",
        "Kategori",
        "Bacaan lebih lanjut",
    ],
    "tr": [
        "Ayrıca bakınız",
        "Kaynakça",
        "Dış bağlantılar",
        "Notlar",
        "Kategoriler",
        "Ek okuma",
    ],
    "cs": [
        "Viz také",
        "Odkazy",
        "Poznámky",
        "Zdroje",
        "Kategorie",
        "Doplňkové čtení",
    ],
    "sv": [
        "Se också",
        "Referenser",
        "Externa länkar",
        "Noter",
        "Källor",
        "Kategorier",
        "Vidare läsning",
    ],
    "ko": [
        "관련 항목",
        "각주",
        "참고 문헌",
        "외부 링크",
        "카테고리",
        "추가 읽기",
    ],
    "fi": [
        "Katso myös",
        "Viitteet",
        "Ulkoiset linkit",
        "Huomautukset",
        "Lähteet",
        "Luokat",
        "Lisälukemisto",
    ],
    "vi": [
        "Xem thêm",
        "Tham khảo",
        "Liên kết ngoài",
        "Ghi chú",
        "Nguồn",
        "Thể loại",
        "Đọc thêm",
    ],
    "hu": [
        "Lásd még",
        "Hivatkozások",
        "Külső linkek",
        "Megjegyzések",
        "Források",
        "Kategóriák",
        "További olvasmányok",
    ],
    "ca": [
        "Vegeu també",
        "Referències",
        "Enllaços externs",
        "Notes",
        "Fonts",
        "Categories",
        "Lectura addicional",
    ],
    "th": [
        "ดูเพิ่มเติม",
        "อ้างอิง",
        "ลิงก์ภายนอก",
        "หมายเหตุ",
        "แหล่งที่มา",
        "หมวดหมู่",
        "อ่านเพิ่มเติม",
    ],
}
all_banned_sections = set(
    [s for language in banned_sections for s in banned_sections[language]]
)


def iterate_items(
    doc: DoclingDocument,  # Changed self to doc based on usage in CustomHierarchicalChunker
    root: Optional[NodeItem] = None,
    with_groups: bool = False,
    traverse_pictures: bool = False,
    page_no: Optional[int] = None,
    _level: int = 0,  # fixed parameter, carries through the node nesting level
) -> typing.Iterable[Tuple[NodeItem, int]]:  # tuple of node and level
    """Iterate over document items, yielding nodes and their nesting level.

    Args:
        doc (DoclingDocument): The document to iterate over.
        root (Optional[NodeItem]): The starting node for iteration. Defaults to doc.body.
        with_groups (bool): Whether to include GroupItem nodes in the output. Defaults to False.
        traverse_pictures (bool): Whether to traverse into the children of PictureItem nodes. Defaults to False.
        page_no (Optional[int]): If set, only yield items from the specified page number. Defaults to None.
        _level (int): Internal parameter for tracking recursion depth.

    Yields:
        typing.Iterable[Tuple[NodeItem, int]]: An iterable of tuples, each containing a NodeItem and its level.
    """
    if not root:
        # If no root is provided, start from the document body
        if not doc.body:
            return  # Nothing to iterate if body is empty
        root = doc.body

    # Check if the current root is a banned section header
    if isinstance(root, SectionHeaderItem) and root.text in all_banned_sections:
        # If it's a banned section, skip this node and all its children entirely
        # logger.info(f"Skipping banned section and its children: {root.text}")
        return

    # Determine if the current node should be yielded based on filters
    should_yield = (
        not isinstance(root, GroupItem) or with_groups
    ) and (  # Yield non-groups or groups if requested
        not isinstance(
            root, DocItem
        )  # Always yield non-DocItems like GroupItem if with_groups=True
        or (
            page_no is None or any(prov.page_no == page_no for prov in root.prov)
        )  # Page filter for DocItems
    )

    # logger.info(f"Processing section: {root}")
    # logger.info(f"Should yield: {should_yield}")
    # if not should_yield:
    #     logger.info(
    #         f"The reason for False yield: isinstance(root, GroupItem): {isinstance(root, GroupItem)}, with_groups: {with_groups}, page_no: {page_no}, root.content_layer: {root.content_layer}"
    #     )
    if should_yield:
        yield root, _level

    # Special handling for PictureItem: only traverse children if requested
    if isinstance(root, PictureItem) and not traverse_pictures:
        return  # Stop traversal here if pictures shouldn't be traversed

    # Recursively traverse children
    for child_ref in root.children:
        child = child_ref.resolve(
            doc
        )  # Resolve the child reference within the document
        if isinstance(child, NodeItem):
            # Pass the document object (doc) to the recursive call
            yield from iterate_items(
                doc,  # Pass the document object
                child,
                with_groups=with_groups,
                traverse_pictures=traverse_pictures,
                page_no=page_no,
                _level=_level + 1,  # Increment level for children
            )


class CustomHierarchicalChunker(HierarchicalChunker):
    r"""Chunker implementation leveraging the document layout.

    Args:
        merge_list_items (bool): Whether to merge successive list items.
            Defaults to True.
        delim (str): Delimiter to use for merging text. Defaults to "\n".
    """

    @classmethod
    def _triplet_serialize(cls, table_df: DataFrame) -> str:
        """Serializes a DataFrame into a Markdown-like table string, wrapped in <Table> tags.

        Performs cleaning on cell content and headers:
        - Replaces newlines with spaces in all cells and headers.
        - Processes headers containing dots:
            - If parts around the last dot are identical (e.g., "Notes .Notes"), keeps the first part ("Notes").
            - Otherwise, replaces the last dot with "> " (e.g., "Notes .Year" becomes "Notes > Year").

        Simplifies rows where all values are identical

        Args:
            table_df: The pandas DataFrame representing the table.

        Returns:
            A string containing the table representation.
        """
        # Replace newlines in cell content with spaces for cleaner output
        df_clean = table_df.astype(str).replace(r"\n", " ", regex=True)

        # Clean headers: Replace newlines with spaces
        cleaned_columns = [str(col).replace("\n", " ") for col in df_clean.columns]

        # Process headers containing dots for potential simplification or clearer hierarchy
        processed_columns = []
        for col in cleaned_columns:
            if "." in col:
                parts = col.rsplit(".", 1)  # Split only on the last dot
                # Check if the parts around the last dot are identical after stripping whitespace
                if len(parts) == 2 and parts[0].strip() == parts[1].strip():
                    processed_columns.append(
                        parts[0].strip()
                    )  # Keep only the unique part
                else:
                    # If parts are different, replace dot with "> " for clarity
                    processed_columns.append(
                        col.replace(".", "> ", 1)
                    )  # Replace only the first occurrence if multiple dots exist, though rsplit handles the last one
            else:
                processed_columns.append(col)  # Keep header as is if no dot

        cleaned_columns = processed_columns  # Use the processed column names
        num_cols = len(cleaned_columns)

        # Process data rows, applying simplification where applicable
        data_rows = []
        for index, row in df_clean.iterrows():
            values = row.tolist()
            row_str = ""
            # Check if simplification is possible (more than one column exists)
            if num_cols > 1:
                if len(set(values)) == 1:
                    # If all values in the row are the same, simplify to one value
                    row_str = f"{values[0]}"
                else:
                    values_to_check = values[1:]  # Values after the first column
                    # Check if all values after the first column are identical
                    if values_to_check and len(set(values_to_check)) == 1:
                        repeated_value = values_to_check[0]
                        # Create the simplified key-value representation
                        row_str = f"{values[0]}: {repeated_value}"
                    else:
                        # Use standard Markdown table row format if values differ
                        row_str = "| " + " | ".join(values) + " |"
            else:
                # Handle tables with 0 or 1 column (no simplification possible)
                row_str = "| " + " | ".join(values) + " |"

            data_rows.append(row_str)

        if len(set(cleaned_columns)) == 1:
            cleaned_columns = [
                cleaned_columns[0]
            ]  # Simplify to a single column if all are the same
        if cleaned_columns == ["0", "1"]:
            header = ""  # Skip header if columns are exactly ["0", "1"]
        else:
            header = "| " + " | ".join(cleaned_columns) + " |"
        # Create the Markdown table separator row
        separator = "| " + " | ".join(["---"] * num_cols) + " |"

        markdown_table = "\n".join([header, separator] + data_rows)
        # Wrap the result in <Table> tags
        output_text = f"<Table>\n{markdown_table}\n</Table>"

        return output_text

    def chunk(self, dl_doc: DoclingDocument, **kwargs: Any) -> Iterator[BaseChunk]:
        r"""Chunk the provided document.

        Args:
            dl_doc (DoclingDocument): document to chunk

        Yields:
            Iterator[Chunk]: iterator over extracted chunks
        """

        heading_by_level: dict[LevelNumber, str] = {}
        list_items: list[TextItem] = []
        for item, level in iterate_items(dl_doc):
            captions = None
            if isinstance(item, DocItem):
                # first handle any merging needed
                if self.merge_list_items:
                    if isinstance(item, ListItem) or (
                        isinstance(item, TextItem)
                        and item.label == DocItemLabel.LIST_ITEM
                    ):
                        list_items.append(item)
                        continue
                    elif list_items:  # need to yield merged list items
                        merged_text = self.delim.join([i.text for i in list_items])

                        yield DocChunk(
                            text=merged_text,
                            meta=DocMeta(
                                doc_items=list_items,
                                headings=[
                                    heading_by_level[k]
                                    for k in sorted(heading_by_level)
                                ]
                                or None,
                                origin=dl_doc.origin,
                            ),
                        )
                        list_items = []  # reset

                if isinstance(item, SectionHeaderItem) or (
                    isinstance(item, TextItem)
                    and item.label in [DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE]
                ):
                    level = (
                        item.level
                        if isinstance(item, SectionHeaderItem)
                        else (0 if item.label == DocItemLabel.TITLE else 1)
                    )

                    heading_by_level[level] = item.text

                    # remove headings of higher level as they just went out of scope
                    keys_to_del = [k for k in heading_by_level if k > level]
                    for k in keys_to_del:
                        heading_by_level.pop(k, None)
                    continue

                if (
                    isinstance(item, TextItem)
                    or ((not self.merge_list_items) and isinstance(item, ListItem))
                    or isinstance(item, CodeItem)
                ):
                    text = item.text
                elif isinstance(item, TableItem):
                    table_df = item.export_to_dataframe()
                    if table_df.shape[0] < 1 or table_df.shape[1] < 2:
                        # at least two cols needed, as first column contains row headers
                        continue
                    text = self._triplet_serialize(table_df=table_df)
                    captions = [
                        c.text for c in [r.resolve(dl_doc) for r in item.captions]
                    ] or None
                else:
                    continue

                c = DocChunk(
                    text=text,
                    meta=DocMeta(
                        doc_items=[item],
                        headings=[heading_by_level[k] for k in sorted(heading_by_level)]
                        or None,
                        captions=captions,
                        origin=dl_doc.origin,
                    ),
                )
                yield c

        if self.merge_list_items and list_items:  # need to yield merged list items
            merged_text = self.delim.join([i.text for i in list_items])

            yield DocChunk(
                text=merged_text,
                meta=DocMeta(
                    doc_items=list_items,
                    headings=[heading_by_level[k] for k in sorted(heading_by_level)]
                    or None,
                    origin=dl_doc.origin,
                ),
            )


docling_chunker = None


def serialize(
    chunk: BaseChunk,
    document_title: str,
) -> Optional[Block]:
    """Serialize the given chunk by separately outputting headers list and main content.

    Args:
        chunk: chunk to serialize

    Returns:
        str: the serialized form with headers, metadata, and content separated
    """
    meta = chunk.meta.export_json_dict()
    headers = meta.get("headings", [])
    content = chunk.text

    # Serialize non-heading metadata.
    meta_items = []
    for k in meta:
        if k not in chunk.meta.excluded_embed and k != "headings":
            val = meta[k]
            if isinstance(val, list):
                meta_items.append(
                    "\n".join([d if isinstance(d, str) else json.dumps(d) for d in val])
                )
            else:
                meta_items.append(json.dumps(val))

    if content is None or not content.strip():
        return None
    block = Block(
        document_title=document_title,
        section_title=" > ".join(headers) if len(headers) > 0 else "",
        content=content,
        last_edit_date=None,  # will be filled in later
        url=None,  # will be filled in later
        num_tokens=0,  # will be filled in later
        block_metadata={},
    )

    return block


class CustomHTMLDocumentBackend(HTMLDocumentBackend):
    @override
    def __init__(
        self,
        in_doc: "CustomInputDocument",
        document_title: str,
        path_or_stream: Union[BytesIO, Path],
        should_translate: bool,
        global_translation_map: Optional[dict[str, str]],
        frequent_words_to_exclude: set,
        inverse_redirection_map: dict,
    ) -> None:
        """This function has been modified from parent to process links for translations, and not require filename as input"""
        super().__init__(in_doc, path_or_stream)
        self.soup: Optional[Tag] = None
        # HTML file:
        self.path_or_stream = path_or_stream
        # Initialise the parents for the hierarchy
        self.max_levels = 10
        self.level = 0
        self.parents: dict[int, Optional[Union[DocItem, GroupItem]]] = {}
        for i in range(0, self.max_levels):
            self.parents[i] = None

        try:
            if isinstance(self.path_or_stream, BytesIO):
                text_stream = self.path_or_stream.getvalue()
                self.soup = BeautifulSoup(text_stream, "lxml")
            if isinstance(self.path_or_stream, Path):
                with open(self.path_or_stream, "rb") as f:
                    html_content = f.read()
                    self.soup = BeautifulSoup(html_content, "lxml")
            # Remove all citations, style tags, Wikipedia notice or metadata boxes, and element with id "p-lang-btn"
            for tag in self.soup.select(
                "sup.reference, style, .ambox, .metadata, #p-lang-btn, .vector-page-toolbar, .vector-column-start, .vector-column-end, .vector-body-before-content"
            ):
                tag.decompose()

            for tag in self.soup.find_all(
                "math",
                alttext=lambda value: value and value.startswith(r"{\displaystyle"),
            ):
                tag.replace_with(
                    NavigableString(tag["alttext"][len(r"{\displaystyle") : -1])
                )
            preprocess_links(
                self.soup,
                document_title,
                should_translate=should_translate,
                global_translation_map=global_translation_map,
                frequent_words_to_exclude=frequent_words_to_exclude,
                inverse_redirection_map=inverse_redirection_map,
            )

            for li in self.soup.find_all("li"):
                # Check if a newline isn't already present between adjacent <li> tags
                next_sibling = li.next_sibling
                # Skip over any whitespace-only strings
                while (
                    next_sibling
                    and isinstance(next_sibling, NavigableString)
                    and not next_sibling.strip()
                ):
                    next_sibling = next_sibling.next_sibling
                if (
                    next_sibling
                    and isinstance(next_sibling, Tag)
                    and next_sibling.name == "li"
                ):
                    li.insert_after(NavigableString("\n"))

        except Exception as e:
            raise RuntimeError(
                "Could not initialize HTML backend for file with "
                f"hash {self.document_hash}."
            ) from e

    @override
    def convert(self) -> DoclingDocument:
        """This function has been modified to use a default filename"""
        # access self.path_or_stream to load stuff
        origin = DocumentOrigin(
            filename="file",
            mimetype="text/html",
            binary_hash=self.document_hash,
        )

        doc = DoclingDocument(name="file", origin=origin)

        if self.is_valid():
            assert self.soup is not None
            content = self.soup.body or self.soup
            # Replace <br> tags with newline characters
            for br in content("br"):
                br.replace_with(NavigableString("\n"))

            headers = content.find(["h1", "h2", "h3", "h4", "h5", "h6"])
            self.content_layer = (
                ContentLayer.BODY if headers is None else ContentLayer.FURNITURE
            )
            self.walk(content, doc)
        else:
            raise RuntimeError(
                f"Cannot convert doc with {self.document_hash} because the backend "
                "failed to init."
            )

        return doc

    @override
    def walk(self, tag: Tag, doc: DoclingDocument) -> None:
        """
        What has changed from parent is that we don't skip the entire article in case of an error
        """
        # Iterate over elements in the body of the document
        text: str = ""
        for element in tag.children:
            if isinstance(element, Tag):
                try:
                    self.analyze_tag(cast(Tag, element), doc)
                except Exception as exc_child:
                    logger.warning(
                        f"Error processing child from tag {tag.name}: {repr(exc_child)}"
                    )
                    continue  # don't skip the entire article
            elif isinstance(element, NavigableString) and not isinstance(
                element, PreformattedString
            ):
                # Floating text outside paragraphs or analyzed tags
                text += element
                siblings: list[Tag] = [
                    item for item in element.next_siblings if isinstance(item, Tag)
                ]
                if element.next_sibling is None or any(
                    [item.name in TAGS_FOR_NODE_ITEMS for item in siblings]
                ):
                    text = text.strip()
                    if text and tag.name in ["div"]:
                        doc.add_text(
                            parent=self.parents[self.level],
                            label=DocItemLabel.TEXT,
                            text=text,
                            content_layer=self.content_layer,
                        )
                    text = ""

        return


class CustomInputDocument(BaseModel):
    document_hash: str = ""
    valid: bool = True
    format: InputFormat = InputFormat.HTML
    file: str = ""

    def __init__(
        self,
        path_or_stream: Union[BytesIO, Path],
    ):
        super().__init__()
        self.document_hash = create_file_hash(path_or_stream)


def convert_html_to_blocks(
    html: str,
    document_title: str,
    should_translate: bool,
    global_translation_map: dict[str, str],
    frequent_words_to_exclude: set[str],
    inverse_redirection_map: dict,
) -> list[Block]:
    global docling_chunker
    if docling_chunker is None:
        max_tokens = get_embedding_model_parameters(DEFAULT_EMBEDDING_MODEL_NAME)[
            "max_sequence_length"
        ]
        docling_chunker = HybridChunker(
            tokenizer=AutoTokenizer.from_pretrained(
                DEFAULT_EMBEDDING_MODEL_NAME, use_fast=True
            ),
            max_tokens=max_tokens,
        )
        docling_chunker._inner_chunker = CustomHierarchicalChunker()

    bio = BytesIO(html.encode("utf-8"))
    dl_doc = CustomHTMLDocumentBackend(
        in_doc=CustomInputDocument(
            path_or_stream=bio,
        ),
        path_or_stream=bio,
        document_title=document_title,
        should_translate=should_translate,
        global_translation_map=global_translation_map,
        frequent_words_to_exclude=frequent_words_to_exclude,
        inverse_redirection_map=inverse_redirection_map,
    ).convert()

    blocks = []
    for chunk in docling_chunker.chunk(dl_doc):
        block = serialize(chunk, document_title)
        if block is not None:
            blocks.append(block)

    return blocks
