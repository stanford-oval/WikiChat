from preprocessing.utils import (
    extract_english_translations,
    replace_except_first,
)


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
