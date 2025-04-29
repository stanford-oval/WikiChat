from datetime import datetime
from enum import Enum
from typing import ClassVar, Optional, Union
from urllib.parse import unquote

from openai import BaseModel
from pydantic import (
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from preprocessing.utils import (
    batch_get_num_tokens,
    extract_english_translations,
    replace_except_first,
)


class BlockLanguage(str, Enum):
    """
    25 Wikipedias in order of the number of active users, according to https://en.wikipedia.org/wiki/List_of_Wikipedias#Active_editions
    """

    ENGLISH = "en"
    FRENCH = "fr"
    GERMAN = "de"
    SPANISH = "es"
    JAPANESE = "ja"
    RUSSIAN = "ru"
    PORTUGUESE = "pt"
    CHINESE = "zh"
    ITALIAN = "it"
    ARABIC = "ar"
    PERSIAN = "fa"
    POLISH = "pl"
    DUTCH = "nl"
    UKRAINIAN = "uk"
    HEBREW = "he"
    INDONESIAN = "id"
    TURKISH = "tr"
    CZECH = "cs"
    SWEDISH = "sv"
    KOREAN = "ko"
    FINNISH = "fi"
    VIETNAMESE = "vi"
    HUNGARIAN = "hu"
    CATALAN = "ca"
    THAI = "th"

    # Smaller Wikipedias used for testing
    KURDISH = "ku"


class BlockType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    INFOBOX = "infobox"


BlockMetadataType = Union[str, int, float, bool, datetime, BlockType, BlockLanguage]


class Block(BaseModel):
    """
    An indexing/retrieval unit. Can be a paragraph, list, linearized table, or linearized Infobox
    """

    MAX_DOCUMENT_TITLE_LENGTH: ClassVar[int] = 500
    # cutting any shorter causes issues with long titles, especially ones that have the translation "(in English: ...)" in them
    MAX_SECTION_TITLE_LENGTH: ClassVar[int] = 400

    document_title: str = Field(..., description="The title of the document")
    section_title: str = Field(
        ...,
        description="The hierarchical section title of the block excluding `document_title`, e.g. 'Land > Central Campus'. Section title can be empty, for instance the first section of Wikipedia articles.",
    )
    content: str = Field(
        ..., description="The content of the block, usually in Markdown format"
    )
    last_edit_date: Optional[datetime] = Field(
        None,
        description="The last edit date of the block in the format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS",
    )
    url: Optional[str] = Field(None, description="The URL of the block")

    num_tokens: int = Field(0, description="The number of tokens in the block content")
    block_metadata: Optional[
        dict[str, Union[BlockMetadataType, list[BlockMetadataType]]]
    ] = Field(None, description="Additional metadata for the block")

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",  # Disallow extra fields
    )

    @field_serializer("last_edit_date")
    def serialize_datetime(self, v: datetime):
        return v.strftime("%Y-%m-%d") if v else None

    @property
    def date_human_readable(self) -> Optional[str]:
        return (
            self.last_edit_date.strftime("%B %d, %Y") if self.last_edit_date else None
        )

    # TODO remove these after the migration
    @property
    def language(self) -> Optional[BlockLanguage]:
        return self.block_metadata.get("language") if self.block_metadata else None

    @language.setter
    def language(self, value: BlockLanguage) -> None:
        if self.block_metadata is None:
            self.block_metadata = {"language": value}
        else:
            self.block_metadata["language"] = value

    @property
    def block_type(self) -> Optional[BlockType]:
        return self.block_metadata.get("type") if self.block_metadata else None

    @block_type.setter
    def block_type(self, value: BlockType) -> None:
        if self.block_metadata is None:
            self.block_metadata = {"type": value}
        else:
            self.block_metadata["type"] = value

    def get_metadata_fields_and_types(
        self,
    ) -> tuple[list[str], list[BlockMetadataType]]:
        """
        Get the metadata fields and types for the block
        """
        field_names = []
        field_types = []
        # TODO if the first block in a collection is an empty list, this may cause an error
        if self.block_metadata:
            for k, v in self.block_metadata.items():
                field_names.append(k)
                if isinstance(v, str):
                    field_types.append(str)
                elif isinstance(v, int):
                    field_types.append(int)
                elif isinstance(v, float):
                    field_types.append(float)
                elif isinstance(v, bool):
                    field_types.append(bool)
                elif isinstance(v, datetime):
                    field_types.append(datetime)
                elif isinstance(v, BlockType):
                    field_types.append(BlockType)
                elif isinstance(v, BlockLanguage):
                    field_types.append(BlockLanguage)
                elif isinstance(v, list):
                    if v:
                        if isinstance(v[0], str):
                            field_types.append(str)
                        elif isinstance(v[0], int):
                            field_types.append(int)
                        elif isinstance(v[0], float):
                            field_types.append(float)
                        elif isinstance(v[0], bool):
                            field_types.append(bool)
                        elif isinstance(v[0], datetime):
                            field_types.append(datetime)
                        elif isinstance(v[0], BlockType):
                            field_types.append(BlockType)
                        elif isinstance(v[0], BlockLanguage):
                            field_types.append(BlockLanguage)
                        else:
                            raise ValueError(f"Invalid metadata type '{type(v[0])}'")
                    else:
                        field_types.append(list)
                else:
                    raise ValueError(f"Invalid metadata type '{type(v)}'")

        return field_names, field_types

    # TODO remove after new parquet files are generated
    @model_validator(mode="before")
    def convert_old_format(cls, values):
        if "block_type" in values:
            block_metadata = values.get("block_metadata", {})
            block_metadata["type"] = values["block_type"]
            values["block_metadata"] = block_metadata
            values.pop("block_type")
        if "language" in values:
            block_metadata = values.get("block_metadata", {})
            block_metadata["language"] = values["language"]
            values["block_metadata"] = block_metadata
            values.pop("language")
        if "id" in values:
            values.pop("id")

        if "section_title" not in values:
            if " > " in values["full_section_title"]:
                values["section_title"] = values["full_section_title"].split(" > ", 1)[
                    -1
                ]
            else:
                values["section_title"] = ""
            values.pop("full_section_title")

        if "content_string" in values:
            values["content"] = values["content_string"]
            values.pop("content_string")

        if "article_title" in values:
            values["document_title"] = values["article_title"]
            values.pop("article_title")

        return values

    @property
    def full_title(self) -> str:
        if not self.section_title:
            return self.document_title
        return self.document_title + " > " + self.section_title

    @property
    def combined_text(self) -> str:
        return self.full_title + " " + self.content

    @property
    def id(self) -> int:
        return abs(hash(self.combined_text))

    @field_validator("content")
    def validate_content(cls, content: str) -> str:
        content = content.strip()
        if not content:
            raise ValueError("`content` cannot be empty or just whitespace")
        return content

    @field_validator("document_title")
    def validate_document_title(cls, document_title: str) -> str:
        document_title = document_title.strip()
        if not document_title:
            raise ValueError("`document_title` cannot be empty or just whitespace")
        document_title = cls.truncate_string(
            document_title, cls.MAX_DOCUMENT_TITLE_LENGTH
        )
        return document_title

    @field_validator("section_title")
    def validate_section_title(cls, section_title: str) -> str:
        """
        Rarely, due to malformed HTML, the full section title can contain the following paragraph too.
        This is a heuristic for Wikipedia Blocks.
        """
        section_title = section_title.strip()
        if "\n" in section_title:
            section_title = section_title.strip().split("\n")[0].strip()
        if len(section_title) > cls.MAX_SECTION_TITLE_LENGTH:
            # If it is still too long, cut from the last " > " to the end
            if section_title.rfind(" > ") > 0:
                section_title = section_title[: section_title.rfind(" > ")]
        if len(section_title) > cls.MAX_SECTION_TITLE_LENGTH:
            # If it is still too long, set it too empty
            section_title = ""

        return section_title

    @field_validator("last_edit_date", mode="before")
    def parse_last_edit_date(cls, last_edit_date):
        if isinstance(last_edit_date, str):
            try:
                return Block.convert_string_to_datetime(last_edit_date)
            except ValueError:
                raise ValueError(
                    f"Invalid date format for `last_edit_date`: '{last_edit_date}'. It should be in the format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ"
                )
        return last_edit_date

    @field_validator("url")
    def validate_url(cls, url: Optional[str]) -> Optional[str]:
        if not url:
            url = None  # Convert empty string to None
            return url
        url = url.strip()
        if url.endswith("#"):
            url = url[:-1]  # TODO remove after the migration
        if "wikipedia.org" in url and "#" in url:
            url = unquote(url)
            url = url.replace(" ", "_")

        if not (
            url.startswith("http") or url.startswith("https") or url.startswith("ftp")
        ):
            raise ValueError(
                "`url` must be either None, or start with 'http', 'https' or 'ftp'"
            )

        return url

    @field_validator("block_metadata", mode="before")
    def validate_block_metadata(cls, value):
        if not value:
            return None  # Convert empty dict to None, because Parquet doesn't support empty dicts
        for k, v in value.items():
            if k in [
                "content",
                "document_title",
                "section_title",
                "last_edit_date",
                "url",
                "num_tokens",
                "block_metadata",
                "metadata",
            ]:
                raise ValueError(f"Invalid metadata field name '{k}'.")
        return value

    # TODO move to utils
    def deduplicate_translations(self) -> None:
        """
        Deduplicates (in English: ...) from each block
        """
        string = (
            self.document_title + " <|> " + self.section_title + " <|> " + self.content
        )
        translation_parenthesis = set(extract_english_translations(string))
        for t in translation_parenthesis:
            string = replace_except_first(string, " " + t, "")

        self.document_title, self.section_title, self.content = tuple(
            string.split(" <|> ", 2)
        )

    @staticmethod
    def truncate_string(string: str, max_length: int) -> str:
        string = string[:max_length]
        last_paren_index = string.rfind(")", max(0, len(string) - 50), len(string))
        if last_paren_index != -1:
            string = string[: last_paren_index + 1]
        return string

    @staticmethod
    def convert_string_to_datetime(date_string: str) -> datetime:
        """
        Convert a string to a datetime object. The string can be in the following formats:
        - YYYY-MM-DDTHH:MM:SSZ
        - YYYY-MM-DDTHH:MM:SS
        - YYYY-MM-DD HH:MM:SS
        - YYYY-MM-DD
        This conversion will ignore the timezone information.
        """
        date_formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
        ]
        for date_format in date_formats:
            try:
                return datetime.strptime(date_string, date_format)
            except ValueError:
                continue
        raise ValueError(
            f"Invalid date format for `last_edit_date`: '{date_string}'. It should be in the format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD HH:MM:SS"
        )

    @staticmethod
    def batch_set_num_tokens(blocks: list["Block"]) -> list[int]:
        for b, num_tokens in zip(
            blocks, batch_get_num_tokens([b.combined_text for b in blocks])
        ):
            b.num_tokens = num_tokens
