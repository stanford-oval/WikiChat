from pydantic import BaseModel, Field


class ChatStarter(BaseModel):
    """
    A chat starter is a suggestion for the user to start a chat with the chatbot.
    It is displayed in the front-end as a button or a link that the user can click on to start a chat.
    """

    display_label: str = Field(
        ...,
        description="Label for the chat starter. This will be displayed in the front-end before the user clicks on the chat starter",
    )
    icon_path: str = Field(..., description="Path to the icon for the chat starter")
    chat_message: str = Field(
        ...,
        description="Message content of the chat starter. This will be sent to the chatbot when the user clicks on the chat starter",
    )


class Corpus(BaseModel):
    name: str = Field(..., description="Name of the corpus")
    corpus_id: str = Field(
        ..., description="Index ID of the corpus in the vector database"
    )
    icon_path: str = Field(..., description="Path to the icon for the corpus")
    llm_corpus_description: str = Field(
        ..., description="Description of the corpus for LLM prompts"
    )
    human_description_markdown: str = Field(
        ..., description="Human-readable description of the corpus"
    )
    chat_starters: list[ChatStarter] = Field(
        ..., description="List of chat starters associated with the corpus"
    )
    overwritten_parameters: dict = Field(
        ...,
        description="Corpus-specific chatbot parameters to overwrite defaults.",
    )


# Corpus objects

wikipedia_corpus = Corpus(
    name="Wikipedia in 25 Languages",
    corpus_id="wikipedia_20250320",
    icon_path="/public/img/wikipedia.png",
    human_description_markdown="""Includes the full text, table and infobox data from Wikipedia in the following languages:
- [English](https://en.wikipedia.org/)
- [French](https://fr.wikipedia.org/)
- [German](https://de.wikipedia.org/)
- [Spanish](https://es.wikipedia.org/)
- [Japanese](https://ja.wikipedia.org/)
- [Russian](https://ru.wikipedia.org/)
- [Portuguese](https://pt.wikipedia.org/)
- [Chinese](https://zh.wikipedia.org/)
- [Italian](https://it.wikipedia.org/)
- [Arabic](https://ar.wikipedia.org/)
- [Persian](https://fa.wikipedia.org/)
- [Polish](https://pl.wikipedia.org/)
- [Dutch](https://nl.wikipedia.org/)
- [Ukrainian](https://uk.wikipedia.org/)
- [Hebrew](https://he.wikipedia.org/)
- [Indonesian](https://id.wikipedia.org/)
- [Turkish](https://tr.wikipedia.org/)
- [Czech](https://cs.wikipedia.org/)
- [Swedish](https://sv.wikipedia.org/)
- [Korean](https://ko.wikipedia.org/)
- [Finnish](https://fi.wikipedia.org/)
- [Vietnamese](https://vi.wikipedia.org/)
- [Hungarian](https://hu.wikipedia.org/)
- [Catalan](https://ca.wikipedia.org/)
- [Thai](https://th.wikipedia.org/).""",
    chat_starters=[
        ChatStarter(
            display_label="Haruki Murakami",
            chat_message="Tell me about Haruki Murakami.",
            icon_path="https://upload.wikimedia.org/wikipedia/commons/7/75/HarukiMurakami.png",
        ),
        ChatStarter(
            display_label="The Oscars",
            chat_message="Who won the 2024 Oscar for Best Picture?",
            icon_path="https://www.paradiseawards.com/mm5/graphics/00000001/3/PA-T38MW.png",
        ),
    ],
    llm_corpus_description="Wikipedia in 25 languages",
    overwritten_parameters={
        "engine": "gpt-4o-mini",
        "do_refine": False,
    },
)

# semantic_scholar_corpus = Corpus(
#     name="Academic Papers",
#     corpus_id="semantic_scholar",
#     icon_path="/public/img/s2_logo.png",
#     human_description_markdown="""A collection of academic papers spanning multiple disciplines, including computer science, physics, mathematics, medicine, and more.
#     This corpus includes peer-reviewed journal articles, conference papers, preprints, and technical reports from [Semantic Scholar](https://www.semanticscholar.org/).
#     """,
#     chat_starters=[
#         ChatStarter(
#             display_label="LLMs",
#             chat_message="How does LLaMA-3 compare against GPT-4o?",
#             icon_path="",  # No icon provided in the original data
#         ),
#         ChatStarter(
#             display_label="COVID-19",
#             chat_message="Tell me about the latest research on COVID-19",
#             icon_path="",  # No icon provided in the original data
#         ),
#     ],
#     llm_corpus_description="Semantic Scholar, a corpus of scientific literature in the fields of computer science, physics, math, medicine etc.",
#     overwritten_parameters={
#         "engine": "gpt-4o-mini",
#         "do_refine": False,
#     },
# )

the_african_times_corpus = Corpus(
    name="The African Times",
    corpus_id="the_african_times",
    icon_path="/public/img/the_african_times.jpg",
    human_description_markdown="The African Times was a newspaper published in London during the late 19th century. Its articles primarily consisted of correspondence from the global African diaspora.",
    chat_starters=[
        ChatStarter(
            display_label="Women in West Africa",
            chat_message="Tell me about the role of women in West Africa in the 1880s.",
            icon_path="https://upload.wikimedia.org/wikipedia/commons/1/15/Africa-countries-WAFU-UFOA.png",
        ),
        ChatStarter(
            display_label="Steamships",
            chat_message="What is the history of steamships?",
            icon_path="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dc/StateLibQld_1_133053_Agamemnon_%28ship%29.jpg/420px-StateLibQld_1_133053_Agamemnon_%28ship%29.jpg",
        ),
    ],
    llm_corpus_description="The African Times, a newspaper published in the late 19th century by the global African diaspora.",
    overwritten_parameters={
        "engine": "gpt-4o-mini",
        "do_refine": False,
    },
)

general_history_of_africa_corpus = Corpus(
    name="General History of Africa",
    corpus_id="general_history_of_africa_volumes_VI_and_VII",
    icon_path="/public/img/general_history_of_africa.png",
    human_description_markdown="""This corpus includes volumes VI and VII of the UNESCO book series "General History of Africa".
Includes the following two volumes:
- [General history of Africa, VI: Africa in the nineteenth century until the 1880s](https://unesdoc.unesco.org/ark:/48223/pf0000184295)
- [General history of Africa, VII: Africa under colonial domination, 1880-1935](https://unesdoc.unesco.org/ark:/48223/pf0000184296)
""",
    llm_corpus_description="General History of Africa, volumes VI and VII",
    chat_starters=[],
    overwritten_parameters={
        "engine": "gpt-4o-mini",
        "do_refine": False,
    },
)


all_corpus_objects = [
    wikipedia_corpus,
    # semantic_scholar_corpus,
    the_african_times_corpus,
    general_history_of_africa_corpus,
]

# add retriever_endpoint to all corpora
for corpus in all_corpus_objects:
    corpus.overwritten_parameters["retriever_endpoint"] = (
        f"https://search.genie.stanford.edu/{corpus.corpus_id}"
    )


def corpus_name_to_corpus_object(corpus_name: str) -> Corpus:
    for corpus in all_corpus_objects:
        if corpus.name == corpus_name:
            return corpus
    raise ValueError(f"Corpus with name '{corpus_name}' not found in corpora.")


def corpus_id_to_corpus_object(corpus_id: str) -> Corpus:
    for corpus in all_corpus_objects:
        if corpus.corpus_id == corpus_id:
            return corpus
    raise ValueError(f"Corpus with id '{corpus_id}' not found in corpora.")


def get_public_indices() -> tuple[list[str], list[str], list[str]]:
    return (
        [corpus.corpus_id for corpus in all_corpus_objects],
        [corpus.name for corpus in all_corpus_objects],
        [corpus.human_description_markdown for corpus in all_corpus_objects],
    )
