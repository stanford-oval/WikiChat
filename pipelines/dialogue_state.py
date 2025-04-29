import re
from typing import Optional

import tiktoken
from pydantic import BaseModel, Field

from retrieval.retrieval_commons import QueryResult, SearchResultBlock


tokenizer = tiktoken.encoding_for_model("gpt-4o")


def jaccard_similarity(tokens1: list[int], tokens2: list[int]) -> float:
    """Compute Jaccard similarity between two tokenized strings"""
    set1 = set(tokens1)
    set2 = set(tokens2)

    intersection = set1.intersection(set2)
    union = set1.union(set2)

    # Return the Jaccard similarity
    return len(intersection) / len(union) if len(union) > 0 else 0


def deduplicate_search_results(
    results: list[SearchResultBlock],
) -> list[SearchResultBlock]:
    tokenized_contents = tokenizer.encode_batch([r.content for r in results])
    deduplicated_indices = []
    for j in range(len(results)):
        similarity = 0
        for idx in deduplicated_indices:
            similarity = max(
                similarity,
                jaccard_similarity(tokenized_contents[j], tokenized_contents[idx]),
            )
            if similarity > 0.8:
                break
        if similarity < 0.8:
            deduplicated_indices.append(j)
    deduplicated_results = [results[i] for i in deduplicated_indices]
    # logger.info(
    #     f"Before deduplication: {len(results)} results, after deduplication: {len(deduplicated_results)} results"
    # )
    assert len(deduplicated_results) <= len(results)

    return deduplicated_results


class DialogueTurn(BaseModel):
    user_utterance: Optional[str] = None
    search_query: list[str] = []
    search_results: list[QueryResult] = []
    llm_claims: list[str] = []
    llm_claim_search_results: list[QueryResult] = []
    filtered_search_results: list[SearchResultBlock] = []
    draft_stage_output: Optional[str] = None
    agent_utterance: Optional[str] = None

    # @model_validator(mode="after")
    # def check_claims_and_results_length(cls, values):
    #     # Access fields directly from the instance (values)
    #     if len(values.llm_claims) != len(values.llm_claim_search_results):
    #         raise ValueError(
    #             "The number of claims must match the number of search results."
    #         )

    #     return values

    @property
    def all_single_search_results(self) -> list[SearchResultBlock]:
        results = []
        for r in self.search_results + self.llm_claim_search_results:
            results.extend(r.results)

        return deduplicate_search_results(results)


class ChatbotConfig(BaseModel):
    """A configuration class for setting up a chatbot with various parameters."""

    engine: str = Field(..., description="The LLM engine to use.")
    do_refine: bool = Field(..., description="Whether to refine the final response.")
    llm_corpus_description: str = Field(
        ...,
        description="The name and description of corpus to search for information, e.g. Multilingual Wikipedia.",
    )
    retriever_endpoint: str = Field(
        ..., description="The endpoint to send retrieval requests to."
    )
    do_reranking: bool = Field(
        ..., description="Whether we should rerank the search results."
    )
    query_pre_reranking_num: int = Field(
        ...,
        description="Number of passages to retrieve before reranking. Will have no effect if `do_reranking` is False.",
    )
    query_post_reranking_num: int = Field(
        ...,
        description="Number of passages to retrieve when searching for information.",
    )
    claim_pre_reranking_num: int = Field(
        ...,
        description="Number of evidences to retrieve per LLM claim before reranking. Will have no effect if `do_reranking` is False.",
    )
    claim_post_reranking_num: int = Field(
        ..., description="Number of evidences to retrieve per claim."
    )


class DialogueState(BaseModel):
    """A class to represent the state of a dialogue with a chatbot."""

    config: ChatbotConfig
    turns: list[DialogueTurn]

    @property
    def current_turn(self) -> DialogueTurn:
        return self.turns[-1]

    def history(self, num_turns: int) -> str:
        history = ""
        for t in self.turns[:-1][-num_turns:]:
            if t.user_utterance:
                history += "User: " + t.user_utterance + "\n"
            if t.agent_utterance:
                history += "Chatbot: " + t.agent_utterance + "\n"
        history += (
            "User: " + self.current_turn.user_utterance
        )  # current turn's user utterance

        # remove all citations, otherwise, chatbot might try to recite them
        citations = re.findall(r"\[\d+\]", history)
        for citation in citations:
            history = history.replace(citation, "")
        return history
