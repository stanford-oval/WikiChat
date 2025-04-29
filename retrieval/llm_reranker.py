"""
Listwise LLM reranking is based on RankGPT: https://github.com/sunnweiwei/RankGPT
"""

import json
import math
import sys
import time
from typing import Optional

import numpy as np
from chainlite import chain, llm_generation_chain
from json_repair import repair_json

from utils.logging import logger
from retrieval.retrieval_commons import QueryResult, SearchResultBlock
from chainlite import run_async_in_parallel

sys.path.insert(0, "./")


llm_reranker = None


class ListwiseLLMReranker:
    LLM_RERANKER_SLIDING_WINDOW_SIZE = 20
    LLM_RERANKER_SLIDING_WINDOW_STEP = 10

    def __init__(
        self,
        engine: str,
        prompt_file: str,
        sliding_window_size: int,
        sliding_window_step: int,
    ):
        """
        sliding_window_size (int, optional): The size of the sliding window. If it is None, defaults to LLM_RERANKER_SLIDING_WINDOW_SIZE.
        sliding_window_step (int, optional): The step size for the sliding window. If it is None, defaults to LLM_RERANKER_SLIDING_WINDOW_STEP.
        """
        self.llm_chain = (
            llm_generation_chain(
                template_file=prompt_file,
                engine=engine,
                max_tokens=150,
                output_json=True,
            )
            | reranker_output_to_list
        )
        if sliding_window_size is None:
            sliding_window_size = ListwiseLLMReranker.LLM_RERANKER_SLIDING_WINDOW_SIZE
        if sliding_window_step is None:
            sliding_window_step = ListwiseLLMReranker.LLM_RERANKER_SLIDING_WINDOW_STEP
        self.sliding_window_size = sliding_window_size
        self.sliding_window_step = sliding_window_step

    async def rerank(
        self,
        query: str,
        query_result: QueryResult,
        query_context: Optional[str] = None,
    ) -> QueryResult:
        """
        Re-ranks the search results using a sliding window approach with a large language model (LLM).

        Args:
            query (str): The search query string.
            query_result (QueryResult): The initial search results to be re-ranked.
            query_context (str, optional): The context to provide to the LLM. Defaults to None.

        Returns:
            QueryResult: The re-ranked search results.
        """
        single_search_results = query_result.results
        end_index = len(single_search_results)
        start_index = end_index - self.sliding_window_size
        while True:
            if start_index < 0:
                start_index = 0
            if end_index <= 0:
                break
            reranked_indices = await self._llm_rerank_window(
                query, single_search_results[start_index:end_index], query_context
            )
            num_missing_indices = len(reranked_indices) - end_index + start_index

            reranked_window = [
                single_search_results[start_index + reranked_indices[i]]
                for i in range(len(reranked_indices))
            ]

            single_search_results = (
                single_search_results[:start_index]
                + reranked_window
                + single_search_results[end_index:]
            )

            if start_index == 0:
                break

            step = max(
                self.sliding_window_step, num_missing_indices
            )  # shift more if there are a lot of missing indices
            end_index -= step
            start_index -= step

        assert len(single_search_results) <= len(query_result.results)
        return QueryResult(results=single_search_results)

    async def _llm_rerank_window(
        self,
        query: str,
        single_search_results: list[SearchResultBlock],
        query_context: str = None,
    ) -> list[int]:
        """
        Might return fewer indices than the input if the LLM does not output an index for a search result.
        This happens when the LLM deems some indices to be completely irrelevant to the query.
        """
        reranked_indices = await self.llm_chain.ainvoke(
            {
                "query": query,
                "search_results": single_search_results,
                "query_context": query_context,
            }
        )

        assert len(reranked_indices) <= len(single_search_results)

        return reranked_indices


class PointwiseLLMReranker:
    def __init__(self, engine: str, prompt_file: str):
        self.llm_chain = (
            llm_generation_chain(
                template_file=prompt_file,
                engine=engine,
                max_tokens=1,
                return_top_logprobs=20,
            )
            | llm_output_to_yes_probability
        )

    async def rerank(
        self,
        query: str,
        query_result: QueryResult,
        query_context: str = None,
    ) -> QueryResult:
        scores = await run_async_in_parallel(
            self.llm_chain.ainvoke,
            [
                {"query": query, "search_result": r, "query_context": query_context}
                for r in query_result.results
            ],
            max_concurrency=50,
            timeout=10,
        )
        assert len(scores) == len(query_result.results)
        # sort by scores in descending order
        query_result.results = [
            query_result.results[i] for i in np.argsort(scores)[::-1]
        ]
        return query_result


def initialize_llm_reranker(
    engine: str,
    reranker_type: str = "listwise",
    prompt_file: str = None,
    sliding_window_size: int = None,
    sliding_window_step: int = None,
):
    """
    Initializes the LLM reranker.

    Args:
        engine (str): The name of the LLM engine to use.
        reranker_type (str, optional): The type of reranker to use, either listwise or pointwise. Defaults to "listwise".
        prompt_file (str, optional): If provided, will override the default prompt file. Defaults to None.
    """
    global llm_reranker
    if reranker_type == "listwise":
        llm_reranker = ListwiseLLMReranker(
            engine,
            (
                "pipelines/prompts/rerank_listwise.prompt"
                if prompt_file is None
                else prompt_file
            ),
            sliding_window_size=sliding_window_size,
            sliding_window_step=sliding_window_step,
        )
    else:
        llm_reranker = PointwiseLLMReranker(
            engine,
            (
                "pipelines/prompts/rerank_pointwise.prompt"
                if prompt_file is None
                else prompt_file
            ),
        )


@chain
def llm_output_to_yes_probability(llm_output: tuple) -> float:
    """
    Converts the LLM output to a probability of the answer being 'yes'.

    Args:
        llm_output (tuple): The output from the LLM, including log probabilities.

    Returns:
        float: The probability of the answer being 'yes'.
    """
    T = 5  # temperature scaling

    logprob_outputs = llm_output[1][0]["top_logprobs"]

    # Filter tokens containing "yes" or "no" (excluding "not")
    filtered_logprobs = [
        lpo
        for lpo in logprob_outputs
        if "yes" in lpo["token"].lower()
        or ("no" in lpo["token"].lower() and "not" not in lpo["token"].lower())
    ]

    # Extract logprobs and apply temperature scaling
    logprobs = np.array([lpo["logprob"] for lpo in filtered_logprobs])
    scaled_probs = np.exp(logprobs / T)
    normalized_probs = scaled_probs / np.sum(scaled_probs)

    # Calculate yes and no probabilities
    yes_prob = sum(
        prob
        for prob, lpo in zip(normalized_probs, filtered_logprobs)
        if "yes" in lpo["token"].lower()
    )
    no_prob = sum(
        prob
        for prob, lpo in zip(normalized_probs, filtered_logprobs)
        if "no" in lpo["token"].lower()
    )

    # Ensure the probabilities sum to 1
    assert math.isclose(yes_prob + no_prob, 1, rel_tol=1e-3)

    return yes_prob


@chain
def reranker_output_to_list(llm_output: str) -> list[int]:
    try:
        ranked_indices = json.loads(llm_output)["ranked_passage_ids"]
    except (KeyError, json.JSONDecodeError):
        logger.error(f"Error parsing LLM reranking output: {llm_output}")

        ranked_indices = repair_json(llm_output, return_objects=True)
        if "ranked_passage_ids" in ranked_indices:
            ranked_indices = ranked_indices["ranked_passage_ids"]
        ranked_indices = []
    ranked_indices = [i - 1 for i in ranked_indices]  # 1-indexed to 0-indexed

    # deduplicate
    new_ranked_indices = []
    for idx in ranked_indices:
        if idx not in new_ranked_indices and idx >= 0:
            new_ranked_indices.append(idx)

    return new_ranked_indices


async def batch_llm_rerank(
    queries: list[str],
    query_results: list[QueryResult],
    query_contexts: Optional[list[str]] = None,
    batch_size: int = 10,
) -> list[QueryResult]:
    start_time = time.time()
    if not query_contexts:
        query_contexts = [None] * len(queries)

    if len(queries) == 1:
        return [
            await llm_reranker.rerank(queries[0], query_results[0], query_contexts[0])
        ]

    # Wrapper function to run rerank in parallel
    async def _run_llm_rerank(_input):
        query, query_result, query_context = _input
        return await llm_reranker.rerank(
            query=query,
            query_result=query_result,
            query_context=query_context,
        )

    reranked_results = await run_async_in_parallel(
        _run_llm_rerank,
        [
            (query, query_result, query_context)
            for query, query_result, query_context in zip(
                queries, query_results, query_contexts
            )
        ],
        max_concurrency=batch_size,
        desc="Reranking",
    )

    assert len(reranked_results) == len(query_results)
    logger.info(
        f"Reranked {len(queries)} queries in {time.time() - start_time:.2f} seconds"
    )
    return reranked_results
