import logging

import requests
from tqdm import trange

from retrieval.retrieval_commons import QueryResult, SearchResultBlock

logger = logging.getLogger(__name__)


def retrieve_via_api(
    queries: str | list[str],
    retriever_endpoint: str,
    do_reranking: bool,
    pre_reranking_num: int,
    post_reranking_num: int,
    languages: str | list[str] = [],
    batch_size: int = 10,
    additional_search_filters: list[dict] = [],
) -> list[QueryResult]:
    """
    Retrieve search results from a retriever API.
    Args:
        queries (str | list[str]): A single query or a list of queries to be sent to the retriever.
        retriever_endpoint (str): The endpoint URL of the retriever API.
        do_reranking (bool): Flag indicating whether to perform reranking on the results.
        pre_reranking_num (int): Number of blocks to consider before reranking.
        post_reranking_num (int): Number of blocks to return after reranking.
        languages (str | list[str], optional): A single language or a list of languages to filter the search results. Defaults to [], meaning search in all languages.
        batch_size (int, optional): Number of queries to send in each batch. Defaults to 10.
        additional_search_filters (list[dict]): Additional search filters. Defaults to [], meaning no addition filters apart from language filters.
    Returns:
        list[QueryResult]: A list of QueryResult objects containing the search results for each query.
    Raises:
        Exception: If the rate limit is reached or if there is an error with the retriever API request.
    """

    if not isinstance(queries, list):
        queries = [queries]
    if not isinstance(languages, list):
        languages = [languages]

    ret = []
    for i in trange(
        0,
        len(queries),
        batch_size,
        disable=len(queries) <= batch_size,
        desc="Retrieving",
    ):
        batch_queries = queries[i : i + batch_size]
        response = requests.post(
            retriever_endpoint,
            json={
                "query": batch_queries,
                "rerank": do_reranking,
                "num_blocks_to_rerank": pre_reranking_num,
                "num_blocks": post_reranking_num,
                "search_filters": [
                    {"field_name": "language", "filter_type": "eq", "field_value": lang}
                    for lang in languages
                ]
                + additional_search_filters,
            },
        )

        if response.status_code == 429:
            raise Exception(
                "You have reached your rate limit for the retrieval server. Please wait and try later, or host your own retriever."
            )
        if response.status_code != 200:
            logger.error(
                f"Error encountered when sending this request to retriever: {response.text}"
            )

        results = response.json()
        assert len(results) == len(batch_queries), (
            f"Number of queries and results do not match. Length of queries: {len(batch_queries)}, Length of results: {len(results)}"
        )
        for j in range(len(batch_queries)):
            search_results = [
                SearchResultBlock(**r) for r in results[j]["results"]
            ]  # convert to pydantic object
            ret.append(QueryResult(results=search_results))

    return ret
