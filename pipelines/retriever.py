import json
import logging
import re
from urllib.parse import quote

import langcodes
import numpy as np
import requests
from langchain_core.runnables import chain

from pipelines.chatbot_config import RerankingMethodEnum

from .langchain_helpers import stage_prompt
from .utils import extract_year

logger = logging.getLogger(__name__)


class RetrievalResult:
    content: str
    title: str
    full_section_title: str
    language: str
    block_type: str
    last_edit_date: str
    score: float
    probability: float

    content_summary: list[
        str
    ]  # (optional) the output of the summarization stage, if any

    def __init__(
        self,
        content: str,
        title: str,
        full_section_title: str,
        language: str,
        block_type: str,
        last_edit_date: str,
        score: float,
        probability: float,
    ):
        self.content = content
        self.title = title
        self.full_section_title = full_section_title
        self.language = language
        self.block_type = block_type
        self.last_edit_date = last_edit_date
        self.score = score
        self.probability = probability
        self.content_summary = []

    @staticmethod
    def retrieval_results_to_list(results: dict):
        outputs = []
        for (
            passage,
            title,
            full_section_title,
            block_type,
            language,
            prob,
            score,
            last_edit_date,
        ) in zip(
            results["text"],
            results["title"],
            results["full_section_title"],
            results["block_type"],
            results["language"],
            results["prob"],
            results["score"],
            results["last_edit_date"],
        ):
            if block_type not in ["text", "table", "infobox", "list"]:
                logger.warning(
                    f"Found invalid block type {str(block_type)} for {passage}."
                )
                continue
            outputs.append(
                RetrievalResult(
                    passage,
                    title,
                    full_section_title,
                    language,
                    block_type,
                    last_edit_date,
                    score,
                    prob,
                )
            )

        return outputs

    def to_markdown(self):
        language = self.language
        content = self.content
        full_section_title = self.full_section_title
        content = content.replace("•", ", ")  # for infoboxes that contain a list

        article_title = full_section_title
        if language != "en":
            content = re.sub("\(in English: ([^)]*)\)", "_(in English: \\1)_", content)
            full_section_title = re.sub(" \(in English:[^)]*\)", "", full_section_title)
        if ">" in full_section_title:
            # extract the article title without the section titles
            title_parts = [p.strip() for p in full_section_title.split(">")]
            article_title = title_parts[0]
            full_section_title = title_parts[0] + "#" + title_parts[-1]
        ret = f"*[{article_title} ({langcodes.Language.get(language).display_name()})](https://{language}.wikipedia.org/wiki/{quote(full_section_title)})*\n{content}"
        if self.content_summary:
            ret += "\n\n**Summary:** \n" + "\n".join(
                [f"- {s}" for s in self.content_summary]
            )
        return ret


async def llm_rerank_window(
    query, retrieval_results: list[RetrievalResult], retrieval_config
):
    reranking_prompt_output = await stage_prompt(retrieval_config).ainvoke(
        {
            "query": query,
            "retrieval_results": retrieval_results,
        }
    )

    reranked_indices = process_reranking_output(reranking_prompt_output)
    reranked_indices = [
        i for i in reranked_indices if i < len(retrieval_results)
    ]  # remove numbers that are outside the range
    logger.debug("reranked_indices = %s", str(reranked_indices))
    return reranked_indices, reranking_prompt_output


def process_reranking_output(response):
    new_response = ""
    for character in response:
        if not character.isdigit():
            new_response += " "
        else:
            new_response += character
    response = new_response.strip()
    response = [int(x) - 1 for x in response.split()]

    # deduplicate
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)

    return new_response


@chain
async def rerank(_input, retrieval_config):
    """
    Reranks the list of results of a single query
    The size of the output is the exact size as the input
    """
    llm_reranker_sliding_window_size = 20
    llm_reranker_sliding_window_step = 10

    query_text = _input["query"].text
    query_time = _input["query"].time
    retrieval_results = _input["query"].retrieval_results

    if retrieval_config.reranking_method == RerankingMethodEnum.none:
        pass
    elif retrieval_config.reranking_method == RerankingMethodEnum.date:
        if query_time == "none":
            pass
        else:
            all_passage_dates = []
            for result in retrieval_results:
                passage_years = extract_year(
                    title=result.full_section_title, content=result.content
                )
                all_passage_dates.append(passage_years)
            if query_time == "recent":
                sort_fn = lambda x: max(
                    x[1] if len(x[1]) > 0 else [0]
                )  # sort based on the latest year mentioned in the paragraph, demoting paragraphs that don't mention a year
            else:
                # query_time is a year
                try:
                    query_year = int(query_time)
                    sort_fn = lambda x: x[1].count(
                        query_year
                    )  # boost the passages that have a matching year with the query, the more they mention the date the more we boost
                except ValueError as e:
                    # raise ValueError('query_time should be none, recent or an integer.')
                    query_year = "none"
                    sort_fn = lambda x: 1  # no change in ordering

            # do the sorting
            retrieval_results, all_passage_dates = list(
                zip(
                    *sorted(
                        zip(
                            retrieval_results,
                            all_passage_dates,
                        ),
                        reverse=True,
                        key=sort_fn,
                    )
                )
            )
    elif retrieval_config.reranking_method == RerankingMethodEnum.llm:
        # LLM reranking is based on RankGPT: https://github.com/sunnweiwei/RankGPT
        end_index = len(retrieval_results)
        start_index = end_index - llm_reranker_sliding_window_size
        original_rank = list(range(len(retrieval_results)))
        while True:
            if start_index < 0:
                start_index = 0
            reranked_indices, reranking_prompt_output = await llm_rerank_window(
                query_text, retrieval_results[start_index:end_index], retrieval_config
            )

            if len(reranked_indices) != (end_index - start_index):
                missing_indices = set(range(end_index - start_index))
                for found_index in reranked_indices:
                    if found_index in missing_indices:
                        missing_indices.remove(found_index)

                logger.warning(
                    "LLM reranking should return the same number of outputs as inputs. Adding missing indices: %s. Prompt output was %s",
                    str(missing_indices),
                    reranking_prompt_output,
                )

                # TODO instead of adding missing indices, shift everything and continue
                # Add missing indices to the end so that we don't crash
                # This is reasonable assuming that if the LLM did not output an index, it probably was not that relevant to the query to begin with
                reranked_indices = reranked_indices + list(missing_indices)

                assert len(reranked_indices) == (end_index - start_index)

            retrieval_results[start_index:end_index] = [
                retrieval_results[start_index + reranked_indices[i]]
                for i in range(len(reranked_indices))
            ]
            original_rank[start_index:end_index] = [
                original_rank[start_index + reranked_indices[i]]
                for i in range(len(reranked_indices))
            ]

            if start_index == 0:
                break
            end_index -= llm_reranker_sliding_window_step
            start_index -= llm_reranker_sliding_window_step
    else:
        raise ValueError("Unsupported reranking method.")

    # choose the top `post_reranking_num` results
    retrieval_results = retrieval_results[: retrieval_config.post_reranking_num]

    _input["query"].retrieval_results = retrieval_results

    return _input["query"]


def _try_to_enforce_block_type_limits(
    results: list[RetrievalResult], block_type_limits: dict, target_num: int
) -> list[RetrievalResult]:
    block_type_limits_copy = {}
    for k in ["table", "text", "list", "infobox"]:
        if k not in block_type_limits:
            block_type_limits_copy[k] = 1e6  # Infinity
        else:
            block_type_limits_copy[k] = block_type_limits[k]

    num_to_remove = len(results) - target_num
    outputs = []
    for r in reversed(results):
        if num_to_remove > 0:
            if block_type_limits_copy[r.block_type] == 0:
                num_to_remove -= 1
                continue  # skip this block
            else:
                block_type_limits_copy[r.block_type] -= 1
        outputs.append(r)
    return list(reversed(outputs))[:target_num]


@chain
def batch_retrieve(_input, retriever_endpoint: str, pre_reranking_num: int):
    top_p = _input.get("top_p", 1.0)
    block_type_limits = _input.get("block_type_limits", {})
    evi_num = pre_reranking_num

    # convert query to list if it is not already
    assert isinstance(_input["query"], list)
    input_queries = _input["query"]

    request_payload = {
        "query": [q.text for q in input_queries],
        "num_blocks": (
            evi_num + 20 if len(block_type_limits) > 0 else evi_num
        ),  # Retrieve more if we need to limit block types
    }
    response = requests.post(
        retriever_endpoint,
        json=request_payload,
    )

    if response.status_code == 429:
        raise Exception(
            "You have reached your rate limit for the retrieval server. Please wait and try later, or host your own retriever."
        )
    if response.status_code != 200:
        logger.error(
            "Error encountered when sending this request to retriever: %s",
            json.dumps(request_payload, ensure_ascii=False, indent=2),
        )
        raise Exception("Retriever Error: %s" % str(response))

    results = response.json()
    assert len(results) == len(input_queries)

    for input_query, result in zip(input_queries, results):
        result = RetrievalResult.retrieval_results_to_list(result)
        result = _try_to_enforce_block_type_limits(result, block_type_limits, evi_num)
        assert (
            len(result) <= evi_num
        ), f"{len(result)}"  # sometimes the retriever can return fewer results than requested

        top_p_cut_off = np.cumsum([r.probability for r in result]) > top_p
        if not np.any(top_p_cut_off):
            # even if we include everything, we don't get to top_p
            top_p_cut_off = len(result)
        else:
            top_p_cut_off = np.argmax(top_p_cut_off) + 1
        result = result[:top_p_cut_off]

        # choose the top `pre_reranking_num` paragraphs
        result = result[:pre_reranking_num]

        input_query.retrieval_results = result
    return [{"query": q} for q in input_queries]


@chain
def retrieve(_input, retriever_endpoint: str, pre_reranking_num: int):
    """
    Retrieve by sending requests to the retriever server
    """
    query_text = _input["query"].text
    assert isinstance(query_text, str)
    top_p = _input.get("top_p", 1.0)
    block_type_limits = _input.get("block_type_limits", {})

    evi_num = pre_reranking_num

    response = requests.post(
        retriever_endpoint,
        json={
            "query": [query_text],
            "num_blocks": (
                evi_num + 20 if len(block_type_limits) > 0 else evi_num
            ),  # Retrieve more if we need to limit block types
        },
    )

    if response.status_code == 429:
        raise Exception(
            "You have reached your rate limit for the retrieval server. Please wait and try later, or host your own retriever."
        )
    if response.status_code != 200:
        raise Exception("Retriever Error: %s" % str(response))

    results = response.json()
    assert len(results) == 1  # one result because we have only one query
    result = results[0]

    result = RetrievalResult.retrieval_results_to_list(result)
    result = _try_to_enforce_block_type_limits(result, block_type_limits, evi_num)
    assert (
        len(result) <= evi_num
    ), f"{len(result)}"  # sometimes the retriever can return fewer results than requested

    top_p_cut_off = np.cumsum([r.probability for r in result]) > top_p
    if not np.any(top_p_cut_off):
        # even if we include everything, we don't get to top_p
        top_p_cut_off = len(result)
    else:
        top_p_cut_off = np.argmax(top_p_cut_off) + 1
    result = result[:top_p_cut_off]

    # choose the top `pre_reranking_num` paragraphs
    result = result[:pre_reranking_num]

    _input["query"].retrieval_results = result
    return _input["query"]
