import json
import re

from chainlite import chain, llm_generation_chain, register_prompt_constants
from chainlite.llm_output import extract_tag_from_llm_output, lines_to_list
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import END, START, StateGraph

from corpora import corpus_id_to_corpus_object
from utils.logging import logger
from pipelines.dialogue_state import ChatbotConfig, DialogueState
from retrieval.retriever_api import retrieve_via_api

register_prompt_constants({"chatbot_name": "WikiChat"})


@chain
async def query_stage(state):
    query_chain = (
        llm_generation_chain("query.prompt", engine=state.config.engine, max_tokens=200)
        | extract_tag_from_llm_output.bind(tags="search_query")  # type: ignore
        | lines_to_list
    )
    search_prompt_output = await query_chain.ainvoke(
        {"dlg": state, "llm_corpus_description": state.config.llm_corpus_description}
    )
    search_prompt_output = [q for q in search_prompt_output if q and q != "None"]

    if not search_prompt_output:
        logger.info("No search needed.")
        state.current_turn.search_query = []
        return
    logger.info(
        f"Search queries: {json.dumps(search_prompt_output, ensure_ascii=False, indent=2)}"
    )

    state.current_turn.search_query = search_prompt_output


@chain
async def search_stage(state):
    query = state.current_turn.search_query
    config = state.config
    if query:
        try:
            search_result = retrieve_via_api(
                state.current_turn.search_query,
                retriever_endpoint=config.retriever_endpoint,
                do_reranking=config.do_reranking,
                pre_reranking_num=config.query_pre_reranking_num,
                post_reranking_num=config.query_post_reranking_num,
            )
            assert len(search_result) == len(query)
        except Exception as e:
            logger.error(f"Error in search: {e}")
            search_result = []
        state.current_turn.search_results = search_result


@chain
async def generate_stage(state):
    llm_claims_chain = (
        llm_generation_chain(
            "generate_split_claims.prompt", engine=state.config.engine, max_tokens=2000
        )
        | lines_to_list
    )

    llm_claims = await llm_claims_chain.ainvoke({"dlg": state})
    if len(llm_claims) == 0 or (len(llm_claims) == 1 and llm_claims[0] == "None"):
        return

    state.current_turn.llm_claims = llm_claims


@chain
async def llm_claim_search_stage(state):
    queries = state.current_turn.llm_claims
    config = state.config
    if queries:
        search_result = retrieve_via_api(
            queries=queries,
            retriever_endpoint=config.retriever_endpoint,
            do_reranking=config.do_reranking,
            pre_reranking_num=config.claim_pre_reranking_num,
            post_reranking_num=config.claim_post_reranking_num,
        )
        assert len(search_result) == len(queries)
        state.current_turn.llm_claim_search_results = search_result


@chain
async def filter_information_stage(state):
    filter_chain = (
        llm_generation_chain(
            "filter_irrelevant_info.prompt", engine=state.config.engine, max_tokens=2000
        )
        | extract_tag_from_llm_output.bind(tags="relevant_content")  # type: ignore
        | lines_to_list
    )
    filtered_info = await filter_chain.abatch(
        [
            {"dlg": state, "result": result}
            for result in state.current_turn.all_single_search_results
        ]
    )
    for result, filtered_result in zip(
        state.current_turn.all_single_search_results, filtered_info
    ):
        # make a deepcopy of result
        if len(filtered_result) == 0 or (
            len(filtered_result) == 1 and filtered_result[0] == "None"
        ):
            continue
        result = result.copy()
        result.summary = filtered_result
        state.current_turn.filtered_search_results.append(result)


@chain
async def draft_stage(state):
    draft_chain = llm_generation_chain(
        "draft_w_citation.prompt", engine=state.config.engine, max_tokens=2000
    ) | extract_tag_from_llm_output.bind(tags="response")  # type: ignore
    draft_output = await draft_chain.ainvoke({"dlg": state})
    state.current_turn.draft_stage_output = draft_output


@chain
async def shift_references(state):
    agent_utterance = state.current_turn.agent_utterance
    references = state.current_turn.filtered_search_results
    # extract all citations in [1], [2], [3] format
    citations = re.findall(r"\[\d+\]", agent_utterance)
    if len(citations) == 0:
        return
    cited_reference_indices = []
    for citation in citations:
        citation_index = int(citation[1:-1]) - 1
        if 0 <= citation_index < len(references):
            cited_reference_indices.append(citation_index)
    cited_reference_indices = list(set(cited_reference_indices))
    cited_reference_indices.sort()

    reference_map = {}
    for i, index in enumerate(cited_reference_indices):
        reference_map[index] = i

    for i, index in enumerate(cited_reference_indices):
        agent_utterance = agent_utterance.replace(f"[{index + 1}]", f"[{i + 1}]")
    references = [references[i] for i in cited_reference_indices]

    state.current_turn.agent_utterance = agent_utterance
    state.current_turn.filtered_search_results = references


@chain
async def refine_stage(state):
    if not state.config.do_refine:
        state.current_turn.agent_utterance = state.current_turn.draft_stage_output
        return
    refine_chain = llm_generation_chain(
        "refine.prompt", engine=state.config.engine, max_tokens=2000
    ) | extract_tag_from_llm_output.bind(tags="revised_response")  # type: ignore
    refine_output = await refine_chain.ainvoke(
        {"dlg": state, "utterance_to_refine": state.current_turn.draft_stage_output}
    )
    state.current_turn.agent_utterance = refine_output


def create_chain(args) -> tuple[CompiledStateGraph, DialogueState]:
    llm_corpus_description = corpus_id_to_corpus_object(
        args.corpus_id
    ).llm_corpus_description
    initial_state = DialogueState(
        turns=[],
        config=ChatbotConfig(
            engine=args.engine,
            do_refine=args.do_refine,
            llm_corpus_description=llm_corpus_description,
            retriever_endpoint=args.retriever_endpoint,
            do_reranking=args.do_reranking,
            query_pre_reranking_num=args.query_pre_reranking_num,
            query_post_reranking_num=args.query_post_reranking_num,
            claim_pre_reranking_num=args.claim_pre_reranking_num,
            claim_post_reranking_num=args.claim_post_reranking_num,
        ),
    )

    graph = StateGraph(DialogueState)

    # nodes
    graph.add_node("query_stage", query_stage)
    graph.add_node("search_stage", search_stage)
    graph.add_node("generate_stage", generate_stage)
    graph.add_node("llm_claim_search_stage", llm_claim_search_stage)
    graph.add_node("filter_information_stage", filter_information_stage)
    graph.add_node("draft_stage", draft_stage)
    graph.add_node("refine_stage", refine_stage)
    graph.add_node("shift_references", shift_references)

    # edges
    graph.add_edge(START, "query_stage")
    graph.add_edge("query_stage", "search_stage")

    graph.add_edge(START, "generate_stage")
    graph.add_edge("generate_stage", "llm_claim_search_stage")

    graph.add_edge("search_stage", "filter_information_stage")
    graph.add_edge("llm_claim_search_stage", "filter_information_stage")

    graph.add_edge("filter_information_stage", "draft_stage")
    graph.add_edge("draft_stage", "refine_stage")
    graph.add_edge("refine_stage", "shift_references")
    graph.add_edge("shift_references", END)

    runnable = graph.compile()

    # runnable.get_graph().print_ascii()

    return runnable, initial_state
