"""
PersuaBot: Zero-shot Persuasive Chatbot with LLM-Generated Strategies and Information Retrieval

Based on the paper:
"Zero-shot Persuasive Chatbots with LLM-Generated Strategies and Information Retrieval"
Furumai et al., 2024 (arXiv:2407.03585)

This implementation follows the paper's two-module architecture:
1. Question Handling Module (QHM): Retrieves information based on user requests
2. Strategy Maintenance Module (SMM): Generates persuasive responses with fact-checking
"""

import json

from chainlite import chain, llm_generation_chain, register_prompt_constants
from chainlite.llm_output import extract_tag_from_llm_output, lines_to_list
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import END, START, StateGraph

from corpora import corpus_id_to_corpus_object
from utils.logging import logger
from pipelines.persuabot_dialogue_state import (
    PersuaBotConfig,
    PersuaBotState,
    PersuaBotSection,
)
from retrieval.retriever_api import retrieve_via_api

register_prompt_constants({"chatbot_name": "PersuaBot"})


# ============================================================================
# Question Handling Module (QHM)
# ============================================================================


@chain
async def qhm_query_stage(state: PersuaBotState):
    """Generate search queries based on user's explicit questions."""
    query_chain = (
        llm_generation_chain(
            "persuabot/qhm_query.prompt", engine=state.config.engine, max_tokens=200
        )
        | extract_tag_from_llm_output.bind(tags="search_query")  # type: ignore
        | lines_to_list
    )
    search_queries = await query_chain.ainvoke(
        {"dlg": state, "llm_corpus_description": state.config.llm_corpus_description}
    )
    search_queries = [q for q in search_queries if q and q != "None"]

    if not search_queries:
        logger.info("QHM: No search needed for user question.")
        state.current_turn.qhm_query = []
        return

    logger.info(f"QHM queries: {json.dumps(search_queries, ensure_ascii=False)}")
    state.current_turn.qhm_query = search_queries


@chain
async def qhm_search_stage(state: PersuaBotState):
    """Retrieve information based on user queries."""
    queries = state.current_turn.qhm_query
    if not queries:
        return

    try:
        search_results = retrieve_via_api(
            queries,
            retriever_endpoint=state.config.retriever_endpoint,
            do_reranking=state.config.do_reranking,
            pre_reranking_num=state.config.query_pre_reranking_num,
            post_reranking_num=state.config.query_post_reranking_num,
        )
        state.current_turn.qhm_search_results = search_results
        logger.info(f"QHM: Retrieved {len(search_results)} result sets")
    except Exception as e:
        logger.error(f"QHM search error: {e}")
        state.current_turn.qhm_search_results = []


# ============================================================================
# Strategy Maintenance Module (SMM)
# ============================================================================


@chain
async def smm_generate_response(state: PersuaBotState):
    """Generate initial persuasive response with strategies."""
    generate_chain = (
        llm_generation_chain(
            "persuabot/smm_generate.prompt", engine=state.config.engine, max_tokens=2000
        )
        | extract_tag_from_llm_output.bind(tags="response")  # type: ignore
    )

    response = await generate_chain.ainvoke(
        {
            "dlg": state,
            "persuasion_domain": state.config.persuasion_domain,
            "target_goal": state.config.target_goal,
        }
    )

    if not response or response == "None":
        logger.warning("SMM: Generated empty response")
        return

    state.current_turn.smm_initial_response = response
    logger.info(f"SMM: Generated initial response ({len(response)} chars)")


@chain
async def smm_decompose_strategies(state: PersuaBotState):
    """Decompose response into sections with distinct strategies."""
    if not state.config.do_strategy_decomposition:
        # Skip decomposition, treat entire response as one section
        if state.current_turn.smm_initial_response:
            section = PersuaBotSection(
                content=state.current_turn.smm_initial_response,
                strategy="general persuasion",
            )
            state.current_turn.smm_sections = [section]
        return

    decompose_chain = llm_generation_chain(
        "persuabot/smm_decompose.prompt", engine=state.config.engine, max_tokens=2000
    )

    decomposed = await decompose_chain.ainvoke(
        {"dlg": state, "response": state.current_turn.smm_initial_response}
    )

    # Parse the decomposed output into sections
    # Expected format: <section strategy="...">content</section>
    sections = []
    import re

    pattern = r'<section strategy="([^"]*)">(.*?)</section>'
    matches = re.findall(pattern, decomposed, re.DOTALL)

    for strategy, content in matches:
        section = PersuaBotSection(
            content=content.strip(), strategy=strategy.strip()
        )
        sections.append(section)

    if not sections:
        # Fallback: treat entire response as one section
        logger.warning("SMM: Could not parse sections, using entire response")
        section = PersuaBotSection(
            content=state.current_turn.smm_initial_response or "",
            strategy="general persuasion",
        )
        sections = [section]

    state.current_turn.smm_sections = sections
    logger.info(f"SMM: Decomposed into {len(sections)} strategy sections")


@chain
async def smm_fact_check(state: PersuaBotState):
    """Fact-check each section of the response."""
    if not state.config.do_fact_checking:
        # Mark all sections as factual
        for section in state.current_turn.smm_sections:
            section.is_factual = True
        return

    fact_check_chain = (
        llm_generation_chain(
            "persuabot/smm_fact_check.prompt",
            engine=state.config.engine,
            max_tokens=500,
        )
        | extract_tag_from_llm_output.bind(tags="verdict")  # type: ignore
    )

    # Fact-check each section in parallel
    verdicts = await fact_check_chain.abatch(
        [{"dlg": state, "section": section} for section in state.current_turn.smm_sections]
    )

    factual_count = 0
    for section, verdict in zip(state.current_turn.smm_sections, verdicts):
        section.is_factual = verdict.strip().lower() in ["true", "factual", "verified"]
        if section.is_factual:
            factual_count += 1

    logger.info(
        f"SMM: Fact-checking complete - {factual_count}/{len(state.current_turn.smm_sections)} sections verified"
    )


@chain
async def smm_retrieve_facts(state: PersuaBotState):
    """For unsubstantiated sections, generate queries and retrieve facts."""
    unverified_sections = [
        s for s in state.current_turn.smm_sections if not s.is_factual
    ]

    if not unverified_sections:
        logger.info("SMM: All sections are factual, no retrieval needed")
        return

    # Generate retrieval queries for unverified sections
    query_chain = (
        llm_generation_chain(
            "persuabot/smm_generate_query.prompt",
            engine=state.config.engine,
            max_tokens=200,
        )
        | extract_tag_from_llm_output.bind(tags="query")  # type: ignore
    )

    queries = await query_chain.abatch(
        [{"dlg": state, "section": section} for section in unverified_sections]
    )

    # Clean queries
    queries = [q.strip() for q in queries if q and q.strip() != "None"]

    if not queries:
        logger.info("SMM: No retrieval queries generated")
        return

    logger.info(f"SMM: Generated {len(queries)} retrieval queries")

    # Retrieve facts
    try:
        search_results = retrieve_via_api(
            queries,
            retriever_endpoint=state.config.retriever_endpoint,
            do_reranking=state.config.do_reranking,
            pre_reranking_num=state.config.claim_pre_reranking_num,
            post_reranking_num=state.config.claim_post_reranking_num,
        )

        # Attach results to sections
        for section, query, results in zip(unverified_sections, queries, search_results):
            section.retrieval_query = query
            section.retrieved_facts = results.results

        logger.info(f"SMM: Retrieved facts for {len(queries)} sections")
    except Exception as e:
        logger.error(f"SMM: Error retrieving facts: {e}")


@chain
async def smm_merge_results(state: PersuaBotState):
    """Merge retrieved facts with strategies to create final response."""
    merge_chain = (
        llm_generation_chain(
            "persuabot/smm_merge.prompt", engine=state.config.engine, max_tokens=2000
        )
        | extract_tag_from_llm_output.bind(tags="response")  # type: ignore
    )

    # Prepare all sections with their facts
    final_response = await merge_chain.ainvoke(
        {
            "dlg": state,
            "sections": state.current_turn.smm_sections,
            "qhm_results": state.current_turn.qhm_search_results,
            "persuasion_domain": state.config.persuasion_domain,
            "target_goal": state.config.target_goal,
        }
    )

    state.current_turn.smm_final_response = final_response
    logger.info(f"SMM: Generated final response ({len(final_response)} chars)")


# ============================================================================
# Graph Construction
# ============================================================================


def create_persuabot_chain(args) -> tuple[CompiledStateGraph, PersuaBotState]:
    """Create the PersuaBot pipeline graph."""

    llm_corpus_description = corpus_id_to_corpus_object(
        args.corpus_id
    ).llm_corpus_description

    initial_state = PersuaBotState(
        turns=[],
        config=PersuaBotConfig(
            engine=args.engine,
            do_refine=False,  # PersuaBot doesn't use refine stage
            llm_corpus_description=llm_corpus_description,
            retriever_endpoint=args.retriever_endpoint,
            do_reranking=args.do_reranking,
            query_pre_reranking_num=args.query_pre_reranking_num,
            query_post_reranking_num=args.query_post_reranking_num,
            claim_pre_reranking_num=args.claim_pre_reranking_num,
            claim_post_reranking_num=args.claim_post_reranking_num,
            # PersuaBot-specific config
            persuasion_domain=getattr(args, "persuasion_domain", "general"),
            target_goal=getattr(
                args, "target_goal", "have a helpful and persuasive conversation"
            ),
            do_fact_checking=getattr(args, "do_fact_checking", True),
            do_strategy_decomposition=getattr(args, "do_strategy_decomposition", True),
        ),
    )

    graph = StateGraph(PersuaBotState)

    # Add QHM nodes
    graph.add_node("qhm_query_stage", qhm_query_stage)
    graph.add_node("qhm_search_stage", qhm_search_stage)

    # Add SMM nodes
    graph.add_node("smm_generate_response", smm_generate_response)
    graph.add_node("smm_decompose_strategies", smm_decompose_strategies)
    graph.add_node("smm_fact_check", smm_fact_check)
    graph.add_node("smm_retrieve_facts", smm_retrieve_facts)
    graph.add_node("smm_merge_results", smm_merge_results)

    # QHM pipeline (runs in parallel with SMM initially)
    graph.add_edge(START, "qhm_query_stage")
    graph.add_edge("qhm_query_stage", "qhm_search_stage")

    # SMM pipeline
    graph.add_edge(START, "smm_generate_response")
    graph.add_edge("smm_generate_response", "smm_decompose_strategies")
    graph.add_edge("smm_decompose_strategies", "smm_fact_check")
    graph.add_edge("smm_fact_check", "smm_retrieve_facts")

    # Both QHM and SMM converge at merge stage
    graph.add_edge("qhm_search_stage", "smm_merge_results")
    graph.add_edge("smm_retrieve_facts", "smm_merge_results")

    graph.add_edge("smm_merge_results", END)

    runnable = graph.compile()

    return runnable, initial_state
