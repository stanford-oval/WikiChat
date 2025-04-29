import argparse
import re
import sys
from typing import Callable

import pytest
from chainlite import write_prompt_logs_to_file

sys.path.insert(0, "./")
from corpora import all_corpus_objects
from pipelines.chatbot import create_chain
from pipelines.dialogue_state import DialogueTurn
from pipelines.pipeline_arguments import (
    add_pipeline_arguments,
    check_pipeline_arguments,
)
from pipelines.utils import dict_to_command_line
from tasks.defaults import CHATBOT_DEFAULT_CONFIG

test_user_utterances = [
    "Hi",  # a turn that doesn't need retrieval
    "Tell me about Jane Austen",  # a turn that needs retrieval
    "what was her first book?",  # a turn that depends on the last turn
    "list all Blue Eye Samurai episodes",  # a turn that requires accessing Wikipedia tables
]


async def pipeline_test_template(
    extra_arguments: dict,
    pipeline_specific_assertions: Callable[[DialogueTurn, int], None] = None,
):
    parser = argparse.ArgumentParser()
    add_pipeline_arguments(parser)

    # set parameters
    args = parser.parse_args(
        dict_to_command_line(CHATBOT_DEFAULT_CONFIG, extra_arguments)
    )
    check_pipeline_arguments(args)

    chatbot, dialogue_state = create_chain(args)

    output_agent_utterances = []
    filtered_search_queries = []

    for turn_id in range(3):
        new_user_utterance = test_user_utterances[turn_id]
        dialogue_state.turns.append(DialogueTurn(user_utterance=new_user_utterance))
        await chatbot.ainvoke(dialogue_state)

        current_turn = dialogue_state.current_turn
        # Specific assertions for each pipeline
        if pipeline_specific_assertions:
            pipeline_specific_assertions(current_turn, turn_id)

        # Same for all pipelines
        assert current_turn.user_utterance == new_user_utterance
        assert current_turn.agent_utterance
        assert len(dialogue_state.turns) == turn_id + 1
        assert len(current_turn.search_query) == len(current_turn.search_results)
        assert len(current_turn.llm_claims) == len(
            current_turn.llm_claim_search_results
        )

        # test that the dialogue history is carried over correctly
        if turn_id > 0:
            previous_turn = dialogue_state.turns[-2]  # -1 is the current turn
            assert previous_turn.agent_utterance == output_agent_utterances[-1]
            assert previous_turn.user_utterance == test_user_utterances[turn_id - 1]
            assert (
                previous_turn.filtered_search_results
                == filtered_search_queries[turn_id - 1]
            )

        output_agent_utterances.append(current_turn.agent_utterance)
        filtered_search_queries.append(current_turn.filtered_search_results)


@pytest.mark.asyncio(scope="session")
async def test_no_refine_pipeline():
    def pipeline_specific_assertions(dialogue_turn: DialogueTurn, turn_id: int) -> None:
        assert dialogue_turn.draft_stage_output
        assert dialogue_turn.agent_utterance

        draft_citations = re.findall(r"\[\d+\]", dialogue_turn.agent_utterance)
        agent_utterance_citations = re.findall(
            r"\[\d+\]", dialogue_turn.agent_utterance
        )
        assert len(draft_citations) == len(agent_utterance_citations)

        draft_without_citations = re.sub(
            r"\[\d+\]", "", dialogue_turn.draft_stage_output
        )
        agent_utterance_without_citations = re.sub(
            r"\[\d+\]", "", dialogue_turn.agent_utterance
        )
        assert (
            draft_without_citations == agent_utterance_without_citations
        )  # because we don't have a refine stage in this test

        # We put these tests in a custom test function because other tests might use non-Wikipedia search corpora, for which the LLM might decide no to search at all
        if turn_id == 0:
            # turn 0 doesn't need retrieval, all other turns do
            # We have already checked that the length of query and results are the same, so we only need to check one here
            assert not dialogue_turn.search_query
            assert not dialogue_turn.llm_claims
            assert not dialogue_turn.filtered_search_results
        else:
            assert dialogue_turn.search_query
            assert dialogue_turn.llm_claims
            assert dialogue_turn.filtered_search_results

    await pipeline_test_template(
        {"do_refine": False},
        pipeline_specific_assertions,
    )


@pytest.mark.asyncio(scope="session")
async def test_chatbot_profiles():
    """
    We just run the chatbot profiles used in the demo to make sure they don't crash
    """
    for co in all_corpus_objects:
        await pipeline_test_template(
            co.overwritten_parameters,
        )


@pytest.fixture(scope="session", autouse=True)
def run_after_tests():
    yield
    write_prompt_logs_to_file()
