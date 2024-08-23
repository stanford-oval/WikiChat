import argparse
import sys
from typing import Callable

import pytest

sys.path.insert(0, "./")
from backend_server import chat_profiles_dict
from tasks.defaults import CHATBOT_DEFAULT_CONFIG
from pipelines.chatbot import create_chain, run_one_turn
from pipelines.dialogue_state import DialogueState
from pipelines.pipeline_arguments import (
    add_pipeline_arguments,
    check_pipeline_arguments,
)
from pipelines.utils import dict_to_command_line

test_user_utterances = [
    "Hi",  # a turn that doesn't need retrieval
    "Tell me about Yōko Ogawa",  # a turn that needs retrieval
    "what is her latest book?",  # a turn that depends on the last turn
    "list all Blue Eye Samurai episodes",  # a turn that requires accessing Wikipedia tables
]


async def pipeline_test_template(
    extra_arguments: dict,
    pipeline_specific_assertions: Callable[[DialogueState, str], None],
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
    initial_search_queries = []

    for turn_id in range(3):
        new_user_utterance = test_user_utterances[turn_id]
        new_agent_utterance, dialogue_state = await run_one_turn(
            chatbot, dialogue_state, new_user_utterance
        )

        # print("=" * 40 + f"\nTurn {turn_id+1}\n" + state_to_string(dialogue_state))

        # Specific assertions for each pipeline
        pipeline_specific_assertions(dialogue_state, new_agent_utterance)

        # Same for all pipelines
        assert dialogue_state["new_user_utterance"] == new_user_utterance
        assert new_agent_utterance
        assert len(dialogue_state["dialogue_history"]) == turn_id + 1
        if turn_id > 0:
            previous_turn = dialogue_state["dialogue_history"][
                -2
            ]  # -1 is the current turn
            assert previous_turn.agent_utterance == output_agent_utterances[-1]
            assert previous_turn.user_utterance == test_user_utterances[turn_id - 1]
            assert (
                previous_turn.initial_search_query
                == initial_search_queries[turn_id - 1]
            )

        output_agent_utterances.append(new_agent_utterance)
        initial_search_queries.append(dialogue_state["initial_search_query"])


@pytest.mark.asyncio(scope="session")
async def test_generate_pipeline():
    def pipeline_specific_assertions(
        dialogue_state: DialogueState, new_agent_utterance: str
    ):
        assert not dialogue_state["draft_stage_output"]
        assert dialogue_state["refine_stage_output"]
        assert new_agent_utterance == dialogue_state["refine_stage_output"]

    await pipeline_test_template(
        {
            "pipeline": "generate",
            "do_refine": True,
            "fuse_claim_splitting": False,
            "generation_prompt": "generate.prompt",
            "refinement_prompt": "refine_w_feedback.prompt",
        },
        pipeline_specific_assertions,
    )


@pytest.mark.asyncio(scope="session")
async def test_retrieve_and_generate_pipeline():
    def pipeline_specific_assertions(
        dialogue_state: DialogueState, new_agent_utterance: str
    ):
        assert dialogue_state["draft_stage_output"]
        assert dialogue_state["refine_stage_output"]
        assert new_agent_utterance == dialogue_state["refine_stage_output"]
        assert not dialogue_state["generate_stage_output"]
        assert not dialogue_state["generation_claims"]

    await pipeline_test_template(
        {
            "pipeline": "retrieve_and_generate",
            "do_refine": True,
            "fuse_claim_splitting": False,
            "generation_prompt": "generate.prompt",
            "refinement_prompt": "refine_w_feedback.prompt",
        },
        pipeline_specific_assertions,
    )


@pytest.mark.asyncio(scope="session")
async def test_distilled_pipeline():
    def pipeline_specific_assertions(
        dialogue_state: DialogueState, new_agent_utterance: str
    ):
        assert dialogue_state["draft_stage_output"]
        assert not dialogue_state["refine_stage_output"]
        assert new_agent_utterance == dialogue_state["draft_stage_output"]
        assert dialogue_state["generate_stage_output"]

    await pipeline_test_template(
        {
            "pipeline": "early_combine",
            "do_refine": False,
            "engine": "gpt-35-turbo-finetuned",
            "fuse_claim_splitting": True,
            "retrieval_reranking_method": "date",
            "retrieval_reranking_num": 9,
            "generation_prompt": "generate.prompt",
            "refinement_prompt": "refine_w_feedback.prompt",
        },
        pipeline_specific_assertions,
    )


@pytest.mark.asyncio(scope="session")
async def test_early_combine_pipeline():
    def pipeline_specific_assertions(
        dialogue_state: DialogueState, new_agent_utterance: str
    ):
        assert dialogue_state["draft_stage_output"]
        assert not dialogue_state["refine_stage_output"]
        assert new_agent_utterance == dialogue_state["draft_stage_output"]
        assert dialogue_state["generate_stage_output"]
        # assert not dialogue_state["generation_claims"]
        assert dialogue_state["generate_stage_output"]

    await pipeline_test_template(
        {"pipeline": "early_combine", "do_refine": False},
        pipeline_specific_assertions,
    )


@pytest.mark.asyncio(scope="session")
async def test_chatbot_profiles():
    """
    We just run the chatbot profiles used in the demo to make sure they don't crash
    """

    def pipeline_specific_assertions(
        dialogue_state: DialogueState, new_agent_utterance: str
    ):
        pass

    for chat_profile in chat_profiles_dict:
        await pipeline_test_template(
            chat_profiles_dict[chat_profile]["overwritten_parameters"],
            pipeline_specific_assertions,
        )
