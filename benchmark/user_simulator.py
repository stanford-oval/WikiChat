"""
Uses an LLM to talk to our chatbot. Used for evaluation and model distillation.
"""

import argparse
import asyncio
import json
import logging
import random
import sys

import spacy
from tqdm import tqdm

sys.path.insert(0, "./")
from chainlite import get_total_cost, llm_generation_chain, write_prompt_logs_to_file

from pipelines.chatbot import create_chain, run_one_turn
from pipelines.dialogue_state import DialogueTurn, state_to_string
from pipelines.pipeline_arguments import (
    add_pipeline_arguments,
    all_configured_engines,
    check_pipeline_arguments,
)
from pipelines.utils import get_logger, make_parent_directories

logger = get_logger(__name__)

spacy_nlp = spacy.load("en_core_web_sm")

user_characteristics = [
    "- Ask interesting follow-up questions when needed, and expand on the chatbot's responses using your life experiences.\n- Never volunteer information, and never correct chatbot's mistakes.",
    "- You are adversarially stress-testing the chatbot.\n- Never volunteer information, and never correct chatbot's mistakes.",
    "- You switch to other topics whenever possible.\n- Keep your inputs short.",
    "- Ask interesting questions about the recent things that happened about the topic.\n- Never volunteer information, and never correct chatbot's mistakes.",
    "- Always disagree with what the chatbot says.",
]


def remove_prefix(utterance: str):
    if utterance.startswith("User:"):
        utterance = utterance[len("User:") :].strip()

    return utterance


def user_simulation_chain(user_engine: str, user_temperature: float, language: str):
    return (
        llm_generation_chain(
            template_file="benchmark/prompts/user_with_passage.prompt",
            engine=user_engine,
            max_tokens=60,
            temperature=user_temperature,
            stop_tokens=["\n"],
            postprocess=False,
            bind_prompt_values={"language": language},
        )
        | remove_prefix
    )


async def simulate_dialog(dialogue_inputs, args) -> list[DialogueTurn]:
    """
    Simulate one dialog
    """
    user_character = random.choice(user_characteristics)
    chatbot, dialogue_state = create_chain(args)

    user_chain = user_simulation_chain(
        args.user_engine, args.user_temperature, args.language
    )
    try:
        for _ in range(args.num_turns):
            if args.mode == "topic":
                new_user_utterance = await user_chain.ainvoke(
                    {
                        "dlg": dialogue_state["dialogue_history"],
                        "user_character": user_character,
                        "topic": dialogue_inputs,
                    }
                )
            elif args.mode == "passage":
                new_user_utterance = await user_chain.ainvoke(
                    {
                        "dlg": dialogue_state["dialogue_history"],
                        "user_character": user_character,
                        "title": dialogue_inputs[0],
                        "passage": dialogue_inputs[1],
                    }
                )
            elif args.mode == "multihop":
                new_user_utterance = await user_chain.ainvoke(
                    {
                        "dlg": dialogue_state["dialogue_history"],
                        "user_character": user_character,
                        "title_1": dialogue_inputs["title_1"],
                        "paragraph_1": dialogue_inputs["paragraph_1"],
                        "title_2": dialogue_inputs["title_2"],
                        "paragraph_2": dialogue_inputs["paragraph_2"],
                    }
                )
            if new_user_utterance == "":
                logger.error("Simulated user utterance is empty.")
                return None
            new_agent_utterance, dialogue_state = await run_one_turn(
                chatbot, dialogue_state, new_user_utterance
            )

    except Exception:
        logger.exception(
            "Skipping dialog due to exception. dialogue_inputs=%s", str(dialogue_inputs)
        )
    return dialogue_state


def repeat_dialogue_inputs(dialogue_inputs, target_num_dialogues):
    """
    repeats dialogue_inputs if we don't have enough of them, truncates if there are too many
    """
    if target_num_dialogues == -1:
        target_num_dialogues = len(dialogue_inputs)
    full_rounds = target_num_dialogues // len(dialogue_inputs)
    dialogue_inputs = (
        dialogue_inputs * full_rounds
        + dialogue_inputs[: target_num_dialogues % len(dialogue_inputs)]
    )
    assert len(dialogue_inputs) == target_num_dialogues
    return dialogue_inputs


def main(args):
    topics = []
    if args.mode == "topic":
        dialogue_inputs = []
        with open(args.input_file) as input_file:
            for line in input_file:
                line = line.strip()
                if len(line) > 0:
                    dialogue_inputs.append(line)

        dialogue_inputs = repeat_dialogue_inputs(dialogue_inputs, args.num_dialogues)
        topics = dialogue_inputs
    elif args.mode == "passage":
        with open(args.input_file) as input_file:
            dialogue_inputs = json.load(input_file)
        # only include the first sentence of the passage
        dialogue_inputs = [
            (title, list(spacy_nlp(passage).sents)[0])
            for title, passage in dialogue_inputs.items()
        ]

        dialogue_inputs = repeat_dialogue_inputs(dialogue_inputs, args.num_dialogues)
        topics = [tp[0] for tp in dialogue_inputs]
    elif args.mode == "multihop":
        with open(args.input_file) as input_file:
            dialogue_inputs = json.load(input_file)
            dialogue_inputs = repeat_dialogue_inputs(
                dialogue_inputs, args.num_dialogues
            )
            topics = [m["title_1"] + " and " + m["title_2"] for m in dialogue_inputs]
    else:
        raise ValueError("Unknown mode: %s" % args.mode)

    all_dialogues = []
    for di in tqdm(dialogue_inputs, desc="Dialogues"):
        dialogue_state = asyncio.run(simulate_dialog(di, args=args))
        all_dialogues.append(dialogue_state)

    make_parent_directories(args.output_file)
    with open(args.output_file, "w") as output_file:
        for idx, dlg in enumerate(all_dialogues):
            if not dlg["dialogue_history"]:
                logger.error('dialog with topic "%s" failed', topics[idx])
                # skip dialogs that failed
                continue
            output_file.write("Topic: " + topics[idx].strip() + "\n")
            for dlg_turn in dlg["dialogue_history"]:
                output_file.write(
                    "User: "
                    + dlg_turn.user_utterance
                    + "\nChatbot: "
                    + dlg_turn.agent_utterance
                    + "\n\n"
                )

                # turn_log = state_to_string(dlg)
                # log_file.write(turn_log)
                # log_file.write("\n")

            # dialog is finished
            output_file.write("=====\n")

    write_prompt_logs_to_file(args.output_file.strip("txt") + "log")
    logger.info("Total LLM cost: $%.2f" % get_total_cost())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_pipeline_arguments(parser)
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        default="topic",
        choices=["topic", "passage", "multihop"],
        help="What type of user simulation to do.",
    )
    parser.add_argument(
        "--user_engine",
        type=str,
        required=True,
        choices=all_configured_engines,
        help="Which LLM to use for user simulator.",
    )
    parser.add_argument(
        "--user_temperature",
        type=float,
        default=0.9,
        help="The temperature to use for the user simulator.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Where to read conversation topics, or passages from.",
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Where to write the outputs"
    )
    parser.add_argument(
        "--num_dialogues",
        type=int,
        required=True,
        help="The number of dialogues to generate. -1 means all topics.",
    )
    parser.add_argument(
        "--num_turns",
        type=int,
        required=True,
        help="The number of turns in each dialog",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Which language the user should speak in. E.g. `en` for English",
    )
    parser.add_argument("--no_logging", action="store_true", help="Disables logging")

    args = parser.parse_args()
    check_pipeline_arguments(args)

    if args.no_logging:
        logging.basicConfig(
            level=logging.ERROR, format=" %(name)s : %(levelname)-8s : %(message)s"
        )
    else:
        logging.basicConfig(
            level=logging.INFO, format=" %(name)s : %(levelname)-8s : %(message)s"
        )

    main(args)
