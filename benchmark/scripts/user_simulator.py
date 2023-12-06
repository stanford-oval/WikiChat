"""
Uses an LLM to talk to our pipelines. Used for evaluation and model distillation.
"""
import argparse
import logging
from typing import List
import json
import random
from functools import partial
import spacy
from tqdm.contrib.concurrent import thread_map
from tqdm.contrib.logging import logging_redirect_tqdm
import sys

sys.path.insert(0, "./")
from pipelines.dialog_turn import DialogueTurn
from pipelines.chatbot import Chatbot
from pipelines.pipeline_arguments import (
    add_pipeline_arguments,
    check_pipeline_arguments,
)
from pipelines.utils import make_parent_directories
from llm.llm_generate import (
    llm_generate,
    write_prompt_logs_to_file,
)
from llm.global_variables import set_debug_mode, get_total_cost


logging.getLogger("openai").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

spacy_nlp = spacy.load("en_core_web_sm")

user_characteristics = [
    "- Ask interesting follow-up questions when needed, and expand on the chatbot's responses using your life experiences.\n- Never volunteer information, and never correct chatbot's mistakes.",
    # "- You are adversarially stress-testing the chatbot.\n- Never volunteer information, and never correct chatbot's mistakes.",
    # "- You switch to other topics whenever possible.\n- Keep your inputs short.",
    # "- Ask interesting follow-up questions when needed.",
    # "- Ask interesting questions about the recent things that happened about the topic.\n- Never volunteer information, and never correct chatbot's mistakes.",
    # "- Always disagree with what the chatbot says.",
]


def user_simulate_topic(
    dlg_history, topic, user_character, user_engine, user_temperature
):
    return llm_generate(
        template_file="benchmark/prompts/user_with_topic.prompt",
        prompt_parameter_values={
            "dlg": dlg_history,
            "topic": topic,
            "user_character": user_character,
        },
        engine=user_engine,
        max_tokens=60,
        temperature=user_temperature,
        stop_tokens=["\n"],
        top_p=0.5,
        frequency_penalty=0.0,
        presence_penalty=0,
        postprocess=True,
    )


def user_simulate_passage(
    dlg_history, title_and_passage, user_character, user_engine, user_temperature
):
    title, passage = title_and_passage

    return llm_generate(
        template_file="benchmark/prompts/user_with_passage.prompt",
        prompt_parameter_values={
            "dlg": dlg_history,
            "title": title,
            "passage": passage,
            "user_character": user_character,
        },
        engine=user_engine,
        max_tokens=60,
        temperature=user_temperature,
        stop_tokens=["\n"],
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0,
        postprocess=True,
    )


def simulate_dialog(dialog_inputs, chatbot, args):
    """
    Simulate one dialog
    """
    dlg_history: List[DialogueTurn] = []
    user_character = random.choice(user_characteristics)
    if args.mode == "topic":
        user_func = user_simulate_topic
    elif args.mode == "passage":
        user_func = user_simulate_passage

    try:
        for _ in range(args.num_turns):
            user_utterance = user_func(
                dlg_history,
                dialog_inputs,
                user_character,
                args.user_engine,
                args.user_temperature,
            )
            if user_utterance == "":
                logger.error("Simulated user utterance is empty.")
                return None
            # logger.info('simulate user utterance: %s', user_utterance)
            new_dlg_turn = chatbot.generate_next_turn(
                dlg_history, user_utterance, pipeline=args.pipeline
            )
            # logger.info('agent response = %s', new_dlg_turn.agent_utterance)
            dlg_history.append(new_dlg_turn)
        return dlg_history
    except Exception:
        logger.exception(
            "Skipping dialog due to exception. dialog_inputs=%s", str(dialog_inputs)
        )


def repeat_dialog_inputs(dialog_inputs, target_num_dialogs):
    """
    repeats dialog_inputs if we don't have enough of them, truncates if there are too many
    """
    if target_num_dialogs == -1:
        target_num_dialogs = len(dialog_inputs)
    full_rounds = target_num_dialogs // len(dialog_inputs)
    dialog_inputs = (
        dialog_inputs * full_rounds
        + dialog_inputs[: target_num_dialogs % len(dialog_inputs)]
    )
    assert len(dialog_inputs) == target_num_dialogs
    return dialog_inputs


def main(args):
    chatbot = Chatbot(args)
    topics = []
    if args.mode == "topic":
        dialog_inputs = []
        with open(args.input_file) as input_file:
            for line in input_file:
                line = line.strip()
                if len(line) > 0:
                    dialog_inputs.append(line)

        dialog_inputs = repeat_dialog_inputs(dialog_inputs, args.num_dialogs)
        topics = dialog_inputs
    elif args.mode == "passage":
        with open(args.input_file) as input_file:
            dialog_inputs = json.load(input_file)
            user_simulate_passage
        # only include the first sentence of the passage
        dialog_inputs = [
            (title, list(spacy_nlp(passage).sents)[0])
            for title, passage in dialog_inputs.items()
        ]

        dialog_inputs = repeat_dialog_inputs(dialog_inputs, args.num_dialogs)
        topics = [tp[0] for tp in dialog_inputs]
    else:
        raise ValueError("Unknown mode: %s" % args.mode)

    with logging_redirect_tqdm():
        all_dialogs = thread_map(
            partial(simulate_dialog, chatbot=chatbot, args=args),
            dialog_inputs,
            max_workers=args.num_workers,
        )

    make_parent_directories(args.output_file)
    with open(args.output_file, "w") as output_file, open(
        args.output_file.strip("txt") + "log", "w"
    ) as log_file:
        for idx, dlg in enumerate(all_dialogs):
            if not dlg or len(dlg) == 0:
                logger.error('dialog with topic "%s" failed', topics[idx])
                # skip dialogs that failed
                continue
            output_file.write("Topic: " + topics[idx].strip() + "\n")
            for dlg_turn in dlg:
                output_file.write(
                    "User(sim): "
                    + dlg_turn.user_utterance
                    + "\nChatbot("
                    + args.pipeline
                    + "): "
                    + dlg_turn.agent_utterance
                    + "\n"
                )

                turn_log = json.dumps(dlg_turn.log(), indent=4)  # , indent=4
                log_file.write(turn_log)
                log_file.write("\n")

            # dialog is finished
            output_file.write("=====\n")

    print("Total LLM cost: $%.2f" % get_total_cost())


if __name__ == "__main__":
    # text generation arguments
    parser = argparse.ArgumentParser()
    add_pipeline_arguments(parser)
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        default="topic",
        choices=["topic", "passage"],
        help="What type of user simulation to do.",
    )
    parser.add_argument(
        "--user_engine",
        type=str,
        required=True,
        choices=[
            "gpt-35-turbo",
            "text-davinci-003",
            "gpt-4",
            "gpt-4-32k",
        ],
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
        "--num_dialogs",
        type=int,
        required=True,
        help="The number of dialogs to generate. -1 means all topics.",
    )
    parser.add_argument(
        "--num_turns",
        type=int,
        required=True,
        help="The number of turns in each dialog",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=5,
        help="The number of threads to run in parallel.",
    )
    parser.add_argument("--no_logging", action="store_true", help="Disables logging")
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Write prompts inputs/outputs to a file for debugging",
    )

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

    if args.debug_mode:
        set_debug_mode()

    main(args)

    if args.debug_mode:
        write_prompt_logs_to_file()
