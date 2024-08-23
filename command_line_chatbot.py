"""
Chat with the chatbot via command line
"""

import argparse
import asyncio
import logging
import readline  # enables keyboard arrows when typing in the terminal

from chainlite import get_total_cost, write_prompt_logs_to_file
from pygments import highlight
from pygments.formatters.terminal256 import Terminal256Formatter
from pygments.lexers.web import JsonLexer

from pipelines.chatbot import create_chain, run_one_turn
from pipelines.dialogue_state import state_to_string
from pipelines.pipeline_arguments import (
    add_pipeline_arguments,
    check_pipeline_arguments,
)
from pipelines.utils import input_user, print_chatbot

logger = logging.getLogger(__name__)


async def main(args):
    chatbot, dialogue_state = create_chain(args)

    while True:
        try:
            new_user_utterance = input_user()
        except EOFError:
            # stop the chatbot
            break

        # check for special commands
        if new_user_utterance in args.quit_commands:
            # stop the chatbot
            break

        new_agent_utterance, dialogue_state = await run_one_turn(
            chatbot, dialogue_state, new_user_utterance
        )

        turn_log = state_to_string(dialogue_state)
        colorful_turn_log = highlight(
            turn_log,
            lexer=JsonLexer(),
            formatter=Terminal256Formatter(style="bw"),
        )
        print("Turn log:", colorful_turn_log)

        print_chatbot(new_agent_utterance)


if __name__ == "__main__":
    # text generation arguments
    parser = argparse.ArgumentParser()
    add_pipeline_arguments(parser)
    parser.add_argument("--no_logging", action="store_true", help="Disables logging")
    parser.add_argument(
        "--quit_commands",
        type=str,
        default=["quit", "q", "Exit", "exit"],
        help="The conversation will continue until this string is typed in.",
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

    asyncio.run(main(args))
    print("Total LLM cost: $%.2f" % get_total_cost())
    write_prompt_logs_to_file()
