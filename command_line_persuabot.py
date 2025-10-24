"""
Chat with PersuaBot via command line

This script allows users to interact with PersuaBot, a persuasive chatbot
based on the paper "Zero-shot Persuasive Chatbots with LLM-Generated
Strategies and Information Retrieval" (Furumai et al., 2024).

PersuaBot uses a two-module architecture:
- Question Handling Module (QHM): Retrieves information based on user requests
- Strategy Maintenance Module (SMM): Generates persuasive responses with fact-checking
"""

import argparse
import asyncio
import json

from chainlite import get_total_cost, write_prompt_logs_to_file
from utils.logging import logger

from pipelines.persuabot import create_persuabot_chain
from pipelines.persuabot_dialogue_state import PersuaBotTurn
from pipelines.pipeline_arguments import (
    add_pipeline_arguments,
    check_pipeline_arguments,
)
from pipelines.utils import input_user, print_chatbot_response


async def main(args):
    # Create the PersuaBot chatbot and initialize the dialogue state
    persuabot, dialogue_state = create_persuabot_chain(args)

    logger.info("=" * 80)
    logger.info("PersuaBot - Persuasive Chatbot with Fact-Checking")
    logger.info("=" * 80)
    logger.info(f"Domain: {args.persuasion_domain}")
    logger.info(f"Goal: {args.target_goal}")
    logger.info("=" * 80)

    # Main loop to process user inputs
    while True:
        try:
            # Get input from the user via the command line
            new_user_utterance = input_user()
        except EOFError:
            break

        # Check if the input matches any of the pre-defined quit commands
        if new_user_utterance in args.quit_commands:
            break

        # Append the new user utterance as a dialogue turn
        dialogue_state.turns.append(PersuaBotTurn(user_utterance=new_user_utterance))

        # Invoke PersuaBot with the updated dialogue state
        await persuabot.ainvoke(dialogue_state)

        # Print the chatbot's response
        print_chatbot_response(dialogue_state.current_turn)

        # Optionally print strategy information in debug mode
        if args.debug:
            print("\n[DEBUG] Strategy Breakdown:")
            for i, section in enumerate(dialogue_state.current_turn.smm_sections, 1):
                print(f"  Section {i}: {section.strategy}")
                print(f"    Factual: {section.is_factual}")
                if section.retrieval_query:
                    print(f"    Retrieved using: {section.retrieval_query}")
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PersuaBot: Persuasive chatbot with fact-checking"
    )
    add_pipeline_arguments(parser)

    # PersuaBot-specific arguments
    parser.add_argument(
        "--persuasion_domain",
        type=str,
        default="general",
        help="The domain for persuasion (e.g., 'donation', 'health', 'recommendation')",
    )
    parser.add_argument(
        "--target_goal",
        type=str,
        default="have a helpful and persuasive conversation",
        help="The specific persuasion goal",
    )
    parser.add_argument(
        "--do_fact_checking",
        action="store_true",
        default=True,
        help="Whether to perform fact-checking on generated content",
    )
    parser.add_argument(
        "--no_fact_checking",
        dest="do_fact_checking",
        action="store_false",
        help="Disable fact-checking",
    )
    parser.add_argument(
        "--do_strategy_decomposition",
        action="store_true",
        default=True,
        help="Whether to decompose response into strategy sections",
    )
    parser.add_argument(
        "--no_strategy_decomposition",
        dest="do_strategy_decomposition",
        action="store_false",
        help="Disable strategy decomposition",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Show strategy breakdown for each response",
    )
    parser.add_argument(
        "--quit_commands",
        type=str,
        default=["quit", "q", "Exit", "exit"],
        help="The conversation will continue until this string is typed in.",
    )

    args = parser.parse_args()
    check_pipeline_arguments(args)

    logger.info(
        f"Starting PersuaBot with arguments: {json.dumps(vars(args), ensure_ascii=False, indent=2)}"
    )

    # Run the main asynchronous function to start the chatbot session
    asyncio.run(main(args))

    # Log the total cost of LLM usage at the end of the session
    logger.info(f"Total LLM cost: ${get_total_cost():.2f}")

    # Write the inputs and outputs of each individual prompt to a file
    write_prompt_logs_to_file()
