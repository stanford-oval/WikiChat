"""
Chat with the chatbot via command line

This script allows users to interact with a chatbot using command line
input. It sets up the chatbot chain, listens for user input, checks if
the user wants to quit, and prints the chatbot's response.
"""

import argparse
import asyncio
import json

# Import cost utilities and logging functions
from chainlite import get_total_cost, write_prompt_logs_to_file
from utils.logging import logger

# Import functions and classes to create the chatbot chain and manage dialogue state
from pipelines.chatbot import create_chain
from pipelines.dialogue_state import DialogueTurn
from pipelines.pipeline_arguments import (
    add_pipeline_arguments,
    check_pipeline_arguments,
)
from pipelines.utils import input_user, print_chatbot_response


async def main(args):
    # Create the chatbot and initialize the dialogue state based on provided args
    chatbot, dialogue_state = create_chain(args)

    # Main loop to process user inputs
    while True:
        try:
            # Get input from the user via the command line
            new_user_utterance = input_user()
        except EOFError:
            # If an EOFError is encountered (e.g., user sends EOF signal), break the loop to stop the chatbot
            break

        # Check if the input matches any of the pre-defined quit commands
        if new_user_utterance in args.quit_commands:
            # Stop the chatbot if a quit command is given
            break

        # Append the new user utterance as a dialogue turn to the conversation history
        dialogue_state.turns.append(DialogueTurn(user_utterance=new_user_utterance))
        # Asynchronously invoke the chatbot with the updated dialogue state
        await chatbot.ainvoke(dialogue_state)

        # Uncomment the following line to print the entire dialogue state for debugging purposes
        # print("Dialogue State:", dialogue_state, end="\n\n")

        # Print the chatbot's response from the current dialogue turn
        print_chatbot_response(dialogue_state.current_turn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_pipeline_arguments(parser)
    parser.add_argument(
        "--quit_commands",
        type=str,
        default=["quit", "q", "Exit", "exit"],
        help="The conversation will continue until this string is typed in.",
    )

    args = parser.parse_args()
    check_pipeline_arguments(args)

    logger.info(
        f"Starting chatbot with arguments: {json.dumps(vars(args), ensure_ascii=False, indent=2)}"
    )

    # Run the main asynchronous function to start the chatbot session
    asyncio.run(main(args))
    # Log the total cost of LLM usage at the end of the session
    logger.info(f"Total LLM cost: ${get_total_cost():.2f}")
    # Write the inputs and outputs of each individual prompt to a file
    write_prompt_logs_to_file()
