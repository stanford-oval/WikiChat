import os
import pathlib

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from pipelines.dialogue_state import DialogueTurn

console = Console()  # rich


def print_chatbot_response(turn: DialogueTurn):
    # Create a Markdown formatted text for the chatbot's utterance
    panel_content = []

    # Format the citations with better styling
    for index, result in enumerate(turn.filtered_search_results, start=1):
        # Citation title in bold and yellow
        reference_title = Text(f"[{index}] {result.full_title}", style="bold yellow")
        # Add a hyperlink to the title
        reference_title.stylize(f"link {result.url}")
        reference_content = Text(result.content, style="dim")

        # Add the citation title and content to the panel
        panel_content.append(reference_title)
        panel_content.append(reference_content)

        if index < len(turn.filtered_search_results):
            panel_content.append(Text(""))

    # Print the chatbot's response
    console.print(
        Panel(Group(*panel_content), title="References", border_style="yellow")
    )

    # Format the chatbot's utterance in bold blue
    chatbot_markdown = Markdown(f"{turn.agent_utterance}", style="bold blue")
    console.print(chatbot_markdown)


def input_user() -> str:
    try:
        # Create a styled text for the prompt
        prompt_text = Text("User: ", style="bold magenta")

        # Get user input using rich's console.input() method
        user_utterance = console.input(prompt_text)

        # Ignore empty inputs
        while not user_utterance.strip():
            user_utterance = console.input(prompt_text)
    finally:
        pass

    return user_utterance.strip()


def make_parent_directories(file_name: str):
    """
    Creates the parent directories of `file_name` if they don't exist
    """
    pathlib.Path(os.path.dirname(file_name)).mkdir(parents=True, exist_ok=True)


def is_everything_verified(ver_out):
    """
    Everything is verified when 1) we have only one claim and it is supported or 2) all claims are supported.
    """
    for label_fix in ver_out:
        if label_fix["label"] != "SUPPORTS":
            return False
    return True


def dict_to_command_line(
    default_parameters: dict, overwritten_parameters: dict
) -> list[str]:
    """
    This function merges the default options set in a dictionary with options
    that need to be overwritten. It then creates a command line argument
    list of key-value options for those parameters. Boolean True values
    are represented only by the key name, False and None values are omitted.

    Parameters:
    - default_parameters (dict): A dictionary of key-value pairs representing
      the default options.
    - overwritten_parameters (dict): A dictionary of key-value pairs
      that need to overwrite the default options.

    Returns:
    - List[str]: A list of strings where each string is a command line
      argument in the form '--key=value'.
    """

    command_line = []
    parameters = default_parameters.copy()
    for k, v in overwritten_parameters.items():
        parameters[k] = v
    for k, v in parameters.items():
        if v is None:
            continue
        if not isinstance(v, bool):
            command_line.append(f"--{k}={v}")
        else:
            if v:
                command_line.append(f"--{k}")
    return command_line
