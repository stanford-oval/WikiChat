import argparse
import random
import string

import chainlit as cl
from chainlit.input_widget import Select
from chainlite import Runnable

from chainlit_callback_handler import ChainlitCallbackHandler
from corpora import all_corpus_objects, corpus_name_to_corpus_object
from database import save_dialogue_to_db
from pipelines.chatbot import create_chain
from pipelines.dialogue_state import DialogueState, DialogueTurn
from pipelines.pipeline_arguments import (
    add_pipeline_arguments,
    check_pipeline_arguments,
)
from pipelines.utils import dict_to_command_line
from tasks.defaults import CHATBOT_DEFAULT_CONFIG
from utils.logging import logger


@cl.set_chat_profiles
async def chat_profile(current_user: cl.User | None):
    ret = []
    for corpus in all_corpus_objects:
        ret.append(
            cl.ChatProfile(
                name=corpus.name,
                icon=corpus.icon_path,
                markdown_description=corpus.human_description_markdown,
                starters=[
                    cl.Starter(
                        label=starter.display_label,
                        message=starter.chat_message,
                        icon=starter.icon_path,
                    )
                    for starter in corpus.chat_starters
                ],
            )
        )

    return ret


@cl.on_chat_start
async def start():
    await cl.ChatSettings(
        [
            Select(
                id="model",
                label="Model",
                values=["gpt-4o-mini", "gpt-4o"],
                initial_index=0,
                description="Select the large language model to use.",
            ),
        ]
    ).send()
    corpus_name = cl.user_session.get("chat_profile")
    assert isinstance(corpus_name, str), "Missing or invalid chat_profile"
    logger.debug(f"Using corpus with name '{corpus_name}'")

    parser = argparse.ArgumentParser()
    add_pipeline_arguments(parser)

    # set parameters
    args = parser.parse_args(
        dict_to_command_line(
            CHATBOT_DEFAULT_CONFIG,
            corpus_name_to_corpus_object(corpus_name).overwritten_parameters,
        )
    )
    check_pipeline_arguments(args)

    chatbot, dialogue_state = create_chain(args)

    cl.user_session.set("chatbot", chatbot)
    cl.user_session.set("dialogue_state", dialogue_state)
    cl.user_session.set(
        "dialogue_id",
        "".join(random.choices(string.ascii_letters + string.digits, k=8)),
    )  # 8-character random string as the unique dialogue_id


@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("model", settings["model"])


@cl.on_message
async def chat(message: cl.Message):
    chatbot_raw = cl.user_session.get("chatbot")
    assert isinstance(chatbot_raw, Runnable), "Missing or invalid chatbot in session"
    chatbot: Runnable = chatbot_raw
    dialogue_state_raw = cl.user_session.get("dialogue_state")
    assert isinstance(
        dialogue_state_raw, DialogueState
    ), "Missing or invalid dialogue_state in session"
    dialogue_state: DialogueState = dialogue_state_raw
    model = cl.user_session.get("model")
    if model:
        dialogue_state.config.engine = model

    dialogue_state.turns.append(DialogueTurn(user_utterance=message.content))
    await chatbot.ainvoke(
        dialogue_state,
        config={"callbacks": [ChainlitCallbackHandler(dialogue_state=dialogue_state)]},
    )
    new_agent_utterance = dialogue_state.current_turn.agent_utterance

    if not new_agent_utterance:
        new_agent_utterance = "I'm sorry, I don't have an answer for that."
    message = cl.Message(content=new_agent_utterance)
    await message.send()

    for ref_id, ref in enumerate(
        dialogue_state.current_turn.filtered_search_results, start=1
    ):
        summary = "\n".join([f"- {s}" for s in ref.summary])  # add bullet points
        m = cl.Text(
            name=f"[{ref_id}]",
            content=f"## [{ref.full_title}]({ref.url})\n\n**Summary:**\n{summary}\n\n**Full text:**\n\n{ref.content}",
            display="side",
        )
        await m.send(for_id=message.id)


@cl.on_chat_end
def on_chat_end():
    dialogue_state: DialogueState = cl.user_session.get("dialogue_state")
    dialogue_id: str = cl.user_session.get("dialogue_id")
    chat_profile: str = cl.user_session.get("chat_profile")
    if dialogue_state and dialogue_state.turns:
        save_dialogue_to_db(dialogue_state, dialogue_id, chat_profile)
