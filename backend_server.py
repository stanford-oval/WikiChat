import argparse
import asyncio
import random
import string

import chainlit as cl

from chainlit_callback_handler import ChainlitCallbackHandler
from database import save_dialogue_to_db
from pipelines.chatbot import create_chain, run_one_turn
from pipelines.pipeline_arguments import (
    add_pipeline_arguments,
    check_pipeline_arguments,
)
from pipelines.utils import dict_to_command_line, get_logger
from tasks.defaults import CHATBOT_DEFAULT_CONFIG

logger = get_logger(__name__)


chat_profiles_dict = {
    "Most Factual": {
        "description": "WikiChat based on GPT-4o",
        "overwritten_parameters": {},
        "step_name_mapping": {
            # "generate": "Generate",
            "query": "Search",
            "search": "",
            "summarize": "",
            "split_claims": "LLM Claims",
            "verify": "",
            "draft": "Draft",
            "refine": "Refine",
        },
    },
    "Balanced": {
        "description": "WikiChat based on GPT-4o-mini",
        "overwritten_parameters": {
            "engine": "gpt-4o-mini",
        },
        "step_name_mapping": {
            # "generate": "Generate",
            "query": "Search",
            "search": "",
            "summarize": "",
            "split_claims": "LLM Claims",
            "verify": "",
            "draft": "Draft",
            "refine": "Refine",
        },
    },
    # "Fastest": {
    #     "description": "WikiChat distilled",
    #     "overwritten_parameters": {
    #         "engine": "gpt-35-turbo-finetuned",
    #         "pipeline": "early_combine",
    #         "generation_prompt": "generate.prompt",
    #         "refinement_prompt": "refine_w_feedback.prompt",
    #         "retrieval_num": 3,
    #         "do_refine": False,
    #         "fuse_claim_splitting": True,
    #         "retrieval_reranking_method": "date",
    #         "retrieval_reranking_num": 9,
    #     },
    #     "step_name_mapping": {
    #         # "generate": "Generate",
    #         "query": "Search",
    #         "search": "",
    #         "summarize": "",
    #         "split_claims": "LLM Claims",
    #         "verify": "",
    #         "draft": "Draft",
    #         "refine": "Refine",
    #     },
    # },
    # "GPT-4o": {
    #     "description": "Vanilla GPT-4o",
    #     "overwritten_parameters": {
    #         "engine": "gpt-4o",
    #         "pipeline": "generate",
    #         "do_refine": False,
    #     },
    #     "step_name_mapping": {
    #         "generate": "Generate",
    #         # "refine": "Refine",
    #     },
    # },
}


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name=k, markdown_description=chat_profiles_dict[k]["description"]
        )
        for k in chat_profiles_dict.keys()
    ]


@cl.on_chat_start
async def start():
    all_step_names = set()
    for profile in chat_profiles_dict:
        all_step_names.update(chat_profiles_dict[profile]["step_name_mapping"].values())

    await asyncio.gather(
        cl.Message(
            content="""Hi! I'm WikiChat.\n
I'll keep our conversations for research purposes.\n
I can talk to you in any language, but I get my information from these Wikipedias: 🇺🇸 English, 🇨🇳 Chinese, 🇪🇸 Spanish, 🇵🇹 Portuguese, 🇷🇺 Russian, 🇩🇪 German, 🇮🇷 Farsi, 🇯🇵 Japanese, 🇫🇷 French, 🇮🇹 Italian.\n
By default, I prioritize factuality over speed. You can try other versions of WikiChat using the above menue.\n
Ask me about anything on Wikipedia!
"""
        ).send(),
        cl.Avatar(
            name="WikiChat",
            path="public/logo_light.png",
        ).send(),
        cl.Avatar(
            name="You",
            path="public/user_avatar.png",
        ).send(),
        *[
            cl.Avatar(
                name=step_name,
                path="public/empty.png",
            ).send()
            for step_name in all_step_names
        ],
        setup_agent(),
    )


@cl.on_settings_update
async def setup_agent():
    chat_profile = cl.user_session.get("chat_profile")
    if chat_profile not in chat_profiles_dict:
        return
    logger.debug("Using chat profile %s", chat_profile)

    parser = argparse.ArgumentParser()
    add_pipeline_arguments(parser)

    # set parameters
    args = parser.parse_args(
        dict_to_command_line(
            CHATBOT_DEFAULT_CONFIG,
            chat_profiles_dict[chat_profile]["overwritten_parameters"],
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


@cl.on_message
async def chat(message: cl.Message):
    chatbot = cl.user_session.get("chatbot")
    chat_profile = cl.user_session.get("chat_profile")
    dialogue_state = cl.user_session.get("dialogue_state")

    dialogue_state["new_user_utterance"] = message.content
    new_agent_utterance, dialogue_state = await run_one_turn(
        chatbot,
        dialogue_state,
        new_user_utterance=message.content,
        callbacks=[
            ChainlitCallbackHandler(
                step_name_mapping=chat_profiles_dict[chat_profile]["step_name_mapping"]
            )
        ],
    )

    # print("dialogue_state = ", dialogue_state)
    await cl.Message(content=new_agent_utterance).send()
    cl.user_session.set("dialogue_state", dialogue_state)


@cl.on_chat_end
def on_chat_end():
    dialogue_state = cl.user_session.get("dialogue_state")
    dialogue_id = cl.user_session.get("dialogue_id")
    chat_profile = cl.user_session.get("chat_profile")
    if dialogue_state and dialogue_state["dialogue_history"]:
        save_dialogue_to_db(dialogue_state, dialogue_id, chat_profile)
