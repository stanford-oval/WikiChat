from datetime import datetime
from typing import List
import pytz

from .global_variables import (
    system_start,
    system_end,
    user_start,
    user_end,
    assistant_start,
    assistant_end,
    local_model_list,
    openai_chat_model_list,
    jinja_environment,
)


def _remove_starting_and_ending_whitespace(text):
    # remove whitespace at the beginning and end of each line
    return "\n".join([line.strip() for line in text.split("\n")])


def _remove_chat_tags(s: str):
    return _remove_starting_and_ending_whitespace(
        s.replace(system_start, "")
        .replace(system_end, "")
        .replace(user_start, "")
        .replace(user_end, "")
        .replace(assistant_start, "")
        .replace(assistant_end, "")
    )  # need to remove starting and ending whitespace because after removing chat tags, whitespace that used to be in the middle might become starting or ending whitespace


def _fill_template(template_file, prompt_parameter_values, get_rendered_blocks=False):
    # logger.info("Filling template %s", template_file)
    template = jinja_environment.get_template(template_file)

    prompt_parameter_values["instruction_start"] = system_start
    prompt_parameter_values["instruction_end"] = system_end
    prompt_parameter_values["input_start"] = user_start
    prompt_parameter_values["input_end"] = user_end
    prompt_parameter_values["output_start"] = assistant_start
    prompt_parameter_values["output_end"] = assistant_end

    # always make these useful constants available in a template
    # make a new function call each time since the date might change during a long-term server deployment
    today = datetime.now(pytz.timezone("US/Pacific")).date()
    prompt_parameter_values["today"] = today.strftime("%B %d, %Y")  # May 30, 2023
    prompt_parameter_values["current_year"] = today.year
    prompt_parameter_values["location"] = "the U.S."
    prompt_parameter_values["chatbot_name"] = "WikiChat"

    filled_prompt = template.render(**prompt_parameter_values)
    filled_prompt = _remove_starting_and_ending_whitespace(filled_prompt)

    # Access the 'content' block and render it
    rendered_blocks = {}
    if get_rendered_blocks:
        for block_name in template.blocks.keys():
            block = template.blocks[block_name](
                template.new_context(vars=prompt_parameter_values)
            )
            rendered = "".join(block)
            rendered = _remove_chat_tags(
                rendered
            )  # blocks are used for logging and local engines, so should never have chat tags
            rendered_blocks[block_name] = rendered

    return filled_prompt, rendered_blocks


def _fill_prompt(template_file, prompt_parameter_values, engine, prompt_format):
    filled_prompt = [
        _fill_template(template_file, p, get_rendered_blocks=True)
        for p in prompt_parameter_values
    ]
    # convert list of tuples to tuple of lists
    filled_prompt, rendered_blocks = tuple(zip(*filled_prompt))
    filled_prompt, rendered_blocks = list(filled_prompt), list(rendered_blocks)

    # remove short_instruction blocks
    for idx, block in enumerate(rendered_blocks):
        if "short_instruction" in block:
            filled_prompt[idx] = filled_prompt[idx].replace(
                block["short_instruction"], ""
            )

    assert engine in local_model_list or prompt_format == "none", "Only local engines can have any prompt formats"
    if engine in local_model_list:
        if prompt_format == "none":
            pass
        elif prompt_format == "alpaca":
            # overwrite filled_prompt
            filled_prompt = [
                (
                    "Below is an instruction that describes a task, paired with an input that provides further context. "
                    "Write a response that appropriately completes the request.\n\n"
                    "### Instruction:\n{short_instruction}\n\n### Input:\n{input}\n\n### Response:"
                ).format_map(block)
                for block in rendered_blocks
            ]
        elif prompt_format == "simple":
            # overwrite filled_prompt
            filled_prompt = [
                "{short_instruction}\n\n{input}\n".format_map(block)
                for block in rendered_blocks
            ]
        else:
            raise ValueError("Unknown prompt format specified for the local engine.")
        
    if engine not in openai_chat_model_list:
        # print("rendered_blocks = ", rendered_blocks)
        filled_prompt = [_remove_chat_tags(f) for f in filled_prompt]

    return filled_prompt, rendered_blocks


def _convert_filled_prompt_to_chat_messages(filled_prompt: List[str]):
    """
    ChatGPT and GPT-4 use ChatCompletion and a different format from older models.
    The following is a naive implementation of few-shot prompting, which may be improved.
    """

    ret = []
    # print("before conversion", json.dumps(filled_prompt, indent=2))
    for fp in filled_prompt:
        # TODO check that start system is unique
        messages = []

        system_s = fp.find(system_start)
        system_e = fp.find(system_end, system_s)
        if system_s < 0:
            # did not find a system message in the prompt, so will put everything inside system for backward compatibility
            messages.append(
                {
                    "role": "system",
                    "content": fp.strip(),
                }
            )
            ret.append(messages)
            continue

        messages.append(
            {
                "role": "system",
                "content": fp[system_s + len(system_start) : system_e].strip(),
            }
        )

        last_index = 0
        while True:
            user_s = fp.find(user_start, last_index)
            assistant_s = fp.find(assistant_start, last_index)
            if (
                user_s >= 0
                and assistant_s < 0
                or (user_s >= 0 and user_s < assistant_s)
            ):
                user_e = fp.find(user_end, user_s)
                assert user_e >= 0, "Missing closing tag for user"
                last_index = user_e
                messages.append(
                    {
                        "role": "user",
                        "content": fp[user_s + len(user_start) : user_e].strip(),
                    }
                )
            elif (
                user_s < 0
                and assistant_s >= 0
                or (assistant_s >= 0 and user_s > assistant_s)
            ):
                assistant_e = fp.find(assistant_end, assistant_s)
                assert assistant_e >= 0, "Missing closing tag for assistant"
                last_index = assistant_e
                messages.append(
                    {
                        "role": "assistant",
                        "content": fp[
                            assistant_s + len(assistant_start) : assistant_e
                        ].strip(),
                    }
                )
            else:
                assert user_s < 0 and assistant_s < 0
                break
        ret.append(messages)

    return ret
