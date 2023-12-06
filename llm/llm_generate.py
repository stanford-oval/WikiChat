"""
Functionality to work with .prompt files
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from typing import List, Union
import copy
import openai
import logging
from typing import List
import openai
from openai import OpenAIError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
from functools import lru_cache, partial, wraps
import together
from llm.load_prompt import _convert_filled_prompt_to_chat_messages, _fill_prompt
import llm.global_variables as global_variables

logger = logging.getLogger(__name__)


def write_prompt_logs_to_file():
    with open(global_variables.prompt_log_file, "w") as f:
        f.write(json.dumps(global_variables.prompt_logs, indent=4, ensure_ascii=False))

def remove_malformed_prompt_logs(is_malformed_function):
    new_prompt_logs = []
    for item in global_variables.prompt_logs:
        if not is_malformed_function(item):
            new_prompt_logs.append(item)

    global_variables.prompt_logs = new_prompt_logs

def lru_cache_if_zero_temperature(maxsize):
    def decorator(func):
        cached_func = lru_cache(maxsize=maxsize)(func)

        @wraps(func)
        def wrapper(original_engine_name, **kwargs):
            if kwargs.get("temperature") == 0:
                # Convert the inputs to be hashable. Necessary for caching to work.
                for a in kwargs:
                    if isinstance(kwargs[a], list):
                        kwargs[a] = tuple(kwargs[a])
                return cached_func(original_engine_name, **kwargs)
            else:
                return func(original_engine_name, **kwargs)

        return wrapper

    return decorator


async def openai_chat_completion(kwargs):
    engine = kwargs["engine"]
    prompt = kwargs["prompt"]
    max_tokens = kwargs["max_tokens"]
    temperature = kwargs["temperature"]
    top_p = kwargs["top_p"]
    frequency_penalty = kwargs["frequency_penalty"]
    presence_penalty = kwargs["presence_penalty"]
    stop = kwargs["stop"]
    logit_bias = kwargs.get("logit_bias", {})

    if kwargs["api_type"] == "azure":
        f = partial(
            openai.ChatCompletion.acreate,
            engine=engine,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            logit_bias=logit_bias,
            api_base=kwargs["api_base"],
            api_key=kwargs["api_key"],
            api_type=kwargs["api_type"],
            api_version=kwargs["api_version"],
        )
    else:
        # openai.com
        # use 'model' instead of 'engine'. no `api_version``
        f = partial(
            openai.ChatCompletion.acreate,
            model=engine,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            logit_bias=logit_bias,
            api_base=kwargs["api_base"],
            api_key=kwargs["api_key"],
            api_type=kwargs["api_type"],
        )

    prompt = _convert_filled_prompt_to_chat_messages(prompt)
    # print("prompt = ", json.dumps(prompt, indent=2, ensure_ascii=False))

    coroutines = [f(messages=p) for p in prompt]
    outputs = await asyncio.gather(*coroutines)

    # make the outputs of these multiple calls to openai look like one call with batching
    ret = outputs[0]
    for i, o in enumerate(outputs[1:]):
        o["choices"][0]["index"] = i + 1
        ret["choices"].append(o["choices"][0])
        ret["usage"]["completion_tokens"] += o["usage"]["completion_tokens"]
        ret["usage"]["prompt_tokens"] += o["usage"]["prompt_tokens"]
        ret["usage"]["total_tokens"] += o["usage"]["total_tokens"]

    for c in ret["choices"]:
        # key 'content' does not exist for gpt-4 when output is empty.
        if c["finish_reason"] == "content_filter":
            c["text"] = None
        else:
            c["text"] = c["message"]["content"]

    return ret


def openai_nonchat_completion(kwargs):
    """
    output looks like
    {
        "choices": [
                {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": null,
                "text": " No.]"
                }
            ],
        "created": 1682393599,
        "id": "cmpl-793iZfhYTA2pAHA3LJSGmy6aUsQO7",
        "model": "text-davinci-003",
        "object": "text_completion",
        "usage": {
            "completion_tokens": 2,
            "prompt_tokens": 937,
            "total_tokens": 939
        }
    }
    """
    if kwargs["api_type"] == "open_ai":
        kwargs["model"] = kwargs["engine"]
        kwargs.pop("engine", None)
        kwargs.pop("api_version", None)
    # print("api_base = ", kwargs["api_base"])
    ret = openai.Completion.create(**kwargs)

    # reorder outputs to have the same order as inputs
    choices_in_correct_order = [{}] * len(ret.choices)
    for choice in ret.choices:
        choices_in_correct_order[choice.index] = choice
    ret.choices = choices_in_correct_order

    return ret


def together_completion(kwargs):
    """
    Together.ai doesn't support batch inference or async calls, so we use multithreading to simulate the behavior of batch inference.
    We use multithreading (instead of multiprocessing) because this method is I/O-bound, mostly waiting for an HTTP response to come back.
    """
    kwargs["model"] = kwargs["engine"]
    engine = kwargs.pop("engine", None)
    kwargs.pop("api_version", None)
    kwargs.pop("api_type", None)
    kwargs.pop("api_key", None)

    max_tokens = kwargs["max_tokens"]
    temperature = kwargs["temperature"]
    top_p = kwargs["top_p"]
    stop = kwargs["stop"]
    if stop is None:
        stop = []

    stop = list(stop) + ["<bot>", "<human>", "=====", "-----"]

    if (
        "presence_penalty" in kwargs
        and kwargs["presence_penalty"] is not None
        and kwargs["presence_penalty"] != 0
    ):
        logger.warning(
            "Ignoring `presence_penalty` since it is not supported by this model."
        )
    if (
        "frequency_penalty" in kwargs
        and kwargs["frequency_penalty"] is not None
        and kwargs["frequency_penalty"] != 0
    ):
        logger.warning(
            "Ignoring `frequency_penalty` since it is not supported by this model."
        )

    with ThreadPoolExecutor(1) as executor: # Together has an extremely low rate limit when using the free tier
        thread_outputs = [
            executor.submit(
                together.Complete.create,
                model=engine,
                prompt=p,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
            )
            for p in kwargs["prompt"]
        ]
    thread_outputs = [o.result() for o in thread_outputs]
    thread_outputs = [o["output"]["choices"][0]["text"] for o in thread_outputs]

    ret = {
        "choices": [
            {
                "finish_reason": "stop",
                "index": i,
                "logprobs": None,
                "text": thread_outputs[i],
            }
            for i in range(len(thread_outputs))
        ],
        "created": 0,
        "id": "",
        "model": engine,
        "object": "text_completion",
        "usage": {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
    }

    return ret


@lru_cache_if_zero_temperature(maxsize=10000)
@retry(
    retry=retry_if_exception_type((OpenAIError, together.error.RateLimitError)),
    wait=wait_exponential(min=1, max=120, exp_base=2),
    stop=stop_after_attempt(7),
       before_sleep=before_sleep_log(logger, logging.ERROR)
)
def _llm_completion_with_backoff_and_cache(original_engine_name, **kwargs):
    try:
        if original_engine_name in global_variables.openai_chat_model_list:
            return asyncio.run(openai_chat_completion(kwargs))
        elif original_engine_name in global_variables.local_model_list:
            return global_variables.local_engine_clients[kwargs["api_base"]].completion(
                kwargs
            )
        elif original_engine_name in global_variables.together_model_list:
            return together_completion(kwargs)
        elif original_engine_name in global_variables.openai_nonchat_model_list:
            return openai_nonchat_completion(kwargs)
        else:
            raise ValueError("This engine is not supported yet.")
    except OpenAIError as e:
        if "Azure OpenAI’s content management policy" in str(e):
            logger.error("No output due to Azure's content filtering.")
            return {
                "choices": [
                    {
                        "finish_reason": "content_filter",
                        "index": i,
                        "logprobs": None,
                        "text": None,
                    }
                    for i in range(len(kwargs["prompt"]))
                ],
                "usage": {
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                    "total_tokens": 0,
                },
            }
        else:
            raise e


def _set_llm_resource_fields(llm_resource, **kwargs):
    kwargs_copy = copy.deepcopy(kwargs)

    if "api_version" in llm_resource:
        kwargs_copy["api_version"] = llm_resource["api_version"]
    if "api_base" in llm_resource:
        kwargs_copy["api_base"] = llm_resource["api_base"]

    if "api_key" in llm_resource:
        kwargs_copy["api_key"] = llm_resource["api_key"]

    if llm_resource["api_type"] == "together":
        together.api_key = llm_resource["api_key"]

    # all providers have these fields
    kwargs_copy["api_type"] = llm_resource["api_type"]
    kwargs_copy["engine"] = llm_resource["engine_map"][kwargs["engine"]]

    # print("setting api_base to ", kwargs_copy["api_base"])

    return kwargs_copy


def _postprocess_generations(generation_output: str) -> str:
    """
    Might output an empty string if generation is not at least one full sentence
    """
    # replace all whitespaces with a single space
    generation_output = " ".join(generation_output.split())
    # print("generation_output = ", generation_output)

    original_generation_output = generation_output
    # remove extra dialog turns, if any
    turn_indicators = [
        "You:",
        "They:",
        "Context:",
        "You said:",
        "They said:",
        "Assistant:",
        "Chatbot:",
    ]
    for t in turn_indicators:
        while generation_output.find(t) > 0:
            generation_output = generation_output[: generation_output.find(t)]

    generation_output = generation_output.strip()
    # delete half sentences
    if len(generation_output) == 0:
        logger.error(
            "LLM output is empty after postprocessing. Before postprocessing it was %s",
            original_generation_output,
        )
        return generation_output

    if generation_output[-1] not in {".", "!", "?"} and generation_output[-2:] != '."':
        # handle preiod inside closing quotation
        last_sentence_end = max(
            generation_output.rfind("."),
            generation_output.rfind("!"),
            generation_output.rfind("?"),
            generation_output.rfind('."') + 1,
        )
        if last_sentence_end > 0:
            generation_output = generation_output[: last_sentence_end + 1]
    return generation_output


def llm_generate(
    template_file: str,
    prompt_parameter_values: Union[dict, List[dict]],
    engine: str,
    max_tokens: int,
    temperature: float,
    stop_tokens,
    top_p: float = 0.9,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
    postprocess: bool = True,
    filled_prompt=None,
):
    """
    Generates continuations for one or more prompts in parallel
    Inputs:
        prompt_parameter_values: dict or list of dict. If the input is a list, the output will be a list as well
        filled_prompt: gives direct access to the underlying model, without having to load a prompt template from a .prompt file. Used for testing.
    """
    if not (
        filled_prompt is None
        and prompt_parameter_values is not None
        and template_file is not None
    ) and not (
        filled_prompt is not None
        and prompt_parameter_values is None
        and template_file is None
    ):
        raise ValueError(
            "Can only use filled_prompt if template_file and prompt_parameter_values are None"
        )

    # Decide which LLM resource to send this request to.
    # Use hash so that each time this function gets called with the same parameters after a backoff, the request gets sent to the same resource
    potential_llm_resources = [
        resource
        for resource in global_variables.all_llm_endpoints
        if engine in resource["engine_map"]
    ]
    llm_resource = potential_llm_resources[
        hash(
            str(
                (
                    template_file,
                    prompt_parameter_values,
                    engine,
                    max_tokens,
                    temperature,
                    stop_tokens,
                    top_p,
                    frequency_penalty,
                    presence_penalty,
                )
            )
        )
        % len(potential_llm_resources)
    ]
    # uniform load balancing instead of hashing
    # llm_resource = potential_llm_resources[random.randrange(len(potential_llm_resources))]

    if llm_resource["api_type"] == "local":
        prompt_format = llm_resource["prompt_format"]
    else:
        prompt_format = "none"

    # convert to a single element list so that the rest of the code only has to deal with a list
    input_was_list = True
    if filled_prompt is None:
        assert prompt_parameter_values is not None
        if not isinstance(prompt_parameter_values, list):
            input_was_list = False
            prompt_parameter_values = [prompt_parameter_values]
        filled_prompt, rendered_blocks = _fill_prompt(
            template_file, prompt_parameter_values, engine, prompt_format
        )
    else:
        if not isinstance(filled_prompt, list):
            input_was_list = False
            filled_prompt = [filled_prompt]

    assert isinstance(filled_prompt, list)

    # Call LLM to generate outputs
    generation_output = _llm_completion_with_backoff_and_cache(
        original_engine_name=engine,
        **_set_llm_resource_fields(
            llm_resource=llm_resource,
            engine=engine,
            prompt=filled_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop_tokens,
        )
    )
    outputs = []
    for choice in generation_output["choices"]:
        if choice["text"]:
            outputs.append(choice["text"])

    logger.info("LLM output: %s", json.dumps(outputs, indent=2, ensure_ascii=False))

    # calculate and record the cost
    cost_prompt, cost_completion = global_variables._model_name_to_cost(engine)
    total_cost = (
        generation_output["usage"]["prompt_tokens"] * cost_prompt
        + generation_output["usage"].get("completion_tokens", 0) * cost_completion
    ) / 1000
    global_variables.add_to_total_cost(total_cost)

    # postprocess the generation outputs
    outputs = [o.strip() for o in outputs]
    if postprocess:
        outputs = [_postprocess_generations(o) for o in outputs]

    # add to prompt logs if needed
    if global_variables.debug_prompts:
        with global_variables.thread_lock:
            for i, o in enumerate(outputs):
                if template_file in global_variables.prompts_to_skip_for_debugging:
                    continue
                global_variables.prompt_logs.append(
                    {
                        "template_name": template_file,
                        "instruction": rendered_blocks[i]["short_instruction"]
                        if "short_instruction" in rendered_blocks[i]
                        else rendered_blocks[i]["instruction"],
                        "input": rendered_blocks[i]["input"],
                        "output": o,
                    }
                )

    if outputs == []:
        outputs = ""

    # convert back to a single item
    if len(outputs) == 1 and not input_was_list:
        outputs = outputs[0]
    return outputs
