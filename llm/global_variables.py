import threading
from jinja2 import Environment, FileSystemLoader, select_autoescape, StrictUndefined
import yaml
import os
from llm.tgi_client import TGIOpenAILikeClient


# singleton
jinja_environment = Environment(
    loader=FileSystemLoader(["./", "./pipelines/", "./pipelines/prompts/"]),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
    line_comment_prefix="#",
    undefined=StrictUndefined,
)  # StrictUndefined raises an exception if a variable in template is not passed to render(), instead of setting it as empty

# define chat tags for chat models
system_start = "<|system_start|>"
system_end = "<|system_end|>"
user_start = "<|user_start|>"
user_end = "<|user_end|>"
assistant_start = "<|assistant_start|>"
assistant_end = "<|assistant_end|>"


llm_config = {}
with open("llm_config.yaml", "r") as config_file:
    llm_config = yaml.unsafe_load(config_file)
    # print(llm_config)
all_llm_endpoints = llm_config["llm_endpoints"]
for a in all_llm_endpoints:
    if "api_key" in a:
        a["api_key"] = os.getenv(a["api_key"])

all_llm_endpoints = [
    a for a in all_llm_endpoints if "api_key" not in a or a["api_key"] is not None
]  # remove resources for which we don't have a key
# print("all_llm_endpoints = ", all_llm_endpoints)

# for prompt debugging
prompt_log_file = "data/prompt_logs.json"
debug_prompts = False
prompt_logs = []
prompts_to_skip_for_debugging = [
    "benchmark/prompts/user_with_passage.prompt",
    "benchmark/prompts/user_with_topic.prompt",
]

# for local engines
local_engine_clients = {}
for endpoint_idx in range(len(all_llm_endpoints)):
    resource = all_llm_endpoints[endpoint_idx]
    if resource["api_type"] == "local":
        assert (
            "api_base" in resource and "prompt_format" in resource
        ), "Local engines should have api_base and prompt_format"
        local_engine_clients[resource["api_base"]] = TGIOpenAILikeClient(
            timeout=10,
            base_url=resource["api_base"],
        )


def set_debug_mode():
    global debug_prompts
    debug_prompts = True


# this code is not safe to use with multiprocessing, only multithreading
thread_lock = threading.Lock()

local_model_list = ["local"]
together_model_list = ["RedPajama-INCITE-7B-Instruct", "llama-2-70b", "llama-2-13b", "llama-2-7b", "llama-2-7b-32k"]
openai_chat_model_list = ["gpt-35-turbo", "gpt-4-32k", "gpt-4"]
openai_nonchat_model_list = [
    "gpt-35-turbo-instruct",
    "text-davinci-002",
    "text-davinci-003",
]

inference_cost_per_1000_tokens = {
    # from https://openai.com/pricing
    "ada": (0.0004, 0.0004),
    "babbage": (0.0005, 0.0005),
    "curie": (0.002, 0.002),
    "davinci": (0.02, 0.02),
    "gpt-35-turbo": (0.0015, 0.002),
    "gpt-4-32k": (0.06, 0.12),
    "gpt-4": (0.03, 0.06),

    # from https://www.together.ai/pricing
    "llama-2-70b": (0.0009, 0.0009),
    "llama-2-13b": (0.000225, 0.000225),
    "llama-2-7b": (0.0002, 0.0002),
    "llama-2-7b-32k": (0.0002, 0.0002),
    "RedPajama-INCITE-7B-Instruct": (0.0002, 0.0002),
}  # (prompt, completion)

total_cost = 0  # in USD


def add_to_total_cost(amount: float):
    global total_cost
    with thread_lock:
        total_cost += amount


def get_total_cost():
    global total_cost
    return total_cost


def _model_name_to_cost(model_name: str) -> float:
    if model_name in local_model_list:
        return 0, 0
    for model_family in inference_cost_per_1000_tokens.keys():
        if model_family in model_name:
            return inference_cost_per_1000_tokens[model_family]
    raise ValueError("Did not recognize OpenAI model name %s" % model_name)
