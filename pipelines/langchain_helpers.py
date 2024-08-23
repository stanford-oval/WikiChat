import sys

from chainlite import llm_generation_chain

sys.path.insert(0, "./")

from pipelines.chatbot_config import StageConfig


def stage_prompt(stage_config: StageConfig, bind_prompt_values: dict = {}):
    return llm_generation_chain(
        template_file=stage_config.template_file,
        engine=stage_config.engine,
        temperature=stage_config.temperature,
        top_p=stage_config.top_p,
        max_tokens=stage_config.max_tokens,
        stop_tokens=stage_config.stop_tokens,
        postprocess=stage_config.postprocess,
        bind_prompt_values=bind_prompt_values,
    )
