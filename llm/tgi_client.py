from text_generation import Client
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class TGIOpenAILikeClient(Client):
    """
    Used for local LLMs served via HuggingFace's text-generation-inference
    """

    def __init__(
        self,
        base_url,
        headers=None,
        cookies=None,
        timeout=10,
    ):
        super().__init__(base_url, headers, cookies, timeout)

    def _generate(
        self,
        prompt,
        max_tokens,
        stop,
        temperature,
        repetition_penalty=None,
        top_k=None,
        top_p=None,
    ):
        if temperature == 0:
            temperature = 1
            do_sample = False
        else:
            do_sample = True

        if top_p == 1:
            top_p = None

        return super().generate(
            prompt,
            do_sample,
            max_tokens,
            best_of=None,
            repetition_penalty=repetition_penalty,
            return_full_text=False,
            seed=None,
            stop_sequences=stop,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            truncate=None,
            typical_p=None,
            watermark=False,
            decoder_input_details=False,
        )

    def completion(self, kwargs):
        max_tokens = kwargs["max_tokens"]
        temperature = kwargs["temperature"]
        top_p = kwargs["top_p"]
        frequency_penalty = kwargs["frequency_penalty"]
        presence_penalty = kwargs["presence_penalty"]
        stop = kwargs["stop"]

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

        with ThreadPoolExecutor(len(kwargs["prompt"])) as executor:
            thread_outputs = [
                executor.submit(
                    self._generate,
                    prompt=p,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=stop,
                )
                for p in kwargs["prompt"]
            ]
        thread_outputs = [o.result().generated_text for o in thread_outputs]
        # print(thread_outputs)
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
            "model": kwargs["engine"],
            "object": "text_completion",
            "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        }
        # print("ret = ", ret)
        return ret
