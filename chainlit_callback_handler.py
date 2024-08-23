import json
import time
from typing import Any, Dict, List, Optional

from chainlit import (
    ChatGeneration,
    CompletionGeneration,
    LangchainCallbackHandler,
    Step,
    Text,
)
from chainlit.context import context_var
from langchain.callbacks.tracers.schemas import Run
from literalai.helper import utc_now
from literalai.step import TrueStepType

from pipelines.dialogue_state import Claim, ClaimLabel, SearchQuery
from pipelines.retriever import RetrievalResult


class ChainlitCallbackHandler(LangchainCallbackHandler):
    def __init__(
        self,
        answer_prefix_tokens: Optional[List[str]] = None,
        stream_final_answer: bool = False,
        force_stream_final_answer: bool = False,
        step_name_mapping: Dict[str, str] = {},
        **kwargs: Any,
    ):
        """
        step_name_mapping: Steps without a mapping are excluded from the front-end
        """

        super().__init__(
            answer_prefix_tokens=answer_prefix_tokens,
            stream_final_answer=stream_final_answer,
            force_stream_final_answer=force_stream_final_answer,
            to_ignore=None,
            to_keep=None,
            **kwargs,
        )
        self.step_name_mapping = step_name_mapping
        self.query_step = None
        self.claim_step = None

    def _display_claims(self, claims: list[Claim], show_with_color: bool) -> None:
        ret = []
        for c in claims:
            if c.label == ClaimLabel.supported:
                color = "#b2d8d8"
            elif c.label == ClaimLabel.refuted:
                color = "#ffcccc"
            else:
                # NEI or not done verifying yet
                color = "lightyellow"
            ret.append(
                "- <span"
                + (f" style='background-color: {color};'" if show_with_color else "")
                + f">{c.text}</span>"
            )
        self.claim_step.output = "\n".join(ret)

    def _display_search_results(self, retrieval_results: list[RetrievalResult]) -> None:
        new_elements = []
        for result in retrieval_results:
            retrieval_result_element = Text(
                content=result.to_markdown(),
                display="inline",
            )
            new_elements.append(retrieval_result_element)
        self.query_step.elements = new_elements
        self._run_sync(self.query_step.update())

    def _update_search_with_summaries(self, retrieval_results: list[RetrievalResult]):
        summary_elements = []
        for r, element in zip(retrieval_results, self.query_step.elements):
            # print(element.content)
            summary_elements.append(
                Text(
                    content=element.content
                    + " \n\n**Summary:**\n"
                    + (
                        "\n".join([f"- {b}" for b in r.content_summary])
                        if r.content_summary
                        else "None"
                    ),
                    display="inline",
                )
            )
            self._run_sync(element.remove())
        self.query_step.elements = summary_elements
        self._run_sync(self.query_step.update())

    def _should_ignore_run(self, run: Run):
        if run.name not in self.step_name_mapping and run.name not in self.to_ignore:
            self.to_ignore.append(run.name)
        ignore, parent_id = super()._should_ignore_run(run)
        return ignore, parent_id

    def _start_trace(self, run: Run) -> None:
        super()._start_trace(run)
        context_var.set(self.context)

        ignore, parent_id = self._should_ignore_run(run)

        if run.run_type in ["chain", "prompt"]:
            self.generation_inputs[str(run.id)] = self.ensure_values_serializable(
                run.inputs
            )

        if ignore:
            return

        step_type: "TrueStepType" = "undefined"
        if run.run_type == "agent":
            step_type = "run"
        elif run.run_type == "chain":
            pass
        elif run.run_type == "llm":
            step_type = "llm"
        elif run.run_type == "retriever":
            step_type = "retrieval"
        elif run.run_type == "tool":
            step_type = "tool"
        elif run.run_type == "embedding":
            step_type = "embedding"

        if not self.steps:
            step_type = "run"

        disable_feedback = not self._is_annotable(run)

        step = Step(
            id=str(run.id),
            name=self.step_name_mapping[run.name],
            type=step_type,
            parent_id=parent_id,
            disable_feedback=disable_feedback,
        )
        step.start = utc_now()
        step.input = run.inputs

        self.steps[str(run.id)] = step

        self._run_sync(step.send())

    def _on_run_update(self, run: Run) -> None:
        """Process a run upon update."""
        context_var.set(self.context)

        ignore, parent_id = self._should_ignore_run(run)

        if ignore:
            return

        current_step = self.steps.get(str(run.id), None)

        if run.run_type == "llm" and current_step:
            provider, model, tools, llm_settings = self._build_llm_settings(
                (run.serialized or {}), (run.extra or {}).get("invocation_params")
            )
            generations = (run.outputs or {}).get("generations", [])
            generation = generations[0][0]
            variables = self.generation_inputs.get(str(run.parent_run_id), {})
            if message := generation.get("message"):
                chat_start = self.chat_generations[str(run.id)]
                duration = time.time() - chat_start["start"]
                if duration and chat_start["token_count"]:
                    throughput = chat_start["token_count"] / duration
                else:
                    throughput = None
                message_completion = self._convert_message(message)
                current_step.generation = ChatGeneration(
                    provider=provider,
                    model=model,
                    tools=tools,
                    variables=variables,
                    settings=llm_settings,
                    duration=duration,
                    token_throughput_in_s=throughput,
                    tt_first_token=chat_start.get("tt_first_token"),
                    messages=[
                        self._convert_message(m) for m in chat_start["input_messages"]
                    ],
                    message_completion=message_completion,
                )

                # find first message with prompt_id
                for m in chat_start["input_messages"]:
                    if m.additional_kwargs.get("prompt_id"):
                        current_step.generation.prompt_id = m.additional_kwargs[
                            "prompt_id"
                        ]
                        if custom_variables := m.additional_kwargs.get("variables"):
                            current_step.generation.variables = custom_variables
                    break

                current_step.language = "json"
                current_step.output = json.dumps(
                    message_completion, indent=4, ensure_ascii=False
                )
            else:
                completion_start = self.completion_generations[str(run.id)]
                completion = generation.get("text", "")
                duration = time.time() - completion_start["start"]
                if duration and completion_start["token_count"]:
                    throughput = completion_start["token_count"] / duration
                else:
                    throughput = None
                current_step.generation = CompletionGeneration(
                    provider=provider,
                    model=model,
                    settings=llm_settings,
                    variables=variables,
                    duration=duration,
                    token_throughput_in_s=throughput,
                    tt_first_token=completion_start.get("tt_first_token"),
                    prompt=completion_start["prompt"],
                    completion=completion,
                )
                current_step.output = completion

            if current_step:
                current_step.end = utc_now()
                self._run_sync(current_step.update())

            if self.final_stream and self.has_streamed_final_answer:
                self._run_sync(self.final_stream.update())

            return

        outputs = run.outputs or {}
        output_keys = list(outputs.keys())
        output = outputs
        if output_keys:
            output = outputs.get(output_keys[0], outputs)

        if current_step and output:
            assert run.name in self.step_name_mapping

            if run.name == "query":
                assert isinstance(output, SearchQuery)
                self.query_step = current_step
                self.query_step.output = output.to_markdown()
                self._run_sync(self.query_step.update())
            elif run.name == "search":
                assert isinstance(output, SearchQuery)
                self._display_search_results(output.retrieval_results)
                self._run_sync(self.query_step.update())
            elif run.name == "summarize":
                assert isinstance(output, SearchQuery)
                self._update_search_with_summaries(output.retrieval_results)
                self._run_sync(self.query_step.update())
            elif run.name == "split_claims":
                self.claim_step = current_step
                self._display_claims(output, show_with_color=False)
                self._run_sync(self.claim_step.update())
            elif run.name == "verify":
                self._display_claims(output, show_with_color=True)
                self._run_sync(self.claim_step.update())

            else:
                assert run.name in [
                    "generate",
                    "draft",
                    "refine",
                ], f"Unknown run name in chainlit: {run.name}"
                current_step.output = str(output)

                self._run_sync(current_step.update())
            current_step.end = utc_now()
