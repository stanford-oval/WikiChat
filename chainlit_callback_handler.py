from typing import Optional

from chainlit import LangchainCallbackHandler, Step
from chainlit.context import context_var
from langchain.callbacks.tracers.schemas import Run
from literalai.helper import utc_now

from pipelines.dialogue_state import DialogueState
from retrieval.retrieval_commons import QueryResult

step_name_mapping = {
    "LangGraph": "Steps",
    "query_stage": "Query",
    "generate_stage": "Generate Claims w/ LLM",
    "search_stage": "Search",
    "llm_claim_search_stage": "Search Claims",
    "filter_information_stage": "Filter Information",
    "draft_stage": "Draft",
    # "refine_stage": "Refine",
}


class ChainlitCallbackHandler(LangchainCallbackHandler):
    def __init__(
        self,
        dialogue_state: DialogueState,
    ):
        """
        step_name_mapping: Steps without a mapping are excluded from the front-end
        """

        super().__init__(
            to_ignore=None,
            to_keep=None,
        )
        self.dialogue_state = dialogue_state

    def _should_ignore_run(self, run: Run) -> tuple[bool, Optional[str]]:
        if run.name not in step_name_mapping or (
            run.tags and not run.tags[0].startswith("graph:")
        ):
            # steps that don't start with graph: are function names associated with graph steps, and are therefore duplicates
            return True, None
        return super()._should_ignore_run(run)

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

        step_type = "undefined"
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

        step = Step(
            id=str(run.id),
            name=(
                step_name_mapping[run.name]
                if run.name in step_name_mapping
                else run.name
            ),
            type=step_type,
            parent_id=parent_id,
            show_input=False,
        )
        step.start = utc_now()

        self.steps[str(run.id)] = step

        self._run_sync(step.send())

    def _on_run_update(self, run: Run) -> None:
        """Process a run upon update."""
        context_var.set(self.context)

        ignore, parent_id = self._should_ignore_run(run)

        if ignore:
            return

        current_step = self.steps.get(str(run.id), None)

        if current_step:
            step_output = None
            if run.name == "query_stage":
                step_output = "\n".join(self.dialogue_state.current_turn.search_query)
                if not step_output:
                    step_output = "_Did not search for anything._"
            elif run.name == "generate_stage":
                step_output = "\n".join(
                    [f"- {c}" for c in self.dialogue_state.current_turn.llm_claims]
                )
                if not step_output:
                    step_output = "_LLM did not generate any claims._"
            elif run.name == "search_stage":
                step_output = "\n".join(
                    [
                        QueryResult.to_markdown(r)
                        for r in self.dialogue_state.current_turn.search_results
                    ]
                )
                if not step_output:
                    step_output = "_No search results._"
            elif run.name == "llm_claim_search_stage":
                step_output = ""
                for claim, search_result in zip(
                    self.dialogue_state.current_turn.llm_claims,
                    self.dialogue_state.current_turn.llm_claim_search_results,
                ):
                    step_output += f"### Claim: {claim}\n{QueryResult.to_markdown(search_result)}\n\n"
                if not step_output:
                    step_output = "_No search results for LLM claims._"
            elif run.name == "filter_information_stage":
                step_output = ""
                for ref in self.dialogue_state.current_turn.filtered_search_results:
                    summary = "\n".join(
                        [f"- {s}" for s in ref.summary]
                    )  # add bullet points
                    step_output += f"#### [{ref.full_title}]({ref.url})\n\n**Summary:**\n{summary}\n\n**Full text:**\n\n{ref.content}\n\n"
            elif run.name == "draft_stage":
                step_output = self.dialogue_state.current_turn.draft_stage_output

            if step_output:
                current_step.output = step_output
                self._run_sync(current_step.update())
            current_step.end = utc_now()
