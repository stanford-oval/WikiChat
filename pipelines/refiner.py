from typing import List
import logging

from .dialog_turn import DialogueTurn
from llm.llm_generate import llm_generate

logger = logging.getLogger(__name__)

class Refiner:
    def __init__(self, prompt, args):
        self.prompt = prompt
        self.temperature = args.temperature
        self.top_p = args.top_p

    def set_refinement_fields(
        self,
        object_dlg_history: List[DialogueTurn],
        new_dlg_turn: DialogueTurn,
        engine_dict,
    ):
        prompt_output = llm_generate(
            template_file=self.prompt,
            prompt_parameter_values={
                "dlg": object_dlg_history,
                "new_dlg_turn": new_dlg_turn,
            },
            engine=engine_dict["default"],
            max_tokens=300,
            temperature=self.temperature,
            top_p=self.top_p,
            stop_tokens=None,
            postprocess=False,
        )
        if self.prompt.endswith("refine_w_feedback.prompt"):
            return Refiner.handle_refinement_with_feedback(new_dlg_turn, prompt_output)
        elif self.prompt.endswith("refine.prompt"):
            return Refiner.handle_refinement_without_feedback(
                new_dlg_turn, prompt_output
            )
        else:
            raise ValueError("Unknown refinement prompt.")

    @staticmethod
    def handle_refinement_without_feedback(new_dlg_turn, prompt_output):
        new_dlg_turn.refined_utterance = prompt_output.strip()
        return new_dlg_turn.refined_utterance

    @staticmethod
    def handle_refinement_with_feedback(new_dlg_turn, prompt_output: str):
        refine_identifiers = [
            "Revised response after applying this feedback:",
            "Response after applying this feedback:",
        ]
        for identifier in refine_identifiers:
            if identifier in prompt_output:
                feedback, prompt_output = prompt_output.split(identifier)

                (
                    new_dlg_turn.feedback,
                    new_dlg_turn.feedback_scores,
                ) = Refiner._parse_feedback(feedback)
                if sum(new_dlg_turn.feedback_scores) == 100 * len(
                    new_dlg_turn.feedback_scores
                ):
                    # skip refinement if it already gets full score
                    new_dlg_turn.refined_utterance = new_dlg_turn.agent_utterance
                else:
                    new_dlg_turn.refined_utterance = prompt_output.strip()
                return new_dlg_turn.refined_utterance

        logger.error(
            "Skipping refinement due to malformatted Refined response: %s",
            prompt_output,
        )
        new_dlg_turn.refined_utterance = new_dlg_turn.agent_utterance
        return new_dlg_turn.refined_utterance

    @staticmethod
    def _parse_feedback(feedback):
        if "User:" in feedback:
            feedback = feedback.split("User:")[0]
        feedback_lines = feedback.strip().split("\n")

        if len(feedback_lines) < 4 or len(feedback_lines) > 5:
            logger.error("Feedback malformatted")
            logger.error(feedback_lines)
            return [], []

        scores = (
            []
        )  # Relevant, Informative, Conversational, Non-Redundant, Temporally Correct scores
        for line in feedback_lines:
            score = line.strip().split(" ")[-1].strip()
            if (
                score == "N/A" or "this criterion is not applicable" in line
            ):  # some models say "not applicable" instead of N/A
                score = 100
            else:
                try:
                    score = int(score.split("/")[0])
                except:
                    logger.error(f"Feedback line malformatted: {line}")
                    score = 100
            scores.append(score)
        logger.info("Feedback scores: %s", scores)
        return feedback_lines, scores
