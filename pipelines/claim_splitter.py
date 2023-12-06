import logging
from typing import List
from llm.llm_generate import llm_generate

logger = logging.getLogger(__name__)


class ClaimSplitter:
    def __init__(self, prompt_template_file: str):
        self.prompt_template_file = prompt_template_file

    def split_claim(
        self,
        dialog_history: List,
        new_user_utterance: str,
        current_agent_utterance: str,
        engine_dict: dict,
        dialog_topic: str = None,
    ):
        """
        dialog_topic: used for splitting claims of a simulated dialog we want to evaluate
        """
        claims_output = llm_generate(
            template_file=self.prompt_template_file,
            prompt_parameter_values={
                "dlg": dialog_history,
                "new_user_utterance": new_user_utterance,
                "current_agent_utterance": current_agent_utterance,
                "dialog_topic": dialog_topic,
            },
            engine=engine_dict["default"],
            max_tokens=300,
            temperature=0,
            stop_tokens=["====="],
            postprocess=False,
        )

        if claims_output.startswith("Yes. "):
            # necessary for distilled models
            claims_output = claims_output[5:]
        all_claims = self._format_output(claims_output)

        return all_claims

    def _format_output(self, output):
        lines = output.split("\n")
        if lines[0].startswith("Nothing."):
            # no claims detected
            return []
        all_claims = []
        try:
            for c in lines:
                claim = c
                cleaned_claim = claim.split(f"- ")[-1].strip()
                if cleaned_claim:
                    split_term = " The year of the results is "
                    if split_term not in cleaned_claim:
                        split_term = " The year of the claim is "
                    splitted = cleaned_claim.split(split_term)
                    if len(splitted) == 2:
                        cleaned_claim, year = splitted
                        year = year[1:-2]
                    else:
                        # sometimes model output may not be well-formatted (e.g. N/A); default to none
                        cleaned_claim = splitted[0]
                        year = "none"
                    all_claims.append((cleaned_claim, year))
        except Exception as e:
            logger.error("Error while parsing claims in %s: %s", output, str(e))
            raise e

        return all_claims

    @staticmethod
    def remove_claims_from_previous_turns(claims: List, object_dlg_history):
        """
        Removes claims that are repeated from the last turn. This is often the result of LLM making a mistake while splitting claims.
        But even if it is actually a claim that the chatbot repeats, removing it here is beneficial as it will reduce repetitiveness.
        """
        previous_turn_claims = []
        for i in range(len(object_dlg_history)):
            previous_turn_claims.extend([c[0] for c in object_dlg_history[i].claims])
        claims = [c for c in claims if c[0] not in previous_turn_claims]

        return claims
