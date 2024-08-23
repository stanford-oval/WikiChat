import json
import operator
from enum import Enum
from typing import Annotated, Sequence, TypedDict

from pipelines.chatbot_config import ChatbotConfig
from pipelines.retriever import RetrievalResult


class SearchQuery:
    text: str
    time: str
    retrieval_results: list[RetrievalResult]

    def __init__(self, query: str, time: str):
        self.text = query
        self.time = time
        self.retrieval_results = []

    def to_dict(self):
        return {
            "text": self.text,
            "time": self.time,
            "retrieval_results": [r.__dict__ for r in self.retrieval_results],
        }

    def to_markdown(self):
        # Chainlit can display Markdown with support for some HTML tags
        return (
            "<center><span style='font-weight:bold'> Query: "
            + self.text
            + "</span></center>"
        )


class DialogueTurn:
    def __init__(
        self,
        agent_utterance: str = None,
        user_utterance: str = None,
        initial_search_query: SearchQuery = None,
        turn_log: dict = None,
    ):
        self.agent_utterance = agent_utterance
        self.user_utterance = user_utterance
        self.initial_search_query = initial_search_query
        self.turn_log = turn_log

    def to_json(self) -> str:
        return json.dumps(self.__dict__)


class ClaimLabel(str, Enum):  # subclassing `str` makes this JSON serializable
    supported = "SUPPORTS"
    refuted = "REFUTES"
    nei = "NOT ENOUGH INFORMATION"


class Claim(SearchQuery):
    label: str
    fixed_claim: ClaimLabel

    def __init__(self, claim: str, time: str):
        super().__init__(query=claim, time=time)
        self.label = ""
        self.fixed_claim = ""

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["label"] = self.label
        for r in d["retrieval_results"]:
            if "content_summary" in r:
                assert len(r["content_summary"]) == 0
                del r[
                    "content_summary"
                ]  # claim retrieval results always have empty summaries
        return d


class Feedback:
    string: str
    scores: dict[str, float]

    def __init__(self, string: str, scores: dict[str, float]):
        self.string = string
        self.scores = scores

    def is_full_score(self) -> bool:
        for criterion, score in self.scores.items():
            if score < 100:
                return False

        return True

    def to_dict(self):
        return self.__dict__

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class DialogueState(TypedDict):
    # The dialogue history excluding the current turn
    dialogue_history: Annotated[Sequence[DialogueTurn], operator.add]

    # For the current turn
    chatbot_config: ChatbotConfig
    new_agent_utterance: Annotated[str, operator.add]
    new_user_utterance: Annotated[str, operator.add]
    generate_stage_output: Annotated[str, operator.add]
    generation_claims: Sequence[Claim]
    draft_stage_output: Annotated[str, operator.add]
    refine_stage_output: Annotated[str, operator.add]
    initial_search_query: SearchQuery
    feedback: Feedback
    wall_time: str  # in seconds


def state_to_dict(state: DialogueState):
    """
    This is defined outside of the DialogueState class because TypedDict classes cannot have methods
    """
    j = dict(state)  # copy
    j["chatbot_config"] = j["chatbot_config"].model_dump()
    if j["generation_claims"]:
        j["generation_claims"] = [c.to_dict() for c in j["generation_claims"]]
    for field in ["initial_search_query", "feedback"]:
        if j[field]:
            j[field] = j[field].to_dict()

    del j["dialogue_history"]

    return j


def state_to_string(state: DialogueState):
    return json.dumps(
        state_to_dict(state), indent=2, ensure_ascii=False, default=vars, sort_keys=True
    )


def clear_current_turn(state: DialogueState):
    state["new_agent_utterance"] = ""
    state["new_user_utterance"] = ""
    state["generate_stage_output"] = ""
    state["generation_claims"] = []
    state["draft_stage_output"] = ""
    state["refine_stage_output"] = ""
    state["initial_search_query"] = None
    state["feedback"] = None
