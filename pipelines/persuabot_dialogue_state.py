from typing import Optional

from pydantic import BaseModel, Field

from pipelines.dialogue_state import (
    DialogueTurn,
    DialogueState,
    ChatbotConfig,
)
from retrieval.retrieval_commons import QueryResult, SearchResultBlock


class PersuaBotSection(BaseModel):
    """A section of the response with its persuasion strategy and fact-check status."""

    content: str = Field(..., description="The content of this section")
    strategy: Optional[str] = Field(
        None, description="The persuasion strategy used in this section"
    )
    is_factual: bool = Field(
        default=False, description="Whether this section passed fact-checking"
    )
    retrieval_query: Optional[str] = Field(
        None, description="Query to retrieve factual information for this section"
    )
    retrieved_facts: list[SearchResultBlock] = Field(
        default_factory=list,
        description="Retrieved factual information for this section",
    )


class PersuaBotTurn(DialogueTurn):
    """Extended dialogue turn for PersuaBot with strategy maintenance fields."""

    # Question Handling Module (QHM) fields
    qhm_query: list[str] = []
    qhm_search_results: list[QueryResult] = []

    # Strategy Maintenance Module (SMM) fields
    smm_initial_response: Optional[str] = None
    smm_sections: list[PersuaBotSection] = []
    smm_final_response: Optional[str] = None

    # Override the agent_utterance to use SMM final response
    @property
    def agent_utterance(self) -> Optional[str]:
        if self.smm_final_response:
            return self.smm_final_response
        return self._agent_utterance

    @agent_utterance.setter
    def agent_utterance(self, value: Optional[str]):
        self._agent_utterance = value

    _agent_utterance: Optional[str] = None


class PersuaBotConfig(ChatbotConfig):
    """Configuration for PersuaBot."""

    persuasion_domain: str = Field(
        default="general",
        description="The domain for persuasion (e.g., 'donation', 'health', 'recommendation')",
    )
    target_goal: str = Field(
        default="persuade the user",
        description="The specific persuasion goal (e.g., 'encourage donation to charity X')",
    )
    do_fact_checking: bool = Field(
        default=True, description="Whether to perform fact-checking on generated content"
    )
    do_strategy_decomposition: bool = Field(
        default=True,
        description="Whether to decompose response into strategy sections",
    )


class PersuaBotState(DialogueState):
    """Dialogue state for PersuaBot."""

    config: PersuaBotConfig
    turns: list[PersuaBotTurn]

    @property
    def current_turn(self) -> PersuaBotTurn:
        return self.turns[-1]
