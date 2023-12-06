from collections import OrderedDict
from typing import List


class DialogueTurn:
    def __init__(
        self,
        agent_utterance: str = None,
        user_utterance: str = None,
        pipeline: str = None,
        engine: str = None,
        generate_engine: str = None,
        draft_engine: str = None,
    ):
        self.engine = engine
        self.generate_engine = generate_engine
        self.draft_engine = draft_engine
        self.pipeline = pipeline
        self.wall_time_seconds = (
            0  # how much time it took to generate this turn, in seconds
        )
        self.agent_utterance = agent_utterance
        self.user_utterance = user_utterance

        # retrieve_and_generate pipeline
        self.initial_search_query = None
        self.initial_search_query_time = None
        self.initial_search_results = []
        self.initial_search_result_titles = []
        self.initial_search_bullets = []

        # generate_and_correct pipeline
        self.llm_utterance = None
        self.claims = []
        self.verification_retrieval_results = {}
        self.verification_result = {}

        # early_combine pipeline
        self.combined_evidences = []
        self.combined_utterance = None
        self.feedback = []
        self.feedback_scores = []
        self.refined_utterance = None

    def _summarize_vc_log(self):
        verification_summary = {}
        assert len(self.verification_result) == len(
            self.verification_retrieval_results
        ), "We need to have retrieved evidence for all claims"
        for key, value in self.verification_retrieval_results.items():
            claim_idx = int(key)
            v_ret_results = []
            for v in value:
                title, paragraph, score = tuple(v)
                v_ret_results.append(
                    {"title": title, "paragraph": paragraph, "score": round(score, 1)}
                )
            verification_summary[self.claims[claim_idx][0]] = OrderedDict(
                {
                    "label": self.verification_result[claim_idx]["label"],
                    "fixed_claim": self.verification_result[claim_idx]["fixed_claim"],
                    "retrieval_results": v_ret_results,
                }
            )
        return verification_summary

    def _summarize_rg_log(self):
        rg_summary = {
            "initial_search_query": self.initial_search_query,
            "initial_search_query_time": self.initial_search_query_time,
            "initial_search_bullets": self.initial_search_bullets,
            "initial_search_results": [],
        }

        for i in range(len(self.initial_search_results)):
            rg_summary["initial_search_results"].append(
                {
                    "title": self.initial_search_result_titles[i],
                    "paragraph": self.initial_search_results[i],
                    # 'bullets': self.initial_search_bullets,
                }
            )

        return rg_summary

    def log(self):
        """
        Returns a json object that contains all information inside `self`
        """
        # combine fields into a more human-readable field
        verification_summary = self._summarize_vc_log()
        rg_summary = self._summarize_rg_log()

        return OrderedDict(
            {
                # retrieve_and_generate pipeline
                "retrieve_and_generate": rg_summary,
                # generate_and_correct pipeline
                "llm_utterance": self.llm_utterance,
                "generate_and_correct": verification_summary,
                # early_combine pipeline
                "combined_evidences": self.combined_evidences,
                "combined_utterance": self.combined_utterance,
                "feedback": self.feedback,
                "feedback_scores": self.feedback_scores,
                "refined_utterance": self.refined_utterance,
                "user_utterance": self.user_utterance,
                "agent_utterance": self.agent_utterance,
                "engine": self.engine,
                "generate_engine": self.generate_engine,
                "draft_engine": self.draft_engine,
                "pipeline": self.pipeline,
                "wall_time_seconds": round(self.wall_time_seconds, 1),
            }
        )

    @staticmethod
    def utterance_list_to_dialog_history(utterance_list: List[str]):
        """
        The resulting dialog history will not have all the fields correctly initialized, since no information about e.g. search queries is available
        """
        dialog_history = []
        assert (
            len(utterance_list) % 2 == 1
        ), "The first turn is always the user, and the turn to be generated is always the agent, so the number of turns should be odd"
        for i in range(0, len(utterance_list) - 2, 2):
            dialog_history.append(
                DialogueTurn(
                    user_utterance=utterance_list[i],
                    agent_utterance=utterance_list[i + 1],
                )
            )
        user_utterance = utterance_list[-1]

        return dialog_history, user_utterance

    @staticmethod
    def dialog_history_to_utterance_list(dialog_history) -> List[str]:
        """
        Convert a list of DialogueTurns to a list of strings
        """
        utterance_list = []
        for turn in dialog_history:
            utterance_list.append(turn.user_utterance)
            utterance_list.append(turn.agent_utterance)
        return utterance_list
