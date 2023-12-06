from concurrent.futures import ThreadPoolExecutor
import time
from typing import List
import re
import requests
import logging
import numpy as np


from .dialog_turn import DialogueTurn
from llm.llm_generate import llm_generate
from .claim_splitter import ClaimSplitter
from .refiner import Refiner
from .utils import is_everything_verified, extract_year

logger = logging.getLogger(__name__)


class Chatbot:
    """
    A stateless chatbot. Stateless means that it does not store the history of the dialog in itself, but requires it as an input
    """

    def __init__(self, args) -> None:
        # Initialize everything, because we can change the pipeline on the fly using system_parameters
        self.claim_splitter = ClaimSplitter(args.claim_prompt_template_file)
        self.evi_num = args.evi_num
        self.colbert_endpoint = args.colbert_endpoint
        self.retrieval_num = args.retrieval_num
        self.refiner = Refiner(prompt=args.refinement_prompt, args=args)

        self.temperature = args.temperature
        self.max_tokens = args.max_tokens
        self.top_p = args.top_p
        self.presence_penalty = args.presence_penalty
        self.frequency_penalty = args.frequency_penalty
        self.skip_verification = args.skip_verification

        # default parameters, can be overridden:
        self.engine = args.engine
        self.generate_engine = args.generate_engine
        self.draft_engine = args.draft_engine
        self.do_refine=args.do_refine

    def generate_next_turn(
        self,
        object_dlg_history: List[DialogueTurn],
        new_user_utterance: str,
        pipeline: str,
        system_parameters: dict = {},
    ):
        """
        Generate the next turn of the dialog
        system_parameters: can override some of the default parameters defined in __init__()
        """
        # throw error if system_parameters contains keys that are not supported
        for key in system_parameters:
            assert key in [
                "engine",
                "generate_engine",
                "draft_engine",
                "do_refine",
            ], f"Unsupported system_parameter key: {key}"

        engine = system_parameters.get("engine", self.engine)
        generate_engine = system_parameters.get("generate_engine", self.generate_engine)
        draft_engine = system_parameters.get("draft_engine", self.draft_engine)
        engine_dict = {"default": engine, "generate": generate_engine, "draft": draft_engine}
        do_refine = system_parameters.get("do_refine", self.do_refine)

        start_time = time.time()

        if pipeline == "generate_and_correct":
            new_dlg_turn = self.generate_and_correct_pipeline(
                object_dlg_history,
                new_user_utterance=new_user_utterance,
                engine_dict=engine_dict,
            )
        elif pipeline == "retrieve_and_generate":
            new_dlg_turn = self.retrieve_and_generate_pipeline(
                object_dlg_history,
                new_user_utterance=new_user_utterance,
                engine_dict=engine_dict,
            )
        elif pipeline == "generate":
            reply = self._generate_only(
                "baseline_chatbot.prompt",
                object_dlg_history,
                new_user_utterance=new_user_utterance,
                engine_dict=engine_dict,
            )
            new_dlg_turn = DialogueTurn(user_utterance=new_user_utterance)
            new_dlg_turn.llm_utterance = reply
            new_dlg_turn.agent_utterance = reply
        elif pipeline == "retrieve_only":
            new_dlg_turn = self.retrieve_only_pipeline(
                object_dlg_history,
                new_user_utterance=new_user_utterance,
                engine_dict=engine_dict,
            )
        elif pipeline == "early_combine":
            new_dlg_turn = self.early_combine_pipeline(
                object_dlg_history,
                new_user_utterance=new_user_utterance,
                engine_dict=engine_dict,
            )
        else:
            raise ValueError

        if do_refine == "True" or do_refine == "true" or do_refine == True:
            do_refine = True
        else:
            do_refine = False

        if do_refine:
            prerefinement_agent_utterance = new_dlg_turn.agent_utterance
            new_dlg_turn.agent_utterance = self.refiner.set_refinement_fields(
                object_dlg_history, new_dlg_turn, engine_dict=engine_dict
            )
            if new_dlg_turn.agent_utterance == prerefinement_agent_utterance:
                logger.info("Refinement did NOT change the agent utterance")

        new_dlg_turn.engine = engine
        new_dlg_turn.generate_engine = generate_engine
        new_dlg_turn.draft_engine = draft_engine
        new_dlg_turn.pipeline = pipeline

        end_time = time.time()
        new_dlg_turn.wall_time_seconds = end_time - start_time

        return new_dlg_turn

    def retrieve_only_pipeline(
        self,
        object_dlg_history: List[DialogueTurn],
        new_user_utterance: str,
        engine_dict: dict,
    ):
        new_dlg_turn = DialogueTurn(user_utterance=new_user_utterance)
        # search based on the history of the dialog so far
        search_prompt_output = llm_generate(
            template_file="query.prompt",
            prompt_parameter_values={
                "dlg": object_dlg_history,
                "new_user_utterance": new_user_utterance,
                "force_search": True,
            },
            engine=engine_dict["default"],
            max_tokens=50,
            temperature=0.0,
            top_p=0.5,
            stop_tokens=["\n"],
            postprocess=False,
        )
        search_prompt_output = (
            "Yes. " + search_prompt_output
        )  # because we are forcing a search
        self._handle_search_prompt_output(
            search_prompt_output=search_prompt_output,
            new_dlg_turn=new_dlg_turn,
            num_paragraphs=1,
            summarize_results=False,
            engine_dict=engine_dict,
        )

        paragraph = new_dlg_turn.initial_search_results[
            0
        ]  # we only retrieve one paragraph
        title = new_dlg_turn.initial_search_result_titles[0]
        new_dlg_turn.agent_utterance = (
            'I found an article titled "' + title + '": ' + paragraph
        )
        return new_dlg_turn

    def retrieve_and_generate_pipeline(
        self,
        object_dlg_history: List[DialogueTurn],
        new_user_utterance: str,
        engine_dict: dict,
    ):
        new_dlg_turn = DialogueTurn(user_utterance=new_user_utterance)
        reply = self._retrieve_and_generate(
            object_dlg_history,
            new_user_utterance,
            new_dlg_turn,
            engine_dict=engine_dict,
        )
        new_dlg_turn.agent_utterance = reply

        return new_dlg_turn

    def generate_and_correct_pipeline(
        self,
        object_dlg_history: List[DialogueTurn],
        new_user_utterance: str,
        engine_dict: str,
    ):
        """
        Verify and correct the last turn of a given dialog using retrieved evidences
        Args:
            - `object_dlg_history` (list): previous dialog turns
            - `new_user_utterance` (str): last user utterance
        Returns:
            - `corrected_reply` (str): corrected LLM response
            - `new_dialog_turn` (DialogTurn)
        """
        original_reply = self._generate_only(
            "generate.prompt",
            object_dlg_history,
            new_user_utterance,
            engine_dict=engine_dict,
        )

        new_dlg_turn = DialogueTurn(user_utterance=new_user_utterance)
        new_dlg_turn.llm_utterance = original_reply

        new_dlg_turn.agent_utterance = self._generate_and_correct_reply(
            object_dlg_history,
            new_user_utterance,
            original_reply,
            new_dlg_turn,
            engine_dict=engine_dict,
        )

        return new_dlg_turn

    def early_combine_pipeline(
        self,
        object_dlg_history: List[DialogueTurn],
        new_user_utterance: str,
        engine_dict: str,
    ):
        new_dlg_turn = DialogueTurn(user_utterance=new_user_utterance)
        original_reply = self._generate_only(
            "generate.prompt",
            object_dlg_history,
            new_user_utterance,
            engine_dict=engine_dict,
        )
        new_dlg_turn.llm_utterance = original_reply

        # gather evidence from two routs in parallel
        with ThreadPoolExecutor(2) as executor:
            search_summary = executor.submit(
                self._search_and_summarize,
                object_dlg_history,
                new_user_utterance,
                new_dlg_turn,
                engine_dict=engine_dict,
            )
            supported_claims = executor.submit(
                self._split_and_fact_check,
                object_dlg_history,
                new_user_utterance,
                original_reply,
                new_dlg_turn,
                engine_dict=engine_dict,
            )
        search_summary = search_summary.result()
        supported_claims = supported_claims.result()

        combined_evi = search_summary + supported_claims
        # logger.info('Combined evidences: %s', new_dlg_turn.combined_evidences)
        new_dlg_turn.combined_evidences = combined_evi

        if not combined_evi:
            logger.info("Combined evidence is empty")
        #     if new_dlg_turn.initial_search_query is None:
        #         new_dlg_turn.combined_utterance = original_reply # no search needed, so return the original chitchat response
        #     else:
        #         new_dlg_turn.combined_utterance = "Sorry, I'm not sure." # will become more conversational after refinement
        # else:
        new_dlg_turn.combined_utterance = self._reply_using_combined_evidence(
            original_reply,
            object_dlg_history,
            new_user_utterance,
            combined_evi,
            engine_dict=engine_dict,
        )
        new_dlg_turn.agent_utterance = new_dlg_turn.combined_utterance

        return new_dlg_turn

    def _handle_search_prompt_output(
        self,
        search_prompt_output: str,
        new_dlg_turn: DialogueTurn,
        num_paragraphs,
        summarize_results: bool,
        engine_dict: dict,
    ):
        """
        Updates `new_dlg_turn` with logs
        A sample output is: Yes. You Google "James E. Webb the administrator of NASA". The year of the results is "none".]
        """
        reranking_factor = 3  # we will retrieve num_paragraphs * reranking_factor paragraphs before reranking them

        search_prompt_output = search_prompt_output.strip()
        search_pattern = (
            r'Yes\. You.*"([^"]*)".* The year of the results is "([^=]*)"\.]?'
        )
        search_match = re.match(search_pattern, search_prompt_output)

        if search_prompt_output.startswith("No"):
            # sometimes LLM outputs No. with extra explanation afterwards instead of ']', or "No search needed". So this more lax condition leads to fewer Exceptions
            logger.info("No search needed.")
        elif search_match:
            search_query = search_match.group(1)
            search_query_time = search_match.group(2)
            y = extract_year(title="", passage=search_query)
            if len(y) > 0:
                logger.info("Overriding query year")
                search_query_time = y[0]
            logger.info("search_query = %s", search_query)
            logger.info("search_query_time = %s", search_query_time)

            # retrieve more paragraphs so that we can do date-based reranking (if needed) and skip "None" summaries (if any)
            paragraphs, scores, titles = self._colbert_retrieve(
                query=search_query,
                num_paragraphs=num_paragraphs * reranking_factor,
                rerank=search_query_time,
            )

            logger.info("Colbert titles: %s", str(titles))

            if summarize_results:
                bullets = []
                not_none_paragraphs = []
                not_none_titles = []
                # summarize in batches, until we reach `num_paragraphs` paragraphs that are deemed relevant
                for start_idx in range(
                    0, num_paragraphs * reranking_factor, num_paragraphs
                ):
                    b, not_none_paragraph_indices = self._summarize_results(
                        search_query,
                        paragraphs[start_idx : start_idx + num_paragraphs],
                        titles[start_idx : start_idx + num_paragraphs],
                        maximum_paragraphs_needed=num_paragraphs
                        - len(not_none_paragraphs),
                        engine_dict=engine_dict,
                    )
                    # print("not_none_paragraph_indices = ", not_none_paragraph_indices)
                    not_none_paragraphs += [
                        paragraphs[start_idx + i] for i in not_none_paragraph_indices
                    ]
                    not_none_titles += [
                        titles[start_idx + i] for i in not_none_paragraph_indices
                    ]
                    bullets = bullets + b
                    assert len(not_none_paragraphs) <= num_paragraphs
                    if len(not_none_paragraphs) == num_paragraphs:
                        break
                titles = not_none_titles
                paragraphs = not_none_paragraphs

            else:
                paragraphs = paragraphs[:num_paragraphs]
                titles = titles[:num_paragraphs]
                bullets = None

            # log everything
            new_dlg_turn.initial_search_query = search_query
            new_dlg_turn.initial_search_query_time = search_query_time
            new_dlg_turn.initial_search_results = paragraphs
            new_dlg_turn.initial_search_result_titles = titles
            new_dlg_turn.initial_search_bullets = bullets
        else:
            raise ValueError(
                "Search prompt's output is invalid: %s" % search_prompt_output
            )
            # logger.error('Search prompt\'s output is invalid: %s' % search_prompt_output)

    def _summarize_results(
        self,
        search_query,
        paragraphs,
        titles,
        maximum_paragraphs_needed,
        engine_dict,
    ):
        """
        Summarizes `paragraphs` and returns the indices of at most `maximum_paragraphs_needed` paragraphs that are deemed relevant to the `query`
        """
        summaries = llm_generate(
            template_file="summarize_and_filter.prompt",
            prompt_parameter_values=[
                {"title": t, "article": p, "query": search_query}
                for (t, p) in zip(titles, paragraphs)
            ],
            engine=engine_dict["default"],
            max_tokens=200,
            temperature=0.0,
            top_p=0.5,
            stop_tokens=None,
            postprocess=False,
        )
        bullets = []
        not_none_paragraph_indices = []
        for paragraph_idx, s in enumerate(summaries):
            if s.startswith("Yes. "):
                # necessary for distilled models
                s = s[5:]
            if s.startswith("None") or s == "- None" or s == "-None":
                # skip the None paragraphs
                logger.info(
                    "This retrieved paragraphs was deemed unrelated: %s",
                    paragraphs[paragraph_idx],
                )
                continue
            not_none_paragraph_indices.append(paragraph_idx)
            for b in s.split("\n-"):
                b = b.strip()
                if len(b) == 0:
                    continue
                if not b.endswith("."):
                    # most likely a partial generation that was cut off because of max_tokens
                    continue
                bullets.append(b.strip("- "))
            if len(not_none_paragraph_indices) == maximum_paragraphs_needed:
                break

        return bullets, not_none_paragraph_indices

    def _retrieve_and_generate(
        self,
        object_dlg_history: List[DialogueTurn],
        new_user_utterance: str,
        new_dlg_turn: DialogueTurn,
        engine_dict: dict,
    ) -> str:
        """
        Retrieves related documents and generates a reply base on them, given the dialog history
        Updates `new_dlg_turn` with logs
        Returns reply
        """
        self._search_and_summarize(
            object_dlg_history, new_user_utterance, new_dlg_turn, engine_dict
        )

        reply = llm_generate(
            template_file="draft.prompt",
            prompt_parameter_values={
                "dlg": object_dlg_history,
                "last_user_utterance": new_user_utterance,
                "evidences": new_dlg_turn.initial_search_bullets,
            },
            engine=engine_dict["default"],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            stop_tokens=["\n"],
            postprocess=True,
        )
        return reply

    def _search_and_summarize(
        self,
        object_dlg_history: List[DialogueTurn],
        new_user_utterance: str,
        new_dlg_turn: DialogueTurn,
        engine_dict: dict,
    ):
        search_prompt_output = llm_generate(
            template_file="query.prompt",
            prompt_parameter_values={
                "dlg": object_dlg_history,
                "new_user_utterance": new_user_utterance,
                "force_search": False,
            },
            engine=engine_dict["default"],
            max_tokens=50,
            temperature=0.0,
            top_p=0.5,
            stop_tokens=["\n"],
            postprocess=False,
        )
        self._handle_search_prompt_output(
            search_prompt_output=search_prompt_output,
            new_dlg_turn=new_dlg_turn,
            num_paragraphs=self.retrieval_num,
            summarize_results=True,
            engine_dict=engine_dict,
        )
        return new_dlg_turn.initial_search_bullets

    def _split_and_fact_check(
        self,
        object_dlg_history: List[DialogueTurn],
        new_user_utterance: str,
        original_reply: str,
        new_dlg_turn: DialogueTurn,
        engine_dict: dict,
    ):
        claims = self.claim_splitter.split_claim(
            dialog_history=object_dlg_history,
            new_user_utterance=new_user_utterance,
            current_agent_utterance=original_reply,
            engine_dict=engine_dict,
        )
        claims = ClaimSplitter.remove_claims_from_previous_turns(claims, object_dlg_history)

        new_dlg_turn.claims = claims
        if not claims:
            logger.info("No claims to check")
            return []

        # retrieve evidence
        ret_output = self._retrieve_evidences(claims)

        # verify claims
        ver_output = self._verify_claims(
            claims,
            ret_output,
            object_dlg_history,
            new_user_utterance,
            original_reply,
            do_correct=False,
            engine_dict=engine_dict,
        )

        new_dlg_turn.verification_retrieval_results = ret_output
        new_dlg_turn.verification_result = ver_output

        # only keep supported claim
        supported_claims = []
        for label_fix in ver_output:
            verification_label, fixed_claim = (
                label_fix["label"],
                label_fix["fixed_claim"],
            )
            if verification_label == "SUPPORTS":
                supported_claims.append(fixed_claim)
        return supported_claims

    def _generate_and_correct_reply(
        self,
        object_dlg_history: List[DialogueTurn],
        new_user_utterance: str,
        original_reply: str,
        new_dlg_turn: DialogueTurn,
        engine_dict: dict,
    ) -> str:
        """
        Verifies and corrects `original_reply` given the dialog history
        Updates `new_dlg_turn` with logs
        Returns corrected reply
        """
        # split claims
        # the returned "claims" is a list of tuples (claim, year)
        claims = self.claim_splitter.split_claim(
            dialog_history=object_dlg_history,
            new_user_utterance=new_user_utterance,
            current_agent_utterance=original_reply,
            engine_dict=engine_dict,
        )
        claims = ClaimSplitter.remove_claims_from_previous_turns(claims, object_dlg_history)
        if not claims:
            logger.info("No claims to check")
            return original_reply
        new_dlg_turn.claims = claims

        # retrieve evidence
        ret_output = self._retrieve_evidences(claims)

        # TODO: use the ret_output together with initial search outputs for verification
        # verify claims
        ver_output = self._verify_claims(
            claims,
            ret_output,
            object_dlg_history,
            new_user_utterance,
            original_reply,
            do_correct=True,
            engine_dict=engine_dict,
        )

        # update dialog turn
        new_dlg_turn.verification_retrieval_results = ret_output
        new_dlg_turn.verification_result = ver_output
        if is_everything_verified(ver_output):
            logger.info("All claims passed verification, nothing to correct")
            return original_reply

        # correction
        corrected_reply = original_reply
        fixed_claims = []
        for label_fix in ver_output:
            verification_label, fixed_claim = (
                label_fix["label"],
                label_fix["fixed_claim"],
            )
            if (
                verification_label == "SUPPORTS"
            ):  # if the claim is already correct, no need to fix
                continue
            fixed_claims.append(fixed_claim)
        assert len(fixed_claims) > 0
        corrected_reply = self._correct(
            original_reply,
            object_dlg_history,
            new_user_utterance,
            fixed_claims,  # corrected claim for REFUTE and "I'm not sure" for NOT ENOUGH INFO claims.
            engine_dict=engine_dict,
        )

        return corrected_reply

    def _generate_only(
        self,
        generation_prompt: str,
        dialog_history: List[DialogueTurn],
        new_user_utterance: str,
        engine_dict: dict,
    ) -> str:
        """
        Generate baseline LLM response
        Args:
            - `generation_prompt` (str): the .prompt file to use for this stage
            - `dialog_history` (list): previous turns
        Returns:
            - `reply`(str): original LLM response
        """
        reply = llm_generate(
            template_file=generation_prompt,
            prompt_parameter_values={
                "dlg": dialog_history,
                "new_user_utterance": new_user_utterance,
                "engine_name": engine_dict["generate"] # used to enforce model knowledge cut-off date for models other than GPT-4
            },
            engine=engine_dict["generate"],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop_tokens=["\n"],
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            postprocess=True,
        )

        return reply

    def _correct(
        self,
        original_reply,
        object_dlg_history,
        last_user_utterance,
        fixed_claims,
        engine_dict: dict,
    ):
        """
        Given context + original response + evidence for a claim, fix the original response

        Args:
            - `original_reply`(str): LLM's original response
            - `object_dlg_history`(list): list of previous DialogueTurns
            - `last_user_utterance` (str): last user utterance
            - `fixed_claims` (list): list of fixed claims
        Returns:
            - `corrected_reply`(str): corrected LLM response
        """
        # correction prompt's context should be in one line
        correction_reply = llm_generate(
            template_file="correction_combiner.prompt",
            prompt_parameter_values={
                "dlg": object_dlg_history,
                "last_user_utterance": last_user_utterance,
                "original_reply": original_reply,
                "fixed_claims": fixed_claims,
            },
            engine=engine_dict["default"],
            max_tokens=self.max_tokens,
            temperature=0,
            stop_tokens=["\n"],
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            postprocess=True,
        )

        return correction_reply

    def _reply_using_combined_evidence(
        self,
        original_reply,
        object_dlg_history,
        last_user_utterance,
        evidences,
        engine_dict: dict,
    ):
        combined_reply = llm_generate(
            template_file="draft.prompt",
            prompt_parameter_values={
                "dlg": object_dlg_history,
                "last_user_utterance": last_user_utterance,
                "original_reply": original_reply,
                "evidences": evidences,
            },
            engine=engine_dict["draft"],
            max_tokens=self.max_tokens,
            temperature=0,
            stop_tokens=None,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            postprocess=True,
        )

        return combined_reply

    def _colbert_retrieve(
        self,
        query: str,
        num_paragraphs: int,
        rerank="none",
        top_p=1,
    ):
        """
        Args:
            `num_paragraphs`: number of paragraphs that will be output
            `rerank` (str): one of 'none', 'recent' or a year like '2005'. 'none' disables reranking. 'recent' retrieves more and returns the most recent ones.
                            '2005' boosts the ranking of results that match 2005. The date of a result is determined by the year numbers it contains.
            `top_p` (float): chooses from the smallest possible set of results whose cumulative probability exceeds top_p
        Returns:
            `passages` (list): a list of passage texts (excluding the title) with the highest similarities to the `query`
            `passage_scores` (list): a list of similarity scores of each passage in `passsages` with `query`
            `passage_titles` (list): a list of passage titles
        """

        # print(self.colbert_endpoint, {'query': query, 'evi_num': num_paragraphs})
        response = requests.get(
            self.colbert_endpoint,
            json={"query": query, "evi_num": num_paragraphs},
        )
        if response.status_code != 200:
            raise Exception("ColBERT Search API Error: %s" % str(response))
        results = response.json()
        passages = []
        passage_titles = []
        for r in results["passages"]:
            r = r.split("|", maxsplit=1)
            passage_titles.append(r[0].strip())
            passages.append(r[1].strip())
        scores = results["passage_scores"]
        probs = results["passage_probs"]
        # print("probs = ", probs)
        top_p_cut_off = np.cumsum(probs) > top_p
        if not np.any(top_p_cut_off):
            # even if we include everything, we don't get to top_p
            top_p_cut_off = len(scores)
        else:
            top_p_cut_off = np.argmax(top_p_cut_off) + 1
        # print("top_p_cut_off = ", top_p_cut_off)
        passages, scores, passage_titles = (
            passages[:top_p_cut_off],
            scores[:top_p_cut_off],
            passage_titles[:top_p_cut_off],
        )

        if rerank == "none":
            pass
        else:
            all_passage_dates = []
            for t, p in zip(passage_titles, passages):
                passage_years = extract_year(title=t, passage=p)
                all_passage_dates.append(passage_years)
            if rerank == "recent":
                sort_fn = lambda x: max(
                    x[3] if len(x[3]) > 0 else [0]
                )  # sort based on the latest year mentioned in the paragraph, demoting paragraphs that don't mention a year
            else:
                # rerank is a year
                try:
                    query_year = int(rerank)
                except ValueError as e:
                    # raise ValueError('rerank should be none, recent or an integer.')
                    logger.error(e)
                    return (
                        passages[:num_paragraphs],
                        scores[:num_paragraphs],
                        passage_titles[:num_paragraphs],
                    )
                sort_fn = lambda x: x[3].count(
                    query_year
                )  # boost the passages that have a matching year with the query, the more they mention the date the more we boost

            # logger.info('Search result dates before date-based reranking: %s', str(all_passage_dates))
            passages, scores, passage_titles, all_passage_dates = list(
                zip(
                    *sorted(
                        zip(passages, scores, passage_titles, all_passage_dates),
                        reverse=True,
                        key=sort_fn,
                    )
                )
            )
            # logger.info('Search result dates after date-based reranking: %s', str(all_passage_dates))

        # choose top num_paragraphs paragraphs
        passages, scores, passage_titles = (
            passages[:num_paragraphs],
            scores[:num_paragraphs],
            passage_titles[:num_paragraphs],
        )

        return passages, scores, passage_titles

    def _retrieve_evidences(self, claims, top_p: float = 1):
        """
        Retrieve evidences
        Args:
            - `claims` (list): list of (claim, year)
            - `top_p` (float): chooses from the smallest possible set of results whose cumulative probability exceeds top_p
        Returns:
            - `ret_output` (dict): a dict from claim_id to a list of `evidence`
                - each `evidence` is a list of length 5: [`title of wikipedia page`, `wikipedia text`, `similarity_score`]
        """
        ret_output = dict()
        for id, (cl, year) in enumerate(claims):
            # if self.args.reranking_method == "none":
            # No re-ranking on evidence. Reranking to match the dates increases the risk of confirmation bias.
            passages, passage_scores, passage_titles = self._colbert_retrieve(
                query=cl, num_paragraphs=self.evi_num, top_p=top_p, rerank="none"
            )
            # else:
            #     # retrieve more so that we can match the dates
            #     passages, passage_scores, passage_titles = self._colbert_retrieve(
            #         query=cl,
            #         num_paragraphs=self.evi_num,
            #         rerank=year,
            #         num_paragraphs_for_reranking=self.evi_num * 3,
            #         top_p=top_p,
            #     )
            evidences = []
            for passage, score, title in zip(passages, passage_scores, passage_titles):
                evidences.append([title, passage, score])
            ret_output[id] = evidences

        return ret_output

    def _verify_claims(
        self,
        claims,
        ret_output,
        object_dlg_history,
        new_user_utterance,
        original_reply,
        do_correct: bool,
        engine_dict: dict,
    ):
        """
        Verify claims using retrieval output
        Args:
            - `claims` (list): list of (claim, year) pairs splitted
            - `ret_output` (dict): a dict from claim_id to a list of `evidence`
                - each `evidence` is a list of length 5: [`title of wikipedia page`, `wikipedia text`, `similarity_score`]
            - `object_dlg_history`(str): list of previous DialogueTurns
            - `last_user_utterance` (str): last user utterance
            - `original_reply`(str): original LLM response
        Returns:
            - `ver_output` (list): a list of verification label ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO") and the fixed claims
        """
        ver_output = []
        parameter_values_list = []

        for claim_id, (cl, year) in enumerate(claims):
            evidences = ret_output[claim_id][: self.evi_num]
            parameter_values_list.append(
                {
                    "dlg": object_dlg_history,
                    "last_user_utterance": new_user_utterance,
                    "original_reply": original_reply,
                    "claim": cl,
                    "evidence_titles": [e[0] for e in evidences],
                    "evidence_texts": [e[1] for e in evidences],
                    "do_correct": do_correct
                }
            )

        # when using gold evidence, we do not split claim so claim is the same with original reply
        if self.skip_verification:
            all_verification_responses = ['is "SUPPORTS"'] * len(claims)
        else:
            all_verification_responses = llm_generate(
                template_file="verify.prompt",
                prompt_parameter_values=parameter_values_list,
                engine=engine_dict["default"],
                max_tokens=200,
                temperature=0,
                stop_tokens=None,
                postprocess=False,
            )

        for (cl, year), verification_response in zip(
            claims, all_verification_responses
        ):
            # logger.info("claim: %s ; verification_response: %s", cl, verification_response)
            # the following handles cases where smaller models like gpt-35-turbo do not follow the few-shot examples' format
            if (
                'is "supports"' in verification_response.lower()
                or "no fact-checking is needed for this claim"
                in verification_response.lower()
                or "the fact-checking result is not applicable to this response"
                in verification_response.lower()
            ):
                verification_label = "SUPPORTS"
                fixed_claim = cl
            elif (
                'the fact-checking result is "not enough info"'
                in verification_response.lower()
            ):
                verification_label = "NOT ENOUGH INFO"
                fixed_claim = ""
            else:
                verification_label = "REFUTES"  # default set to be "REFUTES"
                fixed_claim = ""

            if do_correct and verification_label != "SUPPORTS":
                if "You rewrite your claim:" in verification_response:
                    fixed_claim = verification_response.split(
                        "You rewrite your claim:"
                    )[-1].strip()
                else:
                    logger.error(
                        "verification prompt did not fix a %s. Output: %s"
                        % (verification_label, verification_response)
                    )

            ver_output.append({"label": verification_label, "fixed_claim": fixed_claim})

        return ver_output

    