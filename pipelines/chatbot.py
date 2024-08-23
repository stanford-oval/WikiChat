import re
from operator import itemgetter
from time import time

from chainlite import chain, register_prompt_constants
from langchain.chains.base import Chain
from langchain_core.runnables import (
    RunnableAssign,
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
)
from langgraph.graph import END, StateGraph

from pipelines.chatbot_config import RefineStageConfig, get_chatbot_config
from pipelines.dialogue_state import (
    Claim,
    ClaimLabel,
    DialogueState,
    DialogueTurn,
    Feedback,
    SearchQuery,
    clear_current_turn,
    state_to_dict,
)
from pipelines.langchain_helpers import stage_prompt

from .retriever import batch_retrieve, rerank, retrieve
from .utils import extract_year, get_logger

register_prompt_constants({"chatbot_name": "WikiChat"})

logger = get_logger(__name__)


async def run_one_turn(
    chatbot, dialogue_state: DialogueState, new_user_utterance: str, callbacks=[]
):
    clear_current_turn(dialogue_state)
    dialogue_state["new_user_utterance"] = new_user_utterance
    start_time = time()
    dialogue_state = await chatbot.ainvoke(
        dialogue_state,
        config={"callbacks": callbacks},
    )
    end_time = time()
    new_agent_utterance = dialogue_state["new_agent_utterance"]
    dialogue_state["wall_time"] = (
        str(int((end_time - start_time) * 100) / 100) + " seconds"
    )

    return new_agent_utterance, dialogue_state


@chain
async def process_summary_prompt_output(summary_prompt_output: str) -> list[str]:
    bullet_points = []
    if (
        summary_prompt_output.startswith("None")
        or summary_prompt_output == "- None"
        or summary_prompt_output == "-None"
        or summary_prompt_output.startswith("- There is no information available")
    ):
        return bullet_points
    for b in summary_prompt_output.split("\n-"):
        b = b.strip()
        if len(b) == 0:
            continue
        bullet_points.append(b.strip("- "))
    return bullet_points


@chain
async def process_claim_prompt_output(claim_prompt_output: str) -> list[Claim]:
    # TODO support streaming
    lines = claim_prompt_output.split("\n")
    if lines[0].startswith("Nothing."):
        # no claims detected
        return []
    all_claims = []
    if len(lines) == 1:
        # Sometimes claim_prompt_output is malformed and does not have \n between claims
        lines = lines[0].split("- ")
    try:
        for c in lines:
            claim = c
            cleaned_claim = claim.split("- ")[-1].strip()
            if cleaned_claim:
                split_term = " The year of the results is "
                if split_term not in cleaned_claim:
                    split_term = " The year of the claim is "
                splitted = cleaned_claim.split(split_term)
                if len(splitted) == 2:
                    cleaned_claim, year = splitted
                    year = year[1:-2]
                else:
                    # sometimes model claim_prompt_output may not be well-formatted (e.g. N/A), or the splitting prompt might not include dates; default to none
                    cleaned_claim = splitted[0]
                    year = "none"
                all_claims.append(Claim(claim=cleaned_claim, time=year))
    except Exception as e:
        logger.error(
            "Error while parsing claims in %s: %s", claim_prompt_output, str(e)
        )
        raise e

    return all_claims


@chain
async def process_verification_prompt_output(_input, do_correct: bool) -> Claim:
    claim: Claim = _input["claim"]
    verification_prompt_output: str = _input["verification_prompt_output"]
    fixed_claim = ""
    if (
        'is "supports"' in verification_prompt_output.lower()
        or 'would be "supports"' in verification_prompt_output.lower()
        or "no fact-checking is needed for this claim"
        in verification_prompt_output.lower()
        or "the fact-checking result is not applicable to this response"
        in verification_prompt_output.lower()
    ):
        verification_label = ClaimLabel.supported
    elif (
        'is "not enough info"' in verification_prompt_output.lower()
        or 'would be "not enough info"' in verification_prompt_output.lower()
    ):
        verification_label = ClaimLabel.nei
    else:
        verification_label = ClaimLabel.refuted  # default set to be "REFUTES"

    if do_correct and verification_label != ClaimLabel.supported:
        if "You rewrite your claim:" in verification_prompt_output:
            fixed_claim = verification_prompt_output.split("You rewrite your claim:")[
                -1
            ].strip()
        else:
            logger.error(
                "verification prompt did not fix a %s. Output: %s"
                % (verification_label, verification_prompt_output)
            )

    claim.label = verification_label
    claim.fixed_claim = fixed_claim
    return claim


@chain
async def process_search_prompt_output(
    search_prompt_output: str, force_search: bool, reranking_method
) -> tuple[bool, str, str]:
    if force_search and not search_prompt_output.startswith("Yes."):
        search_prompt_output = (
            "Yes. " + search_prompt_output
        )  # because we are forcing a search, so the "Yes." part is already given in the prompt
    search_prompt_output = search_prompt_output.strip()
    search_pattern = r'Yes\. You.*"([^"]*)".* The year of the results is "([^=]*)"\.]?'

    if search_prompt_output.startswith("No"):
        # sometimes LLM outputs No. with extra explanation afterwards instead of ']', or "No search needed". So this more lax condition leads to fewer Exceptions
        logger.info("No search needed.")
        return False, None, None

    search_match = re.match(search_pattern, search_prompt_output)
    if search_match:
        search_query = search_match.group(1)
        search_query_time = search_match.group(2)
        if reranking_method == "date":
            y = extract_year(title="", content=search_query)
            if len(y) > 0:
                logger.info("Overriding query year")
                search_query_time = y[0]
        logger.debug("search_query = %s", search_query)
        logger.debug("search_query_time = %s", search_query_time)
        return True, search_query, search_query_time
    else:
        logger.error("Search prompt's output is invalid: %s" % search_prompt_output)
        return False, None, None


async def parse_feedback(feedback) -> Feedback | None:
    criteria = ["Relevant", "Temporally Correct", "Natural", "Non-Repetitive"]
    if "User:" in feedback:
        feedback = feedback.split("User:")[0]
    feedback_lines = feedback.strip().split("\n")

    if len(feedback_lines) != len(criteria):
        logger.error("Feedback malformatted")
        logger.error(feedback_lines)
        return None

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

    return Feedback(
        string=feedback_lines, scores=dict([(c, s) for c, s in zip(criteria, scores)])
    )


@chain
async def process_refine_prompt_output(
    _input: dict,
    refine_stage_config: RefineStageConfig,
) -> tuple[str, Feedback | None]:
    utterance_to_refine: str = _input["utterance_to_refine"]
    refine_prompt_output: str = _input["refine_prompt_output"]
    # print("utterance_to_refine = ", utterance_to_refine)
    # TODO add streaming support. If scores are perfect, don't wait for the rest of the output and just yield `draft_prompt_output`
    if refine_stage_config.template_file.endswith("refine_w_feedback.prompt"):
        refine_identifiers = [
            "Revised response after applying this feedback:",
            "Response after applying this feedback:",
            "Improved response after applying this feedback:",
            "Improved response:",
            "Revised response:",
        ]
        for identifier in refine_identifiers:
            if identifier in refine_prompt_output:
                feedback, refined_agent_utterance = refine_prompt_output.split(
                    identifier
                )
                feedback = await parse_feedback(feedback)

                if not feedback or feedback.is_full_score():
                    # skip refinement if it malformed or already gets full score
                    return utterance_to_refine, feedback
                else:
                    return refined_agent_utterance.strip(), feedback

        logger.error(
            "Skipping refinement due to malformatted Refined response: %s",
            refine_prompt_output,
        )
        return utterance_to_refine, None
    else:
        # There is no feedback part to the output
        if refine_prompt_output.startswith("Chatbot:"):
            refine_prompt_output = refine_prompt_output[
                len(refine_prompt_output) :
            ].strip()
        return refine_prompt_output, None


async def query_stage(state):
    config = state["chatbot_config"].query_stage
    if config.skip:
        return
    should_search, query, query_time = await (
        stage_prompt(
            config,
            bind_prompt_values={
                "force_search": config.force_search
            },  # Determine whether to force a search based on the current query stage.
        )
        | process_search_prompt_output.bind(
            force_search=config.force_search,
            reranking_method=state["chatbot_config"].retrieve_stage.reranking_method,
        )
    ).ainvoke(
        {
            "dlg": state["dialogue_history"],
            "new_user_utterance": state["new_user_utterance"],
        }
    )
    if should_search:
        return {"initial_search_query": SearchQuery(query=query, time=query_time)}
    else:
        return {}


async def search_stage(state):
    config = state["chatbot_config"].retrieve_stage
    if config.skip or not state["initial_search_query"]:
        return
    # This sets retrieval_results of the query
    await (
        RunnableAssign(
            {
                "query": retrieve.bind(
                    retriever_endpoint=config.endpoint,
                    pre_reranking_num=config.pre_reranking_num,
                )
            }
        )
        | rerank.bind(retrieval_config=config)
    ).ainvoke({"query": state["initial_search_query"]})

    # state["initial_search_query"].retrieval_results = retrieval_results
    return {
        "initial_search_query": state["initial_search_query"]
    }  # return something so that chainlit can display the outputs


async def summarize_stage(state):
    config = state["chatbot_config"].summarize_stage
    if config.skip or not state["initial_search_query"]:
        return
    output = await (stage_prompt(config) | process_summary_prompt_output).abatch(
        [
            {"result": search_result, "query": state["initial_search_query"].text}
            for search_result in state["initial_search_query"].retrieval_results
        ]
    )
    for search_result, bullet_points in zip(
        state["initial_search_query"].retrieval_results, output
    ):
        search_result.content_summary = bullet_points

    return {
        "initial_search_query": state["initial_search_query"]
    }  # return something so that chainlit can display the outputs


async def generate_stage(state):
    config = state["chatbot_config"].generate_stage
    if config.skip:
        return
    generate_stage_output = await stage_prompt(config).ainvoke(
        {
            "new_user_utterance": state["new_user_utterance"],
            "dlg": state["dialogue_history"],
        }
    )
    return {"generate_stage_output": generate_stage_output}


async def split_claim_stage(state):
    config = state["chatbot_config"].split_claim_stage
    if config.skip:
        return
    if not state["generate_stage_output"]:
        return

    if config.fused and (
        state["generate_stage_output"] == "Nothing."
        or state["generate_stage_output"] == "None"
    ):
        # "Nothing." and "None" are for models that merge generation and claim_splitting
        return

    generation_claims = await RunnableBranch(
        (
            lambda x: not config.fused,
            stage_prompt(config) | process_claim_prompt_output,
        ),
        (
            RunnableLambda(lambda x: x["llm_utterance"]) | process_claim_prompt_output
        ),  # else
    ).ainvoke(
        {
            "dlg": state["dialogue_history"],
            "new_user_utterance": state["new_user_utterance"],
            "llm_utterance": state["generate_stage_output"],
        }
    )

    return {"generation_claims": generation_claims}


async def verify_stage(state):
    config = state["chatbot_config"]
    if config.verify_stage.skip:
        return
    if not state["generation_claims"]:
        # no claims to verify, perhaps the LLM has abstained from responding
        return
    verify_claim = (
        batch_retrieve.bind(  # use batch_retrieve here to handle batching more efficiently
            retriever_endpoint=config.retrieve_evidence_stage.endpoint,
            pre_reranking_num=config.retrieve_evidence_stage.pre_reranking_num,
        )
        | RunnableAssign(
            {"claim": rerank.bind(retrieval_config=config.retrieve_evidence_stage)}
        ).abatch
        | RunnableParallel(
            verification_prompt_output=stage_prompt(
                config.verify_stage, {"do_correct": False}
            ),
            claim=itemgetter("claim"),
        ).abatch
        | process_verification_prompt_output.bind(do_correct=False).abatch
    )
    await verify_claim.ainvoke(
        {"query": [c for c in state["generation_claims"]]}
    )  # updates claim labels in-place

    return {"generation_claims": state["generation_claims"]}


async def draft_stage(state):
    config = state["chatbot_config"].draft_stage
    if config.skip:
        return
    output = await stage_prompt(config).ainvoke(
        {
            "dlg": state["dialogue_history"],
            "evidences": (
                [
                    b
                    for result in state["initial_search_query"].retrieval_results
                    for b in result.content_summary
                ]
                if state["initial_search_query"]
                else []
                + [
                    claim.text
                    for claim in state["generation_claims"]
                    if claim.label == ClaimLabel.supported
                ]
            ),
            "new_user_utterance": state["new_user_utterance"],
        }
    )
    return {"draft_stage_output": output}


def get_first_available_string(l: list[str]) -> str:
    for a in l:
        if a:
            return a
    return None


async def refine_stage(state):
    config = state["chatbot_config"].refine_stage
    if config.skip:
        return
    utterance_to_refine = get_first_available_string(
        [state["draft_stage_output"], state["generate_stage_output"]]
    )

    refined_utterance, feedback = await (
        RunnableAssign({"refine_prompt_output": stage_prompt(config)})
        | process_refine_prompt_output.bind(refine_stage_config=config)
    ).ainvoke(
        {
            "utterance_to_refine": utterance_to_refine,
            "dlg": state["dialogue_history"],
            "new_user_utterance": state["new_user_utterance"],
        },
    )
    return {"refine_stage_output": refined_utterance, "feedback": feedback}


async def add_to_history(state):
    # Order of priority of where to get the agent utterance from
    new_agent_utterance = get_first_available_string(
        [
            state["refine_stage_output"],
            state["draft_stage_output"],
            state["generate_stage_output"],
        ]
    )
    if not new_agent_utterance:
        raise ValueError("Was not able to find the new_agent_utterance from any stages")
    return {
        "new_agent_utterance": new_agent_utterance,
        "dialogue_history": [
            DialogueTurn(
                agent_utterance=new_agent_utterance,
                user_utterance=state["new_user_utterance"],
                initial_search_query=state["initial_search_query"],
                turn_log=state_to_dict(state),
            )
        ],
    }


def create_chain(args) -> Chain:
    config = get_chatbot_config(args)
    initial_state = DialogueState(chatbot_config=config, dialogue_history=[])

    graph = StateGraph(DialogueState)

    # nodes
    graph.add_node("start", lambda x: {})
    graph.set_entry_point("start")
    # retrieval route
    graph.add_node("query", query_stage)
    graph.add_node("search", search_stage)
    graph.add_node("summarize", summarize_stage)
    # generate route
    graph.add_node("generate", generate_stage)
    graph.add_node("split_claims", split_claim_stage)
    graph.add_node("verify", verify_stage)
    # final stages
    graph.add_node("draft", draft_stage)
    graph.add_node("refine", refine_stage)
    graph.add_node("add_to_history", add_to_history)

    # retrieval route nodes
    graph.add_edge("start", "query")
    graph.add_edge(
        "query", "search"
    )  # search and summarize will be skipped if query is empty
    graph.add_edge("search", "summarize")
    # generate route
    graph.add_edge("start", "generate")
    graph.add_edge("generate", "split_claims")
    graph.add_edge("split_claims", "verify")
    # final stages
    graph.add_edge(
        ["summarize", "verify"], "draft"
    )  # add an edge from query to draft so that it waits for it even when we skip summarization
    graph.add_edge("draft", "refine")
    graph.add_edge("refine", "add_to_history")
    graph.add_edge("add_to_history", END)

    runnable = graph.compile()

    # runnable.get_graph().print_ascii()
    return runnable, initial_state
