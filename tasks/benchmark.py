"""
Benchmarking dialogue factual accuracy in the WikiChat paper
"""

from invoke import task

from tasks.defaults import CHATBOT_DEFAULT_CONFIG
from tasks.main import load_api_keys
from tasks.retrieval import get_wikipedia_collection_path


@task(pre=[load_api_keys])
def simulate_users(
    c,
    num_dialogues,  # -1 to simulate all available topics
    num_turns: int,
    simulation_mode: str,  # passage
    subset: str,  # head, recent, tail
    language: str,  # for the topics
    input_file=None,
    user_simulator_engine="gpt-4o",
    user_temperature=1.0,
    # pipeline parameters:
    engine=CHATBOT_DEFAULT_CONFIG["engine"],
    pipeline=CHATBOT_DEFAULT_CONFIG["pipeline"],
    temperature=CHATBOT_DEFAULT_CONFIG["temperature"],
    top_p=CHATBOT_DEFAULT_CONFIG["top_p"],
    retriever_endpoint=CHATBOT_DEFAULT_CONFIG["retriever_endpoint"],
    skip_verification=CHATBOT_DEFAULT_CONFIG["skip_verification"],
    skip_query=CHATBOT_DEFAULT_CONFIG["skip_query"],
    fuse_claim_splitting=CHATBOT_DEFAULT_CONFIG["fuse_claim_splitting"],
    do_refine=CHATBOT_DEFAULT_CONFIG["do_refine"],
    generation_prompt=CHATBOT_DEFAULT_CONFIG["generation_prompt"],
    refinement_prompt=CHATBOT_DEFAULT_CONFIG["refinement_prompt"],
    draft_prompt=CHATBOT_DEFAULT_CONFIG["draft_prompt"],
    retrieval_num=CHATBOT_DEFAULT_CONFIG["retrieval_num"],
    retrieval_reranking_method=CHATBOT_DEFAULT_CONFIG["retrieval_reranking_method"],
    retrieval_reranking_num=CHATBOT_DEFAULT_CONFIG["retrieval_reranking_num"],
    evi_num=CHATBOT_DEFAULT_CONFIG["evi_num"],
    evidence_reranking_method=CHATBOT_DEFAULT_CONFIG["evidence_reranking_method"],
    evidence_reranking_num=CHATBOT_DEFAULT_CONFIG["evidence_reranking_num"],
    generate_engine=CHATBOT_DEFAULT_CONFIG["generate_engine"],
    draft_engine=CHATBOT_DEFAULT_CONFIG["draft_engine"],
    refine_engine=CHATBOT_DEFAULT_CONFIG["refine_engine"],
):
    """
    Simulate user dialogues with a chatbot using specified parameters.

    Accepts all parameters that `inv demo` accepts, plus a few additional parameters for the user simulator.
    """

    pipeline_flags = (
        f"--pipeline {pipeline} "
        f"--engine {engine} "
        f"--claim_prompt split_claims.prompt "
        f"--generation_prompt {generation_prompt} "
        f"--refinement_prompt {refinement_prompt} "
        f"--draft_prompt {draft_prompt} "
        f"--retriever_endpoint {retriever_endpoint} "
        f"--retrieval_num {retrieval_num} "
        f"--retrieval_reranking_method {retrieval_reranking_method} "
        f"--retrieval_reranking_num {retrieval_reranking_num} "
        f"--evi_num {evi_num} "
        f"--evidence_reranking_method {evidence_reranking_method} "
        f"--evidence_reranking_num {evidence_reranking_num} "
        f"--temperature {temperature} "
        f"--top_p {top_p} "
    )
    if generate_engine:
        pipeline_flags += f"--generate_engine {generate_engine} "
    if draft_engine:
        pipeline_flags += f"--draft_engine {draft_engine} "
    if refine_engine:
        pipeline_flags += f"--refine_engine {refine_engine} "

    boolean_pipeline_arguments = {
        "do_refine": do_refine,
        "skip_verification": skip_verification,
        "skip_query": skip_query,
        "fuse_claim_splitting": fuse_claim_splitting,
    }

    for arg, enabled in boolean_pipeline_arguments.items():
        if enabled:
            pipeline_flags += f"--{arg} "

    if not input_file:
        input_file = f"{subset}_articles_{language}.json"

    c.run(
        f"python benchmark/user_simulator.py {pipeline_flags} "
        f"--num_dialogues {num_dialogues} "
        f"--user_engine {user_simulator_engine} "
        f"--user_temperature {user_temperature} "
        f"--mode {simulation_mode} "
        f"--input_file benchmark/topics/{input_file} "
        f"--num_turns {num_turns} "
        f"--output_file benchmark/simulated_dialogues/{pipeline}_{subset}_{language}_{engine}.txt "
        f"--language {language} "
        f"--no_logging"
    )


@task(iterable=["language"])
def benchmark_articles(c, language: str, wikipedia_date: str):
    for lang in language:
        command = (
            "python benchmark/scripts/get_wikipedia_articles_for_benchmark.py "
            f"--collection_path {get_wikipedia_collection_path('workdir', lang, wikipedia_date)} "
            f"--language {lang} "
            f"--recent_output_file benchmark/topics/recent_articles_{lang}.json "
            f"--head_output_file benchmark/topics/head_articles_{lang}.json "
            f"--tail_output_file benchmark/topics/tail_articles_{lang}.json "
            f"--multihop_output_file benchmark/topics/multihop_articles_{lang}.json"
        )

        c.run(command)


@task(pre=[load_api_keys])
def db_to_file(
    c,
    experiment_id="default-experiment",
    output_file=None,
    text_only=True,
    single_system=True,
):
    """Dump database analytics to file"""
    output_file_arg = (
        f"--output_file data/db_analytics/{experiment_id}.txt"
        if output_file is None
        else f"--output_file {output_file}"
    )
    text_only_arg = "--text_only" if text_only else ""
    single_system_arg = "--single_system" if single_system else ""
    c.run(
        f"python benchmark/emnlp_paper/db_analytics.py --experiment_id {experiment_id} {output_file_arg} {text_only_arg} {single_system_arg}"
    )
