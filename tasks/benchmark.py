"""
Benchmarking dialogue factual accuracy in the WikiChat paper
"""

from invoke.tasks import task

from tasks.defaults import CHATBOT_DEFAULT_CONFIG
from tasks.main import load_api_keys
from tasks.retrieval import get_wikipedia_collection_path


@task(pre=[load_api_keys], iterable=["subset", "language"])
def simulate_users(
    c,
    num_dialogues,  # -1 to simulate all available topics
    num_turns: int,
    simulation_mode: str,  # passage
    subset: list[str],  # head, recent, tail
    language: list[str],
    input_file=None,
    user_simulator_engine="gpt-4o",
    user_temperature=1.0,
    # pipeline parameters:
    engine=CHATBOT_DEFAULT_CONFIG["engine"],
    retriever_endpoint=CHATBOT_DEFAULT_CONFIG[
        "retriever_endpoint"
    ],  # TODO use corpus_id instead
    do_refine=CHATBOT_DEFAULT_CONFIG["do_refine"],
    query_post_reranking_num=CHATBOT_DEFAULT_CONFIG["query_post_reranking_num"],
    do_reranking=CHATBOT_DEFAULT_CONFIG["do_reranking"],
    query_pre_reranking_num=CHATBOT_DEFAULT_CONFIG["query_pre_reranking_num"],
    claim_post_reranking_num=CHATBOT_DEFAULT_CONFIG["claim_post_reranking_num"],
    claim_pre_reranking_num=CHATBOT_DEFAULT_CONFIG["claim_pre_reranking_num"],
):
    """
    Simulate user dialogues with a chatbot using specified parameters.

    Accepts all parameters that `inv demo` accepts, plus a few additional parameters for the user simulator.
    """

    if not language or not subset:
        raise ValueError("Specify at least one --language and one --subset")

    pipeline_flags = (
        f"--engine {engine} "
        f"--retriever_endpoint {retriever_endpoint} "
        f"--query_post_reranking_num {query_post_reranking_num} "
        f"--query_pre_reranking_num {query_pre_reranking_num} "
        f"--claim_post_reranking_num {claim_post_reranking_num} "
        f"--claim_pre_reranking_num {claim_pre_reranking_num} "
    )
    boolean_pipeline_arguments = {
        "do_refine": do_refine,
        "do_reranking": do_reranking,
    }

    for arg, enabled in boolean_pipeline_arguments.items():
        if enabled:
            pipeline_flags += f"--{arg} "

    for lang in language:
        for s in subset:
            if not input_file:
                input_file = f"{s}_articles_{lang}.json"

            c.run(
                f"python benchmark/user_simulator.py {pipeline_flags} "
                f"--num_dialogues {num_dialogues} "
                f"--user_engine {user_simulator_engine} "
                f"--user_temperature {user_temperature} "
                f"--mode {simulation_mode} "
                f"--input_file benchmark/topics/{input_file} "
                f"--num_turns {num_turns} "
                f"--output_file benchmark/simulated_dialogues/{s}_{lang}_{engine}.txt "
                f"--language {lang} "
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
