import os

import redis
from invoke import task

from pipelines.utils import get_logger
from tasks.defaults import (
    CHATBOT_DEFAULT_CONFIG,
    DEFAULT_BACKEND_PORT,
    DEFAULT_REDIS_PORT,
)

logger = get_logger(__name__)


@task
def load_api_keys(c):
    """
    Load API keys from a file named 'API_KEYS' and set them as environment variables.

    This function reads the 'API_KEYS' file line by line, extracts key-value pairs,
    and sets them as environment variables. Lines starting with '#' are treated as
    comments and ignored. The expected format for each line in the file is 'KEY=VALUE'.

    Parameters:
    - c: Context, automatically passed by invoke.

    Raises:
    - Exception: If there is an error while reading the 'API_KEYS' file or setting
      the environment variables, an error message is logged.
    """
    try:
        with open("API_KEYS") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = tuple(line.split("=", 1))
                    key, value = key.strip(), value.strip()
                    os.environ[key] = value
                    logger.debug("Loaded API key named %s", key)

    except Exception as e:
        logger.error(
            "Error while loading API keys from API_KEY. Make sure this file exists, and has the correct format. %s",
            str(e),
        )


@task()
def start_redis(c, redis_port: int = DEFAULT_REDIS_PORT):
    """
    Start a Redis server if it is not already running.

    This task attempts to connect to a Redis server on the specified port.
    If the connection fails (indicating that the Redis server is not running),
    it starts a new Redis server on that port.

    Parameters:
    - c: Context, automatically passed by invoke.
    - redis_port (int): The port number on which to start the Redis server. Defaults to DEFAULT_REDIS_PORT.
    """
    try:
        r = redis.Redis(host="localhost", port=redis_port)
        r.ping()
    except redis.exceptions.ConnectionError:
        logger.info("Redis server not found, starting it now...")
        c.run(f"redis-server --port {redis_port} --daemonize yes")
        return

    logger.debug("Redis server is already running.")


@task(pre=[start_redis, load_api_keys], aliases=["chainlit"])
def start_backend(c, backend_port=DEFAULT_BACKEND_PORT):
    """Start backend using chainlit"""
    c.run(f"chainlit run -h --port {backend_port} backend_server.py", pty=True)


@task(pre=[load_api_keys, start_redis], aliases=["test"])
def tests(c):
    """Run tests using pytest"""
    c.run(
        "pytest "
        "-rP "
        "--color=yes "
        "--disable-warnings "
        "./tests/test_pipelines.py "
        "./tests/test_wikipedia_preprocessing.py "
        "./tests/test_retriever.py ",
        pty=True,
    )


@task(pre=[load_api_keys, start_redis])
def demo(
    c,
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
    Start a chatbot with the specified configurations, to interact with using the terminal.

    Parameters:
    - c: Context from invoke task.
    - pipeline: Defines the pipeline to use for processing queries. One of: generate_and_correct, retrieve_and_generate, generate, retrieve_only, early_combine
    - engine: specifies the AI model to use.
    - temperature: Controls randomness in generation for user-facing stages. 0 means greedy decoding.
    - top_p: Top-p sampling for user-facing stages, high values (close to 1.0) mean more diversity.
    - retriever_endpoint: The URL and port number to use for the retriever server.
    - skip_verification: Skips the step of verifying claims (if applicable) and counts all claims as SUPPORTED.
    - skip_query: Skips generating a query, and instead uses the user utterance as the query.
    - fuse_claim_splitting: Determines if claim splitting should be fused with the `generate` stage, meaning that in the generate stage, the model outputs claims in bullet point format.
    - do_refine: Enables refining answers given by the chatbot.
    - generation_prompt: Template file for the generation stage's prompt
    - refinement_prompt: Template file for the refinement stage's prompt.
    - draft_prompt: Template file for the draft stage's prompt.
    - retrieval_reranking_num: Number of documents to be fed to the re-ranker.
    - retrieval_reranking_method: Method to use for re-ranking retrieved documents.
    - retrieval_num: Number of documents to initially retrieve.
    - evidence_reranking_num: Number of evidences to be fed to the re-ranker for each claim.
    - evidence_reranking_method: Method for re-ranking evidence documents. One of "none", "llm" or "date"
    - evi_num: Number of evidences to consider for each sub-claim, after re-ranking is done.
    - generate_engine, draft_engine, refine_engine: Optionally specify different AI models for different stages.
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

    command = f"python command_line_chatbot.py {pipeline_flags}"

    c.run(command)
