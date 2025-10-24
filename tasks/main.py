import os

import redis
from invoke.tasks import task

from utils.logging import logger
from tasks.defaults import (
    CHATBOT_DEFAULT_CONFIG,
    DEFAULT_BACKEND_PORT,
    DEFAULT_REDIS_PORT,
)


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
                    logger.debug(f"Loaded API key named {key}")

    except Exception as e:
        logger.error(
            f"Error while loading API keys from API_KEY. Make sure this file exists, and has the correct format. {str(e)}"
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
        c.run(
            f"docker run --rm -d --name redis-stack -p {redis_port}:6379 -p 8001:8001 redis/redis-stack:latest"
        )
        return

    logger.debug("Redis server is already running.")


@task(pre=[start_redis, load_api_keys], aliases=["chainlit"])
def start_backend(c, backend_port=DEFAULT_BACKEND_PORT):
    """Start backend using chainlit"""
    c.run(f"chainlit run -h --port {backend_port} backend_server.py", pty=True)


@task(pre=[load_api_keys, start_redis], aliases=["test"])
def tests(c):
    """Run tests using pytest (stop after first failure)"""
    c.run(
        "pytest -x "
        "-rP "
        "--color=yes "
        "--disable-warnings "
        "./tests/test_pipelines.py "
        "./tests/test_wikipedia_preprocessing.py "
        "./tests/test_search_query.py "
        "./tests/test_custom_docling.py ",
        pty=True,
    )


@task(pre=[load_api_keys, start_redis])
def demo(
    c,
    engine=CHATBOT_DEFAULT_CONFIG["engine"],
    do_refine=CHATBOT_DEFAULT_CONFIG["do_refine"],
    query_post_reranking_num=CHATBOT_DEFAULT_CONFIG["query_post_reranking_num"],
    do_reranking=CHATBOT_DEFAULT_CONFIG["do_reranking"],
    query_pre_reranking_num=CHATBOT_DEFAULT_CONFIG["query_pre_reranking_num"],
    claim_post_reranking_num=CHATBOT_DEFAULT_CONFIG["claim_post_reranking_num"],
    claim_pre_reranking_num=CHATBOT_DEFAULT_CONFIG["claim_pre_reranking_num"],
    corpus_id=CHATBOT_DEFAULT_CONFIG["corpus_id"],
):
    """
    Start a chatbot with the specified configurations, to interact with using the terminal.

    Parameters:
    - c: Context from invoke task.
    - engine: specifies the AI model to use.
    - do_refine: Enables refining answers given by the chatbot.
    - query_pre_reranking_num: Number of documents to be fed to the re-ranker.
    - do_reranking: Whether to do reranking for retrieved documents.
    - query_post_reranking_num: Number of documents to retrieve.
    - claim_pre_reranking_num: Number of evidences to be fed to the re-ranker for each claim.
    - claim_post_reranking_num: Number of evidences to consider for each sub-claim, after reranking is done.
    - corpus_id: The ID of the corpus to use.
    """

    pipeline_flags = (
        f"--engine {engine} "
        f"--query_post_reranking_num {query_post_reranking_num} "
        f"--query_pre_reranking_num {query_pre_reranking_num} "
        f"--claim_post_reranking_num {claim_post_reranking_num} "
        f"--claim_pre_reranking_num {claim_pre_reranking_num} "
        f'--corpus_id "{corpus_id}" '
    )

    boolean_pipeline_arguments = {
        "do_refine": do_refine,
        "do_reranking": do_reranking,
    }

    for arg, enabled in boolean_pipeline_arguments.items():
        if enabled:
            pipeline_flags += f"--{arg} "

    command = f"python -m command_line_chatbot {pipeline_flags}"

    c.run(command, pty=True)


@task(pre=[load_api_keys, start_redis])
def persuabot(
    c,
    engine=CHATBOT_DEFAULT_CONFIG["engine"],
    query_post_reranking_num=CHATBOT_DEFAULT_CONFIG["query_post_reranking_num"],
    do_reranking=CHATBOT_DEFAULT_CONFIG["do_reranking"],
    query_pre_reranking_num=CHATBOT_DEFAULT_CONFIG["query_pre_reranking_num"],
    claim_post_reranking_num=CHATBOT_DEFAULT_CONFIG["claim_post_reranking_num"],
    claim_pre_reranking_num=CHATBOT_DEFAULT_CONFIG["claim_pre_reranking_num"],
    corpus_id=CHATBOT_DEFAULT_CONFIG["corpus_id"],
    persuasion_domain="general",
    target_goal="have a helpful and persuasive conversation",
    do_fact_checking=True,
    do_strategy_decomposition=True,
    debug=False,
):
    """
    Start PersuaBot, a persuasive chatbot with fact-checking and strategy maintenance.

    PersuaBot is based on the paper "Zero-shot Persuasive Chatbots with LLM-Generated
    Strategies and Information Retrieval" (Furumai et al., 2024).

    Parameters:
    - c: Context from invoke task.
    - engine: specifies the AI model to use.
    - query_pre_reranking_num: Number of documents to be fed to the re-ranker.
    - do_reranking: Whether to do reranking for retrieved documents.
    - query_post_reranking_num: Number of documents to retrieve.
    - claim_pre_reranking_num: Number of evidences to be fed to the re-ranker for each claim.
    - claim_post_reranking_num: Number of evidences to consider for each sub-claim.
    - corpus_id: The ID of the corpus to use.
    - persuasion_domain: Domain for persuasion (e.g., 'donation', 'health', 'recommendation').
    - target_goal: The specific persuasion goal.
    - do_fact_checking: Whether to perform fact-checking on generated content.
    - do_strategy_decomposition: Whether to decompose response into strategy sections.
    - debug: Show strategy breakdown for each response.
    """

    pipeline_flags = (
        f"--engine {engine} "
        f"--query_post_reranking_num {query_post_reranking_num} "
        f"--query_pre_reranking_num {query_pre_reranking_num} "
        f"--claim_post_reranking_num {claim_post_reranking_num} "
        f"--claim_pre_reranking_num {claim_pre_reranking_num} "
        f'--corpus_id "{corpus_id}" '
        f'--persuasion_domain "{persuasion_domain}" '
        f'--target_goal "{target_goal}" '
    )

    boolean_pipeline_arguments = {
        "do_reranking": do_reranking,
        "do_fact_checking": do_fact_checking,
        "do_strategy_decomposition": do_strategy_decomposition,
        "debug": debug,
    }

    for arg, enabled in boolean_pipeline_arguments.items():
        if enabled:
            pipeline_flags += f"--{arg} "

    command = f"python -m command_line_persuabot {pipeline_flags}"

    c.run(command, pty=True)


@task
def format_code(c):
    """Format code using black and isort, excluding folders that start with a dot"""
    c.run("ruff format . --exclude '.*'", pty=True)
    c.run("ruff check . --exclude '.*' --fix", pty=True)
