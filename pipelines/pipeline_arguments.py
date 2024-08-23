"""
Add the common input arguments
"""

from chainlite.llm_config import GlobalVars

from pipelines.chatbot_config import PipelineEnum
from tasks.defaults import CHATBOT_DEFAULT_CONFIG

all_configured_engines = set()
for endpoint in GlobalVars.all_llm_endpoints:
    for engine in endpoint["engine_map"]:
        all_configured_engines.add(engine)


def add_pipeline_arguments(parser):
    # determine components of the pipeline
    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        choices=[e.value for e in PipelineEnum],
        help="The type of pipeline to use. Only used to know which modules to load.",
    )
    parser.add_argument(
        "--generation_prompt",
        default=CHATBOT_DEFAULT_CONFIG["generation_prompt"],
        help="What prompt to use for the generation stage.",
    )
    parser.add_argument(
        "--claim_prompt",
        type=str,
        default="split_claims.prompt",
        help="The path to the file containing the claim LLM prompt.",
    )
    parser.add_argument(
        "--draft_prompt",
        default=CHATBOT_DEFAULT_CONFIG["draft_prompt"],
        help="What prompt to use to draft the final response.",
    )
    parser.add_argument(
        "--refinement_prompt",
        default=CHATBOT_DEFAULT_CONFIG["refinement_prompt"],
        help="What prompt to use to refine the final response.",
    )
    parser.add_argument(
        "--do_refine", action="store_true", help="Whether to refine the final response."
    )
    parser.add_argument(
        "--skip_verification",
        action="store_true",
        help="If True, all claims will be considered correct without fact-checking. Especially useful to speed up debugging of the other parts of the pipeline.",
    )
    parser.add_argument(
        "--skip_query",
        action="store_true",
        help="If True, will just use the last user utterance as the query, instead of generating a query. Useful for single-turn question answering.",
    )
    parser.add_argument(
        "--fuse_claim_splitting",
        action="store_true",
        help="If True, The first claim splitting stage of early_combine pipeline will be fused with the generate stage. Useful for distilled models that have been trained to do this, or when an appropriate generation prompt is used.",
    )
    parser.add_argument(
        "--retriever_endpoint",
        type=str,
        default=CHATBOT_DEFAULT_CONFIG["retriever_endpoint"],
        help="The endpoint to send retrieval requests to.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        required=True,
        choices=all_configured_engines,
        help="The LLM engine to use.",
    )  # choices are from the smallest to the largest model

    parser.add_argument(
        "--generate_engine",
        type=str,
        default=None,
        choices=all_configured_engines,
        help="The LLM engine to use for the 'generate' stage of pipelines. If provided, overrides --engine for that stage.",
    )  # choices are from the smallest to the largest model

    parser.add_argument(
        "--refine_engine",
        type=str,
        default=None,
        choices=all_configured_engines,
        help="The LLM engine to use for the 'refine' stage of pipelines. If provided, overrides --engine for that stage.",
    )  # choices are from the smallest to the largest model

    parser.add_argument(
        "--draft_engine",
        type=str,
        default=None,
        choices=all_configured_engines,
        help="The LLM engine to use for the 'draft' stage of pipelines. If provided, overrides --engine for that stage.",
    )  # choices are from the smallest to the largest model

    # LLM generation hyperparameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=CHATBOT_DEFAULT_CONFIG["temperature"],
        required=False,
        help="Only affects user-facing prompts",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=CHATBOT_DEFAULT_CONFIG["top_p"],
        required=False,
        help="Only affects user-facing prompts",
    )

    # Retrieval parameters
    parser.add_argument(
        "--evi_num",
        type=int,
        default=CHATBOT_DEFAULT_CONFIG["evi_num"],
        help="Number of evidences to retrieve per claim.",
    )

    parser.add_argument(
        "--retrieval_num",
        type=int,
        default=CHATBOT_DEFAULT_CONFIG["retrieval_num"],
        help="Number of passages to retrieve when searching for information.",
    )

    parser.add_argument(
        "--retrieval_reranking_method",
        type=str,
        choices=["none", "date", "llm"],
        default=CHATBOT_DEFAULT_CONFIG["retrieval_reranking_method"],
        help="Reranking method to use for the retrieval stage of pipeline, not evidence retrieval for fact-checking.",
    )

    parser.add_argument(
        "--evidence_reranking_method",
        type=str,
        choices=["none", "date", "llm"],
        default=CHATBOT_DEFAULT_CONFIG["evidence_reranking_method"],
        help="Reranking method to use for evidence retrieval for fact-checking.",
    )

    parser.add_argument(
        "--retrieval_reranking_num",
        type=int,
        default=CHATBOT_DEFAULT_CONFIG["retrieval_reranking_num"],
        help="Number of passages to retrieve before reranking. Will have no effect if `retrieval_reranking_method` is none",
    )

    parser.add_argument(
        "--evidence_reranking_num",
        type=int,
        default=CHATBOT_DEFAULT_CONFIG["evidence_reranking_num"],
        help="Number of evidences to retrieve per claim before reranking. Will have no effect if `evidence_reranking_method` is none",
    )


def check_pipeline_arguments(args):
    if args.retrieval_reranking_method != "none":
        if (
            args.retrieval_reranking_num is None
            or args.retrieval_reranking_num < args.retrieval_num
        ):
            raise ValueError(
                "When retrieval reranking is enabled, you need to specify a `--retrieval_reranking_num` >= `--retrieval_num`"
            )

    if args.evidence_reranking_method != "none":
        if (
            args.evidence_reranking_num is None
            or args.evidence_reranking_num < args.evi_num
        ):
            raise ValueError(
                "When evidence reranking is enabled, you need to specify a `--evidence_reranking_num` >= `--evi_num`"
            )
