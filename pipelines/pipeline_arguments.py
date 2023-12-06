"""
Add the common input arguments
"""

from llm.global_variables import (
    local_model_list,
    together_model_list,
    openai_chat_model_list,
    openai_nonchat_model_list,
)


def add_pipeline_arguments(parser):
    # determine components of the pipeline
    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        choices=[
            "generate_and_correct",
            "retrieve_and_generate",
            "generate",
            "retrieve_only",
            "early_combine",
            "atlas",
        ],
        default="generate_and_correct",
        help="The type of pipeline used to imrpove GPT-3 response. Only used to know which modules to load.",
    )
    parser.add_argument(
        "--claim_prompt_template_file",
        type=str,
        default="split_claims.prompt",
        help="The path to the file containing the claim LLM prompt.",
    )
    parser.add_argument(
        "--refinement_prompt",
        default="refine_w_feedback.prompt",
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
        "--colbert_endpoint",
        type=str,
        default="http://127.0.0.1:5000/search",
        help="whether using colbert for retrieval.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        required=True,
        choices=["atlas"]
        + local_model_list
        + together_model_list
        + openai_chat_model_list
        + openai_nonchat_model_list,
        help="The LLM engine to use.",
    )  # choices are from the smallest to the largest model

    parser.add_argument(
        "--generate_engine",
        type=str,
        default=None,
        choices=["atlas"]
        + local_model_list
        + together_model_list
        + openai_chat_model_list
        + openai_nonchat_model_list,
        help="The LLM engine to use for the 'generate' stage of pipelines. If provided, overrides --engine for that stage.",
    )  # choices are from the smallest to the largest model

    parser.add_argument(
        "--draft_engine",
        type=str,
        default=None,
        choices=["atlas"]
        + local_model_list
        + together_model_list
        + openai_chat_model_list
        + openai_nonchat_model_list,
        help="The LLM engine to use for the 'draft' stage of pipelines. If provided, overrides --engine for that stage.",
    )  # choices are from the smallest to the largest model

    parser.add_argument(
        "--reranking_method",
        type=str,
        choices=["none", "date"],
        default="none",
        help="Only used for retrieve_and_generate pipeline",
    )

    # LLM generation hyperparameters
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=250,
        required=False,
        help="Only affects user-facing prompts",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        required=False,
        help="Only affects user-facing prompts",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        required=False,
        help="Only affects user-facing prompts",
    )
    parser.add_argument(
        "--frequency_penalty",
        type=float,
        default=0.0,
        required=False,
        help="Only affects user-facing prompts",
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0.0,
        required=False,
        help="Only affects user-facing prompts",
    )

    parser.add_argument(
        "--evi_num",
        type=int,
        default=2,
        help="Number of evidences to retrieve per claim.",
    )

    parser.add_argument(
        "--retrieval_num",
        type=int,
        default=3,
        help="Number of passages to retrieve when searching for information.",
    )


def check_pipeline_arguments(args):
    # make sure for ATLAS, both engine and pipeline are set to 'atlas'
    if hasattr(args, "pipeline"):
        if (args.engine == "atlas" and args.pipeline != "atlas") or (
            args.engine != "atlas" and args.pipeline == "atlas"
        ):
            raise ValueError(
                "When using ATLAS, both `engine` and `pipeline` input arguments should be set to 'atlas'."
            )

    if args.generate_engine is None:
        # default to args.engine
        args.generate_engine = args.engine
    if args.draft_engine is None:
        # default to args.engine
        args.draft_engine = args.engine
