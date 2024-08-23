from enum import Enum

from pydantic import BaseModel, Field


class StageConfig(BaseModel):
    template_file: str
    engine: str
    max_tokens: int
    temperature: float = Field(default=0.0)
    top_p: float = Field(default=0.9)
    stop_tokens: list | None = Field(default=None)
    postprocess: bool = Field(default=False)
    skip: bool = Field(default=False)


class RerankingMethodEnum(str, Enum):
    none = "none"
    date = "date"
    llm = "llm"


class RetrievalConfig(StageConfig):
    """
    If reranking_method is none, post_reranking_num is ignored
    LLM parameters are used only when `reranking_method == llm`
    """

    endpoint: str
    reranking_method: RerankingMethodEnum
    pre_reranking_num: int
    post_reranking_num: int


class QueryStageConfig(StageConfig):
    force_search: bool = Field(default=False)


class SplitClaimStageConfig(StageConfig):
    fused: bool = Field(default=False)


class KBStageConfig(StageConfig):
    # template_file here is the verbalizer
    pass


class RefineStageConfig(StageConfig):
    pass


class PipelineEnum(str, Enum):
    generate_and_correct = "generate_and_correct"
    retrieve_and_generate = "retrieve_and_generate"
    generate = "generate"
    retrieve_only = "retrieve_only"
    early_combine = "early_combine"


class ChatbotConfig(BaseModel):
    pipeline: PipelineEnum
    generate_stage: StageConfig
    query_stage: QueryStageConfig
    retrieve_stage: RetrievalConfig
    summarize_stage: StageConfig
    split_claim_stage: SplitClaimStageConfig
    retrieve_evidence_stage: RetrievalConfig
    verify_stage: StageConfig
    draft_stage: StageConfig
    refine_stage: RefineStageConfig


def get_chatbot_config(args):
    generate_stage = StageConfig(
        skip=False,
        template_file="generate.prompt",
        engine=args.engine,
        max_tokens=400,
        temperature=0,
        top_p=1,
    )
    query_stage = QueryStageConfig(
        skip=False,
        template_file="query.prompt",
        engine=args.engine,
        max_tokens=50,
        stop_tokens=["\n"],
        force_search=False,
    )
    retrieve_stage = RetrievalConfig(
        endpoint=args.retriever_endpoint,
        reranking_method=args.retrieval_reranking_method,
        pre_reranking_num=args.retrieval_reranking_num,
        post_reranking_num=args.retrieval_num,
        template_file="rerank.prompt",
        engine=args.engine,
        max_tokens=120,
    )
    # kb_stage = KBStageConfig()
    summarize_stage = StageConfig(
        template_file="summarize_and_filter.prompt",
        engine=args.engine,
        max_tokens=200,
    )
    split_claim_stage = SplitClaimStageConfig(
        template_file="split_claims.prompt",
        engine=args.engine,
        max_tokens=350,
    )
    retrieve_evidence_stage = RetrievalConfig(
        endpoint=args.retriever_endpoint,
        reranking_method=args.evidence_reranking_method,
        pre_reranking_num=args.evidence_reranking_num,
        post_reranking_num=args.evi_num,
        template_file="rerank.prompt",
        engine=args.engine,
        max_tokens=100,
    )
    verify_stage = StageConfig(
        skip=False,
        template_file="verify.prompt",
        engine=args.engine,
        max_tokens=200,
    )
    draft_stage = StageConfig(
        skip=False,
        template_file="draft.prompt",
        engine=args.engine,
        max_tokens=250,
    )
    refine_stage = RefineStageConfig(
        skip=False,
        template_file="refine_w_feedback.prompt",
        engine=args.engine,
        max_tokens=400,
    )

    if args.pipeline == "generate_and_correct":
        generate_stage.skip = True
        query_stage.skip = True
        retrieve_stage.skip = True
        summarize_stage.skip = True
        split_claim_stage.skip = True
        retrieve_evidence_stage.skip = True
        verify_stage.skip = True
        draft_stage.skip = True
    elif args.pipeline == "retrieve_and_generate":
        generate_stage.skip = True
        split_claim_stage.skip = True
        retrieve_evidence_stage.skip = True
        verify_stage.skip = True
    elif args.pipeline == "generate":
        query_stage.skip = True
        retrieve_stage.skip = True
        summarize_stage.skip = True
        split_claim_stage.skip = True
        retrieve_evidence_stage.skip = True
        verify_stage.skip = True
        draft_stage.skip = True
    elif args.pipeline == "retrieve_only":
        generate_stage.skip = True
        summarize_stage.skip = True
        split_claim_stage.skip = True
        retrieve_evidence_stage.skip = True
        verify_stage.skip = True
        draft_stage.skip = True
    elif args.pipeline == "early_combine":
        pass  # do nothing and keep all stages
    else:
        raise ValueError("Pipeline not supported: ", args.pipeline)

    if args.refine_engine is not None:
        refine_stage.engine = args.refine_engine
    if args.generate_engine is not None:
        generate_stage.engine = args.generate_engine
    if args.draft_engine is not None:
        draft_stage.engine = args.draft_engine

    user_facing_stage = [
        stage for stage in [refine_stage, draft_stage, generate_stage] if not stage.skip
    ][0]
    user_facing_stage.temperature = args.temperature
    user_facing_stage.top_p = args.top_p

    split_claim_stage.template_file = args.claim_prompt
    generate_stage.template_file = args.generation_prompt
    refine_stage.template_file = args.refinement_prompt
    draft_stage.template_file = args.draft_prompt
    if args.do_refine:
        refine_stage.skip = False
    else:
        refine_stage.skip = True
    if args.skip_verification:
        retrieve_evidence_stage.skip = True
        verify_stage.skip = True
    if args.skip_query:
        query_stage.skip = True
    if args.fuse_claim_splitting:
        split_claim_stage.fused = True

    config = ChatbotConfig(
        pipeline=args.pipeline,
        generate_stage=generate_stage,
        query_stage=query_stage,
        retrieve_stage=retrieve_stage,
        # kb_stage=kb_stage,
        summarize_stage=summarize_stage,
        split_claim_stage=split_claim_stage,
        retrieve_evidence_stage=retrieve_evidence_stage,
        verify_stage=verify_stage,
        draft_stage=draft_stage,
        refine_stage=refine_stage,
    )

    return config
