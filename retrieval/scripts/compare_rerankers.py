import argparse
import asyncio
import os
import pathlib

import numpy as np
import torch.nn.functional as F
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.reranking import Rerank
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from chainlite import get_total_cost, llm_generation_chain, write_prompt_logs_to_file
from chainlite import run_async_in_parallel
from rich import print
from tqdm import trange
from transformers import AutoModel, AutoTokenizer
import sys

sys.path.insert(0, "./")
from retrieval.llm_reranker import (
    llm_output_to_yes_probability,
    reranker_output_to_list,
)
from retrieval.search_result_block import SearchResultBlock


class PointwiseLLMReranker:
    def __init__(self, engine: str):
        self.llm_chain = (
            llm_generation_chain(
                template_file="pipelines/prompts/rerank_pointwise.prompt",
                engine=engine,
                max_tokens=1,
                return_top_logprobs=20,
                progress_bar_desc="Reranking",
            )
            | llm_output_to_yes_probability
        )

    def predict(
        self, sentences: list[tuple[str, str]], batch_size: int, **kwargs
    ) -> list[float]:
        loop = asyncio.get_event_loop()
        scores_coroutine = run_async_in_parallel(
            self.llm_chain.ainvoke,
            [
                {
                    "query": query,
                    "search_result": SearchResultBlock.construct(
                        document_title="",
                        section_title="",
                        content=doc,
                        similarity_score=0,
                        probability_score=0,
                    ),
                }
                for query, doc in sentences
            ],
            max_concurrency=10,
            desc="Reranking",
        )

        if loop.is_running():
            scores = loop.create_task(scores_coroutine)
        else:
            # If no event loop is running, run the coroutine synchronously
            scores = loop.run_until_complete(scores_coroutine)
        # print("Scores:", scores)
        return scores


# TODO use llm_reranker classes here
class LLMRanker:
    def __init__(self, model_path=None, **kwargs):
        engine = kwargs.get("engine")
        self.llm_reranker_chain = (
            llm_generation_chain(
                template_file="pipelines/prompts/rerank_listwise.prompt",
                engine=engine,
                max_tokens=1500,
                output_json=True,
                # force_skip_cache=True,
            )
            | reranker_output_to_list
        )

    def predict(
        self, sentences: list[tuple[str, str]], batch_size: int, **kwargs
    ) -> list[float]:
        # group sentence by their first element
        query_to_docs = {}
        for sentence in sentences:
            query = sentence[0]
            if query not in query_to_docs:
                query_to_docs[query] = []
            query_to_docs[query].append(sentence[1])

        loop = asyncio.get_event_loop()
        if loop.is_running():
            scores = loop.create_task(self.batch_score_query_document_pairs(sentences))
        else:
            # If no event loop is running, run the coroutine synchronously
            scores = loop.run_until_complete(
                self.batch_score_query_document_pairs(query_to_docs)
            )

        query_docs_to_score = {}
        for query in query_to_docs:
            for doc in query_to_docs[query]:
                query_docs_to_score[(query, doc)] = scores.pop(0)

        scores = [query_docs_to_score[(query, doc)] for query, doc in sentences]

        return scores

    async def batch_score_query_document_pairs(
        self, query_to_docs: dict[str, list[str]]
    ) -> list[float]:
        scores = await run_async_in_parallel(
            self.score_query_document_pair,
            query_to_docs.items(),
            max_concurrency=1,
            desc="Reranking",
        )
        scores = [score for sublist in scores for score in sublist]
        return scores

    async def score_query_document_pair(self, query_docs) -> list[float]:
        query, docs = query_docs
        reranked_indices = await self.llm_rerank_window(
            query,
            [
                SearchResultBlock.construct(
                    document_title="",
                    section_title="",
                    content=d,
                    similarity_score=0,
                    probability_score=0,
                )
                for d in docs
            ],
        )

        query_scores = [0] * len(docs)
        for i, idx in enumerate(reranked_indices):
            query_scores[idx] = 1 - i / len(docs)
        return query_scores

    async def llm_rerank_window(
        self, query: str, single_search_results: list[SearchResultBlock]
    ) -> list[int]:
        """
        Might return fewer indices than the input if the LLM does not output an index for a search result.
        This happens when the LLM deems some indices to be completely irrelevant to the query.
        """
        reranked_indices = await self.llm_reranker_chain.ainvoke(
            {
                "query": query,
                "search_results": single_search_results,
            }
        )

        assert len(reranked_indices) <= len(single_search_results)

        return reranked_indices


class HFEmbedder:
    def __init__(self, model_path=None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.to("cuda")

    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(
        self, queries: list[str], batch_size: int, **kwargs
    ) -> np.ndarray:
        return self._encode_text(queries, batch_size)

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)
    def encode_corpus(
        self, corpus: list[dict[str, str]], batch_size: int, **kwargs
    ) -> np.ndarray:
        return self._encode_text(
            [doc["title"] + "\n" + doc["text"] for doc in corpus], batch_size
        )

    def _encode_text(self, input_texts: list[str], batch_size):
        all_embeddings = []
        for i in trange(0, len(input_texts), batch_size, desc="Encoding"):
            batch_texts = input_texts[i : i + batch_size]
            batch_dict = self.tokenizer(
                batch_texts,
                max_length=1024,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to("cuda")

            outputs = self.model(**batch_dict)

            dimension = 768  # The output dimension of the output embedding, should be in [128, 768]
            embeddings = outputs.last_hidden_state[:, 0][:dimension]

            embeddings = F.normalize(embeddings, p=2, dim=1)
            # convert to numpy
            embeddings = embeddings.detach().cpu().numpy()
            all_embeddings.append(embeddings)

        all_embeddings = np.vstack(all_embeddings)
        return all_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Rerankers using the BEIR benchmark"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "scifact",
            "trec-covid",
            "nfcorpus",
            "robust04",
            "arguana",
            "fever",
        ],  # there are more, but these ones are small and publicly available
        help="Dataset to use",
    )
    parser.add_argument(
        "--engine", type=str, default="gpt-4o-mini", help="The LLM to use"
    )
    parser.add_argument(
        "--reranker",
        type=str,
        required=True,
        choices=["listwise", "pointwise"],
        help="Reranker to use",
    )
    parser.add_argument(
        "--num_to_rerank",
        type=int,
        required=True,
        help="Number of documents to rerank. Must be <= 100",
    )
    args = parser.parse_args()

    # Download the dataset and unzip it
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.dataset}.zip"
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    print("Corpus:", len(corpus), "Queries:", len(queries), "Qrels:", len(qrels))

    # Load the SBERT model and retrieve using cosine-similarity
    model = DRES(HFEmbedder("Alibaba-NLP/gte-multilingual-base"), batch_size=32)
    retriever = EvaluateRetrieval(
        model, score_function="dot"
    )  # or "cos_sim" for cosine similarity

    search_results = retriever.retrieve(corpus, queries)
    ndcg, _map, recall, precision = retriever.evaluate(
        qrels, search_results, retriever.k_values
    )
    print("Before reranking NDCG", ndcg)

    if args.reranker == "listwise":
        reranker = Rerank(LLMRanker(engine=args.engine))
    elif args.reranker == "pointwise":
        reranker = Rerank(PointwiseLLMReranker(engine=args.engine))

    rerank_results = reranker.rerank(
        corpus, queries, search_results, top_k=args.num_to_rerank
    )

    ndcg, _map, recall, precision = retriever.evaluate(
        qrels, rerank_results, retriever.k_values
    )
    print("After reranking NDCG:", ndcg)

    print(f"Total LLM cost: ${get_total_cost():.2f}")
    write_prompt_logs_to_file()
