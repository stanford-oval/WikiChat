import os
import sys

sys.path.insert(0, "./ColBERT/")

import argparse

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nbits", type=int)
    parser.add_argument("--doc_maxlen", type=int)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--index_name", type=str)
    parser.add_argument("--collection", type=str)
    parser.add_argument("--nranks", type=int, help="Number of GPUs to use for indexing")
    args = parser.parse_args()

    print(args)

    collection = Collection(path=args.collection)

    f"Loaded {len(collection):,} passages"

    print("example passage from collection: ", collection[0])
    print()

    with Run().context(
        RunConfig(nranks=args.nranks, experiment=args.experiment_name)
    ):  # nranks specifies the number of GPUs to use.
        config = ColBERTConfig(doc_maxlen=args.doc_maxlen, nbits=args.nbits)

        indexer = Indexer(checkpoint=args.checkpoint, config=config)
        indexer.index(name=args.index_name, collection=collection, overwrite=True)

    print("index location: ", indexer.get_index())

    with Run().context(RunConfig(experiment=args.experiment_name, index_root="")):
        searcher = Searcher(
            index=os.path.join(
                f"experiments/{args.experiment_name}/indexes", args.index_name
            ),
            collection=collection,
        )

    query = "who is Haruki Murakami"
    print(f"#> {query}")

    # Find the top-3 passages for this query
    results = searcher.search(query, k=3)

    # Print out the top-k retrieved passages
    for passage_id, passage_rank, passage_score in zip(*results):
        print(
            f"\t [{passage_rank}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}"
        )
