import argparse
import logging
import sys

from flask import Flask
from flask_cors import CORS
from flask_restful import Api, reqparse
import math
from functools import lru_cache
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "./ColBERT/")

from colbert import Searcher
from colbert.infra import Run, RunConfig
from colbert.infra.config.config import ColBERTConfig

app = Flask(__name__)
CORS(app)
api = Api(app)
logger = logging.getLogger(__name__)


req_parser = reqparse.RequestParser()
req_parser.add_argument("query", type=str, help="The search query")
req_parser.add_argument("evi_num", type=int, help="Number of documents to return")

searcher = None

@lru_cache(maxsize=100000)
def search(query, k):
    search_results = searcher.search(
        query, k=k
    )  # retrieve more results so that our probability estimation is more accurate
    passage_ids, passage_ranks, passage_scores = search_results
    passages = [searcher.collection[passage_id] for passage_id in passage_ids]
    passage_probs = [math.exp(score) for score in passage_scores]
    passage_probs = [prob / sum(passage_probs) for prob in passage_probs]

    results = {
        "passages": passages[:k],
        "passage_ids": passage_ids[:k],
        "passage_ranks": passage_ranks[:k],
        "passage_scores": passage_scores[:k],
        "passage_probs": passage_probs[:k],
    }

    return results


@app.route("/search", methods=["GET", "POST"])
def get():
    args = req_parser.parse_args()
    user_query = args["query"]
    evi_num = args["evi_num"]
    try:
        results = search(user_query, evi_num)
    except Exception as e:
        logger.error(str(e))
        return {}, 500

    return results


def init():
    # print("init() called with args: ", sys.argv)
    global searcher
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--colbert_experiment_name",
        type=str,
        default="wikipedia_all",
        help="name of colbert indexing experiment",
    )
    arg_parser.add_argument(
        "--colbert_index_path",
        type=str,
        help="path to colbert index",
    )
    arg_parser.add_argument(
        "--colbert_checkpoint",
        type=str,
        help="path to the folder containing the colbert model checkpoint",
    )
    arg_parser.add_argument(
        "--colbert_collection_path",
        type=str,
        help="path to colbert document collection",
    )
    # Has the type str since it has to support inputs from gunicorn
    arg_parser.add_argument(
        "--memory_map",
        type=str,
        choices=["True", "False", "true", "false"],
        default="False",
        help="If set, will keep the ColBERT index on disk, so that we use less RAM. But you need to coalesce the index first.",
    )
    args, unknown = arg_parser.parse_known_args()
    # print("args = ", args)

    if args.memory_map.lower() == "true":
        args.memory_map = True
    else:
        args.memory_map = False

    with Run().context(RunConfig(experiment=args.colbert_experiment_name, index_root="")):
        searcher = Searcher(
            index=args.colbert_index_path,
            checkpoint=args.colbert_checkpoint,
            collection=args.colbert_collection_path,
            config=ColBERTConfig(load_index_with_mmap=args.memory_map)
        )

    # warm up ColBERT
    search("Query ColBERT in order to warm it up.", k=5)
    search("Since the first few queries are quite slow", k=5)
    search("Especially with memory_map=True", k=5)

    
# used as a way to pass commandline arguments to gunicorn
def gunicorn_app(*args, **kwargs):
    # Gunicorn CLI args are useless.
    # https://stackoverflow.com/questions/8495367/
    #
    # Start the application in modified environment.
    # https://stackoverflow.com/questions/18668947/
    #
    import sys
    sys.argv = ['--gunicorn']
    for k in kwargs:
        sys.argv.append("--" + k)
        sys.argv.append(kwargs[k])
    init()
    return app

if __name__ == "__main__":
    # Run the app with Flask
    init()
    app.run(port=5000, debug=False, use_reloader=False)
