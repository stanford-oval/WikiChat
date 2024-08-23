import argparse
import os

import requests
from qdrant_client import QdrantClient

from tasks.defaults import DEFAULT_QDRANT_COLLECTION_NAME


def main():
    """
    This script allows for creating snapshots of the collections existing in a Qdrant database, or loading previous snapshots back into the database.
    It takes in two commandline arguments: --action and --collection_name, with an optional --snapshot_path.

    Usage:
    Saving a snapshot: python this_script.py --action save --collection_name collection_name
    Loading a snapshot: python this_script.py --action load --collection_name collection_name --snapshot_path path_to_snapshot

    Args:
    action (str): Required argument. The action to perform, either 'save' to save a snapshot or 'load' to load a snapshot.
    collection_name (str): The name of the Qdrant collection to deal with. Default to `DEFAULT_QDRANT_COLLECTION_NAME`.
    snapshot_path (str, Optional): The path of the snapshot file to load. Required if action is 'load'.
    """

    parser = argparse.ArgumentParser(description="Create a snapshot.")
    parser.add_argument(
        "--action",
        type=str,
        choices=["save", "load"],
        required=True,
        help="Action to perform: save or load.",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default=DEFAULT_QDRANT_COLLECTION_NAME,
        required=False,
        help="The name of the Qdrant collection.",
    )
    parser.add_argument(
        "--snapshot_path",
        type=str,
        default=None,
        required=False,
        help="Action to perform: save or load.",
    )

    args = parser.parse_args()

    qdrant_client = QdrantClient(url="http://localhost", port=6333, timeout=60, prefer_grpc=False)

    if args.action == "save":
        qdrant_client.create_snapshot(collection_name=args.collection_name, wait=False)
    elif args.action == "load":
        if not args.snapshot_path:
            raise ValueError("Need to provide a snapshot_path to load.")
        snapshot_name = os.path.basename(args.snapshot_path)
        requests.post(
            f"http://localhost:6333/collections/{args.collection_name}/snapshots/upload",
            files={"snapshot": (snapshot_name, open(args.snapshot_path, "rb"))},
        )


if __name__ == "__main__":
    main()
