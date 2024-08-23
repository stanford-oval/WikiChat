import argparse

from qdrant_client import QdrantClient, models

from tasks.defaults import DEFAULT_QDRANT_COLLECTION_NAME


def main(language_to_delete, collection_name):
    client = QdrantClient(url="http://localhost", timeout=600, prefer_grpc=False)

    res = client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="language",
                        match=models.MatchValue(value=language_to_delete),
                    ),
                ],
            )
        ),
    )

    print(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete a language from the database.")
    parser.add_argument("--language", type=str, help="The language to delete")
    parser.add_argument(
        "--collection",
        default=DEFAULT_QDRANT_COLLECTION_NAME,
        type=str,
        help="The Qdrant collection to delete from",
    )

    args = parser.parse_args()
    main(args.language, args.collection)
