import argparse

from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

from tasks.defaults import DEFAULT_QDRANT_COLLECTION_NAME

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create payload indices for a Qdrant collection"
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default=DEFAULT_QDRANT_COLLECTION_NAME,
        help="Name of the collection in Qdrant",
    )
    parser.add_argument(
        "--field_names",
        type=str,
        nargs="+",
        default=["language", "block_type"],
        help="List of field names to create payload indices for",
    )

    args = parser.parse_args()

    collection_name = args.collection_name

    qdrant_client = QdrantClient(
        url="http://localhost", port=6333, timeout=60, prefer_grpc=False
    )

    for field_name in args.field_names:
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=PayloadSchemaType.KEYWORD,
            wait=False,
        )
