"""Anonymous Parallel Search MCP retrieval for the chatbot pipeline."""

import asyncio
import json
from urllib.parse import urlsplit

from retrieval.retrieval_commons import QueryResult, SearchResultBlock

MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "WikiChat/0.1 (+https://github.com/stanford-oval/WikiChat)"


def _is_web_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        return (
            parsed.scheme in ("http", "https")
            and bool(parsed.hostname)
            and not parsed.username
            and not parsed.password
            and not any(c.isspace() for c in value)
        )
    except ValueError:
        return False


def _read_results(response, limit: int) -> QueryResult:
    if response.isError:
        message = "\n".join(
            item.text for item in response.content if item.type == "text"
        )
        raise RuntimeError(f"Parallel retrieval failed: {message}")
    payload = response.structuredContent
    if payload is None:
        payload = json.loads(
            "\n".join(item.text for item in response.content if item.type == "text")
        )
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("Parallel returned an invalid results payload")
    if payload.get("errors") and not payload["results"]:
        raise RuntimeError(
            "Parallel page fetching failed; the service returned URL errors"
        )
    blocks = []
    for result in payload["results"]:
        if not isinstance(result, dict):
            raise ValueError("Parallel returned an invalid result")
        url = result.get("url", "")
        if not isinstance(url, str) or not _is_web_url(url):
            continue
        excerpts = result.get("excerpts", [])
        if not isinstance(excerpts, list) or not all(
            isinstance(e, str) for e in excerpts
        ):
            raise ValueError("Parallel returned invalid excerpts")
        content = "\n\n".join(e.strip() for e in excerpts if e.strip())
        if not content:
            continue
        blocks.append(
            SearchResultBlock(
                document_title=result.get("title") or url,
                section_title="",
                content=content,
                url=url,
                # Parallel supplies an ordering, but no comparable vector scores.
                similarity_score=0,
                probability_score=0,
            )
        )
        if len(blocks) >= limit:
            break
    return QueryResult(results=blocks)


async def retrieve_via_parallel(
    queries: list[str],
    post_reranking_num: int,
    session_id: str,
    timeout: float = 30,
) -> list[QueryResult]:
    """Search each objective, or fetch a query that is itself an HTTP(S) URL.

    The caller owns session_id for the whole dialogue, independently of this
    short-lived MCP connection. Explicit failures never select another provider.
    """
    if not queries or post_reranking_num <= 0:
        return [QueryResult() for _ in queries]

    # Keep the SDK optional: existing retriever installations need no new packages.
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError as error:
        raise RuntimeError(
            "Parallel retrieval requires: pip install -r requirements-parallel.txt"
        ) from error

    async with asyncio.timeout(timeout):
        results = []
        failure = None
        async with streamablehttp_client(
            MCP_URL,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            sse_read_timeout=timeout,
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                for query in queries:
                    if _is_web_url(query):
                        name = "web_fetch"
                        arguments = {"urls": [query], "session_id": session_id}
                    else:
                        name = "web_search"
                        arguments = {
                            "objective": query,
                            "search_queries": [query],
                            "session_id": session_id,
                        }
                    response = await session.call_tool(name, arguments)
                    try:
                        results.append(_read_results(response, post_reranking_num))
                    except (RuntimeError, ValueError) as error:
                        failure = error
                        break
        # Raise after SDK task groups close normally so callers receive the
        # actionable tool/parse error, rather than a generic ExceptionGroup.
        if failure is not None:
            raise failure
        return results
