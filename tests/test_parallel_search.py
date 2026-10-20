"""Exercise the real pipeline and MCP SDK with deterministic HTTP responses.

Only model inference and the remote HTTP server are scripted. No LLM credentials,
Redis server, embedding model or live search service is required.
"""

import argparse
import asyncio
import builtins
import json
import sys
from uuid import UUID

import httpx
import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

sys.path.insert(0, "./")
from pipelines import chatbot
from pipelines.dialogue_state import ChatbotConfig, DialogueState, DialogueTurn
from pipelines.pipeline_arguments import (
    add_pipeline_arguments,
    check_pipeline_arguments,
)
from retrieval import parallel_search
from tasks.defaults import CHATBOT_DEFAULT_CONFIG

pytest.importorskip("mcp")


def make_chain(provider="parallel", **overrides):
    parser = argparse.ArgumentParser()
    add_pipeline_arguments(parser)
    flags = ["--engine", "local", "--search_provider", provider]
    if provider == "retriever":
        flags += ["--corpus_id", CHATBOT_DEFAULT_CONFIG["corpus_id"]]
    for key, value in overrides.items():
        flags += [f"--{key}", str(value)]
    args = parser.parse_args(flags)
    check_pipeline_arguments(args)
    return chatbot.create_chain(args)


@pytest.fixture
def model(monkeypatch):
    """Script each model stage while retaining the real graph and consumers."""
    inputs = []

    def generation(prompt, **kwargs):
        def respond(values):
            state = values["dlg"]
            text = state.current_turn.user_utterance
            inputs.append((prompt, values))
            if prompt == "query.prompt":
                query = "None" if text == "Hi" else text
                return f"<search_query>{query}</search_query>"
            if prompt == "generate_split_claims.prompt":
                return "None" if text == "Hi" else "Austen first novel publication"
            if prompt == "filter_irrelevant_info.prompt":
                return (
                    f"<relevant_content>{values['result'].content}</relevant_content>"
                )
            if prompt == "draft_w_citation.prompt":
                results = state.current_turn.filtered_search_results
                return (
                    "<response>"
                    + (results[0].content + " [1]" if results else "Hello")
                    + "</response>"
                )
            raise AssertionError(f"Unexpected model stage: {prompt}")

        return RunnableLambda(respond)

    monkeypatch.setattr(chatbot, "llm_generation_chain", generation)
    return inputs


@pytest.fixture
def server(monkeypatch):
    """Observe real JSON-RPC traffic from the SDK, including discovery."""
    requests = []
    control = {"mode": "ok", "structured": True}
    original_client = httpx.AsyncClient

    async def handle(request):
        assert str(request.url) == parallel_search.MCP_URL
        assert request.headers["user-agent"] == parallel_search.USER_AGENT
        assert "authorization" not in request.headers
        message = json.loads(request.content)
        requests.append(message)
        method = message["method"]
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            result = {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture", "version": "1"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {"name": name, "inputSchema": {"type": "object"}}
                    for name in ("web_search", "web_fetch")
                ]
            }
        elif method == "tools/call":
            mode = control["mode"]
            if mode == "timeout":
                await asyncio.Event().wait()
            if mode == "transport":
                raise httpx.ConnectError("fixture connection failed", request=request)
            if mode == "quota":
                result = {
                    "isError": True,
                    "content": [
                        {"type": "text", "text": "Free anonymous quota exceeded"}
                    ],
                }
            elif mode == "malformed":
                result = {"content": [{"type": "text", "text": "not JSON"}]}
            else:
                arguments = message["params"]["arguments"]
                url = arguments.get("urls", ["https://example.org/austen"])[0]
                rows = control.get(
                    "rows",
                    [
                        {
                            "url": url,
                            "title": "Jane Austen",
                            "excerpts": [
                                "Sense and Sensibility was published in 1811."
                            ],
                        }
                    ],
                )
                payload = {"results": rows}
                if "errors" in control:
                    payload["errors"] = control["errors"]
                result = {"content": [{"type": "text", "text": json.dumps(payload)}]}
                if control["structured"]:
                    result["structuredContent"] = payload
        else:
            raise AssertionError(f"Unexpected MCP method: {method}")
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "id": message["id"], "result": result}
        )

    class Client(original_client):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    return requests, control


def tool_calls(requests):
    return [r["params"] for r in requests if r["method"] == "tools/call"]


@pytest.mark.asyncio
async def test_graph_consumes_search_and_fetch_with_stable_dialogue_id(
    server, model, monkeypatch
):
    monkeypatch.setenv("PARALLEL_API_KEY", "unused-ambient-credential")
    graph, state = make_chain()
    assert state.config.llm_corpus_description == "the public web"
    assert UUID(state.search_session_id).version == 4
    for query in [
        "Jane Austen first novel",
        "https://example.org/austen",
        "What year was it?",
    ]:
        state.turns.append(DialogueTurn(user_utterance=query))
        await graph.ainvoke(state)
        turn = state.current_turn
        assert len(turn.search_results) == len(turn.search_query) == 1
        assert len(turn.llm_claim_search_results) == len(turn.llm_claims) == 1
        assert "1811" in turn.agent_utterance
        assert "[1]" in turn.agent_utterance
        assert turn.filtered_search_results[0].url == "https://example.org/austen"
        assert turn.filtered_search_results[0].summary
    requests, _ = server
    calls = tool_calls(requests)
    assert len(calls) == 6
    assert {call["arguments"]["session_id"] for call in calls} == {
        state.search_session_id
    }
    assert [call["name"] for call in calls].count("web_fetch") == 1
    assert {r["method"] for r in requests} >= {"initialize", "tools/list", "tools/call"}
    assert any(prompt == "filter_irrelevant_info.prompt" for prompt, _ in model)
    assert any(prompt == "draft_w_citation.prompt" for prompt, _ in model)


@pytest.mark.asyncio
async def test_new_and_interleaved_dialogues_keep_separate_ids(server, model):
    graph_a, state_a = make_chain()
    graph_b, state_b = make_chain()
    assert state_a.search_session_id != state_b.search_session_id
    for _ in range(2):
        state_a.turns.append(DialogueTurn(user_utterance="Austen first book"))
        state_b.turns.append(DialogueTurn(user_utterance="Austen first book"))
        await asyncio.gather(graph_a.ainvoke(state_a), graph_b.ainvoke(state_b))
    ids = [c["arguments"]["session_id"] for c in tool_calls(server[0])]
    assert ids.count(state_a.search_session_id) == 4
    assert ids.count(state_b.search_session_id) == 4
    # Both the terminal entrypoint and Chainlit new-chat handler use create_chain.
    _, reset_state = make_chain()
    assert reset_state.search_session_id not in ids
    restored = DialogueState.model_validate(state_a.model_dump())
    assert restored.search_session_id == state_a.search_session_id


@pytest.mark.asyncio
async def test_skipped_and_disabled_search_make_no_requests(server, model):
    graph, state = make_chain()
    state.turns.append(DialogueTurn(user_utterance="Hi"))
    await graph.ainvoke(state)
    assert server[0] == []
    graph, state = make_chain(query_post_reranking_num=0, claim_post_reranking_num=0)
    state.turns.append(DialogueTurn(user_utterance="Austen first book"))
    await graph.ainvoke(state)
    assert server[0] == []
    assert len(state.current_turn.search_results) == 1
    assert not state.current_turn.filtered_search_results


@pytest.mark.asyncio
async def test_default_and_saved_retriever_settings_are_preserved(monkeypatch, model):
    calls = []

    def retrieve(**kwargs):
        calls.append(kwargs)
        return [parallel_search.QueryResult() for _ in kwargs["queries"]]

    async def unexpected(*args, **kwargs):
        pytest.fail("Explicit retriever must not contact Parallel")

    monkeypatch.setattr(chatbot, "retrieve_via_api", retrieve)
    monkeypatch.setattr(chatbot, "retrieve_via_parallel", unexpected)
    parser = argparse.ArgumentParser()
    add_pipeline_arguments(parser)
    args = parser.parse_args(
        ["--engine", "local", "--corpus_id", CHATBOT_DEFAULT_CONFIG["corpus_id"]]
    )
    check_pipeline_arguments(args)
    assert args.search_provider == "retriever"
    graph, state = chatbot.create_chain(args)
    state.turns.append(DialogueTurn(user_utterance="Austen"))
    await graph.ainvoke(state)
    assert {c["retriever_endpoint"] for c in calls} == {
        CHATBOT_DEFAULT_CONFIG["retriever_endpoint"]
    }
    calls.clear()
    graph, state = make_chain(
        "retriever", retriever_endpoint="https://custom.example/search"
    )
    restored = ChatbotConfig.model_validate(state.config.model_dump())
    assert restored.search_provider == "retriever"
    state.turns.append(DialogueTurn(user_utterance="Austen"))
    await graph.ainvoke(state)
    assert {c["retriever_endpoint"] for c in calls} == {"https://custom.example/search"}


@pytest.mark.asyncio
async def test_explicit_parallel_uses_no_custom_corpus_endpoint(
    server, model, monkeypatch
):
    def unexpected(**kwargs):
        pytest.fail("Parallel must not fall back to the corpus endpoint")

    monkeypatch.setattr(chatbot, "retrieve_via_api", unexpected)
    graph, state = make_chain(retriever_endpoint="https://custom.example/search")
    assert state.config.retriever_endpoint == "https://custom.example/search"
    state.turns.append(DialogueTurn(user_utterance="Austen"))
    await graph.ainvoke(state)
    assert len(tool_calls(server[0])) == 2


@pytest.mark.asyncio
async def test_quota_error_and_retry_keep_dialogue_identity(server, model, monkeypatch):
    requests, control = server
    control["mode"] = "quota"
    _, state = make_chain()
    state.turns.append(
        DialogueTurn(
            search_query=["Austen", "Later objective"],
            llm_claims=["Austen first book", "Later claim"],
        )
    )
    messages = []
    monkeypatch.setattr(chatbot.logger, "error", messages.append)

    def unexpected(**kwargs):
        pytest.fail("No implicit provider fallback is allowed")

    monkeypatch.setattr(chatbot, "retrieve_via_api", unexpected)
    # Retain the existing behavior: query search logs failures; claim search raises.
    await chatbot.search_stage.ainvoke(state)
    assert state.current_turn.search_results == []
    assert "Free anonymous quota exceeded" in messages[0]
    assert len(tool_calls(requests)) == 1  # No further objectives after a tool error.
    with pytest.raises(RuntimeError, match="Free anonymous quota exceeded"):
        await chatbot.llm_claim_search_stage.ainvoke(state)
    assert len(tool_calls(requests)) == 2
    control["mode"] = "ok"
    await chatbot.search_stage.ainvoke(state)
    assert state.current_turn.search_results[0].results
    assert {c["arguments"]["session_id"] for c in tool_calls(requests)} == {
        state.search_session_id
    }


@pytest.mark.asyncio
async def test_failed_url_fetch_is_reported_through_claim_stage(server, monkeypatch):
    requests, control = server
    control["rows"] = []
    control["errors"] = [
        {
            "url": "https://example.org/missing",
            "error_type": "http_error",
            "http_status_code": 404,
        }
    ]
    _, state = make_chain()
    state.turns.append(
        DialogueTurn(llm_claims=["https://example.org/missing", "Later claim"])
    )

    def unexpected(**kwargs):
        pytest.fail("A failed page fetch must not change provider")

    monkeypatch.setattr(chatbot, "retrieve_via_api", unexpected)
    with pytest.raises(RuntimeError, match="Parallel page fetching failed"):
        await chatbot.llm_claim_search_stage.ainvoke(state)
    assert state.current_turn.llm_claim_search_results == []
    assert len(tool_calls(requests)) == 1


@pytest.mark.asyncio
async def test_timeout_and_cancellation_reach_actual_caller(server):
    requests, control = server
    control["mode"] = "timeout"
    _, state = make_chain(parallel_timeout=0.1)
    state.turns.append(DialogueTurn(llm_claims=["Austen first book"]))
    with pytest.raises(TimeoutError):
        await chatbot.llm_claim_search_stage.ainvoke(state)
    assert tool_calls(requests)
    state.config.parallel_timeout = 30
    task = asyncio.create_task(chatbot.llm_claim_search_stage.ainvoke(state))
    # Wait for the second invocation to reach the transport before cancellation.
    async with asyncio.timeout(5):
        while len(tool_calls(requests)) < 2:
            await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["transport", "malformed"])
async def test_protocol_failures_are_not_successful_empty_results(server, mode):
    server[1]["mode"] = mode
    _, state = make_chain()
    state.turns.append(DialogueTurn(llm_claims=["Austen"]))
    with pytest.raises(Exception):
        await chatbot.llm_claim_search_stage.ainvoke(state)
    assert tool_calls(server[0])
    assert not state.current_turn.llm_claim_search_results


@pytest.mark.asyncio
@pytest.mark.parametrize("structured", [True, False])
async def test_urls_empty_excerpts_order_and_limits(server, structured):
    requests, control = server
    control["structured"] = structured
    control["rows"] = [
        {"url": "javascript:alert(1)", "excerpts": ["unsafe"]},
        {"url": "https://user:password@example.org", "excerpts": ["unsafe"]},
        {"url": "https://example.org/empty", "excerpts": [" "]},
        {
            "url": "https://example.org/first?a=b#part",
            "title": None,
            "excerpts": [" first ", "second"],
        },
        {"url": "https://example.org/extra", "title": "Extra", "excerpts": ["extra"]},
    ]
    _, state = make_chain(claim_post_reranking_num=1)
    state.turns.append(DialogueTurn(llm_claims=["Austen"]))
    await chatbot.llm_claim_search_stage.ainvoke(state)
    result = state.current_turn.llm_claim_search_results[0]
    assert len(result.results) == 1
    block = result.results[0]
    assert block.url == "https://example.org/first?a=b#part"
    assert block.document_title == block.url
    assert block.content == "first\n\nsecond"
    assert block.similarity_score == block.probability_score == 0
    assert tool_calls(requests)[0]["arguments"]["search_queries"] == ["Austen"]
    control["rows"] = []
    await chatbot.llm_claim_search_stage.ainvoke(state)
    assert state.current_turn.llm_claim_search_results[0].results == []


def test_invalid_provider_and_timeout_are_rejected():
    _, state = make_chain()
    config = state.config.model_dump()
    with pytest.raises(ValidationError):
        ChatbotConfig.model_validate({**config, "search_provider": "unknown"})
    for timeout in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValidationError):
            ChatbotConfig.model_validate({**config, "parallel_timeout": timeout})


@pytest.mark.asyncio
async def test_parallel_ignores_corpus_reranking_options(server, model):
    graph, state = make_chain(query_pre_reranking_num=0, claim_pre_reranking_num=0)
    state.turns.append(DialogueTurn(user_utterance="Austen first novel"))
    await graph.ainvoke(state)
    assert len(tool_calls(server[0])) == 2


@pytest.mark.asyncio
async def test_optional_sdk_is_not_needed_for_existing_retriever(monkeypatch, model):
    original_import = builtins.__import__

    def without_mcp(name, *args, **kwargs):
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError("optional SDK absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_mcp)
    monkeypatch.setattr(
        chatbot,
        "retrieve_via_api",
        lambda **kwargs: [parallel_search.QueryResult() for _ in kwargs["queries"]],
    )
    graph, state = make_chain("retriever")
    state.turns.append(DialogueTurn(user_utterance="Austen"))
    await graph.ainvoke(state)
    _, parallel_state = make_chain()
    parallel_state.turns.append(DialogueTurn(llm_claims=["Austen"]))
    with pytest.raises(RuntimeError, match="requirements-parallel.txt"):
        await chatbot.llm_claim_search_stage.ainvoke(parallel_state)
