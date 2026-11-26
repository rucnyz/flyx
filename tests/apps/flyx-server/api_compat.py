# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""OpenAI/Anthropic/Responses API compatibility: GET /v1/models/:id, reasoning_effort, output_config.effort, validation."""
import os
from pathlib import Path
import re
import sys
import tempfile

from check_server import Client, wait_load
from remembered_model import serve

MODEL_NAME = "org/test-model"

# Deterministic marker lengths make enable_thinking and reasoning_effort directly observable via prompt_tokens.
TEMPLATE = (
    "{% if enable_thinking %}{{ 'think ' * 16 }}{% endif %}"
    "{% if reasoning_effort is defined %}"
    "{% if reasoning_effort == 'high' %}{{ 'hi ' * 32 }}"
    "{% elif reasoning_effort == 'low' %}{{ 'lo ' * 8 }}"
    "{% elif reasoning_effort == 'max' %}{{ 'mx ' * 64 }}"
    "{% elif reasoning_effort == 'none' %}{{ 'no ' * 4 }}"
    "{% endif %}"
    "{% endif %}"
    "{% for m in messages %}{{ m.content }}|{% endfor %}"
)


def check_compat(url):
    cl = Client(url, None)

    # 1. GET /v1/models and GET /v1/models/:id (including slash and percent-encoded slash)
    st, listed = cl.get("/v1/models")
    assert st == 200 and len(listed["data"]) == 1 and listed["data"][0]["id"] == MODEL_NAME, (st, listed)
    expected_ctx = listed["data"][0]["max_model_len"]

    for path in (f"/v1/models/{MODEL_NAME}", "/v1/models/org%2Ftest-model"):
        st, m = cl.get(path)
        assert st == 200 and m["id"] == MODEL_NAME and m["object"] == "model" and m["owned_by"] == "flyx", (path, st, m)
        assert m["max_model_len"] == expected_ctx, (path, m)

    st, err = cl.get("/v1/models/does-not-exist")
    assert st == 404 and err["error"]["code"] == "model_not_found" and err["error"]["type"] == "invalid_request_error", (st, err)

    # The version the build is of, for a bug report to name (`flyx --version` prints the same).
    st, t = cl.get("/v1/telemetry?recent=0")
    assert st == 200 and re.fullmatch(r"\d+\.\d+\.\d+", t["server"]["version"]), (st, t.get("server"))

    # 2. /v1/chat/completions reasoning_effort, enable_thinking, and chat_template_kwargs precedence
    def chat_prompt_tokens(extra):
        body = {"model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "temperature": 0}
        body.update(extra)
        st, r = cl.post("/v1/chat/completions", body)
        assert st == 200, (extra, st, r)
        return r["usage"]["prompt_tokens"]

    tok_none = chat_prompt_tokens({"reasoning_effort": "none"})
    tok_low = chat_prompt_tokens({"reasoning_effort": "low"})
    tok_high = chat_prompt_tokens({"reasoning_effort": "high"})
    # reasoning_effort: "high" + top-level enable_thinking: False alongside chat_template_kwargs without enable_thinking
    tok_high_no_think = chat_prompt_tokens({
        "reasoning_effort": "high",
        "enable_thinking": False,
        "chat_template_kwargs": {"extra_var": 1},
    })
    # chat_template_kwargs.reasoning_effort overrides top-level reasoning_effort
    tok_override_low = chat_prompt_tokens({
        "reasoning_effort": "high",
        "chat_template_kwargs": {"reasoning_effort": "low"},
    })
    assert tok_high > tok_low > tok_none, (tok_none, tok_low, tok_high)
    assert tok_high > tok_high_no_think > tok_none, (tok_none, tok_high_no_think, tok_high)
    assert tok_override_low == tok_low, (tok_override_low, tok_low)

    # Huge max_tokens is clamped cleanly without signed overflow
    st, r = cl.post("/v1/chat/completions", {
        "model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 999999999999, "stop": [" "], "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    assert st == 200, (st, r)

    # Validation -> 400 (never 500 or silent ignore / signed overflow)
    for bad in ({"reasoning_effort": "turbo"}, {"reasoning_effort": 123},
                {"enable_thinking": "false"}, {"chat_template_kwargs": ["bad"]},
                {"n": 4294967297}, {"n": 1.5}, {"n": "1"}, {"n": True},
                {"max_tokens": "50"}, {"max_tokens": 0}, {"max_tokens": -5}, {"max_tokens": 1.5},
                {"max_completion_tokens": "5"}, {"response_format": "json_object"},
                {"parallel_tool_calls": "no"}):
        body = {"model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
        body.update(bad)
        st, r = cl.post("/v1/chat/completions", body)
        assert st == 400 and r["error"]["type"] == "invalid_request_error", (bad, st, r)

    # Valid explicit n=1 and max_completion_tokens override
    st, r = cl.post("/v1/chat/completions", {
        "model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}],
        "n": 1, "max_tokens": None, "max_completion_tokens": 2, "temperature": 0,
    })
    assert st == 200, (st, r)

    # 3. /v1/responses reasoning.effort and validation
    st, r_none = cl.post("/v1/responses", {"model": MODEL_NAME, "input": "hi", "max_output_tokens": 1,
                                           "reasoning": {"effort": "none"}})
    st2, r_high = cl.post("/v1/responses", {"model": MODEL_NAME, "input": "hi", "max_output_tokens": 1,
                                            "reasoning": {"effort": "high"}})
    assert st == 200 and st2 == 200, (st, r_none, st2, r_high)
    assert r_high["usage"]["input_tokens"] > r_none["usage"]["input_tokens"], (r_none, r_high)
    for bad in ({"reasoning": "high"}, {"reasoning": {"effort": "turbo"}},
                {"reasoning": {"effort": 42}}, {"chat_template_kwargs": 123},
                {"max_output_tokens": 0}, {"max_output_tokens": -1}, {"max_output_tokens": "10"},
                {"previous_response_id": 123}, {"instructions": 123},
                {"include": "reasoning.encrypted_content"}, {"include": [123]},
                {"metadata": "bad"}, {"text": "bad"}, {"text": {"format": "json_object"}}):
        body = {"model": MODEL_NAME, "input": "hi", "max_output_tokens": 1}
        body.update(bad)
        st, r = cl.post("/v1/responses", body)
        assert st == 400 and r["error"]["type"] == "invalid_request_error", (bad, st, r)

    # 4. /v1/messages + /v1/messages/count_tokens: output_config.effort and malformed thinking -> 400
    st, c_low = cl.post("/v1/messages/count_tokens", {
        "model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled"}, "output_config": {"effort": "low"},
    })
    st2, c_max = cl.post("/v1/messages/count_tokens", {
        "model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled"}, "output_config": {"effort": "max"},
    })
    assert st == 200 and st2 == 200 and c_max["input_tokens"] > c_low["input_tokens"], (st, c_low, st2, c_max)

    for bad in ({"thinking": "enabled"}, {"thinking": {"type": 123}},
                {"output_config": {"effort": "turbo"}}, {"output_config": {"effort": 42}},
                {"max_tokens": -1}, {"max_tokens": "10"}, {"max_tokens": 1.5},
                {"tool_choice": "auto"}, {"system": 123}, {"tools": "bad"}):
        body = {"model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
        body.update(bad)
        st, r = cl.post("/v1/messages", body)
        assert st == 400 and r["error"]["type"] == "invalid_request_error", (bad, st, r)

    # 5. POST /v1/load rejects non-null wrong JSON types with 400 (never silently ignored)
    for bad in ({"drafter": 123}, {"speculative": "yes"}, {"kv": 16},
                {"name": 42}, {"draft_trees": "on"}, {"lookup": 1}):
        body = {"model": "dummy.flyx"}
        body.update(bad)
        st, err = cl.post("/v1/load", body)
        assert st == 400 and err["error"]["type"] == "invalid_request_error", (bad, st, err)
    # a drafter of the wrong type is told the same choices as an unknown one's, dv among them
    for bad in (123, "eagle"):
        st, err = cl.post("/v1/load", {"model": "dummy.flyx", "drafter": bad})
        want = '"drafter": mtp, dflash2, dspine, dv, all or none'
        assert st == 400 and err["error"]["message"] == want, (bad, st, err)

    # 6. Unloaded model -> GET /v1/models/:id returns 404
    st, _ = cl.post("/v1/unload", {})
    assert st == 200, st
    st, err = cl.get(f"/v1/models/{MODEL_NAME}")
    assert st == 404 and err["error"]["code"] == "model_not_found", (st, err)


def check_kv_default(url, asset):
    """POST /v1/load's "kv": "auto" loads the engine's default cache type, not the server's --kv (the dashboard's
    auto choice); a value of none of the three is refused."""
    cl = Client(url, None)
    for bad in ("fp16", "default"):
        st, err = cl.post("/v1/load", {"model": "dummy.flyx", "kv": bad})
        assert st == 400 and err["error"]["message"] == '"kv": fp8, 16 or auto', (bad, st, err)
    st, s = cl.get("/v1/load")
    assert st == 200 and s["loaded"]["kv"] == "fp8", s
    st, s = cl.post("/v1/load", {"model": str(asset), "kv": "auto", "name": MODEL_NAME, "speculative": False})
    assert st == 202, (st, s)
    s = wait_load(cl, 60)
    assert s.get("state") == "ready" and s["loaded"]["kv"] is None, s


def check_models_auth(url):
    no_key = Client(url, None)
    st, _ = no_key.get(f"/v1/models/{MODEL_NAME}")
    assert st == 401, st
    authed = Client(url, "secret-key")
    st, m = authed.get(f"/v1/models/{MODEL_NAME}")
    assert st == 200 and m["id"] == MODEL_NAME, (st, m)


def check_queued_anthropic_stream_error(url):
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import closing
    import time
    from check_server import wait_telemetry

    cl = Client(url, None)
    # Hold the single session busy past kStartWaitMs (10s) so the queued streaming /v1/messages commits HTTP 200 SSE
    # before prefill, then fails with context overflow (prompt_tokens == 0). It must emit event: error without a
    # bogus 0-token message_start or ping.
    blocker = {"model": MODEL_NAME, "prompt": "0 Count up forever.", "temperature": 0, "max_tokens": 16000, "stream": True}
    too_long = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "word " * 17000}],
        "max_tokens": 8,
        "stream": True,
    }

    def run_queued():
        events = []
        with closing(cl.stream("/v1/messages", too_long)) as s:
            for ev_name, data in s:
                events.append((ev_name, data))
        return events

    with ThreadPoolExecutor(max_workers=1) as pool:
        with closing(cl.stream("/v1/completions", blocker, stop_after=1)) as hold:
            _, first = next(hold)
            assert first["choices"][0]["text"], first
            fut = pool.submit(run_queued)
            wait_telemetry(cl, lambda t: t["state"]["queued"] == 1, 10)
            time.sleep(10.6)
            assert not fut.done(), "queued /v1/messages should still be waiting for the slot past kStartWaitMs"
            next(hold, None)
        events = fut.result(timeout=15)
    names = [name for name, _ in events]
    assert "error" in names, f"expected SSE error event, got {events}"
    assert "message_start" not in names and "ping" not in names, (
        f"bogus message_start/ping emitted before prefill error: {names}"
    )


if __name__ == "__main__":
    exe, asset = (Path(arg).resolve() for arg in sys.argv[1:])
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, APPDATA=tmp, XDG_CONFIG_HOME=tmp, FLYX_BACKEND_DIR=str(exe.parent))
        template = Path(tmp) / "chat.jinja"
        template.write_text(TEMPLATE, encoding="utf-8")
        serve(exe, tmp, env, [asset], check_compat, ctx=4096,
              options=["--name", MODEL_NAME, "--no-spec", "--chat-template", template], expect_name=MODEL_NAME)
        serve(exe, tmp, env, [asset], lambda url: check_kv_default(url, asset), ctx=4096,
              options=["--name", MODEL_NAME, "--no-spec", "--kv", "fp8"], expect_name=MODEL_NAME)
        serve(exe, tmp, env, [asset], check_models_auth, ctx=4096,
              options=["--name", MODEL_NAME, "--no-spec", "--api-key", "secret-key"], expect_name=MODEL_NAME)
        serve(exe, tmp, env, [asset], check_queued_anthropic_stream_error, ctx=16384,
              options=["--name", MODEL_NAME, "--sessions", "1", "--no-spec", "--chat-template", template],
              expect_name=MODEL_NAME)
    print("api_compat: ok")
