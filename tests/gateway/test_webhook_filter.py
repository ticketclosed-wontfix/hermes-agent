"""Unit tests for the pre-LLM webhook filter DSL.

Covers the pure evaluator `gateway.platforms.webhook.evaluate_filter`:
- Leaf predicates: equals, in, regex (with flags), exists
- Combinators: any_of, all_of, not
- Nested combinators
- Missing paths → None semantics
- Malformed specs

Also covers the NOOP sentinel short-circuit in WebhookAdapter.send().
"""

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageType  # noqa: F401 — keeps import parity
from gateway.platforms.webhook import (
    NOOP_SENTINEL,
    WebhookAdapter,
    _resolve_path,
    evaluate_filter,
)


# ---------------------------------------------------------------------------
# _resolve_path
# ---------------------------------------------------------------------------

def test_resolve_path_top_level():
    assert _resolve_path({"action": "opened"}, "action") == "opened"


def test_resolve_path_nested():
    payload = {"pull_request": {"user": {"login": "octocat"}}}
    assert _resolve_path(payload, "pull_request.user.login") == "octocat"


def test_resolve_path_missing_returns_none():
    assert _resolve_path({"action": "opened"}, "label.name") is None


def test_resolve_path_traverses_non_dict_as_none():
    # pull_request.user is a list, cannot traverse into .login
    payload = {"pull_request": {"user": ["octo"]}}
    assert _resolve_path(payload, "pull_request.user.login") is None


def test_resolve_path_empty_or_invalid_returns_none():
    assert _resolve_path({"a": 1}, "") is None
    assert _resolve_path({"a": 1}, None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Leaf predicates
# ---------------------------------------------------------------------------

def test_equals_match():
    payload = {"action": "labeled"}
    ok, reason = evaluate_filter(payload, {"path": "action", "equals": "labeled"})
    assert ok is True
    assert reason == ""


def test_equals_mismatch_has_reason():
    payload = {"action": "opened"}
    ok, reason = evaluate_filter(payload, {"path": "action", "equals": "labeled"})
    assert ok is False
    assert "action" in reason


def test_equals_missing_path_is_false():
    ok, reason = evaluate_filter({}, {"path": "action", "equals": "labeled"})
    assert ok is False
    assert "missing" in reason


def test_in_match():
    payload = {"action": "labeled"}
    spec = {"path": "action", "in": ["labeled", "unlabeled"]}
    ok, _ = evaluate_filter(payload, spec)
    assert ok is True


def test_in_mismatch():
    payload = {"action": "edited"}
    spec = {"path": "action", "in": ["labeled", "unlabeled"]}
    ok, reason = evaluate_filter(payload, spec)
    assert ok is False
    assert "not in allowed" in reason


def test_in_missing_path_is_false():
    spec = {"path": "label.name", "in": ["auto-merge"]}
    ok, _ = evaluate_filter({}, spec)
    assert ok is False


def test_in_numeric_vs_string():
    # Payload has int, allowed list has string — should still match
    payload = {"pr": {"number": 42}}
    spec = {"path": "pr.number", "in": ["42", "43"]}
    ok, _ = evaluate_filter(payload, spec)
    assert ok is True


def test_regex_match_default_case_sensitive():
    payload = {"body": "hello @commit-mcgitface please"}
    spec = {"path": "body", "regex": r"@commit-mcgitface\b"}
    ok, _ = evaluate_filter(payload, spec)
    assert ok is True


def test_regex_case_sensitive_miss():
    payload = {"body": "HELLO @COMMIT-MCGITFACE please"}
    spec = {"path": "body", "regex": r"@commit-mcgitface\b"}
    ok, _ = evaluate_filter(payload, spec)
    assert ok is False


def test_regex_with_i_flag():
    payload = {"body": "HELLO @Commit-McGitface please"}
    spec = {"path": "body", "regex": r"@commit-mcgitface\b", "flags": "i"}
    ok, _ = evaluate_filter(payload, spec)
    assert ok is True


def test_regex_missing_path_is_false():
    spec = {"path": "comment.body", "regex": r"anything"}
    ok, _ = evaluate_filter({}, spec)
    assert ok is False


def test_regex_invalid_pattern_has_reason():
    payload = {"body": "x"}
    spec = {"path": "body", "regex": "("}  # unclosed group
    ok, reason = evaluate_filter(payload, spec)
    assert ok is False
    assert "invalid regex" in reason


def test_exists_true_present():
    payload = {"issue": {"labels": []}}
    ok, _ = evaluate_filter(payload, {"path": "issue.labels", "exists": True})
    assert ok is True


def test_exists_true_missing():
    ok, _ = evaluate_filter({}, {"path": "issue.labels", "exists": True})
    assert ok is False


def test_exists_false_missing_matches():
    ok, _ = evaluate_filter({}, {"path": "issue.labels", "exists": False})
    assert ok is True


def test_exists_false_present_no_match():
    payload = {"issue": {"labels": []}}
    ok, reason = evaluate_filter(payload, {"path": "issue.labels", "exists": False})
    assert ok is False
    assert "present" in reason


# ---------------------------------------------------------------------------
# Combinators
# ---------------------------------------------------------------------------

def test_any_of_matches_first():
    payload = {"action": "labeled"}
    spec = {
        "any_of": [
            {"path": "action", "equals": "labeled"},
            {"path": "action", "equals": "opened"},
        ]
    }
    ok, _ = evaluate_filter(payload, spec)
    assert ok is True


def test_any_of_matches_last():
    payload = {"action": "opened"}
    spec = {
        "any_of": [
            {"path": "action", "equals": "labeled"},
            {"path": "action", "equals": "opened"},
        ]
    }
    ok, _ = evaluate_filter(payload, spec)
    assert ok is True


def test_any_of_no_match_returns_default_reason():
    payload = {"action": "edited"}
    spec = {
        "any_of": [
            {"path": "action", "equals": "labeled"},
            {"path": "action", "equals": "opened"},
        ]
    }
    ok, reason = evaluate_filter(payload, spec)
    assert ok is False
    assert reason == "no filter branch matched"


def test_any_of_empty_rejects():
    ok, reason = evaluate_filter({"a": 1}, {"any_of": []})
    assert ok is False
    assert "no branches" in reason


def test_all_of_match():
    payload = {"action": "labeled", "label": {"name": "auto-merge"}}
    spec = {
        "all_of": [
            {"path": "action", "in": ["labeled", "unlabeled"]},
            {"path": "label.name", "in": ["auto", "auto-merge", "review-me"]},
        ]
    }
    ok, _ = evaluate_filter(payload, spec)
    assert ok is True


def test_all_of_first_fails():
    payload = {"action": "opened", "label": {"name": "auto-merge"}}
    spec = {
        "all_of": [
            {"path": "action", "in": ["labeled"]},
            {"path": "label.name", "in": ["auto-merge"]},
        ]
    }
    ok, reason = evaluate_filter(payload, spec)
    assert ok is False
    assert "action" in reason


def test_not_positive():
    payload = {"action": "opened"}
    ok, _ = evaluate_filter(
        payload, {"not": {"path": "action", "equals": "labeled"}}
    )
    assert ok is True


def test_not_negative():
    payload = {"action": "labeled"}
    ok, reason = evaluate_filter(
        payload, {"not": {"path": "action", "equals": "labeled"}}
    )
    assert ok is False
    assert "negated" in reason


def test_nested_any_inside_all():
    # "opened by a known bot, OR labeled with auto-merge"
    payload = {
        "action": "opened",
        "pull_request": {"user": {"login": "dependabot[bot]"}},
    }
    spec = {
        "any_of": [
            {
                "all_of": [
                    {"path": "action", "in": ["labeled", "unlabeled"]},
                    {"path": "label.name", "in": ["auto-merge"]},
                ]
            },
            {
                "all_of": [
                    {"path": "action", "equals": "opened"},
                    {
                        "path": "pull_request.user.login",
                        "in": ["dependabot[bot]", "renovate[bot]"],
                    },
                ]
            },
        ]
    }
    ok, _ = evaluate_filter(payload, spec)
    assert ok is True


def test_nested_any_inside_all_reject():
    payload = {
        "action": "edited",  # neither opened nor labeled
        "pull_request": {"user": {"login": "octocat"}},
    }
    spec = {
        "any_of": [
            {
                "all_of": [
                    {"path": "action", "in": ["labeled"]},
                    {"path": "label.name", "in": ["auto-merge"]},
                ]
            },
            {
                "all_of": [
                    {"path": "action", "equals": "opened"},
                    {"path": "pull_request.user.login", "in": ["dependabot[bot]"]},
                ]
            },
        ]
    }
    ok, reason = evaluate_filter(payload, spec)
    assert ok is False
    assert reason == "no filter branch matched"


# ---------------------------------------------------------------------------
# Malformed specs
# ---------------------------------------------------------------------------

def test_none_spec_is_pass_through():
    ok, reason = evaluate_filter({"a": 1}, None)
    assert ok is True
    assert reason == ""


def test_non_dict_spec_rejects():
    ok, reason = evaluate_filter({"a": 1}, "garbage")  # type: ignore[arg-type]
    assert ok is False
    assert "object" in reason


def test_path_without_operator_rejects():
    ok, reason = evaluate_filter({"a": 1}, {"path": "a"})
    assert ok is False
    assert "no operator" in reason


def test_unknown_spec_rejects():
    ok, reason = evaluate_filter({"a": 1}, {"xyzzy": 1})
    assert ok is False
    assert "unknown filter spec" in reason


def test_any_of_non_list_rejects():
    ok, reason = evaluate_filter({"a": 1}, {"any_of": "nope"})
    assert ok is False
    assert "list" in reason


# ---------------------------------------------------------------------------
# Real-world-ish: the github-automation filter
# ---------------------------------------------------------------------------

GITHUB_AUTOMATION_FILTER = {
    "any_of": [
        {
            "all_of": [
                {"path": "action", "in": ["labeled", "unlabeled"]},
                {
                    "path": "label.name",
                    "in": ["auto", "auto-merge", "review-me"],
                },
            ]
        },
        {
            "all_of": [
                {"path": "action", "equals": "opened"},
                {
                    "path": "pull_request.user.login",
                    "in": [
                        "dependabot[bot]",
                        "renovate[bot]",
                        "github-actions[bot]",
                    ],
                },
            ]
        },
        {
            "all_of": [
                {"path": "action", "equals": "created"},
                {
                    "path": "comment.body",
                    "regex": r"@commit-mcgitface\b",
                    "flags": "i",
                },
                {"path": "comment.user.login", "in": ["ticketclosedwontfix"]},
            ]
        },
        {
            "all_of": [
                {"path": "action", "equals": "completed"},
                {"path": "check_suite.conclusion", "equals": "failure"},
            ]
        },
    ]
}


def test_github_filter_pr_labeled_auto_merge_passes():
    payload = {
        "action": "labeled",
        "label": {"name": "auto-merge"},
        "pull_request": {"number": 1},
    }
    ok, _ = evaluate_filter(payload, GITHUB_AUTOMATION_FILTER)
    assert ok is True


def test_github_filter_issue_labeled_auto_passes():
    payload = {
        "action": "labeled",
        "label": {"name": "auto"},
        "issue": {"number": 7},
    }
    ok, _ = evaluate_filter(payload, GITHUB_AUTOMATION_FILTER)
    assert ok is True


def test_github_filter_random_comment_blocked():
    payload = {
        "action": "created",
        "comment": {
            "body": "nice work",
            "user": {"login": "some-random-user"},
        },
    }
    ok, reason = evaluate_filter(payload, GITHUB_AUTOMATION_FILTER)
    assert ok is False
    assert reason == "no filter branch matched"


def test_github_filter_mention_from_owner_passes():
    payload = {
        "action": "created",
        "comment": {
            "body": "@commit-mcgitface please rebase",
            "user": {"login": "ticketclosedwontfix"},
        },
    }
    ok, _ = evaluate_filter(payload, GITHUB_AUTOMATION_FILTER)
    assert ok is True


def test_github_filter_mention_from_stranger_blocked():
    payload = {
        "action": "created",
        "comment": {
            "body": "@commit-mcgitface please rebase",
            "user": {"login": "random-person"},
        },
    }
    ok, _ = evaluate_filter(payload, GITHUB_AUTOMATION_FILTER)
    assert ok is False


def test_github_filter_check_suite_success_blocked():
    payload = {
        "action": "completed",
        "check_suite": {"conclusion": "success"},
    }
    ok, _ = evaluate_filter(payload, GITHUB_AUTOMATION_FILTER)
    assert ok is False


def test_github_filter_check_suite_failure_passes():
    payload = {
        "action": "completed",
        "check_suite": {"conclusion": "failure"},
    }
    ok, _ = evaluate_filter(payload, GITHUB_AUTOMATION_FILTER)
    assert ok is True


def test_github_filter_bot_pr_opened_passes():
    payload = {
        "action": "opened",
        "pull_request": {"user": {"login": "dependabot[bot]"}},
    }
    ok, _ = evaluate_filter(payload, GITHUB_AUTOMATION_FILTER)
    assert ok is True


def test_github_filter_human_pr_opened_blocked():
    payload = {
        "action": "opened",
        "pull_request": {"user": {"login": "ticketclosedwontfix"}},
    }
    ok, _ = evaluate_filter(payload, GITHUB_AUTOMATION_FILTER)
    assert ok is False


# ---------------------------------------------------------------------------
# NOOP suppression in WebhookAdapter.send
# ---------------------------------------------------------------------------

def _make_adapter():
    cfg = PlatformConfig(
        enabled=True,
        extra={"host": "127.0.0.1", "port": 0, "routes": {}, "secret": "x"},
    )
    return WebhookAdapter(cfg)


@pytest.mark.asyncio
async def test_send_noop_suppresses_delivery(caplog):
    """Bare 'NOOP' (the original sentinel) still suppresses."""
    adapter = _make_adapter()
    chat = "webhook:test:1"
    adapter._delivery_info[chat] = {
        "deliver": "telegram",
        "deliver_extra": {"chat_id": "-100"},
    }
    # No gateway_runner set — if we fell through to real delivery this
    # would fail with "No gateway runner".  NOOP must short-circuit.
    import logging
    with caplog.at_level(logging.INFO, logger="gateway.platforms.webhook"):
        result = await adapter.send(chat, NOOP_SENTINEL)
    assert result.success is True
    assert any("suppressing delivery" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_send_noop_with_whitespace_still_suppresses():
    adapter = _make_adapter()
    chat = "webhook:test:2"
    adapter._delivery_info[chat] = {"deliver": "telegram", "deliver_extra": {}}
    result = await adapter.send(chat, "  NOOP\n")
    assert result.success is True


@pytest.mark.asyncio
async def test_send_explanation_then_noop_suppresses():
    """Models routinely prepend explanation; trailing NOOP still wins."""
    adapter = _make_adapter()
    chat = "webhook:test:explanation"
    adapter._delivery_info[chat] = {"deliver": "telegram", "deliver_extra": {}}
    content = (
        "PR #15 is open, not draft, CI passed — but no auto-merge label "
        "and author is not a bot. AUTO_MERGE_RECHECK has nothing to act "
        "on.\n\nNOOP"
    )
    result = await adapter.send(chat, content)
    assert result.success is True


@pytest.mark.asyncio
async def test_send_explanation_then_noop_with_whitespace_suppresses():
    """Trailing whitespace on the NOOP line and indentation must still match."""
    adapter = _make_adapter()
    chat = "webhook:test:explanation_ws"
    adapter._delivery_info[chat] = {"deliver": "telegram", "deliver_extra": {}}
    result = await adapter.send(chat, "explanation  \n  NOOP  \n")
    assert result.success is True


@pytest.mark.asyncio
async def test_send_noop_mid_sentence_does_NOT_suppress(caplog):
    """The literal word NOOP appearing mid-content is NOT a sentinel."""
    adapter = _make_adapter()
    chat = "webhook:test:mid_word"
    adapter._delivery_info[chat] = {"deliver": "log", "deliver_extra": {}}
    import logging
    with caplog.at_level(logging.INFO, logger="gateway.platforms.webhook"):
        result = await adapter.send(chat, "NOOP is the sentinel word")
    assert result.success is True
    # Must NOT have logged the suppression banner.
    assert not any("suppressing delivery" in rec.message for rec in caplog.records)
    # Should have logged the normal log-delivery line instead.
    assert any("Response for" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_send_normal_response_does_NOT_suppress(caplog):
    """Plain content with no NOOP at all is delivered normally."""
    adapter = _make_adapter()
    chat = "webhook:test:normal"
    adapter._delivery_info[chat] = {"deliver": "log", "deliver_extra": {}}
    import logging
    with caplog.at_level(logging.INFO, logger="gateway.platforms.webhook"):
        result = await adapter.send(chat, "some work done\nResult: success")
    assert result.success is True
    assert not any("suppressing delivery" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_send_non_noop_still_attempts_delivery():
    adapter = _make_adapter()
    chat = "webhook:test:3"
    # log-delivery path does not need a gateway runner
    adapter._delivery_info[chat] = {"deliver": "log", "deliver_extra": {}}
    result = await adapter.send(chat, "real message body")
    assert result.success is True


# ---------------------------------------------------------------------------
# Integration: filter gating inside _handle_webhook
# ---------------------------------------------------------------------------

import json
from unittest.mock import AsyncMock
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.webhook import _INSECURE_NO_AUTH


def _app_for(adapter):
    app = web.Application()
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


@pytest.mark.asyncio
async def test_handler_filter_rejects_before_agent():
    """Filter rejects payload → 202 filtered:true, handle_message NEVER called."""
    routes = {
        "gh": {
            "secret": _INSECURE_NO_AUTH,
            "events": ["issue_comment"],
            "prompt": "x",
            "filter": {
                "all_of": [
                    {"path": "action", "equals": "created"},
                    {"path": "comment.user.login", "in": ["ticketclosedwontfix"]},
                ]
            },
        }
    }
    cfg = PlatformConfig(
        enabled=True,
        extra={"host": "127.0.0.1", "port": 0, "routes": routes, "secret": ""},
    )
    adapter = WebhookAdapter(cfg)
    adapter.handle_message = AsyncMock()

    async with TestClient(TestServer(_app_for(adapter))) as cli:
        resp = await cli.post(
            "/webhooks/gh",
            json={
                "action": "created",
                "comment": {
                    "body": "nice work",
                    "user": {"login": "drive-by-commenter"},
                },
            },
            headers={"X-GitHub-Event": "issue_comment"},
        )
        assert resp.status == 202
        body = await resp.json()
        assert body["filtered"] is True
        assert "reason" in body
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_handler_filter_accepts_and_spawns():
    """Filter accepts payload → 202 accepted, handle_message IS called."""
    routes = {
        "gh": {
            "secret": _INSECURE_NO_AUTH,
            "events": ["pull_request"],
            "prompt": "x",
            "filter": {
                "all_of": [
                    {"path": "action", "equals": "labeled"},
                    {"path": "label.name", "in": ["auto-merge"]},
                ]
            },
        }
    }
    cfg = PlatformConfig(
        enabled=True,
        extra={"host": "127.0.0.1", "port": 0, "routes": routes, "secret": ""},
    )
    adapter = WebhookAdapter(cfg)
    adapter.handle_message = AsyncMock()

    async with TestClient(TestServer(_app_for(adapter))) as cli:
        resp = await cli.post(
            "/webhooks/gh",
            json={
                "action": "labeled",
                "label": {"name": "auto-merge"},
                "pull_request": {"number": 1},
            },
            headers={"X-GitHub-Event": "pull_request"},
        )
        assert resp.status == 202
        body = await resp.json()
        # Not a filtered response
        assert body.get("filtered") is not True
        assert body["status"] == "accepted"

    # Allow the background task the adapter created to be scheduled
    import asyncio as _asyncio
    await _asyncio.sleep(0.05)
    adapter.handle_message.assert_called_once()


@pytest.mark.asyncio
async def test_handler_without_filter_is_backward_compatible():
    """Routes without a filter field behave exactly as before."""
    routes = {
        "gh": {
            "secret": _INSECURE_NO_AUTH,
            "events": ["pull_request"],
            "prompt": "x",
            # no filter at all
        }
    }
    cfg = PlatformConfig(
        enabled=True,
        extra={"host": "127.0.0.1", "port": 0, "routes": routes, "secret": ""},
    )
    adapter = WebhookAdapter(cfg)
    adapter.handle_message = AsyncMock()

    async with TestClient(TestServer(_app_for(adapter))) as cli:
        resp = await cli.post(
            "/webhooks/gh",
            json={"action": "edited"},
            headers={"X-GitHub-Event": "pull_request"},
        )
        assert resp.status == 202
        body = await resp.json()
        assert body.get("filtered") is not True


# ---------------------------------------------------------------------------
# non_empty DSL operator (added 2026-04-19 to gate check_suite by PR linkage)
# ---------------------------------------------------------------------------

def test_non_empty_true_matches_non_empty_list():
    payload = {"check_suite": {"pull_requests": [{"number": 1}]}}
    spec = {"path": "check_suite.pull_requests", "non_empty": True}
    ok, _ = evaluate_filter(payload, spec)
    assert ok is True


def test_non_empty_true_rejects_empty_list():
    payload = {"check_suite": {"pull_requests": []}}
    spec = {"path": "check_suite.pull_requests", "non_empty": True}
    ok, reason = evaluate_filter(payload, spec)
    assert ok is False
    assert "empty" in reason


def test_non_empty_true_rejects_missing_path():
    payload = {"check_suite": {}}
    spec = {"path": "check_suite.pull_requests", "non_empty": True}
    ok, _ = evaluate_filter(payload, spec)
    assert ok is False


def test_non_empty_true_matches_non_empty_string():
    payload = {"comment": {"body": "hi"}}
    spec = {"path": "comment.body", "non_empty": True}
    ok, _ = evaluate_filter(payload, spec)
    assert ok is True


def test_non_empty_true_rejects_empty_string():
    payload = {"comment": {"body": ""}}
    spec = {"path": "comment.body", "non_empty": True}
    ok, _ = evaluate_filter(payload, spec)
    assert ok is False


def test_non_empty_true_rejects_non_container_value():
    """Booleans/numbers are not 'non-empty' — guard against authoring mistakes."""
    payload = {"x": 42}
    spec = {"path": "x", "non_empty": True}
    ok, _ = evaluate_filter(payload, spec)
    assert ok is False


def test_non_empty_false_matches_empty_or_missing():
    payload = {"check_suite": {"pull_requests": []}}
    spec = {"path": "check_suite.pull_requests", "non_empty": False}
    ok, _ = evaluate_filter(payload, spec)
    assert ok is True

    payload2 = {"check_suite": {}}
    ok2, _ = evaluate_filter(payload2, spec)
    assert ok2 is True


def test_non_empty_combines_with_all_of_for_check_suite_gating():
    """End-to-end DSL: only check_suite.success WITH PRs gets through."""
    spec = {
        "all_of": [
            {"path": "action", "equals": "completed"},
            {"path": "check_suite.conclusion", "equals": "success"},
            {"path": "check_suite.pull_requests", "non_empty": True},
        ]
    }
    # Payload with a PR — should match
    matched, _ = evaluate_filter(
        {
            "action": "completed",
            "check_suite": {
                "conclusion": "success",
                "pull_requests": [{"number": 7}],
            },
        },
        spec,
    )
    assert matched is True

    # Payload with empty PRs — should NOT match
    rejected, reason = evaluate_filter(
        {
            "action": "completed",
            "check_suite": {"conclusion": "success", "pull_requests": []},
        },
        spec,
    )
    assert rejected is False
    assert "empty" in reason


# ---------------------------------------------------------------------------
# Functional: check_suite filter end-to-end through HTTP handler
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handler_check_suite_empty_prs_filtered():
    """check_suite.completed with empty pull_requests array → 202 filtered, no session."""
    routes = {
        "gh": {
            "secret": _INSECURE_NO_AUTH,
            "events": ["check_suite"],
            "prompt": "x",
            "filter": {
                "all_of": [
                    {"path": "action", "equals": "completed"},
                    {"path": "check_suite.conclusion", "equals": "success"},
                    {"path": "check_suite.pull_requests", "non_empty": True},
                ]
            },
        }
    }
    cfg = PlatformConfig(
        enabled=True,
        extra={"host": "127.0.0.1", "port": 0, "routes": routes, "secret": ""},
    )
    adapter = WebhookAdapter(cfg)
    adapter.handle_message = AsyncMock()

    async with TestClient(TestServer(_app_for(adapter))) as cli:
        resp = await cli.post(
            "/webhooks/gh",
            json={
                "action": "completed",
                "check_suite": {
                    "conclusion": "success",
                    "head_branch": "main",
                    "pull_requests": [],
                },
            },
            headers={"X-GitHub-Event": "check_suite"},
        )
        assert resp.status == 202
        body = await resp.json()
        assert body["filtered"] is True
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_handler_check_suite_with_prs_accepted():
    """check_suite.completed WITH pull_requests array → 202 accepted, session spawned."""
    routes = {
        "gh": {
            "secret": _INSECURE_NO_AUTH,
            "events": ["check_suite"],
            "prompt": "x",
            "filter": {
                "all_of": [
                    {"path": "action", "equals": "completed"},
                    {"path": "check_suite.conclusion", "equals": "success"},
                    {"path": "check_suite.pull_requests", "non_empty": True},
                ]
            },
        }
    }
    cfg = PlatformConfig(
        enabled=True,
        extra={"host": "127.0.0.1", "port": 0, "routes": routes, "secret": ""},
    )
    adapter = WebhookAdapter(cfg)
    adapter.handle_message = AsyncMock()

    async with TestClient(TestServer(_app_for(adapter))) as cli:
        resp = await cli.post(
            "/webhooks/gh",
            json={
                "action": "completed",
                "check_suite": {
                    "conclusion": "success",
                    "head_branch": "fix/foo",
                    "pull_requests": [{"number": 42, "head": {"ref": "fix/foo"}}],
                },
            },
            headers={"X-GitHub-Event": "check_suite"},
        )
        assert resp.status == 202
        body = await resp.json()
        assert body.get("filtered") is not True
        assert body["status"] == "accepted"

    import asyncio as _asyncio
    await _asyncio.sleep(0.05)
    adapter.handle_message.assert_called_once()


# ---------------------------------------------------------------------------
# Self-sender + bot-edited guards (re-entry prevention, added 2026-04-19)
#
# Mirrors the live `github-automation` filter in
# ~/.hermes/webhook_subscriptions.json: the dispatch any_of is wrapped in an
# all_of with two fail-safe gates:
#   1. sender.login != "commit-mcgitface[bot]"   (drop our own events)
#   2. NOT (action==edited AND sender.type==Bot) (drop dependabot body-edit churn)
# ---------------------------------------------------------------------------

GITHUB_AUTOMATION_FILTER_V2 = {
    "all_of": [
        {"not": {"path": "sender.login", "equals": "commit-mcgitface[bot]"}},
        {"not": {"all_of": [
            {"path": "action", "equals": "edited"},
            {"path": "sender.type", "equals": "Bot"},
        ]}},
        {"any_of": [
            {"all_of": [
                {"path": "action", "in": ["labeled", "unlabeled"]},
                {"path": "label.name", "in": ["auto", "auto-merge", "review-me"]},
            ]},
            {"all_of": [
                {"path": "action", "equals": "opened"},
                {"path": "pull_request.user.login", "in": [
                    "dependabot[bot]", "renovate[bot]", "github-actions[bot]",
                ]},
            ]},
            {"all_of": [
                {"path": "action", "equals": "created"},
                {"path": "comment.body", "regex": r"@commit-mcgitface\b", "flags": "i"},
                {"path": "comment.user.login", "in": ["ticketclosed-wontfix"]},
            ]},
            {"all_of": [
                {"path": "action", "equals": "completed"},
                {"path": "check_suite.conclusion", "equals": "failure"},
                {"path": "check_suite.pull_requests", "non_empty": True},
            ]},
            {"all_of": [
                {"path": "action", "equals": "completed"},
                {"path": "check_suite.conclusion", "equals": "success"},
                {"path": "check_suite.pull_requests", "non_empty": True},
            ]},
        ]},
    ]
}


def test_self_sender_guard_blocks_own_comment():
    """Our own bot commenting must not re-trigger a session."""
    payload = {
        "action": "created",
        "sender": {"login": "commit-mcgitface[bot]", "type": "Bot"},
        "comment": {
            "body": "Auto-merged PR #7.\n<!-- mcg:AUTO_MERGE_SUCCESS:v1 -->",
            "user": {"login": "commit-mcgitface[bot]"},
        },
    }
    ok, reason = evaluate_filter(payload, GITHUB_AUTOMATION_FILTER_V2)
    assert ok is False
    assert "negated" in reason  # the self-sender `not` matched


def test_self_sender_guard_blocks_own_label_event():
    """The post-facto BOT_PR refusal bug: a label event from our own bot
    must not re-trigger BOT_PR's refuse logic after AUTO_MERGE already ran."""
    payload = {
        "action": "labeled",
        "sender": {"login": "commit-mcgitface[bot]", "type": "Bot"},
        "label": {"name": "auto-merge"},
        "pull_request": {"number": 7, "user": {"login": "dependabot[bot]"}},
    }
    ok, _ = evaluate_filter(payload, GITHUB_AUTOMATION_FILTER_V2)
    assert ok is False


def test_self_sender_guard_allows_human_comment_on_bot_thread():
    """A human replying in a thread where the bot commented previously —
    payload sender is the human, so it passes."""
    payload = {
        "action": "created",
        "sender": {"login": "ticketclosed-wontfix", "type": "User"},
        "comment": {
            "body": "@commit-mcgitface merge it",
            "user": {"login": "ticketclosed-wontfix"},
        },
    }
    ok, _ = evaluate_filter(payload, GITHUB_AUTOMATION_FILTER_V2)
    assert ok is True


def test_self_sender_guard_allows_dependabot_opened_pr():
    """Dependabot opening a PR is the PRIMARY BOT_PR trigger — must still pass."""
    payload = {
        "action": "opened",
        "sender": {"login": "dependabot[bot]", "type": "Bot"},
        "pull_request": {"user": {"login": "dependabot[bot]"}, "number": 7},
    }
    ok, _ = evaluate_filter(payload, GITHUB_AUTOMATION_FILTER_V2)
    assert ok is True


def test_bot_edited_guard_blocks_dependabot_pr_body_edit():
    """Dependabot rebases edit the PR body; no route handles .edited events."""
    payload = {
        "action": "edited",
        "sender": {"login": "dependabot[bot]", "type": "Bot"},
        "pull_request": {"user": {"login": "dependabot[bot]"}, "number": 7},
    }
    ok, _ = evaluate_filter(payload, GITHUB_AUTOMATION_FILTER_V2)
    assert ok is False


def test_bot_edited_guard_blocks_renovate_edit():
    payload = {
        "action": "edited",
        "sender": {"login": "renovate[bot]", "type": "Bot"},
        "pull_request": {"user": {"login": "renovate[bot]"}, "number": 8},
    }
    ok, _ = evaluate_filter(payload, GITHUB_AUTOMATION_FILTER_V2)
    assert ok is False


def test_bot_edited_guard_allows_human_edited_issue_comment():
    """A human editing their own comment — payload action is `edited` on the
    issue_comment resource but sender.type is User, so the bot-edit guard
    doesn't trip. The dispatch any_of still rejects `edited` since no branch
    handles it; this is "filtered: no branch matched", not the edit guard."""
    payload = {
        "action": "edited",
        "sender": {"login": "ticketclosed-wontfix", "type": "User"},
        "comment": {
            "body": "@commit-mcgitface merge",
            "user": {"login": "ticketclosed-wontfix"},
        },
    }
    ok, reason = evaluate_filter(payload, GITHUB_AUTOMATION_FILTER_V2)
    assert ok is False
    # Proof it's the any_of failing, not the edit guard:
    assert reason == "no filter branch matched"


def test_self_sender_guard_missing_sender_fails_safe_open():
    """If sender is absent (shouldn't happen in real GitHub payloads, but
    be defensive): equals against None is False, `not` flips to True, so
    the guard passes through. Then the any_of decides."""
    payload = {
        "action": "opened",
        "pull_request": {"user": {"login": "dependabot[bot]"}, "number": 7},
        # no sender
    }
    ok, _ = evaluate_filter(payload, GITHUB_AUTOMATION_FILTER_V2)
    assert ok is True  # any_of matches the bot-opened PR branch


def test_guards_do_not_block_check_suite_from_github():
    """check_suite.completed events have sender=github (or the actor). Must still pass."""
    payload = {
        "action": "completed",
        "sender": {"login": "github-actions[bot]", "type": "Bot"},
        "check_suite": {
            "conclusion": "success",
            "pull_requests": [{"number": 42}],
        },
    }
    ok, _ = evaluate_filter(payload, GITHUB_AUTOMATION_FILTER_V2)
    assert ok is True


def test_self_sender_guard_matches_live_config_shape():
    """Sanity: the filter structure on disk matches the shape these tests validate."""
    import json
    from pathlib import Path
    cfg = Path.home() / ".hermes" / "webhook_subscriptions.json"
    if not cfg.exists():
        pytest.skip("no live subscription config on this host")
    data = json.loads(cfg.read_text())
    live = data.get("github-automation", {}).get("filter")
    if not live:
        pytest.skip("no github-automation filter configured")
    # Live filter MUST be wrapped in all_of with the self-sender guard.
    assert "all_of" in live, "live filter must be wrapped in all_of"
    branches = live["all_of"]
    # Find the self-sender guard
    found_self_guard = False
    for b in branches:
        if (
            isinstance(b, dict)
            and b.get("not", {}).get("path") == "sender.login"
            and b.get("not", {}).get("equals") == "commit-mcgitface[bot]"
        ):
            found_self_guard = True
            break
    assert found_self_guard, "live filter missing self-sender guard"

