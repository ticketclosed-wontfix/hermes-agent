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

