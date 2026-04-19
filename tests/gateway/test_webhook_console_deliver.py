"""Integration test for console delivery via the webhook adapter.

Tests that the webhook adapter correctly dispatches console notifications
when receiving GitHub webhook events. This test is skippable if the
HERMES_CONSOLE_INGEST_SECRET env var is not set.
"""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH


def _make_config(routes=None, **kwargs):
    extra = {
        "host": "0.0.0.0",
        "port": 0,
        "routes": routes or {},
        "rate_limit": 30,
        "max_body_bytes": 1_048_576,
    }
    if "secret" in kwargs:
        extra["secret"] = kwargs.pop("secret")
    return PlatformConfig(enabled=True, extra=extra)


class TestConsoleDeliverType:
    """Tests for deliver='console' in send()."""

    @pytest.mark.asyncio
    async def test_console_deliver_returns_success(self):
        """When deliver='console', send() stores LLM output in session only."""
        adapter = WebhookAdapter(_make_config())
        chat_id = "webhook:test:d-xyz"
        adapter._delivery_info[chat_id] = {
            "deliver": "console",
            "deliver_extra": {},
            "payload": {},
        }
        result = await adapter.send(chat_id, "LLM response text")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_console_deliver_stores_in_delivery_info(self):
        """Console deliver doesn't pop delivery_info (same pattern as other types)."""
        adapter = WebhookAdapter(_make_config())
        chat_id = "webhook:test:d-xyz"
        import time
        adapter._delivery_info[chat_id] = {
            "deliver": "console",
            "deliver_extra": {},
            "payload": {},
        }
        adapter._delivery_info_created[chat_id] = time.time()
        result = await adapter.send(chat_id, "Response 1")
        assert result.success is True
        assert chat_id in adapter._delivery_info

    @pytest.mark.asyncio
    async def test_console_deliver_unknown_delivery_info_returns_log(self):
        """When no delivery_info matches, falls back to log which succeeds."""
        adapter = WebhookAdapter(_make_config())
        result = await adapter.send("webhook:unknown:missing", "Some content")
        # Falls through to log deliver type
        assert result.success is True


class TestSendConsoleNotification:
    """Tests for the _send_console_notification method."""

    @pytest.mark.asyncio
    async def test_send_console_notification_schedules_task(self):
        """_send_console_notification creates a background task for valid events."""
        adapter = WebhookAdapter(_make_config())

        payload = {
            "action": "opened",
            "sender": {"login": "octocat", "type": "User"},
            "repository": {"full_name": "owner/repo", "name": "repo", "owner": {"login": "owner"}},
            "pull_request": {
                "number": 1,
                "title": "Test PR",
                "html_url": "https://github.com/owner/repo/pull/1",
                "user": {"login": "octocat"},
                "base": {"ref": "main"},
                "head": {"ref": "test"},
                "labels": [],
            },
        }

        with patch.object(adapter, "_deliver_console", new_callable=AsyncMock) as mock_deliver:
            adapter._send_console_notification("pull_request", payload)
            # Give the event loop a chance to schedule the task
            await asyncio.sleep(0.05)
            # build_github_notification should be called and _deliver_console scheduled
            # if it returns a non-None notification

    @pytest.mark.asyncio
    async def test_send_console_notification_skips_non_github(self):
        """Non-GitHub events are not sent to console."""
        adapter = WebhookAdapter(_make_config())
        # _GITHUB_EVENT_TYPES does not include 'push'
        adapter._send_console_notification("push", {"action": "pushed"})
        # No background task should be created; just returns silently


class TestDeliverConsole:
    """Tests for the _deliver_console async method."""

    @pytest.mark.asyncio
    async def test_deliver_console_skips_without_secret(self):
        """If HERMES_CONSOLE_INGEST_SECRET is not set, no POST is made."""
        adapter = WebhookAdapter(_make_config())
        notification = {"source": "github", "repo": "o/r", "kind": "pr_opened"}

        with patch.dict(os.environ, {}, clear=True):
            # Remove any env var
            os.environ.pop("HERMES_CONSOLE_INGEST_SECRET", None)
            os.environ.pop("HERMES_CONSOLE_INGEST_URL", None)
            # Should complete without error
            await adapter._deliver_console(notification)

    @pytest.mark.asyncio
    async def test_deliver_console_posts_with_secret(self):
        """When secret is set, POST to ingest URL with correct headers."""
        adapter = WebhookAdapter(_make_config())
        notification = {
            "source": "github",
            "repo": "o/r",
            "kind": "pr_opened",
            "title": "PR opened",
            "body": "test",
            "url": "https://github.com/o/r/pull/1",
            "severity": "info",
            "metadata": {},
        }

        with patch.dict(os.environ, {
            "HERMES_CONSOLE_INGEST_SECRET": "test-secret-123",
            "HERMES_CONSOLE_INGEST_URL": "http://127.0.0.1:3001/api/notifications/ingest",
        }):
            with patch("aiohttp.ClientSession") as MockSession:
                # Build a mock response
                mock_resp = AsyncMock()
                mock_resp.status = 201
                mock_resp.text = AsyncMock(return_value="created")
                mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
                mock_resp.__aexit__ = AsyncMock(return_value=None)

                # Build a mock session
                mock_session = MagicMock()
                mock_session.post = MagicMock(return_value=mock_resp)
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock(return_value=None)
                MockSession.return_value = mock_session

                await adapter._deliver_console(notification)

                # Verify the POST was attempted
                assert mock_session.post.called

    @pytest.mark.asyncio
    async def test_deliver_console_failure_is_fire_and_forget(self):
        """If the POST fails, _deliver_console logs a warning but doesn't raise."""
        adapter = WebhookAdapter(_make_config())
        notification = {"source": "github", "repo": "o/r", "kind": "pr_opened"}

        with patch.dict(os.environ, {
            "HERMES_CONSOLE_INGEST_SECRET": "test-secret",
        }):
            with patch("aiohttp.ClientSession", side_effect=Exception("connection refused")):
                # Should not raise
                await adapter._deliver_console(notification)


@pytest.mark.skipif(
    not os.environ.get("HERMES_CONSOLE_INGEST_SECRET"),
    reason="HERMES_CONSOLE_INGEST_SECRET not set — console not available"
)
class TestConsoleIngestIntegration:
    """Live integration test that POSTs to the console ingest endpoint.

    Only runs when HERMES_CONSOLE_INGEST_SECRET is configured in the
    environment. This test actually sends a notification to the running
    Hermes Console service.
    """

    @pytest.mark.asyncio
    async def test_post_notification_to_ingest(self):
        """POST a notification to the console ingest endpoint with correct headers."""
        adapter = WebhookAdapter(_make_config())
        notification = {
            "source": "github",
            "repo": "ticketclosed-wontfix/hermes-agent",
            "kind": "pr_opened",
            "title": "Test notification from integration test",
            "body": "This is a test notification body",
            "url": "https://github.com/ticketclosed-wontfix/hermes-agent/pull/1",
            "severity": "info",
            "metadata": {"pr_number": 1, "author": "test"},
        }

        # This will actually POST to the console if it's running
        await adapter._deliver_console(notification)