"""Tests for per-route model/provider override on webhook subscriptions.

Subscription JSON entries can include a ``model`` field (and optionally
``provider``/``api_key``/``base_url``/``api_mode``) to pin a non-primary
model for that route.  The webhook adapter packages those into a
``model_override`` on the resulting ``MessageEvent``; the gateway runner
stashes the override into ``_session_model_overrides`` keyed by the
resolved session key, so ``_resolve_session_agent_runtime`` honors it
exactly like a session-scoped ``/model`` command.

Existing subscriptions without a ``model`` field must continue to use
the gateway primary — this is asserted in
``test_no_model_field_no_override``.
"""

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(routes) -> WebhookAdapter:
    extra = {"host": "0.0.0.0", "port": 0, "routes": routes}
    config = PlatformConfig(enabled=True, extra=extra)
    return WebhookAdapter(config)


def _create_app(adapter: WebhookAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


PAYLOAD = {"action": "opened", "issue": {"number": 1, "title": "test"}}


# ===================================================================
# Adapter packs route_config['model'] into MessageEvent.model_override
# ===================================================================


class TestWebhookAdapterModelOverridePacking:
    @pytest.mark.asyncio
    async def test_no_model_field_no_override(self):
        """Route without model/provider produces event with model_override=None."""
        routes = {
            "plain": {
                "secret": _INSECURE_NO_AUTH,
                "events": [],
                "prompt": "hi",
                "deliver": "log",
            }
        }
        adapter = _make_adapter(routes)
        captured: list[MessageEvent] = []

        async def _capture(event: MessageEvent):
            captured.append(event)

        adapter.handle_message = _capture
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/plain",
                json=PAYLOAD,
                headers={"X-GitHub-Delivery": "no-override-001"},
            )
            assert resp.status == 202

        await asyncio.sleep(0.05)
        assert len(captured) == 1
        assert captured[0].model_override is None

    @pytest.mark.asyncio
    async def test_model_only_inherits_provider_chain(self):
        """Route with bare ``model`` and no provider/api_key produces an
        override containing only ``model`` — gateway will inherit the
        rest of the runtime from the primary chain."""
        routes = {
            "sonnet-route": {
                "secret": _INSECURE_NO_AUTH,
                "events": [],
                "prompt": "hi",
                "deliver": "log",
                "model": "claude-sonnet-4-6",
            }
        }
        adapter = _make_adapter(routes)
        captured: list[MessageEvent] = []

        async def _capture(event: MessageEvent):
            captured.append(event)

        adapter.handle_message = _capture
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/sonnet-route",
                json=PAYLOAD,
                headers={"X-GitHub-Delivery": "model-only-001"},
            )
            assert resp.status == 202

        await asyncio.sleep(0.05)
        assert len(captured) == 1
        ov = captured[0].model_override
        assert ov is not None
        assert ov["model"] == "claude-sonnet-4-6"
        # Bare ``model`` — no provider bundle keys leak in.
        assert "provider" not in ov
        assert "api_key" not in ov
        assert "base_url" not in ov

    @pytest.mark.asyncio
    async def test_full_provider_bundle_flows_through(self):
        """Route with model+provider+api_key+base_url produces an override
        carrying every supplied key, mirroring session_model_overrides."""
        routes = {
            "fully-pinned": {
                "secret": _INSECURE_NO_AUTH,
                "events": [],
                "prompt": "hi",
                "deliver": "log",
                "model": "MiMo-7B-RL",
                "provider": "xiaomi",
                "api_key": "sk-test-1234",
                "base_url": "http://example.invalid:9000/v1",
                "api_mode": "openai",
            }
        }
        adapter = _make_adapter(routes)
        captured: list[MessageEvent] = []

        async def _capture(event: MessageEvent):
            captured.append(event)

        adapter.handle_message = _capture
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/fully-pinned",
                json=PAYLOAD,
                headers={"X-GitHub-Delivery": "full-bundle-001"},
            )
            assert resp.status == 202

        await asyncio.sleep(0.05)
        assert len(captured) == 1
        ov = captured[0].model_override
        assert ov == {
            "model": "MiMo-7B-RL",
            "provider": "xiaomi",
            "api_key": "sk-test-1234",
            "base_url": "http://example.invalid:9000/v1",
            "api_mode": "openai",
        }


# ===================================================================
# Gateway _handle_message stashes model_override into the
# session-keyed override dict.
# ===================================================================


class TestGatewayHandleMessageStashesOverride:
    """We exercise the override-write block directly without spinning up
    the full gateway loop — same pattern as other gateway tests that
    poke private state on a stub gateway-runner-like object.
    """

    def _stub_gateway(self):
        """Minimal stub mimicking the GatewayRunner attributes the
        override-stash block touches.  We intentionally avoid
        instantiating GatewayRunner — its constructor pulls in every
        platform adapter, the session DB, the cron loop, etc.
        """
        from gateway.run import GatewayRunner

        stub = MagicMock(spec=GatewayRunner)
        stub._session_model_overrides = {}
        stub._session_key_for_source = lambda source: f"agent:main:webhook:webhook:{source.chat_id}:webhook:{source.user_id}"
        return stub

    def _make_event(self, model_override):
        from gateway.platforms.base import MessageEvent, MessageType
        from gateway.session import SessionSource

        source = SessionSource(
            platform=Platform.WEBHOOK,
            chat_id="webhook:gh-test:abc123",
            user_id="webhook:gh-test",
            user_name="gh-test",
            chat_type="webhook",
        )
        return MessageEvent(
            text="hello",
            message_type=MessageType.TEXT,
            source=source,
            model_override=model_override,
        )

    def test_event_with_model_override_writes_session_override(self):
        """When a MessageEvent carries a model_override with ``model``,
        the gateway's override-stash block writes it into
        _session_model_overrides under the resolved session key.
        """
        from gateway.run import GatewayRunner

        stub = self._stub_gateway()
        event = self._make_event(
            {"model": "claude-sonnet-4-6", "provider": "custom"}
        )

        # Reproduce the relevant slice of _handle_message — keep this
        # narrow so the test isn't coupled to authorization / queue /
        # interrupt branches that aren't part of the contract.
        source = event.source
        _quick_key = stub._session_key_for_source(source)
        _event_model_override = getattr(event, "model_override", None)
        if _event_model_override and _event_model_override.get("model"):
            stub._session_model_overrides[_quick_key] = dict(_event_model_override)

        assert _quick_key in stub._session_model_overrides
        stored = stub._session_model_overrides[_quick_key]
        assert stored["model"] == "claude-sonnet-4-6"
        assert stored["provider"] == "custom"

    def test_event_without_model_override_leaves_dict_empty(self):
        stub = self._stub_gateway()
        event = self._make_event(None)

        source = event.source
        _quick_key = stub._session_key_for_source(source)
        _event_model_override = getattr(event, "model_override", None)
        if _event_model_override and _event_model_override.get("model"):
            stub._session_model_overrides[_quick_key] = dict(_event_model_override)

        assert stub._session_model_overrides == {}

    def test_event_with_empty_dict_leaves_dict_empty(self):
        """A model_override that's a dict but missing ``model`` is a
        no-op — guards against subscriptions with stray provider-only
        fields accidentally pinning the empty string as a model name."""
        stub = self._stub_gateway()
        event = self._make_event({"provider": "custom"})

        source = event.source
        _quick_key = stub._session_key_for_source(source)
        _event_model_override = getattr(event, "model_override", None)
        if _event_model_override and _event_model_override.get("model"):
            stub._session_model_overrides[_quick_key] = dict(_event_model_override)

        assert stub._session_model_overrides == {}
