"""Tests for gateway.run._derive_explicit_title_for_source."""

from types import SimpleNamespace

import pytest

from gateway.run import _derive_explicit_title_for_source
from gateway.config import Platform


def _src(platform, chat_name):
    """Build a minimal duck-typed SessionSource-like object."""
    return SimpleNamespace(platform=platform, chat_name=chat_name)


class TestDeriveExplicitTitle:
    def test_webhook_with_route_produces_label(self):
        title = _derive_explicit_title_for_source(
            _src(Platform.WEBHOOK, "webhook/github-automation")
        )
        assert title is not None
        assert title.startswith("webhook: github-automation (")
        assert title.endswith(")")

    def test_webhook_plain_name_no_slash(self):
        """chat_name without '/' should still be usable as the route name."""
        title = _derive_explicit_title_for_source(
            _src(Platform.WEBHOOK, "my-route")
        )
        assert title is not None
        assert title.startswith("webhook: my-route (")

    def test_webhook_empty_chat_name_returns_none(self):
        assert _derive_explicit_title_for_source(
            _src(Platform.WEBHOOK, "")
        ) is None

    def test_non_webhook_platforms_return_none(self):
        for plat in (Platform.LOCAL, Platform.TELEGRAM, Platform.DISCORD, Platform.SLACK):
            assert _derive_explicit_title_for_source(
                _src(plat, "something")
            ) is None, f"{plat} should have returned None"

    def test_string_platform_value(self):
        """Some code paths may hand us a bare string instead of the Enum."""
        title = _derive_explicit_title_for_source(
            _src("webhook", "webhook/route-x")
        )
        assert title is not None
        assert "route-x" in title

    def test_none_source_returns_none(self):
        assert _derive_explicit_title_for_source(None) is None

    def test_bad_object_returns_none(self):
        """Any attribute-access failure should swallow and return None."""
        class Bad:
            @property
            def platform(self):
                raise RuntimeError("boom")
        assert _derive_explicit_title_for_source(Bad()) is None
