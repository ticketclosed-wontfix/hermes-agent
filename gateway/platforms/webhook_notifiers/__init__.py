"""Webhook notification builders for structured platform events.

Each submodule extracts notification fields directly from webhook payloads
(no LLM involvement) and returns a dict suitable for posting to a
notification ingest endpoint.
"""

from gateway.platforms.webhook_notifiers.github_notification import (
    build_github_notification,
)

__all__ = ["build_github_notification"]