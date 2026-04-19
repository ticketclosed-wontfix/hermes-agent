"""Generic webhook platform adapter.

Runs an aiohttp HTTP server that receives webhook POSTs from external
services (GitHub, GitLab, JIRA, Stripe, etc.), validates HMAC signatures,
transforms payloads into agent prompts, and routes responses back to the
source or to another configured platform.

Configuration lives in config.yaml under platforms.webhook.extra.routes.
Each route defines:
  - events: which event types to accept (header-based filtering)
  - secret: HMAC secret for signature validation (REQUIRED)
  - prompt: template string formatted with the webhook payload
  - skills: optional list of skills to load for the agent
  - deliver: where to send the response (github_comment, telegram, etc.)
  - deliver_extra: additional delivery config (repo, pr_number, chat_id)

Security:
  - HMAC secret is required per route (validated at startup)
  - Rate limiting per route (fixed-window, configurable)
  - Idempotency cache prevents duplicate agent runs on webhook retries
  - Body size limits checked before reading payload
  - Set secret to "INSECURE_NO_AUTH" to skip validation (testing only)
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import subprocess
import time
from typing import Any, Dict, List, Optional

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8644
_INSECURE_NO_AUTH="***"
_DYNAMIC_ROUTES_FILENAME = "webhook_subscriptions.json"

# Sentinel returned by agents to suppress outbound delivery without error.
# When the final response ends with a line containing only this marker
# (after .strip()), the webhook adapter logs at INFO and short-circuits
# send() with success=True — no delivery.
#
# NOTE: We deliberately match the marker as a TRAILING line, not just
# exact-equal to the trimmed content.  Models routinely prepend an
# explanation before the sentinel ("PR #15 has nothing to act on.\n\nNOOP")
# and fighting that with prompt engineering is a losing battle.  Widening
# the sentinel here is the robust fix.  See test_send_noop_* for coverage.
NOOP_SENTINEL = "NOOP"
_NOOP_TRAILING_RE = re.compile(r"(^|\n)\s*NOOP\s*$")

# Console deliver type — POSTs structured notifications to the Hermes Console
# ingest endpoint.  Fire-and-forget: failures log a warning but don't block
# the LLM pipeline.
_DEFAULT_CONSOLE_INGEST_URL = "http://127.0.0.1:3001/api/notifications/ingest"

# GitHub event types that can produce console notifications
_GITHUB_EVENT_TYPES = frozenset({
    "pull_request", "pull_request_review", "issues",
    "issue_comment", "check_suite", "check_run",
})


def check_webhook_requirements() -> bool:
    """Check if webhook adapter dependencies are available."""
    return AIOHTTP_AVAILABLE


# ---------------------------------------------------------------------------
# Filter DSL evaluator (pre-LLM gating)
# ---------------------------------------------------------------------------
#
# Per-subscription ``filter`` field lets the adapter reject non-actionable
# webhook payloads before any agent session is spawned. Supported DSL:
#
#   {"any_of": [<filter>, ...]}        OR
#   {"all_of": [<filter>, ...]}        AND
#   {"not": <filter>}                  negation
#   {"path": "a.b", "equals": "v"}     exact string match against payload[a][b]
#   {"path": "a.b", "in": ["x","y"]}   value is in the given list
#   {"path": "a.b", "regex": "pat"}    re.search match; "flags":"i" for ignore-case
#   {"path": "a.b", "exists": true}    path resolves to a non-None value
#   {"path": "a.b", "exists": false}   path is missing / None
#   {"path": "a.b", "non_empty": true} path resolves to a non-empty
#                                       list/string/dict (anything where
#                                       ``len() > 0``).  False if missing,
#                                       None, empty, or a non-sized type.
#                                       ``non_empty: false`` matches the
#                                       inverse — missing/empty/non-sized.
#
# Missing paths resolve to None.  equals/in/regex against None are always
# False.  exists:false matches missing paths.
#
# Evaluator is a pure function:
#     evaluate_filter(payload: dict, spec: dict) -> (matched: bool, reason: str)
# When ``matched`` is False, ``reason`` is a short human-readable
# explanation; when True, reason is an empty string.
#
# Subscriptions without a ``filter`` field are pass-through — the evaluator
# is only invoked when a filter is configured, so behaviour is strictly
# backward-compatible.


_FILTER_KEYS = {"any_of", "all_of", "not", "path"}


def _resolve_path(payload: Any, dotted: str) -> Any:
    """Walk ``payload`` using a ``"a.b.c"`` dotted path.  Returns None if any
    segment is missing or a non-dict is encountered mid-walk."""
    if not isinstance(dotted, str) or not dotted:
        return None
    value: Any = payload
    for part in dotted.split("."):
        if isinstance(value, dict):
            value = value.get(part)
            if value is None:
                return None
        else:
            return None
    return value


def _regex_flags(spec_flags: Any) -> int:
    """Translate the ``flags`` string from a regex filter into ``re`` flag
    bits.  Currently supports 'i' (IGNORECASE), 'm' (MULTILINE), 's' (DOTALL).
    Unknown flags are ignored silently — be permissive on config input."""
    if not spec_flags or not isinstance(spec_flags, str):
        return 0
    bits = 0
    for ch in spec_flags.lower():
        if ch == "i":
            bits |= re.IGNORECASE
        elif ch == "m":
            bits |= re.MULTILINE
        elif ch == "s":
            bits |= re.DOTALL
    return bits


def evaluate_filter(payload: Any, spec: Any) -> "tuple[bool, str]":
    """Evaluate a filter spec against a payload.

    Returns (matched, reason).  When ``matched`` is True, ``reason`` is the
    empty string.  When False, ``reason`` is a short human-readable
    explanation used in the filtered HTTP 202 response body.
    """
    # Malformed or empty spec → treat as pass-through (matched).  Consistent
    # with "absent filter = no gating".
    if spec is None:
        return True, ""
    if not isinstance(spec, dict):
        return False, "filter spec must be an object"

    # Combinators
    if "any_of" in spec:
        branches = spec.get("any_of") or []
        if not isinstance(branches, list):
            return False, "any_of must be a list"
        if not branches:
            # Empty any_of rejects by convention (nothing matches "no
            # allowed branches").
            return False, "any_of has no branches"
        for sub in branches:
            ok, _ = evaluate_filter(payload, sub)
            if ok:
                return True, ""
        return False, "no filter branch matched"

    if "all_of" in spec:
        branches = spec.get("all_of") or []
        if not isinstance(branches, list):
            return False, "all_of must be a list"
        for sub in branches:
            ok, reason = evaluate_filter(payload, sub)
            if not ok:
                return False, reason
        return True, ""

    if "not" in spec:
        inner = spec.get("not")
        ok, _ = evaluate_filter(payload, inner)
        if ok:
            return False, "negated filter matched"
        return True, ""

    # Leaf predicate — must have a "path"
    if "path" in spec:
        path = spec.get("path")
        value = _resolve_path(payload, path)

        if "exists" in spec:
            want = bool(spec.get("exists"))
            present = value is not None
            if present == want:
                return True, ""
            if want:
                return False, f"path '{path}' missing"
            return False, f"path '{path}' present but should be absent"

        if "non_empty" in spec:
            want = bool(spec.get("non_empty"))
            # Treat None / non-sized values as empty.  bool(0) is False,
            # but ints aren't sized — len() raises — so we explicitly
            # check for sized container types we care about.
            if value is None:
                non_empty = False
            elif isinstance(value, (list, tuple, dict, str, set)):
                non_empty = len(value) > 0
            else:
                # Booleans, numbers — "non-empty" is meaningless; treat
                # as empty so authors can't accidentally rely on it.
                non_empty = False
            if non_empty == want:
                return True, ""
            if want:
                return False, f"path '{path}' is empty or not a container"
            return False, f"path '{path}' is non-empty but should be empty"

        if "equals" in spec:
            if value is None:
                return False, f"path '{path}' is missing"
            if str(value) == str(spec.get("equals")):
                return True, ""
            return False, f"path '{path}' != expected value"

        if "in" in spec:
            allowed = spec.get("in") or []
            if not isinstance(allowed, list):
                return False, "'in' must be a list"
            if value is None:
                return False, f"path '{path}' is missing"
            # Compare both raw and stringified to handle numeric payload
            # values vs string config entries.
            if value in allowed or str(value) in [str(a) for a in allowed]:
                return True, ""
            return False, f"path '{path}' not in allowed values"

        if "regex" in spec:
            pattern = spec.get("regex")
            if not isinstance(pattern, str):
                return False, "'regex' must be a string"
            if value is None or not isinstance(value, str):
                return False, f"path '{path}' is missing or not a string"
            try:
                if re.search(pattern, value, _regex_flags(spec.get("flags"))):
                    return True, ""
            except re.error as exc:
                return False, f"invalid regex on '{path}': {exc}"
            return False, f"path '{path}' did not match regex"

        return False, f"path predicate on '{path}' has no operator"

    return False, "unknown filter spec (need any_of/all_of/not/path)"


class WebhookAdapter(BasePlatformAdapter):
    """Generic webhook receiver that triggers agent runs from HTTP POSTs."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WEBHOOK)
        self._host: str = config.extra.get("host", DEFAULT_HOST)
        self._port: int = int(config.extra.get("port", DEFAULT_PORT))
        self._global_secret: str = config.extra.get("secret", "")
        self._static_routes: Dict[str, dict] = config.extra.get("routes", {})
        self._dynamic_routes: Dict[str, dict] = {}
        self._dynamic_routes_mtime: float = 0.0
        self._routes: Dict[str, dict] = dict(self._static_routes)
        self._runner = None

        # Delivery info keyed by session chat_id.
        #
        # Read by every send() invocation for the chat_id (status messages
        # AND the final response).  Cleaned up via TTL on each POST so the
        # dict stays bounded — see _prune_delivery_info().  Do NOT pop on
        # send(), or interim status messages (e.g. fallback notifications,
        # context-pressure warnings) will consume the entry before the
        # final response arrives, causing the response to silently fall
        # back to the "log" deliver type.
        self._delivery_info: Dict[str, dict] = {}
        self._delivery_info_created: Dict[str, float] = {}

        # Reference to gateway runner for cross-platform delivery (set externally)
        self.gateway_runner = None

        # Idempotency: TTL cache of recently processed delivery IDs.
        # Prevents duplicate agent runs when webhook providers retry.
        self._seen_deliveries: Dict[str, float] = {}
        self._idempotency_ttl: int = 3600  # 1 hour

        # Rate limiting: per-route timestamps in a fixed window.
        self._rate_counts: Dict[str, List[float]] = {}
        self._rate_limit: int = int(config.extra.get("rate_limit", 30))  # per minute

        # Body size limit (auth-before-body pattern)
        self._max_body_bytes: int = int(
            config.extra.get("max_body_bytes", 1_048_576)
        )  # 1MB

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        # Load agent-created subscriptions before validating
        self._reload_dynamic_routes()

        # Validate routes at startup — secret is required per route
        for name, route in self._routes.items():
            secret = route.get("secret", self._global_secret)
            if not secret:
                raise ValueError(
                    f"[webhook] Route '{name}' has no HMAC secret. "
                    f"Set 'secret' on the route or globally. "
                    f"For testing without auth, set secret to '{_INSECURE_NO_AUTH}'."
                )

        app = web.Application()
        app.router.add_get("/health", self._handle_health)
        app.router.add_post("/webhooks/{route_name}", self._handle_webhook)

        # Port conflict detection — fail fast if port is already in use
        import socket as _socket
        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
                _s.settimeout(1)
                _s.connect(('127.0.0.1', self._port))
            logger.error('[webhook] Port %d already in use. Set a different port in config.yaml: platforms.webhook.port', self._port)
            return False
        except (ConnectionRefusedError, OSError):
            pass  # port is free

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        self._mark_connected()

        route_names = ", ".join(self._routes.keys()) or "(none configured)"
        logger.info(
            "[webhook] Listening on %s:%d — routes: %s",
            self._host,
            self._port,
            route_names,
        )
        return True

    async def disconnect(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._mark_disconnected()
        logger.info("[webhook] Disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Deliver the agent's response to the configured destination.

        chat_id is ``webhook:{route}:{delivery_id}``.  The delivery info
        stored during webhook receipt is read with ``.get()`` (not popped)
        so that interim status messages emitted before the final response
        — fallback-model notifications, context-pressure warnings, etc. —
        do not consume the entry and silently downgrade the final response
        to the ``log`` deliver type.  TTL cleanup happens on POST.
        """
        delivery = self._delivery_info.get(chat_id, {})
        deliver_type = delivery.get("deliver", "log")

        # ── NOOP suppression ─────────────────────────────────────
        # If the agent's final response ends with a line containing only
        # the NOOP sentinel (optionally surrounded by whitespace),
        # swallow the delivery silently.  This lets prompts classify a
        # webhook event as non-actionable (e.g. non-allow-listed
        # commenter, wrong action) and bail without spamming the
        # configured delivery target.  Applies to every delivery type,
        # including github_comment.
        #
        # The match accepts trailing-line NOOP, not just exact-equal
        # trimmed content, because models routinely prepend an
        # explanation before the sentinel.  See _NOOP_TRAILING_RE.
        if content is not None and _NOOP_TRAILING_RE.search(content.strip()):
            # Log the FULL suppressed content at DEBUG and a 200-char
            # preview at INFO so suppressed messages can be audited
            # without spamming production logs.  Without this, the
            # only signal that a real message got eaten by an
            # over-eager sentinel is silence.
            preview = content.strip().replace("\n", " ⏎ ")[:200]
            logger.info(
                "[webhook] Agent returned NOOP; suppressing delivery for %s "
                "(deliver=%s) — preview=%r",
                chat_id,
                deliver_type,
                preview,
            )
            logger.debug(
                "[webhook] Full suppressed NOOP content for %s: %s",
                chat_id,
                content,
            )
            return SendResult(success=True)

        if deliver_type == "log":
            logger.info("[webhook] Response for %s: %s", chat_id, content[:200])
            return SendResult(success=True)

        if deliver_type == "github_comment":
            return await self._deliver_github_comment(content, delivery)

        # Console deliver type — LLM output goes to session DB only.
        # The console notification was already sent from _handle_webhook
        # (extracted from the raw payload, no LLM involvement).
        if deliver_type == "console":
            logger.info("[webhook] Console deliver — LLM output stored in session only: %s", chat_id)
            return SendResult(success=True)

        # Cross-platform delivery — any platform with a gateway adapter
        if self.gateway_runner and deliver_type in (
            "telegram",
            "discord",
            "slack",
            "signal",
            "sms",
            "whatsapp",
            "matrix",
            "mattermost",
            "homeassistant",
            "email",
            "dingtalk",
            "feishu",
            "wecom",
            "wecom_callback",
            "weixin",
            "bluebubbles",
            "qqbot",
        ):
            return await self._deliver_cross_platform(
                deliver_type, content, delivery
            )

        logger.warning("[webhook] Unknown deliver type: %s", deliver_type)
        return SendResult(
            success=False, error=f"Unknown deliver type: {deliver_type}"
        )

    def _prune_delivery_info(self, now: float) -> None:
        """Drop delivery_info entries older than the idempotency TTL.

        Mirrors the cleanup pattern used for ``_seen_deliveries``.  Called
        on each POST so the dict size is bounded by ``rate_limit * TTL``
        even if many webhooks fire and never receive a final response.
        """
        cutoff = now - self._idempotency_ttl
        stale = [
            k
            for k, t in self._delivery_info_created.items()
            if t < cutoff
        ]
        for k in stale:
            self._delivery_info.pop(k, None)
            self._delivery_info_created.pop(k, None)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "webhook"}

    # ------------------------------------------------------------------
    # HTTP handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        """GET /health — simple health check."""
        return web.json_response({"status": "ok", "platform": "webhook"})

    def _reload_dynamic_routes(self) -> None:
        """Reload agent-created subscriptions from disk if the file changed."""
        from hermes_constants import get_hermes_home
        hermes_home = get_hermes_home()
        subs_path = hermes_home / _DYNAMIC_ROUTES_FILENAME
        if not subs_path.exists():
            if self._dynamic_routes:
                self._dynamic_routes = {}
                self._routes = dict(self._static_routes)
                logger.debug("[webhook] Dynamic subscriptions file removed, cleared dynamic routes")
            return
        try:
            mtime = subs_path.stat().st_mtime
            if mtime <= self._dynamic_routes_mtime:
                return  # No change
            data = json.loads(subs_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
            # Merge: static routes take precedence over dynamic ones
            self._dynamic_routes = {
                k: v for k, v in data.items()
                if k not in self._static_routes
            }
            self._routes = {**self._dynamic_routes, **self._static_routes}
            self._dynamic_routes_mtime = mtime
            logger.info(
                "[webhook] Reloaded %d dynamic route(s): %s",
                len(self._dynamic_routes),
                ", ".join(self._dynamic_routes.keys()) or "(none)",
            )
        except Exception as e:
            logger.error("[webhook] Failed to reload dynamic routes: %s", e)

    async def _handle_webhook(self, request: "web.Request") -> "web.Response":
        """POST /webhooks/{route_name} — receive and process a webhook event."""
        # Hot-reload dynamic subscriptions on each request (mtime-gated, cheap)
        self._reload_dynamic_routes()

        route_name = request.match_info.get("route_name", "")
        route_config = self._routes.get(route_name)

        if not route_config:
            return web.json_response(
                {"error": f"Unknown route: {route_name}"}, status=404
            )

        # ── Auth-before-body ─────────────────────────────────────
        # Check Content-Length before reading the full payload.
        content_length = request.content_length or 0
        if content_length > self._max_body_bytes:
            return web.json_response(
                {"error": "Payload too large"}, status=413
            )

        # ── Rate limiting ────────────────────────────────────────
        now = time.time()
        window = self._rate_counts.setdefault(route_name, [])
        window[:] = [t for t in window if now - t < 60]
        if len(window) >= self._rate_limit:
            return web.json_response(
                {"error": "Rate limit exceeded"}, status=429
            )
        window.append(now)

        # Read body
        try:
            raw_body = await request.read()
        except Exception as e:
            logger.error("[webhook] Failed to read body: %s", e)
            return web.json_response({"error": "Bad request"}, status=400)

        # Validate HMAC signature (skip for INSECURE_NO_AUTH testing mode)
        secret = route_config.get("secret", self._global_secret)
        if secret and secret != _INSECURE_NO_AUTH:
            if not self._validate_signature(request, raw_body, secret):
                logger.warning(
                    "[webhook] Invalid signature for route %s", route_name
                )
                return web.json_response(
                    {"error": "Invalid signature"}, status=401
                )

        # Parse payload
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            # Try form-encoded as fallback
            try:
                import urllib.parse

                payload = dict(
                    urllib.parse.parse_qsl(raw_body.decode("utf-8"))
                )
            except Exception:
                return web.json_response(
                    {"error": "Cannot parse body"}, status=400
                )

        # Check event type filter
        event_type = (
            request.headers.get("X-GitHub-Event", "")
            or request.headers.get("X-GitLab-Event", "")
            or payload.get("event_type", "")
            or "unknown"
        )
        allowed_events = route_config.get("events", [])
        if allowed_events and event_type not in allowed_events:
            logger.debug(
                "[webhook] Ignoring event %s for route %s (allowed: %s)",
                event_type,
                route_name,
                allowed_events,
            )
            return web.json_response(
                {"status": "ignored", "event": event_type}
            )

        # ── Pre-LLM filter DSL ───────────────────────────────────
        # If the subscription defines a ``filter`` field, evaluate it
        # against the full payload before spawning an agent session.
        # Rejected payloads return HTTP 202 with filtered:true and the
        # reason the filter gave, so GitHub / the caller treat this as a
        # successful delivery (no retry storms) but no agent runs.
        filter_spec = route_config.get("filter")
        if filter_spec is not None:
            matched, reason = evaluate_filter(payload, filter_spec)
            if not matched:
                logger.info(
                    "[webhook] Filtered event route=%s event=%s reason=%s",
                    route_name,
                    event_type,
                    reason,
                )
                return web.json_response(
                    {
                        "status": "filtered",
                        "filtered": True,
                        "route": route_name,
                        "event": event_type,
                        "reason": reason,
                    },
                    status=202,
                )

        # Format prompt from template
        prompt_template = route_config.get("prompt", "")
        prompt = self._render_prompt(
            prompt_template, payload, event_type, route_name
        )

        # Inject skill content if configured.
        # We call build_skill_invocation_message() directly rather than
        # using /skill-name slash commands — the gateway's command parser
        # would intercept those and break the flow.
        skills = route_config.get("skills", [])
        if skills:
            try:
                from agent.skill_commands import (
                    build_skill_invocation_message,
                    get_skill_commands,
                )

                skill_cmds = get_skill_commands()
                for skill_name in skills:
                    cmd_key = f"/{skill_name}"
                    if cmd_key in skill_cmds:
                        skill_content = build_skill_invocation_message(
                            cmd_key, user_instruction=prompt
                        )
                        if skill_content:
                            prompt = skill_content
                            break  # Load the first matching skill
                    else:
                        logger.warning(
                            "[webhook] Skill '%s' not found", skill_name
                        )
            except Exception as e:
                logger.warning("[webhook] Skill loading failed: %s", e)

        # Build a unique delivery ID
        delivery_id = request.headers.get(
            "X-GitHub-Delivery",
            request.headers.get("X-Request-ID", str(int(time.time() * 1000))),
        )

        # ── Idempotency ─────────────────────────────────────────
        # Skip duplicate deliveries (webhook retries).
        now = time.time()
        # Prune expired entries
        self._seen_deliveries = {
            k: v
            for k, v in self._seen_deliveries.items()
            if now - v < self._idempotency_ttl
        }
        if delivery_id in self._seen_deliveries:
            logger.info(
                "[webhook] Skipping duplicate delivery %s", delivery_id
            )
            return web.json_response(
                {"status": "duplicate", "delivery_id": delivery_id},
                status=200,
            )
        self._seen_deliveries[delivery_id] = now

        # ── Console notification (parallel lane) ──────────────────
        # For GitHub webhooks, extract a structured notification from the
        # raw payload and POST it to the Hermes Console ingest endpoint.
        # This runs BEFORE/ALONGSIDE the LLM session — no LLM involvement.
        # Fire-and-forget: failures log a warning but don't block the pipeline.
        if event_type in _GITHUB_EVENT_TYPES:
            self._send_console_notification(event_type, payload)

        # Use delivery_id in session key so concurrent webhooks on the
        # same route get independent agent runs (not queued/interrupted).
        session_chat_id = f"webhook:{route_name}:{delivery_id}"

        # Store delivery info for send().  Read by every send() invocation
        # for this chat_id (interim status messages and the final response),
        # so we do NOT pop on send.  TTL-based cleanup keeps the dict bounded.
        deliver_config = {
            "deliver": route_config.get("deliver", "log"),
            "deliver_extra": self._render_delivery_extra(
                route_config.get("deliver_extra", {}), payload
            ),
            "payload": payload,
        }
        self._delivery_info[session_chat_id] = deliver_config
        self._delivery_info_created[session_chat_id] = now
        self._prune_delivery_info(now)

        # Build source and event
        source = self.build_source(
            chat_id=session_chat_id,
            chat_name=f"webhook/{route_name}",
            chat_type="webhook",
            user_id=f"webhook:{route_name}",
            user_name=route_name,
        )

        # Per-route model/provider override.  Subscription JSON may pin a
        # specific model (e.g. claude-sonnet-4-6 for github-automation
        # while the global primary stays on Opus).  Mirrors the shape of
        # GatewayRunner._session_model_overrides — only ``model`` is
        # required; provider/api_key/base_url/api_mode are optional and
        # default to whatever the resolved primary chain provides.
        model_override: Optional[Dict[str, str]] = None
        route_model = route_config.get("model")
        if route_model:
            model_override = {"model": str(route_model)}
            for _opt in ("provider", "api_key", "base_url", "api_mode"):
                _val = route_config.get(_opt)
                if _val:
                    model_override[_opt] = str(_val)

        event = MessageEvent(
            text=prompt,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=payload,
            message_id=delivery_id,
            model_override=model_override,
        )

        logger.info(
            "[webhook] %s event=%s route=%s prompt_len=%d delivery=%s",
            request.method,
            event_type,
            route_name,
            len(prompt),
            delivery_id,
        )

        # Non-blocking — return 202 Accepted immediately
        task = asyncio.create_task(self.handle_message(event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        return web.json_response(
            {
                "status": "accepted",
                "route": route_name,
                "event": event_type,
                "delivery_id": delivery_id,
            },
            status=202,
        )

    # ------------------------------------------------------------------
    # Signature validation
    # ------------------------------------------------------------------

    def _validate_signature(
        self, request: "web.Request", body: bytes, secret: str
    ) -> bool:
        """Validate webhook signature (GitHub, GitLab, generic HMAC-SHA256)."""
        # GitHub: X-Hub-Signature-256 = sha256=<hex>
        gh_sig = request.headers.get("X-Hub-Signature-256", "")
        if gh_sig:
            expected = "sha256=" + hmac.new(
                secret.encode(), body, hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(gh_sig, expected)

        # GitLab: X-Gitlab-Token = <plain secret>
        gl_token = request.headers.get("X-Gitlab-Token", "")
        if gl_token:
            return hmac.compare_digest(gl_token, secret)

        # Generic: X-Webhook-Signature = <hex HMAC-SHA256>
        generic_sig = request.headers.get("X-Webhook-Signature", "")
        if generic_sig:
            expected = hmac.new(
                secret.encode(), body, hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(generic_sig, expected)

        # No recognised signature header but secret is configured → reject
        logger.debug(
            "[webhook] Secret configured but no signature header found"
        )
        return False

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def _render_prompt(
        self,
        template: str,
        payload: dict,
        event_type: str,
        route_name: str,
    ) -> str:
        """Render a prompt template with the webhook payload.

        Supports dot-notation access into nested dicts:
        ``{pull_request.title}`` → ``payload["pull_request"]["title"]``

        Special token ``{__raw__}`` dumps the entire payload as indented
        JSON (truncated to 4000 chars).  Useful for monitoring alerts or
        any webhook where the agent needs to see the full payload.
        """
        if not template:
            truncated = json.dumps(payload, indent=2)[:4000]
            return (
                f"Webhook event '{event_type}' on route "
                f"'{route_name}':\n\n```json\n{truncated}\n```"
            )

        def _resolve(match: re.Match) -> str:
            key = match.group(1)
            # Special token: dump the entire payload as JSON
            if key == "__raw__":
                return json.dumps(payload, indent=2)[:4000]
            value: Any = payload
            for part in key.split("."):
                if isinstance(value, dict):
                    value = value.get(part, f"{{{key}}}")
                else:
                    return f"{{{key}}}"
            if isinstance(value, (dict, list)):
                return json.dumps(value, indent=2)[:2000]
            return str(value)

        return re.sub(r"\{([a-zA-Z0-9_.]+)\}", _resolve, template)

    def _render_delivery_extra(
        self, extra: dict, payload: dict
    ) -> dict:
        """Render delivery_extra template values with payload data."""
        rendered: Dict[str, Any] = {}
        for key, value in extra.items():
            if isinstance(value, str):
                rendered[key] = self._render_prompt(value, payload, "", "")
            else:
                rendered[key] = value
        return rendered

    # ------------------------------------------------------------------
    # Response delivery
    # ------------------------------------------------------------------

    async def _deliver_github_comment(
        self, content: str, delivery: dict
    ) -> SendResult:
        """Post agent response as a GitHub PR/issue comment via ``gh`` CLI."""
        extra = delivery.get("deliver_extra", {})
        repo = extra.get("repo", "")
        pr_number = extra.get("pr_number", "")

        if not repo or not pr_number:
            logger.error(
                "[webhook] github_comment delivery missing repo or pr_number"
            )
            return SendResult(
                success=False, error="Missing repo or pr_number"
            )

        try:
            result = subprocess.run(
                [
                    "gh",
                    "pr",
                    "comment",
                    str(pr_number),
                    "--repo",
                    repo,
                    "--body",
                    content,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                logger.info(
                    "[webhook] Posted comment on %s#%s", repo, pr_number
                )
                return SendResult(success=True)
            else:
                logger.error(
                    "[webhook] gh pr comment failed: %s", result.stderr
                )
                return SendResult(success=False, error=result.stderr)
        except FileNotFoundError:
            logger.error(
                "[webhook] 'gh' CLI not found — install GitHub CLI for "
                "github_comment delivery"
            )
            return SendResult(
                success=False, error="gh CLI not installed"
            )
        except Exception as e:
            logger.error("[webhook] github_comment delivery error: %s", e)
            return SendResult(success=False, error=str(e))

    async def _deliver_cross_platform(
        self, platform_name: str, content: str, delivery: dict
    ) -> SendResult:
        """Route response to another platform (telegram, discord, etc.)."""
        if not self.gateway_runner:
            return SendResult(
                success=False,
                error="No gateway runner for cross-platform delivery",
            )

        try:
            target_platform = Platform(platform_name)
        except ValueError:
            return SendResult(
                success=False, error=f"Unknown platform: {platform_name}"
            )

        adapter = self.gateway_runner.adapters.get(target_platform)
        if not adapter:
            return SendResult(
                success=False,
                error=f"Platform {platform_name} not connected",
            )

        # Use home channel if no specific chat_id in deliver_extra
        extra = delivery.get("deliver_extra", {})
        chat_id = extra.get("chat_id", "")
        if not chat_id:
            home = self.gateway_runner.config.get_home_channel(target_platform)
            if home:
                chat_id = home.chat_id
            else:
                return SendResult(
                    success=False,
                    error=f"No chat_id or home channel for {platform_name}",
                )

        # Pass thread_id from deliver_extra so Telegram forum topics work
        metadata = None
        thread_id = extra.get("message_thread_id") or extra.get("thread_id")
        if thread_id:
            metadata = {"thread_id": thread_id}

        result = await adapter.send(chat_id, content, metadata=metadata)

        if result.success:
            try:
                from gateway.mirror import mirror_to_session
                mirror_to_session(
                    platform_name, chat_id, content,
                    source_label="webhook", thread_id=str(thread_id) if thread_id else None,
                )
            except Exception as e:
                logger.debug("[webhook] mirror_to_session failed: %s", e)

        return result

    # ------------------------------------------------------------------
    # Console notification delivery
    # ------------------------------------------------------------------

    def _send_console_notification(self, event_type: str, payload: dict) -> None:
        """Extract a structured notification from a GitHub webhook payload and
        POST it to the Hermes Console ingest endpoint (fire-and-forget).

        This is called from _handle_webhook for every GitHub event before
        the LLM session starts.  It runs synchronously (the async POST is
        scheduled as a background task) so it never blocks the webhook
        response.
        """
        try:
            from gateway.platforms.webhook_notifiers.github_notification import (
                build_github_notification,
            )
        except ImportError:
            logger.debug("[webhook] github_notification module not available")
            return

        notification = build_github_notification(event_type, payload)
        if notification is None:
            return  # Event type/action doesn't warrant a notification

        # Schedule the async POST — don't await it
        task = asyncio.create_task(
            self._deliver_console(notification)
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _deliver_console(self, notification: dict) -> None:
        """POST a notification dict to the Hermes Console ingest endpoint.

        Uses aiohttp with a 5s timeout.  On failure, logs a warning but
        returns success — notifications are fire-and-forget and must not
        block the LLM pipeline.
        """
        ingest_url = os.environ.get(
            "HERMES_CONSOLE_INGEST_URL", _DEFAULT_CONSOLE_INGEST_URL
        )
        ingest_secret = os.environ.get("HERMES_CONSOLE_INGEST_SECRET", "")

        # Fall back to reading from config if env var is not set
        if not ingest_secret:
            try:
                from hermes_constants import get_hermes_home
                env_path = get_hermes_home() / ".env"
                if env_path.exists():
                    for line in env_path.read_text().splitlines():
                        line = line.strip()
                        if line.startswith("HERMES_CONSOLE_INGEST_SECRET="):
                            ingest_secret = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
            except Exception:
                pass

        if not ingest_secret:
            logger.debug("[webhook] No HERMES_CONSOLE_INGEST_SECRET set; skipping console notification")
            return

        try:
            import aiohttp as _aiohttp

            headers = {
                "Content-Type": "application/json",
                "X-Hermes-Ingest-Secret": ingest_secret,
            }
            timeout = _aiohttp.ClientTimeout(total=5)

            async with _aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    ingest_url, json=notification, headers=headers
                ) as resp:
                    if resp.status < 400:
                        logger.info(
                            "[webhook] Console notification sent (status=%d kind=%s repo=%s)",
                            resp.status,
                            notification.get("kind", "?"),
                            notification.get("repo", "?"),
                        )
                    else:
                        body = await resp.text()
                        logger.warning(
                            "[webhook] Console ingest returned %d: %s",
                            resp.status, body[:200],
                        )
        except Exception as e:
            logger.warning(
                "[webhook] Console notification POST failed (fire-and-forget): %s", e
            )
