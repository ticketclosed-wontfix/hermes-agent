"""Verify AIAgent records the ``source`` kwarg in SessionDB rows.

Covers the delegate-tagging feature: child sessions spawned via
delegate_task() pass source='delegate' so the session row is distinguishable
from the parent's (which stays e.g. 'cli' or 'webhook').  Platform is still
inherited from the parent and used for delivery/tool-routing; only the DB
``source`` column is affected.
"""

from unittest.mock import patch

import pytest

from hermes_state import SessionDB


def _construct_agent_light(monkeypatch, session_db, *, source=None, platform="cli"):
    """Construct an AIAgent without firing any LLM/provider resolution.

    We stub out the expensive branches (tool discovery, provider init, memory,
    context compressor) and just confirm the SessionDB row lands with the
    right source value.  AIAgent's __init__ calls SessionDB.create_session
    at the end of the DB-init block, so by the time the constructor returns
    the session row is present.
    """
    from run_agent import AIAgent

    # Avoid heavy init paths (discover toolsets, compressor setup, etc.) by
    # going through the real constructor but with tiny toolsets and no model.
    agent = AIAgent(
        model="anthropic/claude-haiku-4-5",
        max_iterations=1,
        enabled_toolsets=[],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform=platform,
        source=source,
        session_db=session_db,
        persist_session=False,  # avoid file writes
    )
    return agent


def test_source_defaults_to_platform(tmp_path, monkeypatch):
    """When no explicit source is given, platform is used (back-compat)."""
    db = SessionDB(tmp_path / "state.db")
    agent = _construct_agent_light(monkeypatch, db, platform="telegram")

    with db._lock:
        row = db._conn.execute(
            "SELECT source FROM sessions WHERE id = ?", (agent.session_id,)
        ).fetchone()
    assert row is not None
    assert row["source"] == "telegram"


def test_explicit_source_overrides_platform(tmp_path, monkeypatch):
    """source='delegate' lands in DB even when platform is 'cli'."""
    db = SessionDB(tmp_path / "state.db")
    agent = _construct_agent_light(
        monkeypatch, db, source="delegate", platform="cli"
    )

    with db._lock:
        row = db._conn.execute(
            "SELECT source FROM sessions WHERE id = ?", (agent.session_id,)
        ).fetchone()
    assert row is not None
    assert row["source"] == "delegate"
    # platform attr is still the parent's — only the DB source changed
    assert agent.platform == "cli"


def test_source_survives_ensure_session(tmp_path, monkeypatch):
    """ensure_session (safety re-upsert) must also carry the source override."""
    db = SessionDB(tmp_path / "state.db")
    agent = _construct_agent_light(
        monkeypatch, db, source="delegate", platform="cli"
    )

    # Call ensure_session directly — mirrors what run_conversation does
    # at the top of the loop in case create_session hit a transient lock.
    db.ensure_session(
        agent.session_id,
        source=agent._source or agent.platform or "cli",
        model=agent.model,
    )

    with db._lock:
        row = db._conn.execute(
            "SELECT source FROM sessions WHERE id = ?", (agent.session_id,)
        ).fetchone()
    assert row is not None
    assert row["source"] == "delegate"
