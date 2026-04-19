"""Unit tests for github_notification module.

Tests build_github_notification() for each supported event type / action
combination, verifying the returned dict structure and that irrelevant
events return None.
"""

import pytest
from gateway.platforms.webhook_notifiers.github_notification import (
    build_github_notification,
    _check_mentions,
)


# ---------------------------------------------------------------------------
# Realistic GitHub webhook payload fixtures
# ---------------------------------------------------------------------------

def _base_payload(**overrides):
    """Return a minimal GitHub webhook payload with sensible defaults."""
    payload = {
        "action": "opened",
        "sender": {"login": "octocat", "type": "User"},
        "repository": {
            "full_name": "ticketclosed-wontfix/hermes-agent",
            "name": "hermes-agent",
            "owner": {"login": "ticketclosed-wontfix"},
            "html_url": "https://github.com/ticketclosed-wontfix/hermes-agent",
        },
    }
    payload.update(overrides)
    return payload


def _pr_payload(action="opened", **overrides):
    """Return a realistic pull_request webhook payload."""
    pr = {
        "number": 42,
        "title": "Fix auth bug",
        "html_url": "https://github.com/ticketclosed-wontfix/hermes-agent/pull/42",
        "state": "open",
        "draft": False,
        "merged": False,
        "user": {"login": "contributor"},
        "base": {"ref": "main"},
        "head": {"ref": "fix-auth"},
        "labels": [{"name": "bug"}, {"name": "security"}],
        "body": "This PR fixes the auth bug.",
    }
    payload = _base_payload(action=action)
    payload["pull_request"] = pr
    payload.update(overrides)
    return payload


def _issue_payload(action="opened", **overrides):
    """Return a realistic issues webhook payload."""
    issue = {
        "number": 17,
        "title": "Crash on startup",
        "html_url": "https://github.com/ticketclosed-wontfix/hermes-agent/issues/17",
        "state": "open",
        "user": {"login": "bugreporter"},
        "labels": [{"name": "bug"}],
        "body": "App crashes on startup.",
    }
    payload = _base_payload(action=action)
    payload["issue"] = issue
    payload.update(overrides)
    return payload


def _issue_comment_payload(action="created", comment_body="Looks good", **overrides):
    """Return a realistic issue_comment webhook payload."""
    comment = {
        "id": 12345,
        "body": comment_body,
        "html_url": "https://github.com/ticketclosed-wontfix/hermes-agent/issues/17#issuecomment-12345",
        "user": {"login": "commenter"},
    }
    issue = {
        "number": 17,
        "title": "Crash on startup",
        "html_url": "https://github.com/ticketclosed-wontfix/hermes-agent/issues/17",
        "state": "open",
        "user": {"login": "bugreporter"},
        "labels": [],
        "body": "App crashes on startup.",
    }
    payload = _base_payload(action=action)
    payload["comment"] = comment
    payload["issue"] = issue
    payload.update(overrides)
    return payload


def _check_suite_payload(action="completed", conclusion="failure", **overrides):
    """Return a realistic check_suite webhook payload."""
    check_suite = {
        "status": "completed",
        "conclusion": conclusion,
        "head_sha": "abc123def456789012345678901234567890abcd",
        "head_branch": "fix-auth",
        "pull_requests": [
            {
                "number": 42,
                "html_url": "https://github.com/ticketclosed-wontfix/hermes-agent/pull/42",
            }
        ],
    }
    payload = _base_payload(action=action)
    payload["check_suite"] = check_suite
    payload.update(overrides)
    return payload


def _check_run_payload(action="completed", conclusion="failure", **overrides):
    """Return a realistic check_run webhook payload."""
    check_run = {
        "name": "pytest",
        "status": "completed",
        "conclusion": conclusion,
        "head_sha": "abc123def456789012345678901234567890abcd",
        "html_url": "https://github.com/ticketclosed-wontfix/hermes-agent/runs/999",
        "pull_requests": [
            {
                "number": 42,
                "html_url": "https://github.com/ticketclosed-wontfix/hermes-agent/pull/42",
            }
        ],
    }
    payload = _base_payload(action=action)
    payload["check_run"] = check_run
    payload.update(overrides)
    return payload


def _review_payload(action="submitted", **overrides):
    """Return a realistic pull_request_review webhook payload."""
    pr = {
        "number": 42,
        "title": "Fix auth bug",
        "html_url": "https://github.com/ticketclosed-wontfix/hermes-agent/pull/42",
        "state": "open",
    }
    review = {
        "id": 9876,
        "body": "Looks good to me",
        "state": "approved",
        "html_url": "https://github.com/ticketclosed-wontfix/hermes-agent/pull/42#pullrequestreview-9876",
        "user": {"login": "reviewer"},
    }
    payload = _base_payload(action=action)
    payload["pull_request"] = pr
    payload["review"] = review
    payload.update(overrides)
    return payload


# ===================================================================
# pull_request events
# ===================================================================


class TestPullRequestOpened:
    def test_pr_opened_notification(self):
        payload = _pr_payload(action="opened")
        result = build_github_notification("pull_request", payload)
        assert result is not None
        assert result["kind"] == "pr_opened"
        assert result["severity"] == "info"
        assert result["source"] == "github"
        assert result["repo"] == "ticketclosed-wontfix/hermes-agent"
        assert "PR #42 opened" in result["title"]
        assert "Fix auth bug" in result["title"]
        assert result["url"] == "https://github.com/ticketclosed-wontfix/hermes-agent/pull/42"
        assert result["metadata"]["pr_number"] == 42
        assert result["metadata"]["author"] == "contributor"
        assert "bug" in result["metadata"]["labels"]
        assert "security" in result["metadata"]["labels"]

    def test_pr_opened_body_contains_author(self):
        payload = _pr_payload(action="opened")
        result = build_github_notification("pull_request", payload)
        assert "contributor" in result["body"]


class TestPullRequestClosed:
    def test_pr_closed_not_merged(self):
        payload = _pr_payload(action="closed", merged=False)
        # GitHub sets merged=False on close-without-merge
        payload["pull_request"]["merged"] = False
        result = build_github_notification("pull_request", payload)
        assert result is not None
        assert result["kind"] == "pr_closed"
        assert result["severity"] == "info"
        assert "PR #42 closed" in result["title"]

    def test_pr_merged(self):
        payload = _pr_payload(action="closed")
        payload["pull_request"]["merged"] = True
        result = build_github_notification("pull_request", payload)
        assert result is not None
        assert result["kind"] == "pr_merged"
        assert "PR #42 merged" in result["title"]


class TestPullRequestReopened:
    def test_pr_reopened(self):
        payload = _pr_payload(action="reopened")
        result = build_github_notification("pull_request", payload)
        assert result is not None
        assert result["kind"] == "pr_reopened"
        assert "PR #42 reopened" in result["title"]


class TestPullRequestReadyForReview:
    def test_pr_ready_for_review(self):
        payload = _pr_payload(action="ready_for_review")
        result = build_github_notification("pull_request", payload)
        assert result is not None
        assert result["kind"] == "pr_ready_for_review"
        assert "ready for review" in result["title"]


class TestPullRequestEdited:
    def test_pr_edited_returns_none(self):
        payload = _pr_payload(action="edited")
        result = build_github_notification("pull_request", payload)
        assert result is None


class TestPullRequestSynchronize:
    def test_pr_synchronize_returns_none(self):
        payload = _pr_payload(action="synchronize")
        result = build_github_notification("pull_request", payload)
        assert result is None


class TestPullRequestDraftToggle:
    def test_pr_converted_to_draft_returns_none(self):
        payload = _pr_payload(action="converted_to_draft")
        result = build_github_notification("pull_request", payload)
        assert result is None


# ===================================================================
# pull_request_review events
# ===================================================================


class TestPullRequestReview:
    def test_review_submitted(self):
        payload = _review_payload(action="submitted")
        result = build_github_notification("pull_request_review", payload)
        assert result is not None
        assert result["kind"] == "pr_review"
        assert "PR #42 review" in result["title"]
        assert "approved" in result["title"]
        assert result["metadata"]["review_state"] == "approved"
        assert result["metadata"]["author"] == "reviewer"

    def test_review_dismissed_returns_none(self):
        payload = _review_payload(action="dismissed")
        result = build_github_notification("pull_request_review", payload)
        assert result is None


# ===================================================================
# issues events
# ===================================================================


class TestIssuesOpened:
    def test_issue_opened(self):
        payload = _issue_payload(action="opened")
        result = build_github_notification("issues", payload)
        assert result is not None
        assert result["kind"] == "issue_opened"
        assert result["severity"] == "info"
        assert "Issue #17 opened" in result["title"]
        assert "Crash on startup" in result["title"]
        assert result["url"] == "https://github.com/ticketclosed-wontfix/hermes-agent/issues/17"
        assert result["metadata"]["issue_number"] == 17


class TestIssuesClosed:
    def test_issue_closed(self):
        payload = _issue_payload(action="closed")
        result = build_github_notification("issues", payload)
        assert result is not None
        assert result["kind"] == "issue_closed"
        assert "Issue #17 closed" in result["title"]


class TestIssuesReopened:
    def test_issue_reopened(self):
        payload = _issue_payload(action="reopened")
        result = build_github_notification("issues", payload)
        assert result is not None
        assert result["kind"] == "issue_reopened"


class TestIssuesEdited:
    def test_issue_edited_returns_none(self):
        payload = _issue_payload(action="edited")
        result = build_github_notification("issues", payload)
        assert result is None


class TestIssuesLabeled:
    def test_issue_labeled_returns_none(self):
        payload = _issue_payload(action="labeled")
        result = build_github_notification("issues", payload)
        assert result is None


# ===================================================================
# issue_comment events
# ===================================================================


class TestIssueCommentCreated:
    def test_comment_with_commit_mcgitface_mention(self):
        payload = _issue_comment_payload(comment_body="Hey @commit-mcgitface can you check this?")
        result = build_github_notification("issue_comment", payload)
        assert result is not None
        assert result["kind"] == "mention"
        assert result["severity"] == "warning"
        assert "Mention" in result["title"]
        assert "#17" in result["title"]

    def test_comment_with_ticketclosed_wontfix_mention(self):
        payload = _issue_comment_payload(comment_body="@ticketclosed-wontfix please review")
        result = build_github_notification("issue_comment", payload)
        assert result is not None
        assert result["kind"] == "mention"
        assert result["severity"] == "warning"

    def test_comment_without_mention_returns_none(self):
        payload = _issue_comment_payload(comment_body="Just a regular comment")
        result = build_github_notification("issue_comment", payload)
        assert result is None

    def test_comment_edited_returns_none(self):
        payload = _issue_comment_payload(action="edited")
        result = build_github_notification("issue_comment", payload)
        assert result is None

    def test_comment_deleted_returns_none(self):
        payload = _issue_comment_payload(action="deleted")
        result = build_github_notification("issue_comment", payload)
        assert result is None

    def test_comment_on_pr_has_is_pr_metadata(self):
        """issue_comment on a PR should have is_pr=True in metadata."""
        payload = _issue_comment_payload(comment_body="@commit-mcgitface review please")
        # Add pull_request key to the issue dict to signal it's on a PR
        payload["issue"]["pull_request"] = {"url": "https://api.github.com/repos/owner/repo/pulls/42"}
        result = build_github_notification("issue_comment", payload)
        assert result is not None
        assert result["metadata"]["is_pr"] is True

    def test_mention_case_insensitive(self):
        payload = _issue_comment_payload(comment_body="@Commit-McGitface review please")
        result = build_github_notification("issue_comment", payload)
        assert result is not None
        assert result["kind"] == "mention"


# ===================================================================
# check_suite events
# ===================================================================


class TestCheckSuite:
    def test_check_suite_completed_failure(self):
        payload = _check_suite_payload(conclusion="failure")
        result = build_github_notification("check_suite", payload)
        assert result is not None
        assert result["kind"] == "ci_failure"
        assert result["severity"] == "error"
        assert "CI failure" in result["title"]
        assert result["metadata"]["ci_state"] == "failure"

    def test_check_suite_completed_success_returns_none(self):
        payload = _check_suite_payload(conclusion="success")
        result = build_github_notification("check_suite", payload)
        assert result is None

    def test_check_suite_in_progress_returns_none(self):
        payload = _check_suite_payload(action="in_progress")
        result = build_github_notification("check_suite", payload)
        assert result is None

    def test_check_suite_no_pr_returns_none(self):
        """check_suite with no associated PRs should return None."""
        payload = _check_suite_payload(conclusion="failure")
        payload["check_suite"]["pull_requests"] = []
        result = build_github_notification("check_suite", payload)
        assert result is None

    def test_check_suite_url_points_to_pr(self):
        payload = _check_suite_payload(conclusion="failure")
        result = build_github_notification("check_suite", payload)
        assert result is not None
        assert result["url"] == "https://github.com/ticketclosed-wontfix/hermes-agent/pull/42"


# ===================================================================
# check_run events
# ===================================================================


class TestCheckRun:
    def test_check_run_completed_failure_on_pr(self):
        payload = _check_run_payload(conclusion="failure")
        result = build_github_notification("check_run", payload)
        assert result is not None
        assert result["kind"] == "ci_failure"
        assert result["severity"] == "error"
        assert "CI check failed" in result["title"]
        assert "pytest" in result["title"]
        assert result["metadata"]["check_name"] == "pytest"

    def test_check_run_completed_success_returns_none(self):
        payload = _check_run_payload(conclusion="success")
        result = build_github_notification("check_run", payload)
        assert result is None

    def test_check_run_no_pr_returns_none(self):
        """check_run with no associated PRs should return None."""
        payload = _check_run_payload(conclusion="failure")
        payload["check_run"]["pull_requests"] = []
        result = build_github_notification("check_run", payload)
        assert result is None

    def test_check_run_in_progress_returns_none(self):
        payload = _check_run_payload(action="in_progress")
        result = build_github_notification("check_run", payload)
        assert result is None


# ===================================================================
# Unknown / unsupported event types
# ===================================================================


class TestUnsupportedEvents:
    def test_push_event_returns_none(self):
        payload = _base_payload()
        result = build_github_notification("push", payload)
        assert result is None

    def test_fork_event_returns_none(self):
        payload = _base_payload()
        result = build_github_notification("fork", payload)
        assert result is None

    def test_watch_event_returns_none(self):
        payload = _base_payload()
        result = build_github_notification("watch", payload)
        assert result is None

    def test_empty_payload_returns_none(self):
        result = build_github_notification("pull_request", {})
        assert result is None

    def test_none_payload_returns_none(self):
        result = build_github_notification("pull_request", None)
        assert result is None


# ===================================================================
# URL field correctness
# ===================================================================


class TestURLFields:
    def test_pr_url(self):
        payload = _pr_payload(action="opened")
        result = build_github_notification("pull_request", payload)
        assert result["url"].startswith("https://github.com/")
        assert "hermes-agent/pull/42" in result["url"]

    def test_issue_url(self):
        payload = _issue_payload(action="opened")
        result = build_github_notification("issues", payload)
        assert result["url"].startswith("https://github.com/")
        assert "hermes-agent/issues/17" in result["url"]

    def test_comment_url(self):
        payload = _issue_comment_payload(comment_body="@commit-mcgitface check this")
        result = build_github_notification("issue_comment", payload)
        assert result["url"].startswith("https://github.com/")
        assert "issuecomment" in result["url"]

    def test_check_run_url(self):
        payload = _check_run_payload(conclusion="failure")
        result = build_github_notification("check_run", payload)
        assert result["url"] is not None
        assert result["url"].startswith("https://github.com/") or "/runs/" in result["url"]


# ===================================================================
# Helper function tests
# ===================================================================


class TestCheckMentions:
    def test_commit_mcgitface_mention(self):
        assert _check_mentions("@commit-mcgitface please help") is True

    def test_ticketclosed_wontfix_mention(self):
        assert _check_mentions("@ticketclosed-wontfix review please") is True

    def test_case_insensitive(self):
        assert _check_mentions("@Commit-McGitface") is True
        assert _check_mentions("@TicketClosed-WontFix") is True

    def test_no_mention(self):
        assert _check_mentions("Just a regular comment") is False

    def test_empty_string(self):
        assert _check_mentions("") is False

    def test_none_input(self):
        assert _check_mentions(None) is False

    def test_non_string_input(self):
        assert _check_mentions(123) is False


# ===================================================================
# created_at field
# ===================================================================


class TestCreatedAt:
    def test_notification_has_created_at(self):
        payload = _pr_payload(action="opened")
        result = build_github_notification("pull_request", payload)
        assert "created_at" in result
        assert "T" in result["created_at"]  # ISO 8601 has T separator