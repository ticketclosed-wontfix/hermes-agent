"""Build structured notifications from GitHub webhook event payloads.

This module extracts notification fields directly from GitHub webhook
payloads (no LLM involvement). The returned dict is suitable for POSTing
to the Hermes Console ingest endpoint.

Supported event types and actions:
  - pull_request: opened, closed, merged, reopened, ready_for_review
  - pull_request_review: submitted
  - issues: opened, closed, reopened
  - issue_comment: created (only when mentioning specific users/bots)
  - check_suite: completed (failure only)
  - check_run: completed (failure on PRs only)

Events that shouldn't produce notifications (edits, draft toggles,
synchronize, etc.) return None.

Environment variables (read by the webhook adapter, not this module):
  - HERMES_CONSOLE_INGEST_SECRET: shared secret for HMAC auth
  - HERMES_CONSOLE_INGEST_URL: target URL (default http://127.0.0.1:3001/api/notifications/ingest)
"""

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Mentions that trigger a notification from issue_comment
_MENTION_PATTERNS = [
    re.compile(r"@\bticketclosed-wontfix\b", re.IGNORECASE),
    re.compile(r"@\bcommit-mcgitface\b", re.IGNORECASE),
]

# pull_request actions that produce notifications
_PR_NOTIFY_ACTIONS = {"opened", "closed", "reopened", "ready_for_review"}

# issues actions that produce notifications
_ISSUE_NOTIFY_ACTIONS = {"opened", "closed", "reopened"}


def _get_repo(payload: dict) -> str:
    """Extract 'owner/repo' from the payload."""
    repo = payload.get("repository", {})
    full_name = repo.get("full_name", "")
    if full_name:
        return full_name
    owner = repo.get("owner", {})
    login = owner.get("login", "")
    name = repo.get("name", "")
    if login and name:
        return f"{login}/{name}"
    return ""


def _get_sender(payload: dict) -> str:
    """Extract the sender login."""
    sender = payload.get("sender", {})
    return sender.get("login", "unknown")


def _get_labels(payload: dict, key: str = "pull_request") -> List[str]:
    """Extract label names from a payload sub-object."""
    obj = payload.get(key, {})
    labels = obj.get("labels", [])
    if not isinstance(labels, list):
        return []
    return [lbl.get("name", "") for lbl in labels if isinstance(lbl, dict) and lbl.get("name")]


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _base_notification(
    source: str,
    repo: str,
    kind: str,
    title: str,
    body: str,
    url: str,
    severity: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the common notification dict structure."""
    return {
        "source": source,
        "repo": repo,
        "kind": kind,
        "title": title,
        "body": body,
        "url": url,
        "severity": severity,
        "metadata": metadata,
        "created_at": _now_iso(),
    }


def _is_merged(payload: dict) -> bool:
    """Check if a pull_request.closed event is actually a merge."""
    pr = payload.get("pull_request", {})
    # GitHub sets merged=True on the PR object when the action is closed
    # and it was merged (as opposed to closed without merging).
    return bool(pr.get("merged", False))


def _check_mentions(text: str) -> bool:
    """Return True if text contains a monitored @mention."""
    if not text or not isinstance(text, str):
        return False
    for pattern in _MENTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _build_pr_notification(event_type: str, payload: dict) -> Optional[dict]:
    """Handle pull_request events."""
    action = payload.get("action", "")
    if action not in _PR_NOTIFY_ACTIONS:
        return None

    pr = payload.get("pull_request", {})
    if not pr:
        return None

    pr_number = pr.get("number", "")
    pr_title = pr.get("title", "")
    author = pr.get("user", {}).get("login", "unknown")
    html_url = pr.get("html_url", "")
    repo = _get_repo(payload)
    labels = _get_labels(payload, "pull_request")
    is_draft = pr.get("draft", False)
    base_ref = pr.get("base", {}).get("ref", "")
    head_ref = pr.get("head", {}).get("ref", "")

    # Determine kind and severity
    if action == "opened":
        kind = "pr_opened"
        severity = "info"
        verb = "opened"
    elif action == "closed":
        if _is_merged(payload):
            kind = "pr_merged"
            severity = "info"
            verb = "merged"
        else:
            kind = "pr_closed"
            severity = "info"
            verb = "closed"
    elif action == "reopened":
        kind = "pr_reopened"
        severity = "info"
        verb = "reopened"
    elif action == "ready_for_review":
        kind = "pr_ready_for_review"
        severity = "info"
        verb = "marked ready for review"
    else:
        return None

    title = f"PR #{pr_number} {verb}: {pr_title}"
    body_lines = [f"Author: {author}"]
    if is_draft:
        body_lines.append("Draft PR")
    if labels:
        body_lines.append(f"Labels: {', '.join(labels)}")
    if head_ref and base_ref:
        body_lines.append(f"{head_ref} -> {base_ref}")
    body = "\n".join(body_lines)

    metadata = {
        "pr_number": pr_number,
        "author": author,
        "labels": labels,
        "action": action,
    }
    if is_draft:
        metadata["draft"] = True
    if head_ref:
        metadata["head_branch"] = head_ref
    if base_ref:
        metadata["base_branch"] = base_ref

    return _base_notification(
        source="github",
        repo=repo,
        kind=kind,
        title=title,
        body=body,
        url=html_url,
        severity=severity,
        metadata=metadata,
    )


def _build_pr_review_notification(event_type: str, payload: dict) -> Optional[dict]:
    """Handle pull_request_review events."""
    action = payload.get("action", "")
    if action != "submitted":
        return None

    pr = payload.get("pull_request", {})
    review = payload.get("review", {})
    if not pr or not review:
        return None

    pr_number = pr.get("number", "")
    pr_title = pr.get("title", "")
    author = review.get("user", {}).get("login", "unknown")
    html_url = review.get("html_url", "") or pr.get("html_url", "")
    repo = _get_repo(payload)
    state = review.get("state", "")
    body_text = review.get("body", "") or ""

    title = f"PR #{pr_number} review: {state}"
    body_lines = [f"Reviewer: {author}", f"State: {state}"]
    if body_text:
        snippet = body_text[:200]
        body_lines.append(f"Comment: {snippet}")
    body = "\n".join(body_lines)

    metadata = {
        "pr_number": pr_number,
        "author": author,
        "review_state": state,
        "action": action,
    }

    return _base_notification(
        source="github",
        repo=repo,
        kind="pr_review",
        title=title,
        body=body,
        url=html_url,
        severity="info",
        metadata=metadata,
    )


def _build_issues_notification(event_type: str, payload: dict) -> Optional[dict]:
    """Handle issues events."""
    action = payload.get("action", "")
    if action not in _ISSUE_NOTIFY_ACTIONS:
        return None

    issue = payload.get("issue", {})
    if not issue:
        return None

    issue_number = issue.get("number", "")
    issue_title = issue.get("title", "")
    author = issue.get("user", {}).get("login", "unknown")
    html_url = issue.get("html_url", "")
    repo = _get_repo(payload)
    labels = _get_labels(payload, "issue")

    kind = f"issue_{action}"
    verb = action
    severity = "info"

    title = f"Issue #{issue_number} {verb}: {issue_title}"
    body_lines = [f"Author: {author}"]
    if labels:
        body_lines.append(f"Labels: {', '.join(labels)}")
    body = "\n".join(body_lines)

    metadata = {
        "issue_number": issue_number,
        "author": author,
        "labels": labels,
        "action": action,
    }

    return _base_notification(
        source="github",
        repo=repo,
        kind=kind,
        title=title,
        body=body,
        url=html_url,
        severity=severity,
        metadata=metadata,
    )


def _build_issue_comment_notification(event_type: str, payload: dict) -> Optional[dict]:
    """Handle issue_comment events — only when a monitored @mention is present."""
    action = payload.get("action", "")
    if action != "created":
        return None

    comment = payload.get("comment", {})
    if not comment:
        return None

    comment_body = comment.get("body", "") or ""
    if not _check_mentions(comment_body):
        return None

    # Determine if it's on an issue or PR
    issue = payload.get("issue", {})
    if not issue:
        return None

    # Check if the issue is actually a PR (GitHub uses issue_comment for PRs too)
    is_on_pr = "pull_request" in issue

    issue_number = issue.get("number", "")
    issue_title = issue.get("title", "")
    comment_author = comment.get("user", {}).get("login", "unknown")
    comment_url = comment.get("html_url", "")
    repo = _get_repo(payload)

    # Find which mention triggered it
    mention_found = ""
    for pattern in _MENTION_PATTERNS:
        m = pattern.search(comment_body)
        if m:
            mention_found = m.group(0)
            break

    context = "PR" if is_on_pr else "issue"
    title = f"Mention in {context} #{issue_number}: {issue_title}"
    snippet = comment_body[:200].replace("\n", " ")
    body = f"Comment by {comment_author}\n{snippet}"

    metadata = {
        "issue_number": issue_number,
        "author": comment_author,
        "is_pr": is_on_pr,
        "mention": mention_found,
        "action": action,
    }

    return _base_notification(
        source="github",
        repo=repo,
        kind="mention",
        title=title,
        body=body,
        url=comment_url,
        severity="warning",
        metadata=metadata,
    )


def _build_check_suite_notification(event_type: str, payload: dict) -> Optional[dict]:
    """Handle check_suite events — only completed with failure."""
    action = payload.get("action", "")
    if action != "completed":
        return None

    check_suite = payload.get("check_suite", {})
    if not check_suite:
        return None

    conclusion = check_suite.get("conclusion", "")
    if conclusion != "failure":
        return None  # success is too noisy for user notifications

    prs = check_suite.get("pull_requests", [])
    # Only notify if associated with PRs
    if not prs:
        return None

    repo = _get_repo(payload)
    head_sha = check_suite.get("head_sha", "")[:12]
    head_branch = check_suite.get("head_branch", "")

    # Use the first PR's URL if available
    first_pr = prs[0] if prs else {}
    pr_number = first_pr.get("number", "")
    pr_url = first_pr.get("html_url", "") or ""

    title = f"CI failure on {repo}"
    if pr_number:
        title = f"CI failure on PR #{pr_number}"
    body_lines = [f"Conclusion: {conclusion}"]
    if head_sha:
        body_lines.append(f"Head SHA: {head_sha}...")
    if head_branch:
        body_lines.append(f"Branch: {head_branch}")
    body = "\n".join(body_lines)

    metadata = {
        "ci_state": conclusion,
        "head_sha": check_suite.get("head_sha", ""),
        "head_branch": head_branch,
        "action": action,
    }
    if pr_number:
        metadata["pr_number"] = pr_number

    # Fallback URL: link to the repo commits if no PR URL
    url = pr_url
    if not url and repo:
        url = f"https://github.com/{repo}/commit/{check_suite.get('head_sha', '')}"

    return _base_notification(
        source="github",
        repo=repo,
        kind="ci_failure",
        title=title,
        body=body,
        url=url,
        severity="error",
        metadata=metadata,
    )


def _build_check_run_notification(event_type: str, payload: dict) -> Optional[dict]:
    """Handle check_run events — only completed with failure on PRs."""
    action = payload.get("action", "")
    if action != "completed":
        return None

    check_run = payload.get("check_run", {})
    if not check_run:
        return None

    conclusion = check_run.get("conclusion", "")
    if conclusion != "failure":
        return None

    # Only notify if associated with PRs
    prs = check_run.get("pull_requests", [])
    if not prs:
        return None

    repo = _get_repo(payload)
    check_name = check_run.get("name", "unknown check")
    head_sha = check_run.get("head_sha", "")[:12]

    first_pr = prs[0] if prs else {}
    pr_number = first_pr.get("number", "")
    pr_url = first_pr.get("html_url", "") or ""

    title = f"CI check failed: {check_name}"
    if pr_number:
        title = f"CI check failed on PR #{pr_number}: {check_name}"
    body_lines = [f"Check: {check_name}", f"Conclusion: {conclusion}"]
    if head_sha:
        body_lines.append(f"Head SHA: {head_sha}...")
    body = "\n".join(body_lines)

    metadata = {
        "ci_state": conclusion,
        "check_name": check_name,
        "head_sha": check_run.get("head_sha", ""),
        "action": action,
    }
    if pr_number:
        metadata["pr_number"] = pr_number

    url = check_run.get("html_url", "") or pr_url
    if not url and repo:
        url = f"https://github.com/{repo}/commit/{check_run.get('head_sha', '')}"

    return _base_notification(
        source="github",
        repo=repo,
        kind="ci_failure",
        title=title,
        body=body,
        url=url,
        severity="error",
        metadata=metadata,
    )


def build_github_notification(event_type: str, payload: dict) -> Optional[dict]:
    """Build a notification dict from a GitHub webhook event payload.

    Args:
        event_type: The X-GitHub-Event header value (e.g. 'pull_request',
            'issues', 'check_suite').
        payload: The parsed JSON webhook payload.

    Returns:
        A notification dict with keys: source, repo, kind, title, body, url,
        severity, metadata, created_at.  Returns None for events that
        shouldn't produce a notification (edits, draft toggles, non-failure
        CI, comments without mentions, etc.).
    """
    if not isinstance(payload, dict) or not payload:
        return None

    builders = {
        "pull_request": _build_pr_notification,
        "pull_request_review": _build_pr_review_notification,
        "issues": _build_issues_notification,
        "issue_comment": _build_issue_comment_notification,
        "check_suite": _build_check_suite_notification,
        "check_run": _build_check_run_notification,
    }

    builder = builders.get(event_type)
    if builder is None:
        return None

    try:
        return builder(event_type, payload)
    except Exception:
        # Never let notification extraction break the webhook pipeline.
        # Log at the call site if needed; return None here.
        return None