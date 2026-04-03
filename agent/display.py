"""CLI presentation -- spinner, kawaii faces, tool preview formatting.

Pure display functions and classes with no AIAgent dependency.
Used by AIAgent._execute_tool_calls for CLI feedback.
"""

import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from difflib import unified_diff
from pathlib import Path

# ANSI escape codes for coloring tool failure indicators
_RED = "\033[31m"
_RESET = "\033[0m"

logger = logging.getLogger(__name__)

_ANSI_RESET = "\033[0m"
_ANSI_DIM = "\033[38;2;150;150;150m"
_ANSI_FILE = "\033[38;2;180;160;255m"
_ANSI_HUNK = "\033[38;2;120;120;140m"
_ANSI_MINUS = "\033[38;2;255;255;255;48;2;120;20;20m"
_ANSI_PLUS = "\033[38;2;255;255;255;48;2;20;90;20m"
_MAX_INLINE_DIFF_FILES = 6
_MAX_INLINE_DIFF_LINES = 80


@dataclass
class LocalEditSnapshot:
    """Pre-tool filesystem snapshot used to render diffs locally after writes."""
    paths: list[Path] = field(default_factory=list)
    before: dict[str, str | None] = field(default_factory=dict)

# =========================================================================
# Configurable tool preview length (0 = no limit)
# Set once at startup by CLI or gateway from display.tool_preview_length config.
# =========================================================================
_tool_preview_max_len: int = 0  # 0 = unlimited


def set_tool_preview_max_len(n: int) -> None:
    """Set the global max length for tool call previews. 0 = no limit."""
    global _tool_preview_max_len
    _tool_preview_max_len = max(int(n), 0) if n else 0


def get_tool_preview_max_len() -> int:
    """Return the configured max preview length (0 = unlimited)."""
    return _tool_preview_max_len


# =========================================================================
# Skin-aware helpers (lazy import to avoid circular deps)
# =========================================================================

def _get_skin():
    """Get the active skin config, or None if not available."""
    try:
        from hermes_cli.skin_engine import get_active_skin
        return get_active_skin()
    except Exception:
        return None


def get_skin_faces(key: str, default: list) -> list:
    """Get spinner face list from active skin, falling back to default."""
    skin = _get_skin()
    if skin:
        faces = skin.get_spinner_list(key)
        if faces:
            return faces
    return default


def get_skin_verbs() -> list:
    """Get thinking verbs from active skin."""
    skin = _get_skin()
    if skin:
        verbs = skin.get_spinner_list("thinking_verbs")
        if verbs:
            return verbs
    return KawaiiSpinner.THINKING_VERBS


def get_skin_tool_prefix() -> str:
    """Get tool output prefix character from active skin."""
    skin = _get_skin()
    if skin:
        return skin.tool_prefix
    return "┊"


def get_tool_emoji(tool_name: str, default: str = "⚡") -> str:
    """Get the display emoji for a tool.

    Resolution order:
    1. Active skin's ``tool_emojis`` overrides (if a skin is loaded)
    2. Tool registry's per-tool ``emoji`` field
    3. *default* fallback
    """
    # 1. Skin override
    skin = _get_skin()
    if skin and skin.tool_emojis:
        override = skin.tool_emojis.get(tool_name)
        if override:
            return override
    # 2. Registry default
    try:
        from tools.registry import registry
        emoji = registry.get_emoji(tool_name, default="")
        if emoji:
            return emoji
    except Exception:
        pass
    # 3. Hardcoded fallback
    return default


# =========================================================================
# Tool preview (one-line summary of a tool call's primary argument)
# =========================================================================

def _oneline(text: str) -> str:
    """Collapse whitespace (including newlines) to single spaces."""
    return " ".join(text.split())


def build_tool_preview(tool_name: str, args: dict, max_len: int | None = None) -> str | None:
    """Build a short preview of a tool call's primary argument for display.

    *max_len* controls truncation.  ``None`` (default) defers to the global
    ``_tool_preview_max_len`` set via config; ``0`` means unlimited.
    """
    if max_len is None:
        max_len = _tool_preview_max_len
    if not args:
        return None
    primary_args = {
        "terminal": "command", "web_search": "query", "web_extract": "urls",
        "read_file": "path", "write_file": "path", "patch": "path",
        "search_files": "pattern", "browser_navigate": "url",
        "browser_click": "ref", "browser_type": "text",
        "image_generate": "prompt", "text_to_speech": "text",
        "vision_analyze": "question", "mixture_of_agents": "user_prompt",
        "skill_view": "name", "skills_list": "category",
        "cronjob": "action",
        "execute_code": "code", "delegate_task": "goal",
        "clarify": "question", "skill_manage": "name",
    }

    if tool_name == "process":
        action = args.get("action", "")
        sid = args.get("session_id", "")
        data = args.get("data", "")
        timeout_val = args.get("timeout")
        parts = [action]
        if sid:
            parts.append(sid[:16])
        if data:
            parts.append(f'"{_oneline(data[:20])}"')
        if timeout_val and action == "wait":
            parts.append(f"{timeout_val}s")
        return " ".join(parts) if parts else None

    if tool_name == "todo":
        todos_arg = args.get("todos")
        merge = args.get("merge", False)
        if todos_arg is None:
            return "reading task list"
        elif merge:
            return f"updating {len(todos_arg)} task(s)"
        else:
            return f"planning {len(todos_arg)} task(s)"

    if tool_name == "session_search":
        query = _oneline(args.get("query", ""))
        return f"recall: \"{query[:25]}{'...' if len(query) > 25 else ''}\""

    if tool_name == "memory":
        action = args.get("action", "")
        target = args.get("target", "")
        if action == "add":
            content = _oneline(args.get("content", ""))
            return f"+{target}: \"{content[:25]}{'...' if len(content) > 25 else ''}\""
        elif action == "replace":
            return f"~{target}: \"{_oneline(args.get('old_text', '')[:20])}\""
        elif action == "remove":
            return f"-{target}: \"{_oneline(args.get('old_text', '')[:20])}\""
        return action

    if tool_name == "send_message":
        target = args.get("target", "?")
        msg = _oneline(args.get("message", ""))
        if len(msg) > 20:
            msg = msg[:17] + "..."
        return f"to {target}: \"{msg}\""

    if tool_name.startswith("rl_"):
        rl_previews = {
            "rl_list_environments": "listing envs",
            "rl_select_environment": args.get("name", ""),
            "rl_get_current_config": "reading config",
            "rl_edit_config": f"{args.get('field', '')}={args.get('value', '')}",
            "rl_start_training": "starting",
            "rl_check_status": args.get("run_id", "")[:16],
            "rl_stop_training": f"stopping {args.get('run_id', '')[:16]}",
            "rl_get_results": args.get("run_id", "")[:16],
            "rl_list_runs": "listing runs",
            "rl_test_inference": f"{args.get('num_steps', 3)} steps",
        }
        return rl_previews.get(tool_name)

    key = primary_args.get(tool_name)
    if not key:
        for fallback_key in ("query", "text", "command", "path", "name", "prompt", "code", "goal"):
            if fallback_key in args:
                key = fallback_key
                break

    if not key or key not in args:
        return None

    value = args[key]
    if isinstance(value, list):
        value = value[0] if value else ""

    preview = _oneline(str(value))
    if not preview:
        return None
    if max_len > 0 and len(preview) > max_len:
        preview = preview[:max_len - 3] + "..."
    return preview


# =========================================================================
# Inline diff previews for write actions
# =========================================================================

def _resolved_path(path: str) -> Path:
    """Resolve a possibly-relative filesystem path against the current cwd."""
    candidate = Path(os.path.expanduser(path))
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def _snapshot_text(path: Path) -> str | None:
    """Return UTF-8 file content, or None for missing/unreadable files."""
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, IsADirectoryError, UnicodeDecodeError, OSError):
        return None


def _display_diff_path(path: Path) -> str:
    """Prefer cwd-relative paths in diffs when available."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


def _resolve_skill_manage_paths(args: dict) -> list[Path]:
    """Resolve skill_manage write targets to filesystem paths."""
    action = args.get("action")
    name = args.get("name")
    if not action or not name:
        return []

    from tools.skill_manager_tool import _find_skill, _resolve_skill_dir

    if action == "create":
        skill_dir = _resolve_skill_dir(name, args.get("category"))
        return [skill_dir / "SKILL.md"]

    existing = _find_skill(name)
    if not existing:
        return []

    skill_dir = Path(existing["path"])
    if action in {"edit", "patch"}:
        file_path = args.get("file_path")
        return [skill_dir / file_path] if file_path else [skill_dir / "SKILL.md"]
    if action in {"write_file", "remove_file"}:
        file_path = args.get("file_path")
        return [skill_dir / file_path] if file_path else []
    if action == "delete":
        files = [path for path in sorted(skill_dir.rglob("*")) if path.is_file()]
        return files
    return []


def _resolve_local_edit_paths(tool_name: str, function_args: dict | None) -> list[Path]:
    """Resolve local filesystem targets for write-capable tools."""
    if not isinstance(function_args, dict):
        return []

    if tool_name == "write_file":
        path = function_args.get("path")
        return [_resolved_path(path)] if path else []

    if tool_name == "patch":
        path = function_args.get("path")
        return [_resolved_path(path)] if path else []

    if tool_name == "skill_manage":
        return _resolve_skill_manage_paths(function_args)

    return []


def capture_local_edit_snapshot(tool_name: str, function_args: dict | None) -> LocalEditSnapshot | None:
    """Capture before-state for local write previews."""
    paths = _resolve_local_edit_paths(tool_name, function_args)
    if not paths:
        return None

    snapshot = LocalEditSnapshot(paths=paths)
    for path in paths:
        snapshot.before[str(path)] = _snapshot_text(path)
    return snapshot


def _result_succeeded(result: str | None) -> bool:
    """Conservatively detect whether a tool result represents success."""
    if not result:
        return False
    try:
        data = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(data, dict):
        return False
    if data.get("error"):
        return False
    if "success" in data:
        return bool(data.get("success"))
    return True


def _diff_from_snapshot(snapshot: LocalEditSnapshot | None) -> str | None:
    """Generate unified diff text from a stored before-state and current files."""
    if not snapshot:
        return None

    chunks: list[str] = []
    for path in snapshot.paths:
        before = snapshot.before.get(str(path))
        after = _snapshot_text(path)
        if before == after:
            continue

        display_path = _display_diff_path(path)
        diff = "".join(
            unified_diff(
                [] if before is None else before.splitlines(keepends=True),
                [] if after is None else after.splitlines(keepends=True),
                fromfile=f"a/{display_path}",
                tofile=f"b/{display_path}",
            )
        )
        if diff:
            chunks.append(diff)

    if not chunks:
        return None
    return "".join(chunk if chunk.endswith("\n") else chunk + "\n" for chunk in chunks)


def extract_edit_diff(
    tool_name: str,
    result: str | None,
    *,
    function_args: dict | None = None,
    snapshot: LocalEditSnapshot | None = None,
) -> str | None:
    """Extract a unified diff from a file-edit tool result."""
    if tool_name == "patch" and result:
        try:
            data = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            data = None
        if isinstance(data, dict):
            diff = data.get("diff")
            if isinstance(diff, str) and diff.strip():
                return diff

    if tool_name not in {"write_file", "patch", "skill_manage"}:
        return None
    if not _result_succeeded(result):
        return None
    return _diff_from_snapshot(snapshot)


def _emit_inline_diff(diff_text: str, print_fn) -> bool:
    """Emit rendered diff text through the CLI's prompt_toolkit-safe printer."""
    if print_fn is None or not diff_text:
        return False
    try:
        print_fn("  ┊ review diff")
        for line in diff_text.rstrip("\n").splitlines():
            print_fn(line)
        return True
    except Exception:
        return False


def _render_inline_unified_diff(diff: str) -> list[str]:
    """Render unified diff lines in Hermes' inline transcript style."""
    rendered: list[str] = []
    from_file = None
    to_file = None

    for raw_line in diff.splitlines():
        if raw_line.startswith("--- "):
            from_file = raw_line[4:].strip()
            continue
        if raw_line.startswith("+++ "):
            to_file = raw_line[4:].strip()
            if from_file or to_file:
                rendered.append(f"{_ANSI_FILE}{from_file or 'a/?'} → {to_file or 'b/?'}{_ANSI_RESET}")
            continue
        if raw_line.startswith("@@"):
            rendered.append(f"{_ANSI_HUNK}{raw_line}{_ANSI_RESET}")
            continue
        if raw_line.startswith("-"):
            rendered.append(f"{_ANSI_MINUS}{raw_line}{_ANSI_RESET}")
            continue
        if raw_line.startswith("+"):
            rendered.append(f"{_ANSI_PLUS}{raw_line}{_ANSI_RESET}")
            continue
        if raw_line.startswith(" "):
            rendered.append(f"{_ANSI_DIM}{raw_line}{_ANSI_RESET}")
            continue
        if raw_line:
            rendered.append(raw_line)

    return rendered


def _split_unified_diff_sections(diff: str) -> list[str]:
    """Split a unified diff into per-file sections."""
    sections: list[list[str]] = []
    current: list[str] = []

    for line in diff.splitlines():
        if line.startswith("--- ") and current:
            sections.append(current)
            current = [line]
            continue
        current.append(line)

    if current:
        sections.append(current)

    return ["\n".join(section) for section in sections if section]


def _summarize_rendered_diff_sections(
    diff: str,
    *,
    max_files: int = _MAX_INLINE_DIFF_FILES,
    max_lines: int = _MAX_INLINE_DIFF_LINES,
) -> list[str]:
    """Render diff sections while capping file count and total line count."""
    sections = _split_unified_diff_sections(diff)
    rendered: list[str] = []
    omitted_files = 0
    omitted_lines = 0

    for idx, section in enumerate(sections):
        if idx >= max_files:
            omitted_files += 1
            omitted_lines += len(_render_inline_unified_diff(section))
            continue

        section_lines = _render_inline_unified_diff(section)
        remaining_budget = max_lines - len(rendered)
        if remaining_budget <= 0:
            omitted_lines += len(section_lines)
            omitted_files += 1
            continue

        if len(section_lines) <= remaining_budget:
            rendered.extend(section_lines)
            continue

        rendered.extend(section_lines[:remaining_budget])
        omitted_lines += len(section_lines) - remaining_budget
        omitted_files += 1 + max(0, len(sections) - idx - 1)
        for leftover in sections[idx + 1:]:
            omitted_lines += len(_render_inline_unified_diff(leftover))
        break

    if omitted_files or omitted_lines:
        summary = f"… omitted {omitted_lines} diff line(s)"
        if omitted_files:
            summary += f" across {omitted_files} additional file(s)/section(s)"
        rendered.append(f"{_ANSI_HUNK}{summary}{_ANSI_RESET}")

    return rendered


def render_edit_diff_with_delta(
    tool_name: str,
    result: str | None,
    *,
    function_args: dict | None = None,
    snapshot: LocalEditSnapshot | None = None,
    print_fn=None,
) -> bool:
    """Render an edit diff inline without taking over the terminal UI."""
    diff = extract_edit_diff(
        tool_name,
        result,
        function_args=function_args,
        snapshot=snapshot,
    )
    if not diff:
        return False
    try:
        rendered_lines = _summarize_rendered_diff_sections(diff)
    except Exception as exc:
        logger.debug("Could not render inline diff: %s", exc)
        return False
    return _emit_inline_diff("\n".join(rendered_lines), print_fn)


# =========================================================================
# KawaiiSpinner
# =========================================================================

class KawaiiSpinner:
    """Animated spinner with kawaii faces for CLI feedback during tool execution."""

    SPINNERS = {
        'dots': ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'],
        'bounce': ['⠁', '⠂', '⠄', '⡀', '⢀', '⠠', '⠐', '⠈'],
        'grow': ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█', '▇', '▆', '▅', '▄', '▃', '▂'],
        'arrows': ['←', '↖', '↑', '↗', '→', '↘', '↓', '↙'],
        'star': ['✶', '✷', '✸', '✹', '✺', '✹', '✸', '✷'],
        'moon': ['🌑', '🌒', '🌓', '🌔', '🌕', '🌖', '🌗', '🌘'],
        'pulse': ['◜', '◠', '◝', '◞', '◡', '◟'],
        'brain': ['🧠', '💭', '💡', '✨', '💫', '🌟', '💡', '💭'],
        'sparkle': ['⁺', '˚', '*', '✧', '✦', '✧', '*', '˚'],
    }

    KAWAII_WAITING = [
        "(｡◕‿◕｡)", "(◕‿◕✿)", "٩(◕‿◕｡)۶", "(✿◠‿◠)", "( ˘▽˘)っ",
        "♪(´ε` )", "(◕ᴗ◕✿)", "ヾ(＾∇＾)", "(≧◡≦)", "(★ω★)",
    ]

    KAWAII_THINKING = [
        "(｡•́︿•̀｡)", "(◔_◔)", "(¬‿¬)", "( •_•)>⌐■-■", "(⌐■_■)",
        "(´･_･`)", "◉_◉", "(°ロ°)", "( ˘⌣˘)♡", "ヽ(>∀<☆)☆",
        "٩(๑❛ᴗ❛๑)۶", "(⊙_⊙)", "(¬_¬)", "( ͡° ͜ʖ ͡°)", "ಠ_ಠ",
    ]

    THINKING_VERBS = [
        "pondering", "contemplating", "musing", "cogitating", "ruminating",
        "deliberating", "mulling", "reflecting", "processing", "reasoning",
        "analyzing", "computing", "synthesizing", "formulating", "brainstorming",
    ]

    def __init__(self, message: str = "", spinner_type: str = 'dots', print_fn=None):
        self.message = message
        self.spinner_frames = self.SPINNERS.get(spinner_type, self.SPINNERS['dots'])
        self.running = False
        self.thread = None
        self.frame_idx = 0
        self.start_time = None
        self.last_line_len = 0
        # Optional callable to route all output through (e.g. a no-op for silent
        # background agents).  When set, bypasses self._out entirely so that
        # agents with _print_fn overridden remain fully silent.
        self._print_fn = print_fn
        # Capture stdout NOW, before any redirect_stdout(devnull) from
        # child agents can replace sys.stdout with a black hole.
        self._out = sys.stdout

    def _write(self, text: str, end: str = '\n', flush: bool = False):
        """Write to the stdout captured at spinner creation time.

        If a print_fn was supplied at construction, all output is routed through
        it instead — allowing callers to silence the spinner with a no-op lambda.
        """
        if self._print_fn is not None:
            try:
                self._print_fn(text)
            except Exception:
                pass
            return
        try:
            self._out.write(text + end)
            if flush:
                self._out.flush()
        except (ValueError, OSError):
            pass

    @property
    def _is_tty(self) -> bool:
        """Check if output is a real terminal, safe against closed streams."""
        try:
            return hasattr(self._out, 'isatty') and self._out.isatty()
        except (ValueError, OSError):
            return False

    def _is_patch_stdout_proxy(self) -> bool:
        """Return True when stdout is prompt_toolkit's StdoutProxy.

        patch_stdout wraps sys.stdout in a StdoutProxy that queues writes and
        injects newlines around each flush().  The \\r overwrite never lands on
        the correct line — each spinner frame ends up on its own line.

        The CLI already drives a TUI widget (_spinner_text) for spinner display,
        so KawaiiSpinner's \\r-based animation is redundant under StdoutProxy.
        """
        try:
            from prompt_toolkit.patch_stdout import StdoutProxy
            return isinstance(self._out, StdoutProxy)
        except ImportError:
            return False

    def _animate(self):
        # When stdout is not a real terminal (e.g. Docker, systemd, pipe),
        # skip the animation entirely — it creates massive log bloat.
        # Just log the start once and let stop() log the completion.
        if not self._is_tty:
            self._write(f"  [tool] {self.message}", flush=True)
            while self.running:
                time.sleep(0.5)
            return

        # When running inside prompt_toolkit's patch_stdout context the CLI
        # renders spinner state via a dedicated TUI widget (_spinner_text).
        # Driving a \r-based animation here too causes visual overdraw: the
        # StdoutProxy injects newlines around each flush, so every frame lands
        # on a new line and overwrites the status bar.
        if self._is_patch_stdout_proxy():
            while self.running:
                time.sleep(0.1)
            return

        # Cache skin wings at start (avoid per-frame imports)
        skin = _get_skin()
        wings = skin.get_spinner_wings() if skin else []

        while self.running:
            if os.getenv("HERMES_SPINNER_PAUSE"):
                time.sleep(0.1)
                continue
            frame = self.spinner_frames[self.frame_idx % len(self.spinner_frames)]
            elapsed = time.time() - self.start_time
            if wings:
                left, right = wings[self.frame_idx % len(wings)]
                line = f"  {left} {frame} {self.message} {right} ({elapsed:.1f}s)"
            else:
                line = f"  {frame} {self.message} ({elapsed:.1f}s)"
            pad = max(self.last_line_len - len(line), 0)
            self._write(f"\r{line}{' ' * pad}", end='', flush=True)
            self.last_line_len = len(line)
            self.frame_idx += 1
            time.sleep(0.12)

    def start(self):
        if self.running:
            return
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()

    def update_text(self, new_message: str):
        self.message = new_message

    def print_above(self, text: str):
        """Print a line above the spinner without disrupting animation.

        Clears the current spinner line, prints the text, and lets the
        next animation tick redraw the spinner on the line below.
        Thread-safe: uses the captured stdout reference (self._out).
        Works inside redirect_stdout(devnull) because _write bypasses
        sys.stdout and writes to the stdout captured at spinner creation.
        """
        if not self.running:
            self._write(f"  {text}", flush=True)
            return
        # Clear spinner line with spaces (not \033[K) to avoid garbled escape
        # codes when prompt_toolkit's patch_stdout is active — same approach
        # as stop(). Then print text; spinner redraws on next tick.
        blanks = ' ' * max(self.last_line_len + 5, 40)
        self._write(f"\r{blanks}\r  {text}", flush=True)

    def stop(self, final_message: str = None):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)

        is_tty = self._is_tty
        if is_tty:
            # Clear the spinner line with spaces instead of \033[K to avoid
            # garbled escape codes when prompt_toolkit's patch_stdout is active.
            blanks = ' ' * max(self.last_line_len + 5, 40)
            self._write(f"\r{blanks}\r", end='', flush=True)
        if final_message:
            elapsed = f" ({time.time() - self.start_time:.1f}s)" if self.start_time else ""
            if is_tty:
                self._write(f"  {final_message}", flush=True)
            else:
                self._write(f"  [done] {final_message}{elapsed}", flush=True)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


# =========================================================================
# Kawaii face arrays (used by AIAgent._execute_tool_calls for spinner text)
# =========================================================================

KAWAII_SEARCH = [
    "♪(´ε` )", "(｡◕‿◕｡)", "ヾ(＾∇＾)", "(◕ᴗ◕✿)", "( ˘▽˘)っ",
    "٩(◕‿◕｡)۶", "(✿◠‿◠)", "♪～(´ε｀ )", "(ノ´ヮ`)ノ*:・゚✧", "＼(◎o◎)／",
]
KAWAII_READ = [
    "φ(゜▽゜*)♪", "( ˘▽˘)っ", "(⌐■_■)", "٩(｡•́‿•̀｡)۶", "(◕‿◕✿)",
    "ヾ(＠⌒ー⌒＠)ノ", "(✧ω✧)", "♪(๑ᴖ◡ᴖ๑)♪", "(≧◡≦)", "( ´ ▽ ` )ノ",
]
KAWAII_TERMINAL = [
    "ヽ(>∀<☆)ノ", "(ノ°∀°)ノ", "٩(^ᴗ^)۶", "ヾ(⌐■_■)ノ♪", "(•̀ᴗ•́)و",
    "┗(＾0＾)┓", "(｀・ω・´)", "＼(￣▽￣)／", "(ง •̀_•́)ง", "ヽ(´▽`)/",
]
KAWAII_BROWSER = [
    "(ノ°∀°)ノ", "(☞゚ヮ゚)☞", "( ͡° ͜ʖ ͡°)", "┌( ಠ_ಠ)┘", "(⊙_⊙)？",
    "ヾ(•ω•`)o", "(￣ω￣)", "( ˇωˇ )", "(ᵔᴥᵔ)", "＼(◎o◎)／",
]
KAWAII_CREATE = [
    "✧*。٩(ˊᗜˋ*)و✧", "(ﾉ◕ヮ◕)ﾉ*:・ﾟ✧", "ヽ(>∀<☆)ノ", "٩(♡ε♡)۶", "(◕‿◕)♡",
    "✿◕ ‿ ◕✿", "(*≧▽≦)", "ヾ(＾-＾)ノ", "(☆▽☆)", "°˖✧◝(⁰▿⁰)◜✧˖°",
]
KAWAII_SKILL = [
    "ヾ(＠⌒ー⌒＠)ノ", "(๑˃ᴗ˂)ﻭ", "٩(◕‿◕｡)۶", "(✿╹◡╹)", "ヽ(・∀・)ノ",
    "(ノ´ヮ`)ノ*:・ﾟ✧", "♪(๑ᴖ◡ᴖ๑)♪", "(◠‿◠)", "٩(ˊᗜˋ*)و", "(＾▽＾)",
    "ヾ(＾∇＾)", "(★ω★)/", "٩(｡•́‿•̀｡)۶", "(◕ᴗ◕✿)", "＼(◎o◎)／",
    "(✧ω✧)", "ヽ(>∀<☆)ノ", "( ˘▽˘)っ", "(≧◡≦) ♡", "ヾ(￣▽￣)",
]
KAWAII_THINK = [
    "(っ°Д°;)っ", "(；′⌒`)", "(・_・ヾ", "( ´_ゝ`)", "(￣ヘ￣)",
    "(。-`ω´-)", "( ˘︹˘ )", "(¬_¬)", "ヽ(ー_ー )ノ", "(；一_一)",
]
KAWAII_GENERIC = [
    "♪(´ε` )", "(◕‿◕✿)", "ヾ(＾∇＾)", "٩(◕‿◕｡)۶", "(✿◠‿◠)",
    "(ノ´ヮ`)ノ*:・ﾟ✧", "ヽ(>∀<☆)ノ", "(☆▽☆)", "( ˘▽˘)っ", "(≧◡≦)",
]


# =========================================================================
# Cute tool message (completion line that replaces the spinner)
# =========================================================================

def _detect_tool_failure(tool_name: str, result: str | None) -> tuple[bool, str]:
    """Inspect a tool result string for signs of failure.

    Returns ``(is_failure, suffix)`` where *suffix* is an informational tag
    like ``" [exit 1]"`` for terminal failures, or ``" [error]"`` for generic
    failures.  On success, returns ``(False, "")``.
    """
    if result is None:
        return False, ""

    if tool_name == "terminal":
        try:
            data = json.loads(result)
            exit_code = data.get("exit_code")
            if exit_code is not None and exit_code != 0:
                return True, f" [exit {exit_code}]"
        except (json.JSONDecodeError, TypeError, AttributeError):
            logger.debug("Could not parse terminal result as JSON for exit code check")
        return False, ""

    # Memory-specific: distinguish "full" from real errors
    if tool_name == "memory":
        try:
            data = json.loads(result)
            if data.get("success") is False and "exceed the limit" in data.get("error", ""):
                return True, " [full]"
        except (json.JSONDecodeError, TypeError, AttributeError):
            logger.debug("Could not parse memory result as JSON for capacity check")

    # Generic heuristic for non-terminal tools
    lower = result[:500].lower()
    if '"error"' in lower or '"failed"' in lower or result.startswith("Error"):
        return True, " [error]"

    return False, ""


def get_cute_tool_message(
    tool_name: str, args: dict, duration: float, result: str | None = None,
) -> str:
    """Generate a formatted tool completion line for CLI quiet mode.

    Format: ``| {emoji} {verb:9} {detail}  {duration}``

    When *result* is provided the line is checked for failure indicators.
    Failed tool calls get a red prefix and an informational suffix.
    """
    dur = f"{duration:.1f}s"
    is_failure, failure_suffix = _detect_tool_failure(tool_name, result)
    skin_prefix = get_skin_tool_prefix()

    def _trunc(s, n=40):
        s = str(s)
        if _tool_preview_max_len == 0:
            return s  # no limit
        return (s[:n-3] + "...") if len(s) > n else s

    def _path(p, n=35):
        p = str(p)
        if _tool_preview_max_len == 0:
            return p  # no limit
        return ("..." + p[-(n-3):]) if len(p) > n else p

    def _wrap(line: str) -> str:
        """Apply skin tool prefix and failure suffix."""
        if skin_prefix != "┊":
            line = line.replace("┊", skin_prefix, 1)
        if not is_failure:
            return line
        return f"{line}{failure_suffix}"

    if tool_name == "web_search":
        return _wrap(f"┊ 🔍 search    {_trunc(args.get('query', ''), 42)}  {dur}")
    if tool_name == "web_extract":
        urls = args.get("urls", [])
        if urls:
            url = urls[0] if isinstance(urls, list) else str(urls)
            domain = url.replace("https://", "").replace("http://", "").split("/")[0]
            extra = f" +{len(urls)-1}" if len(urls) > 1 else ""
            return _wrap(f"┊ 📄 fetch     {_trunc(domain, 35)}{extra}  {dur}")
        return _wrap(f"┊ 📄 fetch     pages  {dur}")
    if tool_name == "web_crawl":
        url = args.get("url", "")
        domain = url.replace("https://", "").replace("http://", "").split("/")[0]
        return _wrap(f"┊ 🕸️  crawl     {_trunc(domain, 35)}  {dur}")
    if tool_name == "terminal":
        return _wrap(f"┊ 💻 $         {_trunc(args.get('command', ''), 42)}  {dur}")
    if tool_name == "process":
        action = args.get("action", "?")
        sid = args.get("session_id", "")[:12]
        labels = {"list": "ls processes", "poll": f"poll {sid}", "log": f"log {sid}",
                  "wait": f"wait {sid}", "kill": f"kill {sid}", "write": f"write {sid}", "submit": f"submit {sid}"}
        return _wrap(f"┊ ⚙️  proc      {labels.get(action, f'{action} {sid}')}  {dur}")
    if tool_name == "read_file":
        return _wrap(f"┊ 📖 read      {_path(args.get('path', ''))}  {dur}")
    if tool_name == "write_file":
        return _wrap(f"┊ ✍️  write     {_path(args.get('path', ''))}  {dur}")
    if tool_name == "patch":
        return _wrap(f"┊ 🔧 patch     {_path(args.get('path', ''))}  {dur}")
    if tool_name == "search_files":
        pattern = _trunc(args.get("pattern", ""), 35)
        target = args.get("target", "content")
        verb = "find" if target == "files" else "grep"
        return _wrap(f"┊ 🔎 {verb:9} {pattern}  {dur}")
    if tool_name == "browser_navigate":
        url = args.get("url", "")
        domain = url.replace("https://", "").replace("http://", "").split("/")[0]
        return _wrap(f"┊ 🌐 navigate  {_trunc(domain, 35)}  {dur}")
    if tool_name == "browser_snapshot":
        mode = "full" if args.get("full") else "compact"
        return _wrap(f"┊ 📸 snapshot  {mode}  {dur}")
    if tool_name == "browser_click":
        return _wrap(f"┊ 👆 click     {args.get('ref', '?')}  {dur}")
    if tool_name == "browser_type":
        return _wrap(f"┊ ⌨️  type      \"{_trunc(args.get('text', ''), 30)}\"  {dur}")
    if tool_name == "browser_scroll":
        d = args.get("direction", "down")
        arrow = {"down": "↓", "up": "↑", "right": "→", "left": "←"}.get(d, "↓")
        return _wrap(f"┊ {arrow}  scroll    {d}  {dur}")
    if tool_name == "browser_back":
        return _wrap(f"┊ ◀️  back      {dur}")
    if tool_name == "browser_press":
        return _wrap(f"┊ ⌨️  press     {args.get('key', '?')}  {dur}")
    if tool_name == "browser_close":
        return _wrap(f"┊ 🚪 close     browser  {dur}")
    if tool_name == "browser_get_images":
        return _wrap(f"┊ 🖼️  images    extracting  {dur}")
    if tool_name == "browser_vision":
        return _wrap(f"┊ 👁️  vision    analyzing page  {dur}")
    if tool_name == "todo":
        todos_arg = args.get("todos")
        merge = args.get("merge", False)
        if todos_arg is None:
            return _wrap(f"┊ 📋 plan      reading tasks  {dur}")
        elif merge:
            return _wrap(f"┊ 📋 plan      update {len(todos_arg)} task(s)  {dur}")
        else:
            return _wrap(f"┊ 📋 plan      {len(todos_arg)} task(s)  {dur}")
    if tool_name == "session_search":
        return _wrap(f"┊ 🔍 recall    \"{_trunc(args.get('query', ''), 35)}\"  {dur}")
    if tool_name == "memory":
        action = args.get("action", "?")
        target = args.get("target", "")
        if action == "add":
            return _wrap(f"┊ 🧠 memory    +{target}: \"{_trunc(args.get('content', ''), 30)}\"  {dur}")
        elif action == "replace":
            return _wrap(f"┊ 🧠 memory    ~{target}: \"{_trunc(args.get('old_text', ''), 20)}\"  {dur}")
        elif action == "remove":
            return _wrap(f"┊ 🧠 memory    -{target}: \"{_trunc(args.get('old_text', ''), 20)}\"  {dur}")
        return _wrap(f"┊ 🧠 memory    {action}  {dur}")
    if tool_name == "skills_list":
        return _wrap(f"┊ 📚 skills    list {args.get('category', 'all')}  {dur}")
    if tool_name == "skill_view":
        return _wrap(f"┊ 📚 skill     {_trunc(args.get('name', ''), 30)}  {dur}")
    if tool_name == "image_generate":
        return _wrap(f"┊ 🎨 create    {_trunc(args.get('prompt', ''), 35)}  {dur}")
    if tool_name == "text_to_speech":
        return _wrap(f"┊ 🔊 speak     {_trunc(args.get('text', ''), 30)}  {dur}")
    if tool_name == "vision_analyze":
        return _wrap(f"┊ 👁️  vision    {_trunc(args.get('question', ''), 30)}  {dur}")
    if tool_name == "mixture_of_agents":
        return _wrap(f"┊ 🧠 reason    {_trunc(args.get('user_prompt', ''), 30)}  {dur}")
    if tool_name == "send_message":
        return _wrap(f"┊ 📨 send      {args.get('target', '?')}: \"{_trunc(args.get('message', ''), 25)}\"  {dur}")
    if tool_name == "cronjob":
        action = args.get("action", "?")
        if action == "create":
            skills = args.get("skills") or ([] if not args.get("skill") else [args.get("skill")])
            label = args.get("name") or (skills[0] if skills else None) or args.get("prompt", "task")
            return _wrap(f"┊ ⏰ cron      create {_trunc(label, 24)}  {dur}")
        if action == "list":
            return _wrap(f"┊ ⏰ cron      listing  {dur}")
        return _wrap(f"┊ ⏰ cron      {action} {args.get('job_id', '')}  {dur}")
    if tool_name.startswith("rl_"):
        rl = {
            "rl_list_environments": "list envs", "rl_select_environment": f"select {args.get('name', '')}",
            "rl_get_current_config": "get config", "rl_edit_config": f"set {args.get('field', '?')}",
            "rl_start_training": "start training", "rl_check_status": f"status {args.get('run_id', '?')[:12]}",
            "rl_stop_training": f"stop {args.get('run_id', '?')[:12]}", "rl_get_results": f"results {args.get('run_id', '?')[:12]}",
            "rl_list_runs": "list runs", "rl_test_inference": "test inference",
        }
        return _wrap(f"┊ 🧪 rl        {rl.get(tool_name, tool_name.replace('rl_', ''))}  {dur}")
    if tool_name == "execute_code":
        code = args.get("code", "")
        first_line = code.strip().split("\n")[0] if code.strip() else ""
        return _wrap(f"┊ 🐍 exec      {_trunc(first_line, 35)}  {dur}")
    if tool_name == "delegate_task":
        tasks = args.get("tasks")
        if tasks and isinstance(tasks, list):
            return _wrap(f"┊ 🔀 delegate  {len(tasks)} parallel tasks  {dur}")
        return _wrap(f"┊ 🔀 delegate  {_trunc(args.get('goal', ''), 35)}  {dur}")

    preview = build_tool_preview(tool_name, args) or ""
    return _wrap(f"┊ ⚡ {tool_name[:9]:9} {_trunc(preview, 35)}  {dur}")


# =========================================================================
# Honcho session line (one-liner with clickable OSC 8 hyperlink)
# =========================================================================

_DIM = "\033[2m"
_SKY_BLUE = "\033[38;5;117m"
_ANSI_RESET = "\033[0m"


def honcho_session_url(workspace: str, session_name: str) -> str:
    """Build a Honcho app URL for a session."""
    from urllib.parse import quote
    return (
        f"https://app.honcho.dev/explore"
        f"?workspace={quote(workspace, safe='')}"
        f"&view=sessions"
        f"&session={quote(session_name, safe='')}"
    )


def _osc8_link(url: str, text: str) -> str:
    """OSC 8 terminal hyperlink (clickable in iTerm2, Ghostty, WezTerm, etc.)."""
    return f"\033]8;;{url}\033\\{text}\033]8;;\033\\"


def honcho_session_line(workspace: str, session_name: str) -> str:
    """One-line session indicator: `Honcho session: <clickable name>`."""
    url = honcho_session_url(workspace, session_name)
    linked_name = _osc8_link(url, f"{_SKY_BLUE}{session_name}{_ANSI_RESET}")
    return f"{_DIM}Honcho session:{_ANSI_RESET} {linked_name}"


def write_tty(text: str) -> None:
    """Write directly to /dev/tty, bypassing stdout capture."""
    try:
        fd = os.open("/dev/tty", os.O_WRONLY)
        os.write(fd, text.encode("utf-8"))
        os.close(fd)
    except OSError:
        sys.stdout.write(text)
        sys.stdout.flush()


# =========================================================================
# Context pressure display (CLI user-facing warnings)
# =========================================================================

# ANSI color codes for context pressure tiers
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_BOLD = "\033[1m"
_DIM_ANSI = "\033[2m"

# Bar characters
_BAR_FILLED = "▰"
_BAR_EMPTY = "▱"
_BAR_WIDTH = 20


def format_context_pressure(
    compaction_progress: float,
    threshold_tokens: int,
    threshold_percent: float,
    compression_enabled: bool = True,
) -> str:
    """Build a formatted context pressure line for CLI display.

    The bar and percentage show progress toward the compaction threshold,
    NOT the raw context window.  100% = compaction fires.

    Args:
        compaction_progress: How close to compaction (0.0–1.0, 1.0 = fires).
        threshold_tokens: Compaction threshold in tokens.
        threshold_percent: Compaction threshold as a fraction of context window.
        compression_enabled: Whether auto-compression is active.
    """
    pct_int = min(int(compaction_progress * 100), 100)
    filled = min(int(compaction_progress * _BAR_WIDTH), _BAR_WIDTH)
    bar = _BAR_FILLED * filled + _BAR_EMPTY * (_BAR_WIDTH - filled)

    threshold_k = f"{threshold_tokens // 1000}k" if threshold_tokens >= 1000 else str(threshold_tokens)
    threshold_pct_int = int(threshold_percent * 100)

    color = f"{_BOLD}{_YELLOW}"
    icon = "⚠"
    if compression_enabled:
        hint = "compaction approaching"
    else:
        hint = "no auto-compaction"

    return (
        f"  {color}{icon} context {bar} {pct_int}% to compaction{_ANSI_RESET}"
        f"  {_DIM_ANSI}{threshold_k} threshold ({threshold_pct_int}%) · {hint}{_ANSI_RESET}"
    )


def format_context_pressure_gateway(
    compaction_progress: float,
    threshold_percent: float,
    compression_enabled: bool = True,
) -> str:
    """Build a plain-text context pressure notification for messaging platforms.

    No ANSI — just Unicode and plain text suitable for Telegram/Discord/etc.
    The percentage shows progress toward the compaction threshold.
    """
    pct_int = min(int(compaction_progress * 100), 100)
    filled = min(int(compaction_progress * _BAR_WIDTH), _BAR_WIDTH)
    bar = _BAR_FILLED * filled + _BAR_EMPTY * (_BAR_WIDTH - filled)

    threshold_pct_int = int(threshold_percent * 100)

    icon = "⚠️"
    if compression_enabled:
        hint = f"Context compaction approaching (threshold: {threshold_pct_int}% of window)."
    else:
        hint = "Auto-compaction is disabled — context may be truncated."

    return f"{icon} Context: {bar} {pct_int}% to compaction\n{hint}"


# =========================================================================
# Inline read_file preview (show file content in CLI transcript)
# =========================================================================

_ANSI_LINE_NUM = "\033[38;2;100;100;120m"
_MAX_INLINE_READ_LINES = 60


def render_read_file_inline(
    function_result: str,
    *,
    print_fn=None,
    max_lines: int = _MAX_INLINE_READ_LINES,
) -> bool:
    """Render read_file content inline in the CLI transcript.

    Parses the JSON tool result from read_file and emits the file content
    with dimmed line numbers, similar to how inline diffs are rendered for
    write_file/patch.  Returns True if content was emitted.

    When the file exceeds *max_lines*, the first portion is shown with a
    trailing summary of how many lines were omitted.
    """
    if print_fn is None or not function_result:
        return False
    try:
        data = json.loads(function_result)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(data, dict):
        return False
    content = data.get("content")
    if not content or not isinstance(content, str):
        return False

    # Skip rendering for binary/image files
    if data.get("is_binary") or data.get("is_image"):
        return False

    lines = content.splitlines()
    total = len(lines)
    show = lines[:max_lines]

    try:
        skin_prefix = get_skin_tool_prefix()
        for line in show:
            # Lines from read_file are already formatted as "NUM|CONTENT"
            parts = line.split("|", 1)
            if len(parts) == 2 and parts[0].strip().isdigit():
                num = parts[0]
                text = parts[1]
                print_fn(f"  {skin_prefix} {_ANSI_LINE_NUM}{num}{_ANSI_RESET}|{_ANSI_DIM}{text}{_ANSI_RESET}")
            else:
                print_fn(f"  {skin_prefix} {_ANSI_DIM}{line}{_ANSI_RESET}")

        if total > max_lines:
            omitted = total - max_lines
            print_fn(f"  {skin_prefix} {_ANSI_HUNK}… {omitted} more line(s) (of {total} total){_ANSI_RESET}")

        return True
    except Exception:
        return False
