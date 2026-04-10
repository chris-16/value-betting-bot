"""Claude CLI adapter for the w5-football-prediction consensus engine.

Replaces Gemini as the LLM provider by invoking the Claude CLI (Max plan)
via subprocess. Returns structured JSON predictions.
"""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

# Default timeout for Claude CLI calls (seconds)
_DEFAULT_TIMEOUT = 120


class ClaudeCLIError(Exception):
    """Raised when the Claude CLI returns an error or unparseable output."""


def query_claude(
    prompt: str,
    *,
    model: str = "claude-sonnet-4-20250514",
    timeout: int = _DEFAULT_TIMEOUT,
) -> str:
    """Send a prompt to Claude via the CLI and return the raw text response.

    Uses ``claude -p`` (pipe mode) for non-interactive single-turn queries.
    Requires the ``claude`` CLI to be installed and authenticated (Max plan).

    Args:
        prompt: The full prompt text to send.
        model: Claude model identifier to use.
        timeout: Maximum seconds to wait for a response.

    Returns:
        The raw text response from Claude.

    Raises:
        ClaudeCLIError: If the CLI process fails or times out.
    """
    cmd = ["claude", "-p", "--model", model, "--output-format", "text"]
    logger.debug("Invoking Claude CLI (model=%s, timeout=%ds)", model, timeout)

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ClaudeCLIError(f"Claude CLI timed out after {timeout}s") from exc
    except FileNotFoundError as exc:
        raise ClaudeCLIError(
            "Claude CLI not found — install with: npm install -g @anthropic-ai/claude-code"
        ) from exc

    if result.returncode != 0:
        stderr = result.stderr.strip()[:500]
        raise ClaudeCLIError(f"Claude CLI exited with code {result.returncode}: {stderr}")

    return result.stdout.strip()


def query_claude_json(
    prompt: str,
    *,
    model: str = "claude-sonnet-4-20250514",
    timeout: int = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Send a prompt to Claude and parse the response as JSON.

    Extracts the first JSON object found in the response text,
    handling cases where Claude wraps JSON in markdown code blocks.

    Args:
        prompt: The full prompt text (should instruct Claude to return JSON).
        model: Claude model identifier.
        timeout: Maximum seconds to wait.

    Returns:
        Parsed JSON dictionary.

    Raises:
        ClaudeCLIError: If the response cannot be parsed as JSON.
    """
    raw = query_claude(prompt, model=model, timeout=timeout)

    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = text.index("\n")
        text = text[first_newline + 1 :]
        # Remove closing fence
        if text.endswith("```"):
            text = text[:-3].strip()

    try:
        return json.loads(text)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                pass
        raise ClaudeCLIError(f"Failed to parse Claude response as JSON: {raw[:300]}")
