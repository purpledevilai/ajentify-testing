"""
Prompt registry with local override support.

All framework prompts are defined here as defaults. If the user places an
``ajentify_prompts.py`` file in their working directory, any matching names
are used instead.  Run ``ajentify-test init --with-prompts`` to scaffold
the override file with all current defaults.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


# ── Default prompts ─────────────────────────────────────────────

BOOLEAN_ASSESSOR_PROMPT = (
    "You are a strict test evaluator. You will receive a conversation "
    "transcript and a statement to evaluate.\n\n"
    "Conversation Transcript:\n{conversation}\n\n"
    "Statement to evaluate:\n{statement}\n\n"
    "Determine whether the statement is TRUE or FALSE based solely on "
    "evidence in the conversation.\n"
    "Be strict — only mark as true if the conversation clearly supports it."
)

SCORE_ASSESSOR_PROMPT = (
    "You are a strict test evaluator. You will receive a conversation "
    "transcript and criteria to score.\n\n"
    "Conversation Transcript:\n{conversation}\n\n"
    "Criteria to score:\n{criteria}\n\n"
    "Rate how well the criteria was met on a scale of 0.0 to 1.0, where:\n"
    "- 0.0 = not met at all\n"
    "- 0.5 = partially met\n"
    "- 1.0 = fully and excellently met\n\n"
    "Be strict and justify your score."
)

EXTRACT_PROMPT_TEMPLATE = (
    "You are analysing a conversation transcript. Extract the requested "
    "information based on the output schema provided.\n\n"
    "Conversation Transcript:\n{conversation}"
)

ASSESS_ALL_PROMPT_TEMPLATE = (
    "You are a strict test evaluator. Evaluate each assertion below against the "
    "conversation transcript. For every assertion return whether it passed and your reasoning.\n\n"
    "Rules:\n"
    "- Base every judgment solely on evidence present in the transcript.\n"
    "- For TRUE assertions: only pass if the transcript clearly supports the statement.\n"
    "- For FALSE assertions: only pass if the transcript clearly does NOT support the statement.\n"
    "- For SCORE assertions: only pass if the quality clearly meets or exceeds the threshold.\n\n"
    "Assertions:\n{assertions}\n\n"
    "Conversation Transcript:\n{conversation}"
)

SIM_AGENT_PREAMBLE = (
    "You are a simulated user in an automated test. Your job is to play the role "
    "of a real person having a natural conversation with an AI assistant.\n\n"
    "CRITICAL RULES:\n"
    "- You are the USER. You ask questions and describe your situation. You do NOT "
    "act as an assistant.\n"
    "- NEVER say things like 'How can I help you?' or 'What would you like to know?' "
    "— those are assistant phrases.\n"
    "- Once the assistant gives you a clear, complete answer, call the end_test tool "
    "with a brief summary of the conversation.\n"
    "- If the assistant asks a clarifying question, respond naturally.\n\n"
)

END_TEST_TOOL_DESCRIPTION = (
    "Call this tool when the assistant has given you a clear, complete answer "
    "to your question, OR when the conversation has reached a natural conclusion. "
    "Provide a summary of what happened."
)

END_TEST_SUMMARY_PARAM_DESCRIPTION = (
    "A brief summary of the conversation: what was discussed, "
    "what actions the assistant took, and any notable observations."
)


# ── Names that can be overridden ────────────────────────────────

_OVERRIDABLE = [
    "BOOLEAN_ASSESSOR_PROMPT",
    "SCORE_ASSESSOR_PROMPT",
    "EXTRACT_PROMPT_TEMPLATE",
    "ASSESS_ALL_PROMPT_TEMPLATE",
    "SIM_AGENT_PREAMBLE",
    "END_TEST_TOOL_DESCRIPTION",
    "END_TEST_SUMMARY_PARAM_DESCRIPTION",
]


def _apply_overrides() -> None:
    """Import ``ajentify_prompts`` from CWD if it exists, and override globals."""
    config_path = Path.cwd() / "ajentify_prompts.py"
    if not config_path.exists():
        return

    cwd_str = str(Path.cwd())
    if cwd_str not in sys.path:
        sys.path.insert(0, cwd_str)

    try:
        user_mod = importlib.import_module("ajentify_prompts")
        importlib.reload(user_mod)
    except Exception:
        return

    g = globals()
    for name in _OVERRIDABLE:
        if hasattr(user_mod, name):
            g[name] = getattr(user_mod, name)


_apply_overrides()
