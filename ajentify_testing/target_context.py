from __future__ import annotations
import json
from typing import Optional, TYPE_CHECKING

from ajentify_testing.models import CheckResult, CheckType, CheckStatus
from ajentify_testing.exceptions import AssertionFailed, AssessmentFailed

if TYPE_CHECKING:
    from ajentify_testing.session import TestSession


def _format_conversation(messages: list[dict]) -> str:
    """Turn context messages into a human-readable transcript."""
    lines: list[str] = []
    for msg in messages:
        msg_type = msg.get("type") or msg.get("sender")
        if msg_type == "human":
            lines.append(f"[User]: {msg.get('message', '')}")
        elif msg_type == "ai":
            lines.append(f"[Agent]: {msg.get('message', '')}")
        elif msg_type == "tool_call":
            tool_input = json.dumps(msg.get("tool_input", {}))
            lines.append(f"[Agent Tool Call]: {msg.get('tool_name', '')}({tool_input})")
        elif msg_type == "tool_response":
            lines.append(f"[Tool Response]: {msg.get('tool_output', '')}")
    return "\n".join(lines)


class TargetContext:
    """Wrapper around the agent-under-test's Ajentify context.

    Provides:
      - chat() / invoke()        — drive the conversation
      - load_transcript()        — fetch the full conversation with tool calls
      - assert_*()               — deterministic checks on the transcript
      - assess_*()               — LLM-powered checks via SRE

    Args:
        session: The active TestSession.
        agent_id: The Ajentify agent ID of the agent under test.
        prompt_args: Optional prompt_args dict passed to context creation.
        user_defined: Optional user_defined dict passed to context creation.
        invoke_on_create: If True, invoke the agent immediately on context creation
            so it produces a greeting message.
    """

    def __init__(
        self,
        session: TestSession,
        agent_id: str,
        prompt_args: Optional[dict] = None,
        user_defined: Optional[dict] = None,
        invoke_on_create: bool = True,
    ):
        self.session = session
        self.client = session.client

        ctx = self.client.create_context(
            agent_id=agent_id,
            invoke_agent_message=invoke_on_create,
            prompt_args=prompt_args,
            user_defined=user_defined,
        )
        self.context_id: str = ctx["context_id"]

        self.greeting: str = ""
        if invoke_on_create and ctx.get("messages"):
            for m in ctx["messages"]:
                if (m.get("sender") or m.get("type")) == "ai":
                    self.greeting = m.get("message", "")
                    break

        if invoke_on_create and not self.greeting:
            invoke_resp = self.client.invoke(self.context_id)
            self.greeting = invoke_resp.get("response", "")

        self.messages: list[dict] = []
        self.turn_count: int = 0
        self.checks: list[CheckResult] = []

        session._register_resource(self)

    # ── Conversation ────────────────────────────────────────────

    def chat(self, message: str) -> str:
        """Send a user message and return the agent's text response."""
        resp = self.client.chat(self.context_id, message)
        return resp.get("response", "")

    def invoke(self) -> str:
        """Invoke the agent (no user message) and return its response."""
        resp = self.client.invoke(self.context_id)
        return resp.get("response", "")

    def load_transcript(self):
        """Fetch the full context transcript (with tool calls) from Ajentify."""
        full_ctx = self.client.get_context(self.context_id, with_tool_calls=True)
        self.messages = full_ctx.get("messages", [])

    def get_transcript_text(self) -> str:
        """Return the transcript as formatted text."""
        return _format_conversation(self.messages)

    # ── assert_* (deterministic) ────────────────────────────────

    def assert_called_tool(
        self,
        tool_name: str,
        with_params: Optional[dict] = None,
    ):
        """Assert the agent called a specific tool during the conversation.

        Args:
            tool_name: The name of the tool that should have been called.
            with_params: If provided, at least one call to tool_name must
                contain all of these key-value pairs in its tool_input.
        """
        self._ensure_transcript()

        matching_calls = [
            msg for msg in self.messages
            if (msg.get("type") or msg.get("sender")) == "tool_call"
            and msg.get("tool_name") == tool_name
        ]

        if not matching_calls:
            check = CheckResult(
                check_type=CheckType.ASSERT,
                name=f"assert_called_tool({tool_name!r})",
                status=CheckStatus.FAILED,
                detail=f"Tool '{tool_name}' was never called",
            )
            self.checks.append(check)
            raise AssertionFailed(
                f"assert_called_tool({tool_name!r})",
                f"Tool '{tool_name}' was never called",
            )

        if with_params is not None:
            for call in matching_calls:
                tool_input = call.get("tool_input", {})
                if all(tool_input.get(k) == v for k, v in with_params.items()):
                    check = CheckResult(
                        check_type=CheckType.ASSERT,
                        name=f"assert_called_tool({tool_name!r}, with_params={with_params!r})",
                        status=CheckStatus.PASSED,
                    )
                    self.checks.append(check)
                    return

            actual_inputs = [c.get("tool_input", {}) for c in matching_calls]
            check = CheckResult(
                check_type=CheckType.ASSERT,
                name=f"assert_called_tool({tool_name!r}, with_params={with_params!r})",
                status=CheckStatus.FAILED,
                detail=f"Tool was called but params didn't match. Actual inputs: {actual_inputs}",
            )
            self.checks.append(check)
            raise AssertionFailed(
                f"assert_called_tool({tool_name!r}, with_params={with_params!r})",
                f"Tool was called but params didn't match. Actual: {actual_inputs}",
            )

        check = CheckResult(
            check_type=CheckType.ASSERT,
            name=f"assert_called_tool({tool_name!r})",
            status=CheckStatus.PASSED,
        )
        self.checks.append(check)

    def assert_not_called_tool(self, tool_name: str):
        """Assert the agent did NOT call a specific tool."""
        self._ensure_transcript()

        called = any(
            (msg.get("type") or msg.get("sender")) == "tool_call"
            and msg.get("tool_name") == tool_name
            for msg in self.messages
        )

        if called:
            check = CheckResult(
                check_type=CheckType.ASSERT,
                name=f"assert_not_called_tool({tool_name!r})",
                status=CheckStatus.FAILED,
                detail=f"Tool '{tool_name}' was called when it should not have been",
            )
            self.checks.append(check)
            raise AssertionFailed(
                f"assert_not_called_tool({tool_name!r})",
                f"Tool '{tool_name}' was called when it should not have been",
            )

        check = CheckResult(
            check_type=CheckType.ASSERT,
            name=f"assert_not_called_tool({tool_name!r})",
            status=CheckStatus.PASSED,
        )
        self.checks.append(check)

    def assert_message_contains(self, substring: str):
        """Assert at least one agent message contains the substring (case-insensitive)."""
        self._ensure_transcript()
        lower = substring.lower()

        found = any(
            (msg.get("type") or msg.get("sender")) == "ai"
            and lower in (msg.get("message", "") or "").lower()
            for msg in self.messages
        )

        name = f"assert_message_contains({substring!r})"
        if not found:
            check = CheckResult(
                check_type=CheckType.ASSERT, name=name, status=CheckStatus.FAILED,
                detail=f"No agent message contained '{substring}'",
            )
            self.checks.append(check)
            raise AssertionFailed(name, f"No agent message contained '{substring}'")

        self.checks.append(CheckResult(check_type=CheckType.ASSERT, name=name, status=CheckStatus.PASSED))

    def assert_message_not_contains(self, substring: str):
        """Assert no agent message contains the substring (case-insensitive)."""
        self._ensure_transcript()
        lower = substring.lower()

        found = any(
            (msg.get("type") or msg.get("sender")) == "ai"
            and lower in (msg.get("message", "") or "").lower()
            for msg in self.messages
        )

        name = f"assert_message_not_contains({substring!r})"
        if found:
            check = CheckResult(
                check_type=CheckType.ASSERT, name=name, status=CheckStatus.FAILED,
                detail=f"An agent message contained '{substring}' when it should not have",
            )
            self.checks.append(check)
            raise AssertionFailed(name, f"An agent message contained '{substring}'")

        self.checks.append(CheckResult(check_type=CheckType.ASSERT, name=name, status=CheckStatus.PASSED))

    def assert_turn_count(self, *, min: int | None = None, max: int | None = None):
        """Assert the conversation turn count is within bounds."""
        name = f"assert_turn_count(min={min}, max={max})"

        if min is not None and self.turn_count < min:
            check = CheckResult(
                check_type=CheckType.ASSERT, name=name, status=CheckStatus.FAILED,
                detail=f"Turn count {self.turn_count} < minimum {min}",
            )
            self.checks.append(check)
            raise AssertionFailed(name, f"Turn count {self.turn_count} < minimum {min}")

        if max is not None and self.turn_count > max:
            check = CheckResult(
                check_type=CheckType.ASSERT, name=name, status=CheckStatus.FAILED,
                detail=f"Turn count {self.turn_count} > maximum {max}",
            )
            self.checks.append(check)
            raise AssertionFailed(name, f"Turn count {self.turn_count} > maximum {max}")

        self.checks.append(CheckResult(check_type=CheckType.ASSERT, name=name, status=CheckStatus.PASSED))

    # ── assess_* (LLM-powered) ─────────────────────────────────

    def assess_true(self, statement: str):
        """Assess (via LLM) that a statement is true about the conversation."""
        self._ensure_transcript()
        conversation_text = self.get_transcript_text()

        result = self.session.client.run_sre(self.session.boolean_sre_id, {
            "conversation": conversation_text,
            "statement": statement,
        })

        is_true = result.get("result", False)
        reasoning = result.get("reasoning", "")

        name = f"assess_true({statement!r})"
        if not is_true:
            check = CheckResult(
                check_type=CheckType.ASSESS, name=name, status=CheckStatus.FAILED,
                reasoning=reasoning,
            )
            self.checks.append(check)
            raise AssessmentFailed(statement, reasoning)

        self.checks.append(CheckResult(
            check_type=CheckType.ASSESS, name=name, status=CheckStatus.PASSED, reasoning=reasoning,
        ))

    def assess_false(self, statement: str):
        """Assess (via LLM) that a statement is false about the conversation."""
        self._ensure_transcript()
        conversation_text = self.get_transcript_text()

        result = self.session.client.run_sre(self.session.boolean_sre_id, {
            "conversation": conversation_text,
            "statement": statement,
        })

        is_true = result.get("result", False)
        reasoning = result.get("reasoning", "")

        name = f"assess_false({statement!r})"
        if is_true:
            check = CheckResult(
                check_type=CheckType.ASSESS, name=name, status=CheckStatus.FAILED,
                reasoning=reasoning,
            )
            self.checks.append(check)
            raise AssessmentFailed(f"Expected false: {statement}", reasoning)

        self.checks.append(CheckResult(
            check_type=CheckType.ASSESS, name=name, status=CheckStatus.PASSED, reasoning=reasoning,
        ))

    def assess_score(self, criteria: str, *, min: float = 0.0):
        """Assess (via LLM) a score for how well criteria was met.

        Args:
            criteria: The criteria to evaluate.
            min: Minimum acceptable score (0.0 to 1.0). Fails if below.
        """
        self._ensure_transcript()
        conversation_text = self.get_transcript_text()

        result = self.session.client.run_sre(self.session.score_sre_id, {
            "conversation": conversation_text,
            "criteria": criteria,
        })

        score = float(result.get("score", 0.0))
        reasoning = result.get("reasoning", "")

        name = f"assess_score({criteria!r}, min={min})"
        if score < min:
            check = CheckResult(
                check_type=CheckType.ASSESS, name=name, status=CheckStatus.FAILED,
                reasoning=reasoning, score=score,
            )
            self.checks.append(check)
            raise AssessmentFailed(
                f"{criteria} — score {score:.2f} < minimum {min:.2f}",
                reasoning,
            )

        self.checks.append(CheckResult(
            check_type=CheckType.ASSESS, name=name, status=CheckStatus.PASSED,
            reasoning=reasoning, score=score,
        ))

    # ── Cleanup ─────────────────────────────────────────────────

    def cleanup(self):
        """Delete the target context from Ajentify."""
        if self.context_id:
            try:
                self.client.delete_context(self.context_id)
            except Exception:
                pass
            self.context_id = None

    # ── Internal ────────────────────────────────────────────────

    def _ensure_transcript(self):
        """Load transcript if not already loaded."""
        if not self.messages:
            self.load_transcript()
