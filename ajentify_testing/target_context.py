from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, TYPE_CHECKING

from ajentify_testing.models import CheckResult, CheckType, CheckStatus
from ajentify_testing.exceptions import AssertionFailed, AssessmentFailed
from ajentify_testing.params import Param

if TYPE_CHECKING:
    from ajentify_testing.session import TestSession


# ── Prompt templates for assess_* ────────────────────────────────

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

BOOLEAN_PARAMS = [
    Param.boolean("result", "Whether the statement is true"),
    Param.string("reasoning", "Explanation of why the statement is true or false"),
]

SCORE_PARAMS = [
    Param.number("score", "Score from 0.0 to 1.0"),
    Param.string("reasoning", "Explanation of the score"),
]


# ── Assessment descriptor classes ────────────────────────────────

class AssessTrue:
    """Descriptor for assess_all: statement should be true."""
    def __init__(self, statement: str):
        self.statement = statement


class AssessFalse:
    """Descriptor for assess_all: statement should be false."""
    def __init__(self, statement: str):
        self.statement = statement


class AssessScore:
    """Descriptor for assess_all: criteria should meet a minimum score."""
    def __init__(self, criteria: str, *, min: float = 0.0):
        self.criteria = criteria
        self.min = min


# ── Helpers ──────────────────────────────────────────────────────

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


# ── TargetContext ────────────────────────────────────────────────

class TargetContext:
    """Wrapper around the agent-under-test's Ajentify context.

    Provides:
      - chat() / invoke()        — drive the conversation
      - load_transcript()        — fetch the full conversation with tool calls
      - assert_*()               — deterministic checks on the transcript
      - assess_*()               — LLM-powered checks via inline SRE
      - assess_all()             — run multiple assessments, collecting all results
      - extract()                — ad-hoc structured extraction from the conversation

    Args:
        session: The active TestSession.
        agent_id: The Ajentify agent ID of the agent under test.
        prompt_args: Optional prompt_args dict passed to context creation.
        user_defined: Optional user_defined dict passed to context creation.
    """

    def __init__(
        self,
        session: TestSession,
        agent_id: str,
        prompt_args: Optional[dict] = None,
        user_defined: Optional[dict] = None,
    ):
        self.session = session
        self.client = session.client

        ctx = self.client.create_context(
            agent_id=agent_id,
            invoke_agent_message=True,
            prompt_args=prompt_args,
            user_defined=user_defined,
        )
        self.context_id: str = ctx["context_id"]

        self.greeting: str = ""
        if ctx.get("messages"):
            for m in ctx["messages"]:
                if (m.get("sender") or m.get("type")) == "ai":
                    self.greeting = m.get("message", "")
                    break

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

    # ── assess_* (LLM-powered via inline SRE) ───────────────────

    def assess_true(self, statement: str):
        """Assess (via LLM) that a statement is true about the conversation."""
        self._ensure_transcript()
        conversation_text = self.get_transcript_text()

        prompt = BOOLEAN_ASSESSOR_PROMPT.format(
            conversation=conversation_text, statement=statement,
        )
        result = self.client.run_sre_inline(prompt, BOOLEAN_PARAMS)

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

        prompt = BOOLEAN_ASSESSOR_PROMPT.format(
            conversation=conversation_text, statement=statement,
        )
        result = self.client.run_sre_inline(prompt, BOOLEAN_PARAMS)

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

        prompt = SCORE_ASSESSOR_PROMPT.format(
            conversation=conversation_text, criteria=criteria,
        )
        result = self.client.run_sre_inline(prompt, SCORE_PARAMS)

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

    # ── assess_all (batch assessments, no early exit) ────────────

    def assess_all(self, assessments: list[AssessTrue | AssessFalse | AssessScore]):
        """Run multiple LLM assessments in parallel, collecting all results before failing.

        Unlike calling assess_true / assess_false / assess_score individually
        (which raise on the first failure), assess_all fires every assessment
        concurrently and only raises after all are complete — giving a full picture.

        Args:
            assessments: List of AssessTrue, AssessFalse, or AssessScore descriptors.

        Raises:
            AssessmentFailed: If any assessment in the batch failed, with a
                summary of all failures.
        """
        self._ensure_transcript()
        conversation_text = self.get_transcript_text()

        def _run(assessment):
            if isinstance(assessment, AssessTrue):
                return self._run_boolean_assessment(
                    conversation_text, assessment.statement, expect_true=True,
                )
            elif isinstance(assessment, AssessFalse):
                return self._run_boolean_assessment(
                    conversation_text, assessment.statement, expect_true=False,
                )
            elif isinstance(assessment, AssessScore):
                return self._run_score_assessment(
                    conversation_text, assessment.criteria, min_score=assessment.min,
                )

        results: list[tuple[CheckResult, str | None]] = []
        with ThreadPoolExecutor(max_workers=len(assessments)) as executor:
            futures = {executor.submit(_run, a): i for i, a in enumerate(assessments)}
            indexed: dict[int, tuple[CheckResult, str | None]] = {}
            for future in as_completed(futures):
                idx = futures[future]
                indexed[idx] = future.result()
            for i in range(len(assessments)):
                results.append(indexed[i])

        failures: list[str] = []
        for check, failure_msg in results:
            self.checks.append(check)
            if failure_msg is not None:
                failures.append(failure_msg)

        if failures:
            summary = "; ".join(failures)
            raise AssessmentFailed(f"{len(failures)} assessment(s) failed", summary)

    def _run_boolean_assessment(
        self, conversation_text: str, statement: str,
        *, expect_true: bool,
    ) -> tuple[CheckResult, str | None]:
        """Run a single boolean assessment. Thread-safe (no shared mutation)."""
        prompt = BOOLEAN_ASSESSOR_PROMPT.format(
            conversation=conversation_text, statement=statement,
        )
        result = self.client.run_sre_inline(prompt, BOOLEAN_PARAMS)
        is_true = result.get("result", False)
        reasoning = result.get("reasoning", "")

        kind = "assess_true" if expect_true else "assess_false"
        name = f"{kind}({statement!r})"
        passed = is_true if expect_true else not is_true

        status = CheckStatus.PASSED if passed else CheckStatus.FAILED
        check = CheckResult(
            check_type=CheckType.ASSESS, name=name, status=status, reasoning=reasoning,
        )
        failure_msg = None if passed else f"{name}: {reasoning}"
        return check, failure_msg

    def _run_score_assessment(
        self, conversation_text: str, criteria: str,
        *, min_score: float,
    ) -> tuple[CheckResult, str | None]:
        """Run a single score assessment. Thread-safe (no shared mutation)."""
        prompt = SCORE_ASSESSOR_PROMPT.format(
            conversation=conversation_text, criteria=criteria,
        )
        result = self.client.run_sre_inline(prompt, SCORE_PARAMS)
        score = float(result.get("score", 0.0))
        reasoning = result.get("reasoning", "")

        name = f"assess_score({criteria!r}, min={min_score})"
        passed = score >= min_score

        status = CheckStatus.PASSED if passed else CheckStatus.FAILED
        check = CheckResult(
            check_type=CheckType.ASSESS, name=name, status=status,
            reasoning=reasoning, score=score,
        )
        failure_msg = None if passed else f"{name} (score={score:.2f}): {reasoning}"
        return check, failure_msg

    # ── extract (ad-hoc structured extraction) ───────────────────

    def extract(self, parameters: list[dict], prompt: str | None = None) -> dict:
        """Run an ad-hoc structured extraction against the conversation.

        Uses the inline SRE endpoint. If no prompt is provided, uses a
        default extraction prompt with the conversation transcript.

        Args:
            parameters: Parameter definition dicts (use Param helpers).
            prompt: Optional custom prompt. The conversation transcript is
                always available — if you omit prompt, a default is used.

        Returns:
            Dict with fields matching the parameter definitions.
        """
        self._ensure_transcript()
        conversation_text = self.get_transcript_text()

        if prompt is None:
            full_prompt = EXTRACT_PROMPT_TEMPLATE.format(conversation=conversation_text)
        else:
            full_prompt = prompt.format(conversation=conversation_text)

        return self.client.run_sre_inline(full_prompt, parameters)

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
