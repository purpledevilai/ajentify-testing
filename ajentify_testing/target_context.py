from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, TYPE_CHECKING

from ajentify_testing.models import CheckResult, CheckType, CheckStatus
from ajentify_testing.exceptions import AssertionFailed, AssessmentFailed, TestFailure
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

BOOLEAN_PARAMS = [
    Param.boolean("result", "Whether the statement is true"),
    Param.string("reasoning", "Explanation of why the statement is true or false"),
]

SCORE_PARAMS = [
    Param.number("score", "Score from 0.0 to 1.0"),
    Param.string("reasoning", "Explanation of the score"),
]


# ── Check descriptor classes (for check_all) ─────────────────────

class AssertCalledTool:
    """Descriptor for check_all: agent called a specific tool."""
    def __init__(self, tool_name: str, with_params: dict | None = None):
        self.tool_name = tool_name
        self.with_params = with_params


class AssertNotCalledTool:
    """Descriptor for check_all: agent did NOT call a specific tool."""
    def __init__(self, tool_name: str):
        self.tool_name = tool_name


class AssertMessageContains:
    """Descriptor for check_all: at least one agent message contains text."""
    def __init__(self, substring: str):
        self.substring = substring


class AssertMessageNotContains:
    """Descriptor for check_all: no agent message contains text."""
    def __init__(self, substring: str):
        self.substring = substring


class AssertTurnCount:
    """Descriptor for check_all: conversation turn count within bounds."""
    def __init__(self, *, min: int | None = None, max: int | None = None):
        self.min = min
        self.max = max


class AssessTrue:
    """Descriptor for check_all: statement should be true (LLM-powered)."""
    def __init__(self, statement: str):
        self.statement = statement


class AssessFalse:
    """Descriptor for check_all: statement should be false (LLM-powered)."""
    def __init__(self, statement: str):
        self.statement = statement


class AssessScore:
    """Descriptor for check_all: criteria should meet a minimum score (LLM-powered)."""
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
      - check_all()              — run mixed assert/assess checks, collecting all results
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

    def assess_all(self, checks: list) -> None:
        """Evaluate all assessments in a single LLM call, collecting all results before failing.

        Each assertion is returned as an object with a result enum ("PASS"/"FAIL")
        and a reasoning string:

            {
                "assertion_1": {"result": "PASS", "reasoning": "..."},
                "assertion_2": {"result": "FAIL", "reasoning": "..."},
                ...
            }

        Args:
            checks: List of AssessTrue, AssessFalse, or AssessScore descriptors.

        Raises:
            TestFailure: If any assertion failed, with a combined summary.
        """
        self._ensure_transcript()
        conversation_text = self.get_transcript_text()

        _RESULT_ENUM = [
            {"name": "PASS", "description": "The assertion passed", "type": "string"},
            {"name": "FAIL", "description": "The assertion failed", "type": "string"},
        ]

        assertion_lines: list[str] = []
        parameters: list[dict] = []

        for i, check in enumerate(checks):
            n = i + 1
            if isinstance(check, AssessTrue):
                line = f"{n}. [TRUE] {check.statement}"
                obj_desc = f"Evaluation of assertion {n}: passes if TRUE — {check.statement}"
            elif isinstance(check, AssessFalse):
                line = f"{n}. [FALSE] {check.statement}"
                obj_desc = f"Evaluation of assertion {n}: passes if FALSE (does not apply) — {check.statement}"
            elif isinstance(check, AssessScore):
                line = f"{n}. [SCORE >= {check.min}] {check.criteria}"
                obj_desc = f"Evaluation of assertion {n}: passes if quality score >= {check.min} — {check.criteria}"
            else:
                raise TypeError(
                    f"assess_all only accepts AssessTrue, AssessFalse, AssessScore — got {type(check)}"
                )

            assertion_lines.append(line)
            parameters.append({
                "name": f"assertion_{n}",
                "description": obj_desc,
                "type": "object",
                "parameters": [
                    {
                        "name": "result",
                        "description": "PASS if the assertion passed, FAIL if it did not",
                        "type": "enum",
                        "parameters": _RESULT_ENUM,
                    },
                    {
                        "name": "reasoning",
                        "description": "Explanation of why this assertion passed or failed",
                        "type": "string",
                        "parameters": [],
                    },
                ],
            })

        prompt = ASSESS_ALL_PROMPT_TEMPLATE.format(
            assertions="\n".join(assertion_lines),
            conversation=conversation_text,
        )

        result = self.client.run_sre_inline(prompt, parameters)

        failures: list[str] = []
        for i, check in enumerate(checks):
            n = i + 1
            assertion_result = result.get(f"assertion_{n}") or {}
            passed = assertion_result.get("result", "FAIL") == "PASS"
            reasoning = assertion_result.get("reasoning", "")

            if isinstance(check, AssessTrue):
                name = f"assess_true({check.statement!r})"
            elif isinstance(check, AssessFalse):
                name = f"assess_false({check.statement!r})"
            else:
                name = f"assess_score({check.criteria!r}, min={check.min})"

            status = CheckStatus.PASSED if passed else CheckStatus.FAILED
            self.checks.append(CheckResult(
                check_type=CheckType.ASSESS,
                name=name,
                status=status,
                reasoning=reasoning,
            ))

            if not passed:
                failures.append(f"{name}: {reasoning}")

        if failures:
            raise TestFailure(f"{len(failures)} assessment(s) failed: {'; '.join(failures)}")

    # ── check_all (batch assert + assess, no early exit) ───────────

    _ASSERT_TYPES = (
        AssertCalledTool, AssertNotCalledTool,
        AssertMessageContains, AssertMessageNotContains, AssertTurnCount,
    )
    _ASSESS_TYPES = (AssessTrue, AssessFalse, AssessScore)

    def check_all(self, checks: list):
        """Run a mix of assert and assess checks, collecting all results before failing.

        Deterministic asserts run instantly. LLM-powered assessments run in
        parallel. No check short-circuits — every check in the list is
        executed regardless of earlier failures.

        Args:
            checks: List of descriptor objects — any combination of
                AssertCalledTool, AssertNotCalledTool, AssertMessageContains,
                AssertMessageNotContains, AssertTurnCount, AssessTrue,
                AssessFalse, AssessScore.

        Raises:
            TestFailure: If any check in the batch failed, with a summary
                of all failures.
        """
        self._ensure_transcript()
        conversation_text = self.get_transcript_text()

        assert_checks = [(i, c) for i, c in enumerate(checks) if isinstance(c, self._ASSERT_TYPES)]
        assess_checks = [(i, c) for i, c in enumerate(checks) if isinstance(c, self._ASSESS_TYPES)]

        indexed: dict[int, tuple[CheckResult, str | None]] = {}

        for i, check in assert_checks:
            indexed[i] = self._run_assert_check(check)

        if assess_checks:
            with ThreadPoolExecutor(max_workers=len(assess_checks)) as executor:
                futures = {
                    executor.submit(self._run_assess_check, c, conversation_text): i
                    for i, c in assess_checks
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    indexed[idx] = future.result()

        failures: list[str] = []
        for i in range(len(checks)):
            result, failure_msg = indexed[i]
            self.checks.append(result)
            if failure_msg is not None:
                failures.append(failure_msg)

        if failures:
            summary = "; ".join(failures)
            raise TestFailure(f"{len(failures)} check(s) failed: {summary}")

    def _run_assert_check(self, check) -> tuple[CheckResult, str | None]:
        """Run a single assert descriptor. Returns (CheckResult, failure_msg | None)."""
        if isinstance(check, AssertCalledTool):
            return self._check_called_tool(check.tool_name, check.with_params)
        elif isinstance(check, AssertNotCalledTool):
            return self._check_not_called_tool(check.tool_name)
        elif isinstance(check, AssertMessageContains):
            return self._check_message_contains(check.substring, expect=True)
        elif isinstance(check, AssertMessageNotContains):
            return self._check_message_contains(check.substring, expect=False)
        elif isinstance(check, AssertTurnCount):
            return self._check_turn_count(check.min, check.max)

    def _run_assess_check(self, check, conversation_text: str) -> tuple[CheckResult, str | None]:
        """Run a single assess descriptor. Thread-safe."""
        if isinstance(check, AssessTrue):
            return self._run_boolean_assessment(conversation_text, check.statement, expect_true=True)
        elif isinstance(check, AssessFalse):
            return self._run_boolean_assessment(conversation_text, check.statement, expect_true=False)
        elif isinstance(check, AssessScore):
            return self._run_score_assessment(conversation_text, check.criteria, min_score=check.min)

    def _check_called_tool(self, tool_name: str, with_params: dict | None) -> tuple[CheckResult, str | None]:
        matching = [
            m for m in self.messages
            if (m.get("type") or m.get("sender")) == "tool_call"
            and m.get("tool_name") == tool_name
        ]
        if not matching:
            name = f"assert_called_tool({tool_name!r})"
            detail = f"Tool '{tool_name}' was never called"
            return CheckResult(check_type=CheckType.ASSERT, name=name, status=CheckStatus.FAILED, detail=detail), detail

        if with_params is not None:
            name = f"assert_called_tool({tool_name!r}, with_params={with_params!r})"
            for call in matching:
                tool_input = call.get("tool_input", {})
                if all(tool_input.get(k) == v for k, v in with_params.items()):
                    return CheckResult(check_type=CheckType.ASSERT, name=name, status=CheckStatus.PASSED), None
            actual = [c.get("tool_input", {}) for c in matching]
            detail = f"Tool called but params didn't match. Actual: {actual}"
            return CheckResult(check_type=CheckType.ASSERT, name=name, status=CheckStatus.FAILED, detail=detail), detail

        name = f"assert_called_tool({tool_name!r})"
        return CheckResult(check_type=CheckType.ASSERT, name=name, status=CheckStatus.PASSED), None

    def _check_not_called_tool(self, tool_name: str) -> tuple[CheckResult, str | None]:
        name = f"assert_not_called_tool({tool_name!r})"
        called = any(
            (m.get("type") or m.get("sender")) == "tool_call" and m.get("tool_name") == tool_name
            for m in self.messages
        )
        if called:
            detail = f"Tool '{tool_name}' was called when it should not have been"
            return CheckResult(check_type=CheckType.ASSERT, name=name, status=CheckStatus.FAILED, detail=detail), detail
        return CheckResult(check_type=CheckType.ASSERT, name=name, status=CheckStatus.PASSED), None

    def _check_message_contains(self, substring: str, *, expect: bool) -> tuple[CheckResult, str | None]:
        kind = "assert_message_contains" if expect else "assert_message_not_contains"
        name = f"{kind}({substring!r})"
        lower = substring.lower()
        found = any(
            (m.get("type") or m.get("sender")) == "ai"
            and lower in (m.get("message", "") or "").lower()
            for m in self.messages
        )
        passed = found if expect else not found
        if not passed:
            detail = f"No agent message contained '{substring}'" if expect else f"An agent message contained '{substring}' when it should not have"
            return CheckResult(check_type=CheckType.ASSERT, name=name, status=CheckStatus.FAILED, detail=detail), detail
        return CheckResult(check_type=CheckType.ASSERT, name=name, status=CheckStatus.PASSED), None

    def _check_turn_count(self, min_val: int | None, max_val: int | None) -> tuple[CheckResult, str | None]:
        name = f"assert_turn_count(min={min_val}, max={max_val})"
        if min_val is not None and self.turn_count < min_val:
            detail = f"Turn count {self.turn_count} < minimum {min_val}"
            return CheckResult(check_type=CheckType.ASSERT, name=name, status=CheckStatus.FAILED, detail=detail), detail
        if max_val is not None and self.turn_count > max_val:
            detail = f"Turn count {self.turn_count} > maximum {max_val}"
            return CheckResult(check_type=CheckType.ASSERT, name=name, status=CheckStatus.FAILED, detail=detail), detail
        return CheckResult(check_type=CheckType.ASSERT, name=name, status=CheckStatus.PASSED), None

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
