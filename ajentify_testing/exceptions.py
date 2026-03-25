class TestFailure(Exception):
    """Base class for all test failures."""


class AssertionFailed(TestFailure):
    """A deterministic assertion (assert_*) failed."""

    def __init__(self, check: str, detail: str = ""):
        self.check = check
        self.detail = detail
        msg = f"Assertion failed: {check}"
        if detail:
            msg += f" — {detail}"
        super().__init__(msg)


class AssessmentFailed(TestFailure):
    """An LLM-powered assessment (assess_*) failed."""

    def __init__(self, statement: str, reasoning: str = ""):
        self.statement = statement
        self.reasoning = reasoning
        msg = f"Assessment failed: {statement}"
        if reasoning:
            msg += f"\n  Reasoning: {reasoning}"
        super().__init__(msg)
