from dataclasses import dataclass, field
from enum import Enum


class CheckType(str, Enum):
    ASSERT = "assert"
    ASSESS = "assess"


class CheckStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"


@dataclass
class CheckResult:
    """Result of a single assert_* or assess_* call."""
    check_type: CheckType
    name: str
    status: CheckStatus
    detail: str = ""
    reasoning: str = ""
    score: float | None = None


@dataclass
class TestResult:
    """Full result from running a single test."""
    test_name: str
    description: str
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    conversation: list[dict] = field(default_factory=list)
    turn_count: int = 0
    elapsed_seconds: float = 0.0
    error: str = ""
