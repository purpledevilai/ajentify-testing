"""
Test discovery, parallel execution, and reporting.
"""

import argparse
import importlib
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from ajentify_testing.session import TestSession
from ajentify_testing.models import TestResult, CheckStatus
from ajentify_testing.target_context import _format_conversation
from ajentify_testing.exceptions import TestFailure

TESTS_DIR = Path.cwd() / "tests"

_print_lock = threading.Lock()


def _log(msg: str):
    with _print_lock:
        print(msg, flush=True)


# ── Discovery ───────────────────────────────────────────────────

def discover_tests(tests_dir: Path | None = None) -> dict[str, dict]:
    """Import all test_*.py modules and collect {name, description, run}.

    Each test module must export:
      - name: str
      - description: str
      - run: Callable[[TestSession], None]
    """
    directory = tests_dir or TESTS_DIR
    tests: dict[str, dict] = {}

    for test_file in sorted(directory.glob("test_*.py")):
        module_name = f"tests.{test_file.stem}"
        module = importlib.import_module(module_name)

        if not (hasattr(module, "name") and hasattr(module, "run")):
            continue

        tests[module.name] = {
            "name": module.name,
            "description": getattr(module, "description", ""),
            "run": module.run,
            "module": module_name,
        }

    return tests


# ── Single test execution ───────────────────────────────────────

def _run_one(
    session: TestSession,
    test_info: dict,
    progress: dict,
) -> TestResult:
    name = test_info["name"]
    description = test_info["description"]
    run_fn = test_info["run"]

    _log(f"[STARTED]  {name}")
    start = time.time()

    passed = False
    error = ""

    try:
        run_fn(session)
        passed = True

    except TestFailure as exc:
        error = str(exc)

    except Exception as exc:
        error = f"Unexpected error: {exc}"

    finally:
        elapsed = time.time() - start

        resources = session._get_thread_resources()
        all_checks: list = []
        conversation: list[dict] = []
        turn_count = 0

        for obj in resources:
            if hasattr(obj, "checks"):
                all_checks.extend(obj.checks)
            if hasattr(obj, "messages") and obj.messages:
                conversation = obj.messages
                turn_count = getattr(obj, "turn_count", 0)
            obj.cleanup()

        session._clear_thread_resources()

    status = "PASS" if passed else ("FAIL" if "Unexpected" not in error else "ERROR")
    _report(name, status, elapsed, progress)

    return TestResult(
        test_name=name,
        description=description,
        passed=passed,
        checks=all_checks,
        conversation=conversation,
        turn_count=turn_count,
        elapsed_seconds=elapsed,
        error=error,
    )


def _report(name: str, status: str, elapsed: float, progress: dict):
    with progress["lock"]:
        progress["completed"] += 1
        done = progress["completed"]
        total = progress["total"]
    _log(f"[{status}]  {name} ({elapsed:.1f}s) — {done}/{total} complete")


# ── Summary ─────────────────────────────────────────────────────

def build_summary(results: list[TestResult]) -> str:
    if not results:
        return "No tests run."

    name_w = max(len(r.test_name) for r in results) + 2

    header = f"{'Test':<{name_w}} {'Result':>6}  {'Time':>6}  Error"
    lines = [header, "-" * (len(header) + 30)]

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        error = r.error or ""
        lines.append(f"{r.test_name:<{name_w}} {status:>6}  {r.elapsed_seconds:>5.1f}s  {error}")

    passed = sum(1 for r in results if r.passed)
    lines.append("-" * (len(header) + 30))
    lines.append(f"Total: {passed}/{len(results)} passed")

    return "\n".join(lines)


# ── Markdown report ─────────────────────────────────────────────

def save_results(results: list[TestResult], results_dir: Path | None = None) -> Path:
    out_dir = results_dir or (Path.cwd() / "results")
    out_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = out_dir / f"results_{timestamp}.md"

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    failures = [r for r in results if not r.passed]
    passes = [r for r in results if r.passed]
    ordered = failures + passes

    md: list[str] = []
    md.append("# Test Results")
    md.append("")
    md.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"**Result:** {passed}/{total} passed")
    md.append("")

    md.append("## Summary")
    md.append("")
    md.append("| Test | Result | Time | Checks | Error |")
    md.append("|------|--------|------|--------|-------|")
    for r in ordered:
        status = "PASS" if r.passed else "FAIL"
        checks_passed = sum(1 for c in r.checks if c.status == CheckStatus.PASSED)
        checks_total = len(r.checks)
        checks_str = f"{checks_passed}/{checks_total}" if checks_total else "-"
        error_raw = r.error or ""
        error_line = error_raw.split("\n")[0]
        if len(error_line) > 80:
            error_line = error_line[:77] + "..."
        error = error_line.replace("|", "\\|")
        md.append(f"| {r.test_name} | {status} | {r.elapsed_seconds:.1f}s | {checks_str} | {error} |")
    md.append("")

    for r in ordered:
        status = "PASS" if r.passed else "FAIL"
        md.append("---")
        md.append("")
        md.append(f"## {r.test_name} — {status} ({r.elapsed_seconds:.1f}s)")
        md.append("")

        if r.description:
            md.append(f"**Description:** {r.description}")
            md.append("")

        if r.error:
            md.append(f"**Error:** {r.error}")
            md.append("")

        if r.checks:
            md.append("### Checks")
            md.append("")
            for c in r.checks:
                icon = "PASS" if c.status == CheckStatus.PASSED else "FAIL"
                line = f"- **[{icon}]** {c.name}"
                if c.score is not None:
                    line += f" (score: {c.score:.2f})"
                md.append(line)
                if c.detail:
                    md.append(f"  - {c.detail}")
                if c.reasoning:
                    md.append(f"  - *Reasoning:* {c.reasoning}")
            md.append("")

        if r.conversation:
            md.append(f"### Conversation ({r.turn_count} turns)")
            md.append("")
            md.append("```")
            md.append(_format_conversation(r.conversation))
            md.append("```")
            md.append("")

    filepath.write_text("\n".join(md))
    return filepath


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ajentify Testing Framework")
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Run a single test by name",
    )
    parser.add_argument(
        "--tests-dir",
        type=str,
        default=None,
        help="Path to tests directory (default: ./tests)",
    )
    args = parser.parse_args()

    tests_dir = Path(args.tests_dir) if args.tests_dir else None
    all_tests = discover_tests(tests_dir)

    if not all_tests:
        print("No tests discovered. Make sure your tests/ directory has test_*.py files.")
        sys.exit(1)

    if args.test:
        if args.test not in all_tests:
            print(f"Unknown test: {args.test}")
            print(f"Available: {', '.join(all_tests)}")
            sys.exit(1)
        tests_to_run = {args.test: all_tests[args.test]}
    else:
        tests_to_run = all_tests

    session = TestSession()

    progress = {
        "total": len(tests_to_run),
        "completed": 0,
        "lock": threading.Lock(),
    }

    print(f"Running {len(tests_to_run)} test(s) in parallel...\n")

    results: list[TestResult] = []

    try:
        with ThreadPoolExecutor(max_workers=len(tests_to_run)) as executor:
            futures = {
                executor.submit(_run_one, session, info, progress): name
                for name, info in tests_to_run.items()
            }
            for future in as_completed(futures):
                results.append(future.result())

        test_order = list(tests_to_run.keys())
        results.sort(key=lambda r: test_order.index(r.test_name))

        summary = build_summary(results)
        print("\n" + "=" * 72)
        print("TEST RESULTS")
        print("=" * 72)
        print(summary)
        print()

        filepath = save_results(results)
        print(f"Results saved to {filepath}")

    finally:
        session.cleanup()

    all_passed = all(r.passed for r in results)
    sys.exit(0 if all_passed else 1)
