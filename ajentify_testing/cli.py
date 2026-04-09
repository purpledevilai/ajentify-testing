"""
CLI entry point for ajentify-test.

Usage:
    ajentify-test                       # run all tests
    ajentify-test --test my_test        # run a single test
    ajentify-test init                  # scaffold tests/, .env.example
    ajentify-test init --with-prompts   # also create ajentify_prompts.py
"""

import argparse
import sys
import textwrap
from pathlib import Path

from ajentify_testing import prompts as _prompts


# ── Templates ───────────────────────────────────────────────────

_ENV_EXAMPLE = textwrap.dedent("""\
    AJENTIFY_API_KEY=your-api-key-here

    # The agent you want to test
    AGENT_ID=your-agent-id-here
""")

_TEST_EXAMPLE = textwrap.dedent('''\
    """
    Example test — demonstrates basic framework features.

    Replace AGENT_ID in your .env with the agent you want to test,
    then customise this file or create new test_*.py files alongside it.
    """

    from ajentify_testing import SimAgent, TargetContext, run_conversation

    name = "example_greeting"
    description = "Verify the agent greets the user and responds to a basic question"


    def run(session):
        sim = SimAgent(
            session,
            persona="A curious new user exploring the assistant for the first time",
            first_message="Hi there! What can you help me with?",
        )

        target = TargetContext(
            session,
            agent_id=session.env("AGENT_ID"),
        )

        run_conversation(sim, target, max_turns=10)

        target.assert_turn_count(min=1, max=10)

        target.assess_true("The agent introduced itself or described what it can do")
        target.assess_score("The agent was friendly and helpful", min=0.6)
''')

_PROMPTS_HEADER = textwrap.dedent("""\
    \"\"\"
    Ajentify Testing — Prompt Overrides

    Any variable defined here overrides the framework default.
    Delete or comment out lines you don't want to customise.
    \"\"\"

""")


def _generate_prompts_file() -> str:
    """Build the contents of an ajentify_prompts.py with all current defaults."""
    lines = [_PROMPTS_HEADER]

    for name in _prompts._OVERRIDABLE:
        value = getattr(_prompts, name)
        lines.append(f"{name} = (")
        for part in value.split("\n"):
            lines.append(f"    {part!r}")
        lines.append(")\n\n")

    return "\n".join(lines)


# ── Commands ────────────────────────────────────────────────────

def _cmd_init(args: argparse.Namespace) -> None:
    """Scaffold a test directory in the current working directory."""
    cwd = Path.cwd()

    tests_dir = cwd / "tests"
    tests_dir.mkdir(exist_ok=True)

    init_file = tests_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("")
        print(f"  Created {init_file.relative_to(cwd)}")

    example = tests_dir / "test_example.py"
    if not example.exists():
        example.write_text(_TEST_EXAMPLE)
        print(f"  Created {example.relative_to(cwd)}")
    else:
        print(f"  Skipped {example.relative_to(cwd)} (already exists)")

    env_example = cwd / ".env.example"
    if not env_example.exists():
        env_example.write_text(_ENV_EXAMPLE)
        print(f"  Created {env_example.relative_to(cwd)}")
    else:
        print(f"  Skipped {env_example.relative_to(cwd)} (already exists)")

    env_file = cwd / ".env"
    if not env_file.exists():
        env_file.write_text(_ENV_EXAMPLE)
        print(f"  Created {env_file.relative_to(cwd)} (fill in your credentials)")
    else:
        print(f"  Skipped {env_file.relative_to(cwd)} (already exists)")

    if args.with_prompts:
        prompts_file = cwd / "ajentify_prompts.py"
        if not prompts_file.exists():
            prompts_file.write_text(_generate_prompts_file())
            print(f"  Created {prompts_file.relative_to(cwd)}")
        else:
            print(f"  Skipped {prompts_file.relative_to(cwd)} (already exists)")

    print("\nDone! Next steps:")
    print("  1. Edit .env with your Ajentify credentials")
    print("  2. Write tests in tests/test_*.py")
    print("  3. Run: ajentify-test")


def _cmd_run(args: argparse.Namespace) -> None:
    """Run tests (delegates to the existing runner)."""
    from ajentify_testing.runner import main as runner_main
    runner_main()


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="ajentify-test",
        description="Ajentify Testing Framework — test any Ajentify agent",
    )
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser(
        "init",
        help="Scaffold a test directory in the current project",
    )
    init_parser.add_argument(
        "--with-prompts",
        action="store_true",
        help="Also create ajentify_prompts.py with default prompts for customisation",
    )

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

    if args.command == "init":
        _cmd_init(args)
    else:
        _cmd_run(args)
