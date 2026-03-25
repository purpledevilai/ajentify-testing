# Ajentify Testing Framework

Test any Ajentify agent with simulated conversations, deterministic assertions, and LLM-powered assessments.

## Quick Start

```bash
# 1. Clone and install
cd ajentify-testing
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your Ajentify credentials and target agent ID

# 3. Run tests
python run_tests.py
```

## Core Concepts

### SimAgent

A throwaway agent that role-plays a user scenario against your agent under test. It automatically includes an `end_test` tool so the conversation terminates naturally.

```python
sim = SimAgent(
    session,
    persona="A first home buyer asking about a property in Richmond",
    first_message="Hi, what's the price guide for 42 Smith St?",
)
```

### TargetContext

A wrapper around your agent's Ajentify context. Exposes two families of checks:

- **`assert_*()`** — Deterministic, instant, zero-cost. Inspects the conversation data directly.
- **`assess_*()`** — LLM-powered via SRE. Handles subjective/semantic checks.

```python
target = TargetContext(
    session,
    agent_id=session.env("AGENT_ID"),
    prompt_args={"region": "VIC"},
    user_defined={"api_url": "https://..."},
)
```

### run_conversation()

Orchestrates the multi-turn dialogue between a SimAgent and TargetContext. Ends when the sim agent calls `end_test`, `max_turns` is reached, or either side returns an empty message.

```python
run_conversation(sim, target, max_turns=15)
```

## Writing Tests

Each test is a Python file in `tests/` with three exports:

```python
# tests/test_property_lookup.py
from ajentify_testing import SimAgent, TargetContext, run_conversation

name = "property_lookup"
description = "Agent should look up property details when asked"

def run(session):
    sim = SimAgent(session,
        persona="A buyer asking about a 3-bedroom house",
        first_message="What's the price for 42 Smith St in Richmond?",
    )

    target = TargetContext(session,
        agent_id=session.env("AGENT_ID"),
        prompt_args={"region": "VIC"},
    )

    run_conversation(sim, target, max_turns=15)

    # Deterministic assertions
    target.assert_called_tool("lookup_property")
    target.assert_called_tool("lookup_property", with_params={"suburb": "Richmond"})
    target.assert_turn_count(max=10)

    # LLM-powered assessments
    target.assess_true("Gave the user a price guide")
    target.assess_false("Offered a private inspection without being asked")
    target.assess_score("Followed professional sales approach", min=0.7)
```

Resource cleanup is handled automatically by the framework. SimAgents and TargetContexts are cleaned up after each test, even if assertions fail.

## Assertion Reference

### assert_* (Deterministic)

| Method | Description |
|--------|-------------|
| `assert_called_tool(name)` | Agent called the named tool |
| `assert_called_tool(name, with_params={...})` | Agent called the tool with matching params |
| `assert_not_called_tool(name)` | Agent did NOT call the named tool |
| `assert_message_contains(text)` | At least one agent message contains the text |
| `assert_message_not_contains(text)` | No agent message contains the text |
| `assert_turn_count(min=N, max=N)` | Conversation length is within bounds |

### assess_* (LLM-Powered)

| Method | Description |
|--------|-------------|
| `assess_true(statement)` | Statement is true about the conversation |
| `assess_false(statement)` | Statement is false about the conversation |
| `assess_score(criteria, min=0.7)` | Score (0.0–1.0) meets minimum threshold |

## Running Tests

```bash
# Run all tests in parallel
python run_tests.py

# Run a single test by name
python run_tests.py --test property_lookup

# Specify a custom tests directory
python run_tests.py --tests-dir path/to/my/tests
```

Results are printed to the console and saved as a markdown report in `results/`.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AJENTIFY_BASE_URL` | Yes | Ajentify API base URL |
| `AJENTIFY_API_KEY` | Yes | Your Ajentify API key |
| `AGENT_ID` | No | Default agent ID (read via `session.env("AGENT_ID")`) |

Access any env var in tests with `session.env("MY_VAR")`.

## Project Structure

```
ajentify-testing/
  ajentify_testing/        # Framework package
    __init__.py            # Public API
    client.py              # Ajentify HTTP client
    session.py             # TestSession (shared SREs, end_test tool)
    sim_agent.py           # SimAgent
    target_context.py      # TargetContext with assert_*/assess_*
    conversation.py        # run_conversation()
    models.py              # TestResult, CheckResult
    exceptions.py          # AssertionFailed, AssessmentFailed
    runner.py              # Discovery, parallel execution, reporting
  tests/                   # Your test files
    test_example.py        # Example test
  results/                 # Generated markdown reports
  run_tests.py             # CLI entry point
```
