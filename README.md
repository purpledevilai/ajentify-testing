# Ajentify Testing Framework

Test any Ajentify agent with simulated conversations, deterministic assertions, and LLM-powered assessments.

## Installation

```bash
pip install ajentify-testing
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/purpledevilai/ajentify-testing.git
```

## Quick Start

```bash
# Scaffold a test directory in your project
cd your-project/
ajentify-test init

# Configure credentials
# Edit .env with your Ajentify API key and agent ID

# Run tests
ajentify-test
```

`ajentify-test init` creates:

```
your-project/
  tests/
    __init__.py
    test_example.py      # Example test to get started
  .env.example           # Credential template
  .env                   # Your credentials (gitignored)
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
- **`assess_*()`** — LLM-powered via inline SRE. Handles subjective/semantic checks.

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
    target.assert_message_contains("Hi, thank you for choosing NewLiving Realestate")

    # LLM-powered assessments (raise on first failure)
    target.assess_true("Gave the user a price guide")
    target.assess_false("Offered a private inspection without being asked")
    target.assess_score("Followed professional sales approach", min=0.7)
```

Resource cleanup is handled automatically by the framework. SimAgents and TargetContexts are cleaned up after each test, even if assertions fail.

## Batch Assessments with assess_all

Individual `assess_*` calls raise on the first failure and each makes its own LLM call. `assess_all` evaluates every assessment in **one single LLM call**, collects all pass/fail results with reasoning, then raises a combined failure if anything didn't pass:

```python
from ajentify_testing import AssessTrue, AssessFalse, AssessScore

target.assess_all([
    AssessTrue("Greeted the user by name"),
    AssessTrue("Provided a price guide"),
    AssessFalse("Offered a private inspection without being asked"),
    AssessScore("Followed professional sales approach", min=0.7),
])
```

The prompt lists every assertion with its type and threshold. The LLM returns a structured object per assertion with a `result` (`"PASS"` / `"FAIL"`) and `reasoning`. If any fail, a single `TestFailure` is raised with a combined summary — every assertion runs regardless of earlier failures.

## Mixed Assert + Assess with check_all

When you want to combine deterministic assertions and LLM assessments in one collected batch, use `check_all`. Deterministic checks run instantly; LLM assessments run in parallel. All results are collected before any failure is raised:

```python
from ajentify_testing import (
    AssertCalledTool, AssertTurnCount,
    AssessTrue, AssessFalse, AssessScore,
)

target.check_all([
    AssertCalledTool("lookup_property", with_params={"suburb": "Richmond"}),
    AssertTurnCount(max=10),
    AssessTrue("Gave the user a price guide"),
    AssessFalse("Offered a private inspection without being asked"),
    AssessScore("Followed professional sales approach", min=0.7),
])
```

Each `Assert*` descriptor maps to an `assert_*` call; each `Assess*` descriptor maps to an `assess_*` call (run in parallel via `ThreadPoolExecutor`).

## Structured Extraction with extract

Use `extract` for ad-hoc structured data extraction from the conversation, powered by the inline SRE endpoint:

```python
from ajentify_testing import Param

result = target.extract([
    Param.boolean("passed", "Whether the agent completed the task"),
    Param.number("score", "Quality score from 0.0 to 1.0"),
    Param.string("summary", "Brief summary of what happened"),
])

if result["passed"]:
    print(f"Score: {result['score']}")
```

You can provide a custom prompt (use `{conversation}` as a placeholder for the transcript):

```python
result = target.extract(
    [Param.string("bin_color", "The bin color the agent recommended")],
    prompt="What bin color did the agent recommend?\n\n{conversation}",
)
```

## Param Helper

Build parameter definitions without writing dicts by hand:

```python
from ajentify_testing import Param

Param.string("name", "Person's name")
Param.number("score", "Quality score")
Param.boolean("passed", "Whether it passed")
Param.array("topics", "Key topics", items=Param.string("topic", "A topic"))
Param.object("person", "A person", children=[
    Param.string("name", "Name"),
    Param.number("age", "Age"),
])
Param.enum("sentiment", "Sentiment", values=["positive", "negative", "neutral"])
```

## Customising Prompts

The framework uses built-in prompts for sim agents, assessors, and extraction. To customise any prompt without modifying the package:

```bash
ajentify-test init --with-prompts
```

This creates an `ajentify_prompts.py` in your project root with all the default prompts. Edit any prompt you want to change — delete lines you don't need to override:

```python
# ajentify_prompts.py

SIM_AGENT_PREAMBLE = (
    "You are a simulated customer testing our support bot. "
    "Be realistic and occasionally confused...\n\n"
)

# Only override what you need — everything else uses built-in defaults
```

Available prompt overrides:

| Variable | Used by | Purpose |
|----------|---------|---------|
| `SIM_AGENT_PREAMBLE` | `SimAgent` | System prompt preamble for simulated users |
| `BOOLEAN_ASSESSOR_PROMPT` | `assess_true()`, `assess_false()` | Prompt for true/false evaluation |
| `SCORE_ASSESSOR_PROMPT` | `assess_score()` | Prompt for score evaluation |
| `EXTRACT_PROMPT_TEMPLATE` | `extract()` | Default extraction prompt |
| `ASSESS_ALL_PROMPT_TEMPLATE` | `assess_all()` | Batch assessment prompt |
| `END_TEST_TOOL_DESCRIPTION` | `TestSession` | Description for the end_test tool |
| `END_TEST_SUMMARY_PARAM_DESCRIPTION` | `TestSession` | Description for the summary parameter |

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
| `assess_all([...])` | One LLM call evaluates all AssessTrue/False/Score checks, collects all results before failing |

### check_all (Mixed)

| Method | Description |
|--------|-------------|
| `check_all([...])` | Mix of Assert* and Assess* descriptors — asserts run instantly, assessments run in parallel, all results collected before failing |

**Descriptors:** `AssertCalledTool`, `AssertNotCalledTool`, `AssertMessageContains`, `AssertMessageNotContains`, `AssertTurnCount`, `AssessTrue`, `AssessFalse`, `AssessScore`

### extract

| Method | Description |
|--------|-------------|
| `extract(parameters)` | Extract structured data from the conversation |
| `extract(parameters, prompt=...)` | Extract with a custom prompt (use `{conversation}`) |

## Running Tests

```bash
# Run all tests in parallel
ajentify-test

# Run a single test by name
ajentify-test --test property_lookup

# Specify a custom tests directory
ajentify-test --tests-dir path/to/my/tests
```

The legacy `python run_tests.py` entry point still works if you're running from the framework source.

Results are printed to the console and saved as a markdown report in `results/`.

> **Tip:** Add `.env` and `results/` to your `.gitignore` to keep credentials and test output out of version control.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AJENTIFY_API_KEY` | Yes | Your Ajentify API key |
| `AGENT_ID` | No | Default agent ID (read via `session.env("AGENT_ID")`) |

Access any env var in tests with `session.env("MY_VAR")`.

## Project Structure (when using pip)

```
your-project/
  tests/                     # Your test files (committed to your repo)
    __init__.py
    test_example.py
  ajentify_prompts.py        # Optional prompt overrides
  .env                       # Credentials (gitignored)
  .env.example               # Credential template
  requirements.txt           # includes ajentify-testing
  results/                   # Generated markdown reports (gitignored)
```
