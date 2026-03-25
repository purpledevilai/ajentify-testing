"""
Example test — demonstrates all framework features.

Replace AGENT_ID in your .env with the agent you want to test,
then customise this file or create new test_*.py files alongside it.
"""

from ajentify_testing import SimAgent, TargetContext, run_conversation

name = "example_greeting"
description = "Verify the agent greets the user and responds to a basic question"


def run(session):
    # 1. Create a simulated user
    sim = SimAgent(
        session,
        persona="A curious new user exploring the assistant for the first time",
        first_message="Hi there! What can you help me with?",
    )

    # 2. Create a context for the agent under test
    target = TargetContext(
        session,
        agent_id=session.env("AGENT_ID"),
    )

    # 3. Run the conversation (SimAgent talks to the target agent)
    run_conversation(sim, target, max_turns=10)

    # 4. Deterministic assertions — instant, zero-cost checks
    target.assert_turn_count(min=1, max=10)

    # 5. LLM-powered assessments — semantic checks via SRE
    target.assess_true("The agent introduced itself or described what it can do")
    target.assess_score("The agent was friendly and helpful", min=0.6)
