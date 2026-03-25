from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ajentify_testing.sim_agent import SimAgent
    from ajentify_testing.target_context import TargetContext

DEFAULT_MAX_TURNS = 20


def run_conversation(
    sim: SimAgent,
    target: TargetContext,
    max_turns: int = DEFAULT_MAX_TURNS,
):
    """Orchestrate a multi-turn conversation between a SimAgent and a TargetContext.

    If the target agent has agent_speaks_first=True, the flow is:
      1. Target's greeting is sent to the sim agent.
      2. Sim responds, its response is sent back to the target.
      3. Repeat until end_test, max_turns, or empty message.

    If the target agent does NOT speak first (no greeting), the sim agent
    initiates using its first_message or an invoke, and the first message
    goes to the target to start the loop.

    After completion, the target's transcript is loaded and turn_count is set.

    Args:
        sim: The simulated user agent.
        target: The agent under test.
        max_turns: Maximum conversation turns before stopping.
    """
    turn_count = 0

    if target.greeting:
        current_target_message = target.greeting
    else:
        # Agent doesn't speak first — invoke the sim agent to produce the
        # opening message. If first_message was set, the sim's prompt
        # instructs it to use that. Invoking ensures the message is recorded
        # in the sim's context history.
        sim_resp = sim.client.invoke(sim.context_id)
        opening = sim_resp.get("response", "")

        if not opening:
            target.turn_count = 0
            target.load_transcript()
            return

        turn_count = 1
        target_resp = target.client.chat(target.context_id, opening)
        current_target_message = target_resp.get("response", "")

        if not current_target_message:
            target.turn_count = turn_count
            target.load_transcript()
            return

    for turn in range(max_turns):
        turn_count += 1

        sim_resp = sim.respond(current_target_message)

        cst_calls = sim_resp.get("client_side_tool_calls") or []
        end_call = next((c for c in cst_calls if c["tool_name"] == "end_test"), None)
        if end_call:
            break

        sim_message = sim_resp.get("response", "")
        if not sim_message:
            break

        target_resp = target.client.chat(target.context_id, sim_message)
        current_target_message = target_resp.get("response", "")

        if not current_target_message:
            break

    target.turn_count = turn_count
    target.load_transcript()
