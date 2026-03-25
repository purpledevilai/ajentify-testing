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

    Flow:
      1. Target agent's greeting is sent to the sim agent.
      2. Sim agent responds (or uses its first_message on the first turn).
      3. Sim's response is sent to the target agent.
      4. Repeat until the sim agent calls end_test, max_turns is reached,
         or either side returns an empty message.

    After completion, the target's transcript is loaded and turn_count is set.

    Args:
        sim: The simulated user agent.
        target: The agent under test.
        max_turns: Maximum conversation turns before stopping.
    """
    current_target_message = target.greeting
    turn_count = 0

    for turn in range(max_turns):
        turn_count = turn + 1

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
