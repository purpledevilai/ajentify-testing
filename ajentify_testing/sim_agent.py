from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ajentify_testing.session import TestSession

SIM_AGENT_PREAMBLE = (
    "You are a simulated user in an automated test. Your job is to play the role "
    "of a real person having a natural conversation with an AI assistant.\n\n"
    "CRITICAL RULES:\n"
    "- You are the USER. You ask questions and describe your situation. You do NOT "
    "act as an assistant.\n"
    "- NEVER say things like 'How can I help you?' or 'What would you like to know?' "
    "— those are assistant phrases.\n"
    "- Once the assistant gives you a clear, complete answer, call the end_test tool "
    "with a brief summary of the conversation.\n"
    "- If the assistant asks a clarifying question, respond naturally.\n\n"
)


class SimAgent:
    """A throwaway agent that role-plays a user scenario.

    The SimAgent automatically gets the session's end_test client-side tool
    and includes a standard preamble instructing it to behave like a real user.

    Args:
        session: The active TestSession.
        persona: Description of who the simulated user is and what they want.
        first_message: Optional explicit first message. If provided, this is sent
            as the opening message instead of letting the agent generate one.
    """

    def __init__(
        self,
        session: TestSession,
        persona: str,
        first_message: str | None = None,
    ):
        self.session = session
        self.client = session.client
        self.persona = persona
        self.first_message = first_message

        self.agent_id: str | None = None
        self.context_id: str | None = None

        prompt = SIM_AGENT_PREAMBLE + f"Your persona:\n{persona}\n"
        if first_message:
            prompt += (
                f"\nYour first message MUST be exactly or very close to:\n"
                f'"{first_message}"\n'
            )

        agent = self.client.create_agent(
            agent_name="sim-agent",
            agent_description="Automated simulated user for testing",
            prompt=prompt,
            is_public=False,
            agent_speaks_first=False,
            tools=[session.end_test_tool_id],
        )
        self.agent_id = agent["agent_id"]

        ctx = self.client.create_context(agent_id=self.agent_id)
        self.context_id = ctx["context_id"]

        session._register_resource(self)

    def respond(self, message: str) -> dict:
        """Send a message to the sim agent and return the full API response."""
        return self.client.chat(self.context_id, message)

    def cleanup(self):
        """Delete the agent's context and agent from Ajentify."""
        if self.context_id:
            try:
                self.client.delete_context(self.context_id)
            except Exception:
                pass
            self.context_id = None

        if self.agent_id:
            try:
                self.client.delete_agent(self.agent_id)
            except Exception:
                pass
            self.agent_id = None
