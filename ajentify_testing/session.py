import os
import threading
from pathlib import Path
from dotenv import load_dotenv
from ajentify_testing.client import AjentifyClient


# ── Prompt templates ────────────────────────────────────────────

END_TEST_TOOL_DESCRIPTION = (
    "Call this tool when the assistant has given you a clear, complete answer "
    "to your question, OR when the conversation has reached a natural conclusion. "
    "Provide a summary of what happened."
)

END_TEST_SUMMARY_PARAM_DESCRIPTION = (
    "A brief summary of the conversation: what was discussed, "
    "what actions the assistant took, and any notable observations."
)

BOOLEAN_ASSESSOR_PROMPT = """You are a strict test evaluator. You will receive a conversation transcript and a statement to evaluate.

Conversation Transcript:
{conversation}

Statement to evaluate:
{statement}

Determine whether the statement is TRUE or FALSE based solely on evidence in the conversation.
Be strict — only mark as true if the conversation clearly supports it."""

SCORE_ASSESSOR_PROMPT = """You are a strict test evaluator. You will receive a conversation transcript and criteria to score.

Conversation Transcript:
{conversation}

Criteria to score:
{criteria}

Rate how well the criteria was met on a scale of 0.0 to 1.0, where:
- 0.0 = not met at all
- 0.5 = partially met
- 1.0 = fully and excellently met

Be strict and justify your score."""


class TestSession:
    """Manages shared Ajentify resources for a test run.

    Creates once-per-session:
      - end_test client-side tool (SimAgents use this to signal conversation end)
      - Boolean assessor SRE  (powers assess_true / assess_false)
      - Score assessor SRE    (powers assess_score)

    All resources are tracked and torn down on cleanup().
    """

    def __init__(self, env_path: str | None = None):
        path = Path(env_path) if env_path else Path.cwd() / ".env"
        if not path.exists():
            raise FileNotFoundError(
                f"Environment file not found at {path}. "
                f"Copy .env.example to .env and fill in your credentials."
            )
        load_dotenv(path)

        self.base_url = os.environ["AJENTIFY_BASE_URL"]
        self.api_key = os.environ["AJENTIFY_API_KEY"]
        self.client = AjentifyClient(self.base_url, self.api_key)

        self._resource_stack: list[tuple[str, str]] = []
        self._thread_targets: dict[int, list] = {}
        self._targets_lock = threading.Lock()

        self.end_test_tool_id: str = ""
        self.boolean_sre_id: str = ""
        self.score_sre_id: str = ""

        self._setup()

    def env(self, key: str, default: str | None = None) -> str:
        """Read an environment variable (convenience for tests)."""
        val = os.environ.get(key, default)
        if val is None:
            raise KeyError(f"Missing required environment variable: {key}")
        return val

    # ── Setup ───────────────────────────────────────────────────

    def _track(self, resource_type: str, resource_id: str):
        self._resource_stack.append((resource_type, resource_id))

    def _setup(self):
        print("Setting up test session...")

        self._create_end_test_tool()
        self._create_boolean_assessor()
        self._create_score_assessor()

        print(f"  end_test tool:      {self.end_test_tool_id}")
        print(f"  boolean assessor:   {self.boolean_sre_id}")
        print(f"  score assessor:     {self.score_sre_id}")
        print("Session ready.\n")

    def _create_end_test_tool(self):
        pd = self.client.create_pd(parameters=[
            {
                "name": "summary",
                "description": END_TEST_SUMMARY_PARAM_DESCRIPTION,
                "type": "string",
            },
        ])
        self._track("pd", pd["pd_id"])

        tool = self.client.create_tool(
            name="end_test",
            description=END_TEST_TOOL_DESCRIPTION,
            pd_id=pd["pd_id"],
            is_client_side_tool=True,
        )
        self.end_test_tool_id = tool["tool_id"]
        self._track("tool", self.end_test_tool_id)

    def _create_boolean_assessor(self):
        pd = self.client.create_pd(parameters=[
            {"name": "result", "description": "Whether the statement is true", "type": "boolean"},
            {"name": "reasoning", "description": "Explanation of why the statement is true or false", "type": "string"},
        ])
        self._track("pd", pd["pd_id"])

        sre = self.client.create_sre(
            name="ajentify_testing_boolean_assessor",
            description="Evaluates whether a statement is true about a conversation",
            pd_id=pd["pd_id"],
            prompt_template=BOOLEAN_ASSESSOR_PROMPT,
        )
        self.boolean_sre_id = sre["sre_id"]
        self._track("sre", self.boolean_sre_id)

    def _create_score_assessor(self):
        pd = self.client.create_pd(parameters=[
            {"name": "score", "description": "Score from 0.0 to 1.0", "type": "number"},
            {"name": "reasoning", "description": "Explanation of the score", "type": "string"},
        ])
        self._track("pd", pd["pd_id"])

        sre = self.client.create_sre(
            name="ajentify_testing_score_assessor",
            description="Scores how well criteria was met in a conversation",
            pd_id=pd["pd_id"],
            prompt_template=SCORE_ASSESSOR_PROMPT,
        )
        self.score_sre_id = sre["sre_id"]
        self._track("sre", self.score_sre_id)

    # ── Per-test target tracking (used by the runner) ──────────

    def _register_resource(self, obj):
        """Register a TargetContext or SimAgent for the current thread (called automatically).

        The runner will call cleanup() on all registered objects in its finally block,
        so tests don't need to worry about cleanup even if assertions fail mid-test.
        """
        tid = threading.get_ident()
        with self._targets_lock:
            self._thread_targets.setdefault(tid, []).append(obj)

    def _get_thread_resources(self) -> list:
        """Get all registered objects for the current thread."""
        tid = threading.get_ident()
        with self._targets_lock:
            return list(self._thread_targets.get(tid, []))

    def _clear_thread_resources(self):
        """Clear registered resources for the current thread."""
        tid = threading.get_ident()
        with self._targets_lock:
            self._thread_targets.pop(tid, None)

    # ── Cleanup ─────────────────────────────────────────────────

    def cleanup(self):
        print("\nCleaning up test session...")
        deleters = {
            "sre": self.client.delete_sre,
            "tool": self.client.delete_tool,
            "pd": self.client.delete_pd,
        }
        for resource_type, resource_id in reversed(self._resource_stack):
            try:
                deleters[resource_type](resource_id)
            except Exception as exc:
                print(f"  Warning: failed to delete {resource_type} {resource_id}: {exc}")
        self._resource_stack.clear()
        print("Session cleanup complete.")
