import os
import threading
from pathlib import Path
from dotenv import load_dotenv
from ajentify_testing.client import AjentifyClient


# ── Prompt constants ─────────────────────────────────────────────

END_TEST_TOOL_DESCRIPTION = (
    "Call this tool when the assistant has given you a clear, complete answer "
    "to your question, OR when the conversation has reached a natural conclusion. "
    "Provide a summary of what happened."
)

END_TEST_SUMMARY_PARAM_DESCRIPTION = (
    "A brief summary of the conversation: what was discussed, "
    "what actions the assistant took, and any notable observations."
)


class TestSession:
    """Manages shared Ajentify resources for a test run.

    Creates once-per-session:
      - end_test client-side tool (SimAgents use this to signal conversation end)

    Assessments (assess_true, assess_false, assess_score) use the inline
    SRE endpoint directly — no pre-created SREs or PDs needed.

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

        self.base_url = "https://api.ajentify.com"
        self.api_key = os.environ["AJENTIFY_API_KEY"]
        self.client = AjentifyClient(self.base_url, self.api_key)

        self._resource_stack: list[tuple[str, str]] = []
        self._thread_targets: dict[int, list] = {}
        self._targets_lock = threading.Lock()

        self.end_test_tool_id: str = ""

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

        print(f"  end_test tool: {self.end_test_tool_id}")
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
