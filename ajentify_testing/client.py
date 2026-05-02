import requests
from typing import Optional


class AjentifyClient:
    """Thin HTTP wrapper around the Ajentify REST API."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": api_key,
        }

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _post(self, path: str, body: dict) -> dict:
        resp = requests.post(self._url(path), headers=self.headers, json=body)
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        resp = requests.get(self._url(path), headers=self.headers, params=params)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str) -> dict:
        resp = requests.delete(self._url(path), headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    # ── Agents ──────────────────────────────────────────────────

    def create_agent(
        self,
        agent_name: str,
        agent_description: str,
        prompt: str,
        *,
        is_public: bool = False,
        agent_speaks_first: bool = False,
        tools: Optional[list[str]] = None,
        uses_prompt_args: bool = False,
        prompt_arg_names: Optional[list[str]] = None,
    ) -> dict:
        body: dict = {
            "agent_name": agent_name,
            "agent_description": agent_description,
            "prompt": prompt,
            "is_public": is_public,
            "agent_speaks_first": agent_speaks_first,
            "uses_prompt_args": uses_prompt_args,
        }
        if tools:
            body["tools"] = tools
        if prompt_arg_names:
            body["prompt_arg_names"] = prompt_arg_names
        return self._post("/agent", body)

    def delete_agent(self, agent_id: str) -> dict:
        return self._delete(f"/agent/{agent_id}")

    # ── Contexts ────────────────────────────────────────────────

    def create_context(
        self,
        agent_id: str,
        *,
        invoke_agent_message: bool = False,
        prompt_args: Optional[dict] = None,
        user_defined: Optional[dict] = None,
    ) -> dict:
        body: dict = {
            "agent_id": agent_id,
            "invoke_agent_message": invoke_agent_message,
        }
        if prompt_args is not None:
            body["prompt_args"] = prompt_args
        if user_defined is not None:
            body["user_defined"] = user_defined
        return self._post("/context", body)

    def get_context(self, context_id: str, *, with_tool_calls: bool = False) -> dict:
        params = {}
        if with_tool_calls:
            params["with_tool_calls"] = "true"
        return self._get(f"/context/{context_id}", params=params)

    def delete_context(self, context_id: str) -> dict:
        return self._delete(f"/context/{context_id}")

    # ── Chat ────────────────────────────────────────────────────

    def chat(self, context_id: str, message: str, *, save_ai_messages: bool = True) -> dict:
        return self._post("/chat", {
            "context_id": context_id,
            "message": message,
            "save_ai_messages": save_ai_messages,
        })

    def invoke(self, context_id: str, *, save_ai_messages: bool = True) -> dict:
        return self._post("/chat/invoke", {
            "context_id": context_id,
            "save_ai_messages": save_ai_messages,
        })

    # ── Parameter Definitions ───────────────────────────────────

    def create_pd(self, schema: dict) -> dict:
        """Create a Parameter Definition from a JSON Schema (Draft 2020-12).

        Build the schema with :func:`ajentify_testing.params._build_object_schema`
        on top of `Param.*` fragments, or pass a hand-rolled schema dict.
        """
        return self._post("/parameter-definition", {"schema": schema})

    def delete_pd(self, pd_id: str) -> dict:
        return self._delete(f"/parameter-definition/{pd_id}")

    # ── Tools ───────────────────────────────────────────────────

    def create_tool(
        self,
        name: str,
        description: str,
        *,
        pd_id: Optional[str] = None,
        code: Optional[str] = None,
        is_client_side_tool: bool = False,
    ) -> dict:
        body: dict = {
            "name": name,
            "description": description,
            "is_client_side_tool": is_client_side_tool,
        }
        if pd_id:
            body["pd_id"] = pd_id
        if code:
            body["code"] = code
        return self._post("/tool", body)

    def delete_tool(self, tool_id: str) -> dict:
        return self._delete(f"/tool/{tool_id}")

    # ── Structured Response Endpoints ───────────────────────────

    def create_sre(
        self,
        name: str,
        description: str,
        pd_id: str,
        prompt_template: str,
        *,
        is_public: bool = False,
        variable_names: Optional[list[str]] = None,
        model_id: Optional[str] = None,
    ) -> dict:
        """Create a Structured Response Endpoint.

        ``variable_names`` is the new-style template substitution: pass the
        explicit list of tokens that appear verbatim in ``prompt_template``
        (e.g. ``["{{conversation}}"]``). The backend does direct string
        replacement at run time. Omit it to fall back to the legacy
        ``{variable}`` regex placeholder behavior.
        """
        body: dict = {
            "name": name,
            "description": description,
            "pd_id": pd_id,
            "prompt_template": prompt_template,
            "is_public": is_public,
        }
        if variable_names is not None:
            body["variable_names"] = variable_names
        if model_id is not None:
            body["model_id"] = model_id
        return self._post("/sre", body)

    def run_sre(self, sre_id: str, args: dict) -> dict:
        return self._post(f"/run-sre/{sre_id}", args)

    def run_sre_inline(self, prompt: str, schema: dict, model: Optional[str] = None) -> dict:
        """Run a one-off structured extraction.

        ``schema`` must be a complete JSON Schema (Draft 2020-12) describing the
        desired output object — typically built via
        :func:`ajentify_testing.params._build_object_schema`.
        """
        body: dict = {"prompt": prompt, "schema": schema}
        if model:
            body["model"] = model
        return self._post("/run-sre", body)

    def delete_sre(self, sre_id: str) -> dict:
        return self._delete(f"/sre/{sre_id}")
