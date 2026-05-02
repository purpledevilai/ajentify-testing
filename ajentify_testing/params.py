"""Parameter helpers that emit JSON Schema fragments.

The Ajentify backend now stores Parameter Definitions as canonical JSON
Schema (Draft 2020-12) and the inline SRE endpoint validates against the
same. To keep user tests readable we expose the same `Param.string(...)`
DSL but each helper now returns a small `{name, schema}` carrier dict where
`schema` is a JSON Schema property fragment. Call sites assemble those
fragments into a top-level object schema via `_build_object_schema(...)`
before sending to the API.

Public usage stays identical to the old API:

    Param.string("name", "Person's name")
    Param.boolean("passed", "Whether the test passed")
    Param.number("score", "Quality score from 0.0 to 1.0")
    Param.array("topics", "Key topics", items=Param.string("topic", "A topic"))
    Param.object("person", "A person", children=[
        Param.string("name", "Name"),
        Param.number("age", "Age"),
    ])
    Param.enum("sentiment", "Overall sentiment", values=["positive", "negative", "neutral"])
"""

from typing import Any


class Param:
    """Convenience factory for parameter definition fragments.

    Each helper returns a dict shaped like ``{"name": <str>, "schema": <JSON Schema>}``.
    The ``schema`` value is a valid JSON Schema *property* fragment — combine many
    fragments into a top-level object schema with :func:`_build_object_schema`.
    """

    @staticmethod
    def string(name: str, description: str) -> dict:
        return {"name": name, "schema": {"type": "string", "description": description}}

    @staticmethod
    def number(name: str, description: str) -> dict:
        return {"name": name, "schema": {"type": "number", "description": description}}

    @staticmethod
    def boolean(name: str, description: str) -> dict:
        return {"name": name, "schema": {"type": "boolean", "description": description}}

    @staticmethod
    def object(name: str, description: str, children: list[dict]) -> dict:
        # Mirror the top-level object shape produced by `_build_object_schema`
        # so nested objects look identical to the root. additionalProperties is
        # locked off to nudge the LLM into emitting only the requested fields.
        properties = {child["name"]: child["schema"] for child in children}
        required = [child["name"] for child in children]
        return {
            "name": name,
            "schema": {
                "type": "object",
                "description": description,
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        }

    @staticmethod
    def array(name: str, description: str, items: dict) -> dict:
        # `items` is itself a Param fragment — pull its schema out.
        return {
            "name": name,
            "schema": {
                "type": "array",
                "description": description,
                "items": items["schema"],
            },
        }

    @staticmethod
    def enum(name: str, description: str, values: list[Any]) -> dict:
        # Stored as a string-typed schema with an enum constraint, which is the
        # most common shape the LLM extractor handles cleanly.
        return {
            "name": name,
            "schema": {
                "type": "string",
                "description": description,
                "enum": list(values),
            },
        }


def _build_object_schema(params: list[dict]) -> dict:
    """Wrap a list of `Param.*` fragments into a top-level JSON Schema object.

    Used internally before posting to `/parameter-definition` or `/run-sre`.
    Sets every fragment as required and disables extra properties so the LLM
    can't quietly drop or add fields.
    """
    return {
        "type": "object",
        "properties": {p["name"]: p["schema"] for p in params},
        "required": [p["name"] for p in params],
        "additionalProperties": False,
    }
