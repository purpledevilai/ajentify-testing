class Param:
    """Convenience factory for parameter definition dicts.

    Usage:
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

    @staticmethod
    def string(name: str, description: str) -> dict:
        return {"name": name, "description": description, "type": "string", "parameters": []}

    @staticmethod
    def number(name: str, description: str) -> dict:
        return {"name": name, "description": description, "type": "number", "parameters": []}

    @staticmethod
    def boolean(name: str, description: str) -> dict:
        return {"name": name, "description": description, "type": "boolean", "parameters": []}

    @staticmethod
    def object(name: str, description: str, children: list[dict]) -> dict:
        return {"name": name, "description": description, "type": "object", "parameters": children}

    @staticmethod
    def array(name: str, description: str, items: dict) -> dict:
        return {"name": name, "description": description, "type": "array", "parameters": [items]}

    @staticmethod
    def enum(name: str, description: str, values: list[str]) -> dict:
        return {
            "name": name,
            "description": description,
            "type": "enum",
            "parameters": [{"name": v, "description": v, "type": "string"} for v in values],
        }
