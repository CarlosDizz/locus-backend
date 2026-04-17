def get_knowledge_tool_manifest() -> list[dict]:
    return [
        {
            "type": "function",
            "name": "resolve_poi_facts",
            "description": "Pide datos concretos y prudentes sobre un POI sin inventar hechos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "poi_name": {"type": "string"},
                    "question": {"type": "string"},
                },
                "required": ["poi_name"],
                "additionalProperties": False
            },
            "strict": True,
        }
    ]
