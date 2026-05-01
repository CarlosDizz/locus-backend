def get_knowledge_tool_manifest(*, include_web_research_tool: bool = True) -> list[dict]:
    tools = [
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
                "required": ["poi_name", "question"],
                "additionalProperties": False
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "search_wikipedia",
            "description": "Busca en Wikipedia la historia, origen o significado de un lugar, monumento o concepto por nombre libre. Usala cuando el usuario pregunte sobre historia o contexto de algo concreto y no tengas esa informacion en el contexto actual.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Nombre o descripcion del lugar o concepto a buscar. Se mas especifico si sabes el nombre local (ej: 'Colonna dell Immacolata Roma' en lugar de 'Column of Peace')."
                    },
                },
                "required": ["query"],
                "additionalProperties": False
            },
            "strict": True,
        },
    ]
    if include_web_research_tool:
        tools.append(
            {
            "type": "function",
            "name": "search_web_facts",
            "description": "Investiga en internet historia local, curiosidades, personajes, agenda o contexto actual de un lugar cuando Wikipedia se quede corta o la pregunta requiera abrir la mano a la web.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Consulta libre sobre el lugar, persona o curiosidad que quieres investigar."
                    },
                    "preferred_domains": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Dominios a priorizar si quieres favorecer fuentes oficiales, turismo local o prensa local fiable."
                    },
                    "max_results": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "maximum": 8,
                        "description": "Numero maximo de fuentes a devolver en el resumen."
                    }
                },
                "required": ["query", "preferred_domains", "max_results"],
                "additionalProperties": False
            },
            "strict": True,
            },
        )
    return tools
