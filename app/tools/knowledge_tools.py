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
        {
            "type": "function",
            "name": "search_web_facts",
            "description": "Busca informacion en la web sobre historia local, curiosidades, personajes, agenda, web oficial o contexto actual de un lugar. Prefierela a Wikipedia cuando la pregunta sea local, poco documentada, actual o muy concreta.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Consulta libre. Incluye nombre local, ciudad o tipo de dato si ayuda a afinar."
                    },
                    "preferred_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Dominios preferidos si quieres priorizar web oficial, turismo local, prensa local o una institucion concreta."
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                        "description": "Numero maximo de resultados utiles a devolver."
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            },
            "strict": True,
        },
    ]
