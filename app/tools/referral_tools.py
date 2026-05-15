def get_referral_tool_manifest() -> list[dict]:
    return [
        {
            "type": "function",
            "name": "search_access_referrals",
            "description": (
                "Busca enlaces concretos y curados de entrada, pase, acceso, atraccion o experiencia fisica reservable "
                "que Locus no puede sustituir con su guia realtime. Si no existe un enlace fiable, devuelve error "
                "no_reliable_referral_link; no genera busquedas genericas de GetYourGuide. No la uses para visitas "
                "guiadas culturales normales, free tours ni tours a pie cuyo valor principal sea un guia humano."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Busqueda concreta: entradas museo, pase atraccion, bus turistico, barco, teleferico, etc.",
                    },
                    "poi_name": {"type": "string"},
                    "city_name": {"type": "string"},
                    "intent": {
                        "type": "string",
                        "description": "ticket, pass, transport, attraction o access segun la necesidad inferida.",
                    },
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "strict": False,
        }
    ]
