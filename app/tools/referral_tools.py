def get_referral_tool_manifest() -> list[dict]:
    return [
        {
            "type": "function",
            "name": "search_access_referrals",
            "description": (
                "Busca enlaces utiles de GetYourGuide para entradas, pases, tours, free tours, visitas guiadas, "
                "excursiones, transporte turistico y experiencias reservables. Prioriza paginas concretas de producto; "
                "si no hay producto fiable, puede devolver una busqueda sugerida claramente marcada como fallback."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Busqueda concreta: entradas museo, tour por la ciudad, free tour, pase atraccion, bus turistico, barco, teleferico, etc.",
                    },
                    "poi_name": {"type": "string"},
                    "city_name": {"type": "string"},
                    "intent": {
                        "type": "string",
                        "description": "ticket, pass, tour, free_tour, transport, attraction, activity o access segun la necesidad inferida.",
                    },
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "strict": False,
        }
    ]
