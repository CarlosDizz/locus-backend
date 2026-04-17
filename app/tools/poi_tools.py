def get_poi_tool_manifest() -> list[dict]:
    return [
        {
            "type": "function",
            "name": "get_nearby_pois",
            "description": "Busca puntos de interés cercanos a partir de la ubicación de la sesión.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "lat": {"type": "number"},
                    "lng": {"type": "number"},
                },
                "required": ["query"],
                "additionalProperties": False
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "get_poi_summary",
            "description": "Obtiene un resumen prudente de un POI a partir de una fuente documental.",
            "parameters": {
                "type": "object",
                "properties": {"poi_name": {"type": "string"}},
                "required": ["poi_name"],
                "additionalProperties": False
            },
            "strict": True,
        },
    ]
