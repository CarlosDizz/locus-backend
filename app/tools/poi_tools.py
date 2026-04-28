def get_poi_tool_manifest() -> list[dict]:
    return [
        {
            "type": "function",
            "name": "get_nearby_pois",
            "description": "Refresca o amplía los lugares turísticos ya visibles en el mapa base cuando el contexto actual no baste.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "lat": {"type": "number"},
                    "lng": {"type": "number"},
                },
                "required": ["query", "lat", "lng"],
                "additionalProperties": False
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "search_tourism_candidates",
            "description": "Úsala cuando el usuario pregunte por un monumento, edificio histórico, plaza, iglesia, museo o lugar singular que podría merecer estar en el mapa turístico. Devuelve candidatos turísticos para evaluar, pero no los marca automáticamente en el mapa ni los guarda en base de datos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "near_poi_name": {"type": "string"},
                    "lat": {"type": "number"},
                    "lng": {"type": "number"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
                "additionalProperties": False
            },
            "strict": False,
        },
        {
            "type": "function",
            "name": "search_contextual_recommendations",
            "description": "Úsala para recomendaciones contextuales no estrictamente turísticas: hostelería, bebidas, transporte, farmacia, cajero, supermercado u otras necesidades prácticas del momento. Devuelve resultados para evaluar, pero no los marca automáticamente en el mapa ni los guarda en base de datos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "need": {"type": "string"},
                    "lat": {"type": "number"},
                    "lng": {"type": "number"},
                    "limit": {"type": "integer"},
                },
                "required": ["need"],
                "additionalProperties": False
            },
            "strict": False,
        },
        {
            "type": "function",
            "name": "identify_map_landmark",
            "description": "Úsala cuando el usuario describa algo que tiene delante o cerca y quiera identificarlo o localizarlo en el mapa. Sirve para expresiones como 'esto que tengo enfrente', 'el monumento de la plaza', 'la torre junto a la iglesia' o 'el obelisco frente a la puerta principal'. Devuelve candidatos para evaluar, pero no los marca automáticamente en el mapa.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reference_text": {"type": "string"},
                    "near_poi_name": {"type": "string"},
                    "lat": {"type": "number"},
                    "lng": {"type": "number"},
                    "limit": {"type": "integer"},
                },
                "required": ["reference_text"],
                "additionalProperties": False
            },
            "strict": False,
        },
        {
            "type": "function",
            "name": "mark_pois_on_map",
            "description": "Úsala cuando ya tengas uno o varios resultados válidos y quieras que aparezcan en el mapa de Locus como recomendaciones efímeras de la sesión. Sirve tanto para candidatos turísticos como para recomendaciones contextuales prácticas. No guarda nada en base de datos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "poi_names": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "replace_existing": {"type": "boolean"},
                    "reason": {"type": "string"}
                },
                "required": ["poi_names"],
                "additionalProperties": False
            },
            "strict": False,
        },
        {
            "type": "function",
            "name": "promote_poi_to_catalog",
            "description": "Convierte un candidato o recomendación efímera en POI fijo del catálogo si cumple criterios mínimos: relevancia turística/cultural, coordenadas fiables y encaje real en la ciudad activa. Úsala cuando el usuario pida que un lugar pase a ser una visita fija, aparezca entre los POIs normales o se pueda abrir como visita/llamada desde ficha. Si no se puede promocionar con seguridad, devuelve el motivo y no finjas que ya se ha añadido.",
            "parameters": {
                "type": "object",
                "properties": {
                    "poi_name": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["poi_name"],
                "additionalProperties": False
            },
            "strict": False,
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
