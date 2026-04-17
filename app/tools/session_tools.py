def get_session_tool_manifest() -> list[dict]:
    return [
        {
            "type": "function",
            "name": "get_session_profile",
            "description": "Lee el perfil actual del grupo y las preferencias de la sesión.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            "strict": True,
        },
        {
            "type": "function",
            "name": "set_active_poi",
            "description": "Actualiza el POI activo de la sesión cuando el usuario cambia de lugar.",
            "parameters": {
                "type": "object",
                "properties": {"poi_name": {"type": "string"}},
                "required": ["poi_name"],
                "additionalProperties": False
            },
            "strict": True,
        },
    ]
