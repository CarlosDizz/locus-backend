"""Las skills del planificador de Borea: criterio declarativo que se carga al usarlo.

Cada skill es un fichero en formato abierto SKILL.md (trips/skills/*.md): nombre,
cuándo usarla, la lista de tools que activa y las instrucciones. El prompt base
solo lleva el nombre y la línea de «cuándo usarla»; el modelo llama a
cargar_skill y recibe el resto en ese turno. Así no se paga en cada mensaje lo
que no hace falta, cada área se afina sin tocar las demás y, como el formato no
es de OpenAI, las skills sirven igual si se cambia de modelo.

No se usan las skills nativas de la API de OpenAI: van por un contenedor con
terminal que no puede tocar nuestra base de datos (decidido el 2026-10-24).
"""

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    tools: tuple[str, ...]
    body: str


def parse_skill(text: str) -> Skill:
    """Lee la cabecera (entre las dos líneas «---») y el cuerpo de un SKILL.md."""
    head, body = "", text
    if text.startswith("---"):
        _, head, body = text.split("---", 2)
    fields: dict[str, str] = {}
    for line in head.strip().splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    tools = tuple(
        t.strip() for t in fields.get("tools", "").strip("[]").split(",") if t.strip()
    )
    return Skill(
        name=fields.get("name", ""),
        description=fields.get("description", ""),
        tools=tools,
        body=body.strip(),
    )


@lru_cache
def load_skills() -> dict[str, Skill]:
    folder = files("locus_v2.trips").joinpath("skills")
    skills = [
        parse_skill(entry.read_text(encoding="utf-8"))
        for entry in folder.iterdir()
        if entry.name.endswith(".md")
    ]
    return {skill.name: skill for skill in sorted(skills, key=lambda s: s.name) if skill.name}


def skills_index() -> str:
    """La lista que va en el prompt: nombre y cuándo usarla, nada más."""
    lines = [f"- {s.name}: {s.description}" for s in load_skills().values()]
    return (
        "Skills disponibles (cárgalas con cargar_skill antes de trabajar en su área; "
        "cada una trae su criterio y activa sus herramientas):\n" + "\n".join(lines)
    )
