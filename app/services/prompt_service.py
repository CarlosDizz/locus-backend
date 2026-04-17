import json
from pathlib import Path


class PromptService:
    def __init__(self) -> None:
        self.base_path = Path(__file__).resolve().parent.parent / "prompts"

    def load_prompt(self, name: str) -> dict:
        path = self.base_path / name
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def render(self, prompt_name: str, context: dict) -> str:
        shared = self.load_prompt("shared_rules.json")
        prompt = self.load_prompt(prompt_name)

        sections = [
            f"AGENTE:\n{prompt['identity']['name']}",
            f"ROL:\n{prompt['identity']['role']}",
        ]
        sections.append("OBJETIVOS:\n" + "\n".join(f"- {item}" for item in prompt["objectives"]))
        sections.append("REGLAS DURAS:\n" + "\n".join(f"- {item}" for item in shared["hard_rules"] + prompt["hard_rules"]))
        sections.append("USO DE TOOLS:\n" + "\n".join(f"- {item}" for item in prompt["tool_rules"]))
        sections.append("ESTILO:\n" + "\n".join(f"- {item}" for item in prompt["style"]))

        dynamic_lines = []
        for key, label in prompt["dynamic_context"].items():
            value = context.get(key)
            dynamic_lines.append(f"{label}:\n{value or '(sin dato)'}")
        sections.append("CONTEXTO DINAMICO:\n" + "\n\n".join(dynamic_lines))
        return "\n\n".join(sections)


prompt_service = PromptService()
