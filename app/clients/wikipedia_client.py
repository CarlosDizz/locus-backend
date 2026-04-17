import wikipedia

from app.config import settings


class WikipediaClient:
    def __init__(self) -> None:
        wikipedia.set_lang(settings.wikipedia_language)

    def get_summary(self, query: str, sentences: int = 4) -> str:
        if not query:
            return ""
        candidates = [query]
        if not query.lower().startswith("pasaje de "):
            candidates.append(f"Pasaje de {query}")

        for candidate in candidates:
            try:
                return wikipedia.summary(candidate, sentences=sentences, auto_suggest=False)
            except wikipedia.exceptions.DisambiguationError as exc:
                for option in exc.options[:5]:
                    try:
                        return wikipedia.summary(option, sentences=sentences, auto_suggest=False)
                    except Exception:
                        continue
            except wikipedia.exceptions.PageError:
                continue
            except Exception:
                continue
        return ""
