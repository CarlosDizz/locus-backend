import wikipedia

from app.config import settings


class WikipediaClient:
    def __init__(self) -> None:
        self._lang = settings.wikipedia_language

    def get_summary(self, query: str, sentences: int = 5) -> str:
        if not query:
            return ""
        result = self._fetch(query, self._lang, sentences)
        if not result and self._lang != "en":
            result = self._fetch(query, "en", sentences)
        return result

    def _fetch(self, query: str, lang: str, sentences: int) -> str:
        wikipedia.set_lang(lang)
        for candidate in self._candidates(query):
            try:
                return wikipedia.summary(candidate, sentences=sentences, auto_suggest=True)
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

    def _candidates(self, query: str) -> list[str]:
        return [query]
