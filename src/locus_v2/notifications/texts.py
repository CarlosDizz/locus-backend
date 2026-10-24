"""Los textos de los avisos, en el idioma de quien los recibe (los 9 del grupo Locus)."""

TEXTS: dict[str, dict[str, str]] = {
    "es": {
        "trip": "Vuestro viaje",
        "joined": "{name} se ha unido al viaje",
        "added_one": "{name} ha añadido «{item}»",
        "added_many": "{name} ha añadido «{item}» y {more} más",
        "changed": "{name} ha cambiado el plan",
    },
    "en": {
        "trip": "Your trip",
        "joined": "{name} joined the trip",
        "added_one": "{name} added “{item}”",
        "added_many": "{name} added “{item}” and {more} more",
        "changed": "{name} changed the plan",
    },
    "fr": {
        "trip": "Votre voyage",
        "joined": "{name} a rejoint le voyage",
        "added_one": "{name} a ajouté « {item} »",
        "added_many": "{name} a ajouté « {item} » et {more} de plus",
        "changed": "{name} a modifié le programme",
    },
    "it": {
        "trip": "Il vostro viaggio",
        "joined": "{name} si è unito al viaggio",
        "added_one": "{name} ha aggiunto «{item}»",
        "added_many": "{name} ha aggiunto «{item}» e altri {more}",
        "changed": "{name} ha cambiato il programma",
    },
    "de": {
        "trip": "Eure Reise",
        "joined": "{name} ist der Reise beigetreten",
        "added_one": "{name} hat „{item}“ hinzugefügt",
        "added_many": "{name} hat „{item}“ und {more} weitere hinzugefügt",
        "changed": "{name} hat den Plan geändert",
    },
    "pt": {
        "trip": "A vossa viagem",
        "joined": "{name} juntou-se à viagem",
        "added_one": "{name} adicionou «{item}»",
        "added_many": "{name} adicionou «{item}» e mais {more}",
        "changed": "{name} alterou o plano",
    },
    "zh": {
        "trip": "你们的旅行",
        "joined": "{name} 加入了旅行",
        "added_one": "{name} 添加了「{item}」",
        "added_many": "{name} 添加了「{item}」等 {more} 项",
        "changed": "{name} 修改了行程",
    },
    "ja": {
        "trip": "みんなの旅行",
        "joined": "{name}さんが旅行に参加しました",
        "added_one": "{name}さんが「{item}」を追加しました",
        "added_many": "{name}さんが「{item}」ほか{more}件を追加しました",
        "changed": "{name}さんがプランを変更しました",
    },
    "ar": {
        "trip": "رحلتكم",
        "joined": "انضم {name} إلى الرحلة",
        "added_one": "أضاف {name} «{item}»",
        "added_many": "أضاف {name} «{item}» و{more} أخرى",
        "changed": "غيّر {name} الخطة",
    },
}


def text(locale: str, key: str, **values: object) -> str:
    language = (locale or "es")[:2].lower()
    return (TEXTS.get(language) or TEXTS["es"])[key].format(**values)
