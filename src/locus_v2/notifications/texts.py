"""Los textos de los avisos, en el idioma de quien los recibe (los 9 del grupo Locus)."""

TEXTS: dict[str, dict[str, str]] = {
    "es": {
        "pre_trip": "Repasamos tu viaje a {city}: sales en dos días. ¿Lo dejamos cerrado?",
        "today": "Hoy: {items}",
        "and": " y ",
        "trip": "Vuestro viaje",
        "joined": "{name} se ha unido al viaje",
        "added_one": "{name} ha añadido «{item}»",
        "added_many": "{name} ha añadido «{item}» y {more} más",
        "changed": "{name} ha cambiado el plan",
    },
    "en": {
        "pre_trip": (
            "Let’s go over your trip to {city}: "
            "you leave in two days. Shall we wrap it up?"
        ),
        "today": "Today: {items}",
        "and": " and ",
        "trip": "Your trip",
        "joined": "{name} joined the trip",
        "added_one": "{name} added “{item}”",
        "added_many": "{name} added “{item}” and {more} more",
        "changed": "{name} changed the plan",
    },
    "fr": {
        "pre_trip": "On révise ton voyage à {city} : tu pars dans deux jours. On le boucle ?",
        "today": "Aujourd’hui : {items}",
        "and": " et ",
        "trip": "Votre voyage",
        "joined": "{name} a rejoint le voyage",
        "added_one": "{name} a ajouté « {item} »",
        "added_many": "{name} a ajouté « {item} » et {more} de plus",
        "changed": "{name} a modifié le programme",
    },
    "it": {
        "pre_trip": "Ripassiamo il tuo viaggio a {city}: parti tra due giorni. Lo chiudiamo?",
        "today": "Oggi: {items}",
        "and": " e ",
        "trip": "Il vostro viaggio",
        "joined": "{name} si è unito al viaggio",
        "added_one": "{name} ha aggiunto «{item}»",
        "added_many": "{name} ha aggiunto «{item}» e altri {more}",
        "changed": "{name} ha cambiato il programma",
    },
    "de": {
        "pre_trip": (
            "Lass uns deine Reise nach {city} durchgehen: "
            "In zwei Tagen geht’s los. Machen wir sie fertig?"
        ),
        "today": "Heute: {items}",
        "and": " und ",
        "trip": "Eure Reise",
        "joined": "{name} ist der Reise beigetreten",
        "added_one": "{name} hat „{item}“ hinzugefügt",
        "added_many": "{name} hat „{item}“ und {more} weitere hinzugefügt",
        "changed": "{name} hat den Plan geändert",
    },
    "pt": {
        "pre_trip": (
            "Vamos rever a tua viagem a {city}: "
            "partes daqui a dois dias. Fechamos o plano?"
        ),
        "today": "Hoje: {items}",
        "and": " e ",
        "trip": "A vossa viagem",
        "joined": "{name} juntou-se à viagem",
        "added_one": "{name} adicionou «{item}»",
        "added_many": "{name} adicionou «{item}» e mais {more}",
        "changed": "{name} alterou o plano",
    },
    "zh": {
        "pre_trip": "来看看你的{city}之旅：两天后出发，我们把行程定下来吧？",
        "today": "今天：{items}",
        "and": "和",
        "trip": "你们的旅行",
        "joined": "{name} 加入了旅行",
        "added_one": "{name} 添加了「{item}」",
        "added_many": "{name} 添加了「{item}」等 {more} 项",
        "changed": "{name} 修改了行程",
    },
    "ja": {
        "pre_trip": "{city}への旅を見直しましょう。出発まであと2日です。プランを仕上げましょうか？",
        "today": "今日：{items}",
        "and": "と",
        "trip": "みんなの旅行",
        "joined": "{name}さんが旅行に参加しました",
        "added_one": "{name}さんが「{item}」を追加しました",
        "added_many": "{name}さんが「{item}」ほか{more}件を追加しました",
        "changed": "{name}さんがプランを変更しました",
    },
    "ar": {
        "pre_trip": "لنراجع رحلتك إلى {city}: تسافر بعد يومين. هل ننهي الخطة؟",
        "today": "اليوم: {items}",
        "and": " و",
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
