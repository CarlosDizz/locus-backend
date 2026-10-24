# ruff: noqa: E501 - el HTML de las páginas va tal cual, con líneas largas.
"""La invitación a un viaje de Borea: /borea/unirse/<código>.

Es el enlace que se comparte por WhatsApp y el que lleva el QR. En un iPhone con
Borea lo abre la app directamente (enlace universal, en el Caddyfile); si no la
tiene, llega aquí: quién invita, adónde y cuándo, y el botón de la App Store.
iOS no recuerda el enlace durante la instalación, así que al volver basta con
tocar «Ya tengo Borea», que abre la app con borea://unirse/<código>.
"""

from datetime import date
from html import escape

from fastapi import APIRouter, Header
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from locus_v2.api.auth import SessionDep, SettingsDep
from locus_v2.notifications.service import first_name
from locus_v2.shared.apps import ClientApp, visible_in
from locus_v2.trips.models import Trip

router = APIRouter(tags=["borea"])

_TEXTS: dict[str, dict[str, str]] = {
    "es": {"invites": "{name} te invita a su viaje", "trip": "Un viaje con Borea",
           "store": "Descargar Borea", "open": "Ya tengo Borea", "invalid":
           "Este enlace ya no vale. Pide uno nuevo a quien te invitó.",
           "hint": "Instala Borea, entra y vuelve aquí para unirte."},
    "en": {"invites": "{name} invites you to their trip", "trip": "A trip with Borea",
           "store": "Download Borea", "open": "I already have Borea", "invalid":
           "This link no longer works. Ask whoever invited you for a new one.",
           "hint": "Install Borea, sign in and come back here to join."},
    "fr": {"invites": "{name} t’invite à son voyage", "trip": "Un voyage avec Borea",
           "store": "Télécharger Borea", "open": "J’ai déjà Borea", "invalid":
           "Ce lien ne fonctionne plus. Demande-en un nouveau.",
           "hint": "Installe Borea, connecte-toi et reviens ici pour rejoindre le voyage."},
    "it": {"invites": "{name} ti invita al suo viaggio", "trip": "Un viaggio con Borea",
           "store": "Scarica Borea", "open": "Ho già Borea", "invalid":
           "Questo link non è più valido. Chiedine uno nuovo.",
           "hint": "Installa Borea, accedi e torna qui per unirti."},
    "de": {"invites": "{name} lädt dich zur Reise ein", "trip": "Eine Reise mit Borea",
           "store": "Borea laden", "open": "Ich habe Borea schon", "invalid":
           "Dieser Link gilt nicht mehr. Bitte um einen neuen.",
           "hint": "Installiere Borea, melde dich an und komm hierher zurück."},
    "pt": {"invites": "{name} convida-te para a viagem", "trip": "Uma viagem com Borea",
           "store": "Descarregar Borea", "open": "Já tenho Borea", "invalid":
           "Este link já não funciona. Pede um novo.",
           "hint": "Instala o Borea, entra e volta aqui para te juntares."},
    "zh": {"invites": "{name} 邀请你一起旅行", "trip": "Borea 旅行",
           "store": "下载 Borea", "open": "我已安装 Borea", "invalid":
           "此链接已失效，请向邀请你的人索取新链接。",
           "hint": "安装 Borea 并登录，然后回到这里加入。"},
    "ja": {"invites": "{name}さんから旅行への招待", "trip": "Boreaの旅行",
           "store": "Boreaをダウンロード", "open": "Boreaを持っています", "invalid":
           "このリンクは無効です。新しいリンクをもらってください。",
           "hint": "Boreaをインストールしてログインし、ここに戻って参加してください。"},
    "ar": {"invites": "يدعوك {name} إلى رحلته", "trip": "رحلة مع Borea",
           "store": "تنزيل Borea", "open": "لدي Borea", "invalid":
           "هذا الرابط لم يعد صالحًا. اطلب رابطًا جديدًا.",
           "hint": "ثبّت Borea وسجّل الدخول ثم عد إلى هنا للانضمام."},
}

_FONTS = (
    "https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;600"
    "&family=Syne:wght@800&display=swap"
)

_STYLES = """
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; min-height: 100vh; display: grid; place-items: center; padding: 32px 16px;
    font-family: 'Instrument Sans', -apple-system, system-ui, sans-serif; color: #eef0fa;
    background: radial-gradient(120% 45% at 15% -10%, rgba(92,242,160,.22), transparent 60%),
      radial-gradient(90% 40% at 90% -5%, rgba(139,107,255,.2), transparent 60%), #0d0b1f; }
  main { width: 100%; max-width: 380px; text-align: center; }
  .marca { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 28px;
    letter-spacing: -.01em; }
  .tarjeta { margin: 28px 0 22px; padding: 26px 20px; border-radius: 22px; background: #17142e;
    border: 1px solid #2b2650; }
  .quien { color: #a3a6c8; font-size: 15px; margin: 0 0 6px; }
  h1 { font-size: 34px; line-height: 1.1; margin: 0; font-weight: 600; }
  .fechas { color: #5cf2a0; font-weight: 600; margin: 10px 0 0; }
  a.boton { display: block; padding: 15px; border-radius: 999px; font-weight: 600; font-size: 16px;
    text-decoration: none; margin-top: 10px; }
  .principal { background: #5cf2a0; color: #062416; }
  .secundario { border: 1px solid #2b2650; color: #eef0fa; background: #17142e; }
  .nota { color: #a3a6c8; font-size: 13px; line-height: 1.5; margin-top: 18px; }
"""


def _language(accept_language: str | None) -> str:
    for part in (accept_language or "").split(","):
        code = part.strip()[:2].lower()
        if code in _TEXTS:
            return code
    return "es"


_MONTHS: dict[str, list[str]] = {
    "es": "ene feb mar abr may jun jul ago sept oct nov dic".split(),
    "en": "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(),
    "fr": "janv. févr. mars avr. mai juin juil. août sept. oct. nov. déc.".split(),
    "it": "gen feb mar apr mag giu lug ago set ott nov dic".split(),
    "de": "Jan. Feb. März Apr. Mai Juni Juli Aug. Sept. Okt. Nov. Dez.".split(),
    "pt": "jan fev mar abr mai jun jul ago set out nov dez".split(),
    "ar": "يناير فبراير مارس أبريل مايو يونيو يوليو أغسطس سبتمبر أكتوبر نوفمبر ديسمبر".split(),
}


def _day(value: date, language: str, year: bool) -> str:
    if language in ("zh", "ja"):
        texto = f"{value.month}月{value.day}日"
        return f"{value.year}年{texto}" if year else texto
    mes = _MONTHS[language][value.month - 1]
    dia = f"{mes} {value.day}" if language == "en" else f"{value.day} {mes}"
    return f"{dia} {value.year}" if year else dia


def _dates(trip: Trip, language: str) -> str:
    if trip.starts_at is None:
        return ""
    inicio = trip.starts_at.date()
    fin = (trip.ends_at or trip.starts_at).date()
    if inicio == fin:
        return _day(inicio, language, year=True)
    return f"{_day(inicio, language, year=False)} – {_day(fin, language, year=True)}"


@router.get("/borea/unirse/{code}", response_class=HTMLResponse)
async def borea_invite(
    code: str, session: SessionDep, settings: SettingsDep,
    accept_language: str | None = Header(default=None),
) -> HTMLResponse:
    language = _language(accept_language)
    t = _TEXTS[language]
    trip = await session.scalar(
        select(Trip).options(selectinload(Trip.owner)).where(Trip.invite_code == code[:16])
    )
    store = f"https://apps.apple.com/app/id{settings.borea_app_store_app_apple_id}"
    if trip is None or not visible_in(trip.apps, ClientApp.BOREA):
        cuerpo = f'<div class="tarjeta"><p class="quien">{escape(t["invalid"])}</p></div>'
        status = 404
    else:
        nombre = first_name(trip.owner) if trip.owner is not None else ""
        cuerpo = f"""
      <div class="tarjeta">
        <p class="quien">{escape(t["invites"].format(name=nombre))}</p>
        <h1>{escape(trip.city or t["trip"])}</h1>
        <p class="fechas">{escape(_dates(trip, language))}</p>
      </div>
      <a class="boton principal" href="{escape(store)}">{escape(t["store"])}</a>
      <a class="boton secundario" href="borea://unirse/{escape(code)}">{escape(t["open"])}</a>
      <p class="nota">{escape(t["hint"])}</p>"""
        status = 200
    direction = "rtl" if language == "ar" else "ltr"
    html = f"""<!doctype html>
<html lang="{language}" dir="{direction}">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="robots" content="noindex" />
    <meta name="apple-itunes-app" content="app-id={settings.borea_app_store_app_apple_id}" />
    <title>Borea</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link href="{_FONTS}" rel="stylesheet" />
    <style>{_STYLES}</style>
  </head>
  <body>
    <main>
      <div class="marca">Borea</div>{cuerpo}
    </main>
  </body>
</html>
"""
    return HTMLResponse(html, status_code=status)


# --- La web informativa de Borea: /borea ----------------------------------------------
# Una sola página, sin app web (Borea es solo iPhone). Lleva el script de Travelpayouts
# Drive, que su alta exige verificar en la web (2026-10-24, decidido por Carlos): va
# solo aquí, nunca en la PWA de Locus, para no meter contenido de terceros en ella
# (locusguide.es ya estuvo marcado por Google por eso; ver el Caddyfile).

_TRAVELPAYOUTS_DRIVE = """
    <script nowprocket data-noptimize="1" data-cfasync="false" data-wpfc-render="false" seraph-accel-crit="1" data-no-defer="1" data-cmp-ab="2">
      (function () {
          var script = document.createElement("script");
          script.async = 1;
          script.setAttribute("data-cmp-ab","2");
          script.src = 'https://emrldtp.com/NTkwNTQz.js?t=590543';
          document.head.appendChild(script);
      })();
    </script>"""

_HOME = """<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Borea Travel Planner, de Locus</title>
    <meta name="description" content="Prepara tu viaje conversando: Borea monta el plan día a día, con el tiempo, el mapa y dónde comer como la gente del sitio." />
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link href="{fonts}" rel="stylesheet" />
    <style>{styles}
      .claim {{ font-size: 30px; line-height: 1.15; font-weight: 600; margin: 26px 0 10px; }}
      .sub {{ color: #a3a6c8; font-size: 16px; line-height: 1.55; margin: 0 0 26px; }}
      ul {{ list-style: none; padding: 0; margin: 0 0 28px; text-align: left; display: grid; gap: 12px; }}
      li {{ background: #17142e; border: 1px solid #2b2650; border-radius: 16px; padding: 14px 16px; font-size: 15px; line-height: 1.45; }}
      li b {{ color: #5cf2a0; font-weight: 600; }}
      footer {{ margin-top: 30px; font-size: 13px; color: #a3a6c8; }}
      footer a {{ color: #a3a6c8; }}
    </style>{drive}
  </head>
  <body>
    <main>
      <div class="marca">Borea</div>
      <p class="claim">Tu viaje, montado conversando.</p>
      <p class="sub">Cuéntale a Borea a dónde vas y con quién, o pégale el correo de una reserva. Ella monta el plan día a día.</p>
      <ul>
        <li><b>El plan, día a día.</b> Con el tiempo de cada día, el mapa y el camino a pie entre paradas.</li>
        <li><b>Comer como la gente del sitio.</b> Recomendaciones con valoraciones reales, sin turistadas y sin que nadie pague por salir.</li>
        <li><b>Viajes compartidos.</b> Invita con un QR o un enlace y planead juntos; avisos cuando alguien cambia algo.</li>
        <li><b>De Locus.</b> Lleva el plan a Locus Guide y escucha cada sitio con tu guía.</li>
      </ul>
      <a class="boton principal" href="{store}">Próximamente en la App Store</a>
      <footer>
        <a href="https://api.locusguide.es/borea/privacy-policy">Privacidad</a> ·
        <a href="https://api.locusguide.es/borea/terms">Términos</a> ·
        <a href="https://app.locusguide.es">Locus Guide</a>
      </footer>
    </main>
  </body>
</html>
"""


@router.get("/borea", response_class=HTMLResponse)
@router.get("/borea/", response_class=HTMLResponse)
async def borea_home(settings: SettingsDep) -> HTMLResponse:
    store = f"https://apps.apple.com/app/id{settings.borea_app_store_app_apple_id}"
    return HTMLResponse(_HOME.format(
        fonts=_FONTS, styles=_STYLES, drive=_TRAVELPAYOUTS_DRIVE, store=escape(store),
    ))
