"""Static legal pages, ported verbatim from app/routes/legal.py.

No DB, no auth — the app store review process and the Ionic app's in-app
links to these pages need them reachable with no dependencies at all. The
same property is what a payment provider's reviewer needs: Paddle is the
merchant of record for web top-ups, so it carries the legal responsibility
towards the buyer and will not approve an account whose terms, refunds and
contact details are behind a login.
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, Response

router = APIRouter(tags=["legal"])

CONTACT_EMAIL = "dizz01@gmail.com"

_STYLES = """
      :root { color-scheme: light; font-family: Georgia, 'Times New Roman', serif; color: #1f2a2e; background: #f6f0e7; }
      body { margin: 0; padding: 32px 18px; }
      main { max-width: 840px; margin: 0 auto; background: rgba(255,255,255,.72); border: 1px solid rgba(47,93,98,.16); border-radius: 24px; padding: 28px; box-shadow: 0 24px 54px rgba(31,42,46,.08); }
      h1, h2 { line-height: 1.15; }
      h1 { font-size: clamp(2rem, 5vw, 3.2rem); margin: 0 0 8px; }
      h2 { margin-top: 28px; color: #2f5d62; }
      p, li { font-size: 1rem; line-height: 1.65; }
      .updated { color: #667074; margin-top: 0; }
      a { color: #2f5d62; font-weight: 700; }
      table { border-collapse: collapse; width: 100%; margin-top: 12px; }
      th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid rgba(47,93,98,.16); font-size: 1rem; }
      th { color: #2f5d62; }
"""


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title} | Locus</title>
    <style>{_STYLES}</style>
  </head>
  <body>
    <main>
{body}
    </main>
  </body>
</html>
"""

_PRIVACY_POLICY_HTML = """
<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Política de privacidad | Locus</title>
    <style>
      :root { color-scheme: light; font-family: Georgia, 'Times New Roman', serif; color: #1f2a2e; background: #f6f0e7; }
      body { margin: 0; padding: 32px 18px; }
      main { max-width: 840px; margin: 0 auto; background: rgba(255,255,255,.72); border: 1px solid rgba(47,93,98,.16); border-radius: 24px; padding: 28px; box-shadow: 0 24px 54px rgba(31,42,46,.08); }
      h1, h2 { line-height: 1.15; }
      h1 { font-size: clamp(2rem, 5vw, 3.2rem); margin: 0 0 8px; }
      h2 { margin-top: 28px; color: #2f5d62; }
      p, li { font-size: 1rem; line-height: 1.65; }
      .updated { color: #667074; margin-top: 0; }
      a { color: #2f5d62; font-weight: 700; }
    </style>
  </head>
  <body>
    <main>
      <h1>Política de privacidad de Locus</h1>
      <p class="updated">Última actualización: 31 de mayo de 2026</p>

      <p>
        Locus es una aplicación móvil de guía urbana, mapas, recomendaciones y chat asistido por IA.
        Esta política explica qué datos tratamos, para qué los usamos y cómo puedes contactar con nosotros.
      </p>

      <h2>Datos que tratamos</h2>
      <ul>
        <li>Datos de cuenta, como nombre, correo electrónico e identificador de Google cuando inicias sesión con Google.</li>
        <li>Ubicación aproximada o precisa cuando das permiso a la aplicación, para mostrar la ciudad actual, puntos de interés cercanos y recomendaciones contextuales.</li>
        <li>Contenido que envías al chat o a funciones de guía, para generar respuestas y mantener el contexto de la sesión.</li>
        <li>Datos técnicos básicos, como idioma, plataforma, versión de la app, errores y registros necesarios para seguridad y diagnóstico.</li>
        <li>Información de uso y compras o saldo interno cuando uses funciones de facturación dentro de la app.</li>
      </ul>

      <h2>Finalidades</h2>
      <ul>
        <li>Prestar las funciones principales de Locus: mapa, guía, chat, recomendaciones, llamadas y puntos de interés.</li>
        <li>Autenticar usuarios y proteger cuentas.</li>
        <li>Mejorar estabilidad, seguridad, rendimiento y experiencia de producto.</li>
        <li>Gestionar compras, saldo, consumo y soporte.</li>
      </ul>

      <h2>Servicios de terceros</h2>
      <p>
        Locus puede apoyarse en proveedores como Google Sign-In, Google Maps, servicios de infraestructura cloud,
        bases de datos, sistemas de pago y proveedores de IA para ofrecer sus funcionalidades. Estos proveedores
        tratan datos únicamente en la medida necesaria para prestar el servicio.
      </p>

      <h2>Ubicación</h2>
      <p>
        La ubicación se usa para situarte en el mapa y adaptar la experiencia a la ciudad en la que estás.
        Puedes retirar el permiso de ubicación desde los ajustes del sistema operativo. Sin ubicación, algunas
        funciones de recomendaciones cercanas pueden no funcionar correctamente.
      </p>

      <h2>Conservación</h2>
      <p>
        Conservamos los datos mientras tu cuenta esté activa o mientras sean necesarios para prestar el servicio,
        cumplir obligaciones legales, resolver incidencias o prevenir abusos. Podemos eliminar o anonimizar datos
        cuando ya no sean necesarios.
      </p>

      <h2>Tus derechos</h2>
      <p>
        Puedes solicitar acceso, rectificación, eliminación u oposición al tratamiento de tus datos escribiendo a
        <a href="mailto:dizz01@gmail.com">dizz01@gmail.com</a>. También puedes solicitar la eliminación de tu cuenta
        y datos asociados.
      </p>

      <h2>Menores</h2>
      <p>
        Locus no está dirigida a menores de 13 años. Si detectamos que una cuenta pertenece a un menor sin autorización
        válida, podremos eliminarla.
      </p>

      <h2>Cambios</h2>
      <p>
        Podemos actualizar esta política para reflejar cambios legales, técnicos o de producto. Publicaremos la versión
        vigente en esta misma página.
      </p>

      <h2>Contacto</h2>
      <p>
        Para cualquier consulta sobre privacidad o protección de datos: <a href="mailto:dizz01@gmail.com">dizz01@gmail.com</a>.
      </p>
    </main>
  </body>
</html>
"""

_ACCOUNT_DELETION_HTML = """
<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Eliminar cuenta | Locus</title>
    <style>
      :root { color-scheme: light; font-family: Georgia, 'Times New Roman', serif; color: #1f2a2e; background: #f6f0e7; }
      body { margin: 0; padding: 32px 18px; }
      main { max-width: 840px; margin: 0 auto; background: rgba(255,255,255,.72); border: 1px solid rgba(47,93,98,.16); border-radius: 24px; padding: 28px; box-shadow: 0 24px 54px rgba(31,42,46,.08); }
      h1, h2 { line-height: 1.15; }
      h1 { font-size: clamp(2rem, 5vw, 3.2rem); margin: 0 0 8px; }
      h2 { margin-top: 28px; color: #2f5d62; }
      p, li { font-size: 1rem; line-height: 1.65; }
      .updated { color: #667074; margin-top: 0; }
      .cta { display: inline-block; margin: 12px 0 4px; padding: 12px 18px; border-radius: 999px; background: #2f5d62; color: #fff; text-decoration: none; font-weight: 700; }
      a { color: #2f5d62; font-weight: 700; }
      .cta:visited { color: #fff; }
    </style>
  </head>
  <body>
    <main>
      <h1>Eliminar cuenta y datos de Locus</h1>
      <p class="updated">Última actualización: 31 de mayo de 2026</p>

      <p>
        Puedes solicitar la eliminación de tu cuenta de Locus y de los datos personales asociados en cualquier momento.
        Actualmente gestionamos estas solicitudes por correo electrónico para poder verificar la titularidad de la cuenta.
      </p>

      <h2>Cómo solicitar la eliminación</h2>
      <ol>
        <li>Escríbenos desde el correo asociado a tu cuenta de Locus.</li>
        <li>Indica en el asunto: “Eliminar mi cuenta de Locus”.</li>
        <li>Incluye el correo de tu cuenta y, si iniciaste sesión con Google, el mismo correo de Google.</li>
      </ol>

      <p>
        <a class="cta" href="mailto:dizz01@gmail.com?subject=Eliminar%20mi%20cuenta%20de%20Locus&body=Solicito%20la%20eliminaci%C3%B3n%20de%20mi%20cuenta%20de%20Locus%20y%20de%20los%20datos%20personales%20asociados.%0A%0ACorreo%20de%20la%20cuenta%3A%20">Solicitar eliminación de cuenta</a>
      </p>

      <h2>Qué datos se eliminan</h2>
      <ul>
        <li>Datos de cuenta, como nombre, correo electrónico e identificadores de autenticación.</li>
        <li>Preferencias y perfil de usuario.</li>
        <li>Sesiones, mensajes de chat y datos asociados al uso de la guía, cuando estén vinculados a tu cuenta.</li>
        <li>Datos de ubicación guardados que estén vinculados a tu cuenta, si existen.</li>
      </ul>

      <h2>Datos que podemos conservar temporalmente</h2>
      <p>
        Podemos conservar durante el tiempo legalmente necesario registros mínimos relacionados con facturación,
        seguridad, prevención de fraude, cumplimiento normativo o resolución de incidencias. Cuando ya no sean
        necesarios, se eliminarán o anonimizarán.
      </p>

      <h2>Plazo</h2>
      <p>
        Procesaremos la solicitud tan pronto como sea razonablemente posible. En condiciones normales responderemos
        al correo de solicitud para confirmar la recepción y completar la verificación de titularidad.
      </p>

      <h2>Contacto</h2>
      <p>
        Para solicitudes de eliminación de cuenta o privacidad: <a href="mailto:dizz01@gmail.com">dizz01@gmail.com</a>.
      </p>

      <p>
        También puedes consultar la <a href="/privacy-policy">política de privacidad de Locus</a>.
      </p>
    </main>
  </body>
</html>
"""


@router.head("/privacy-policy")
async def privacy_policy_head() -> Response:
    return Response(media_type="text/html")


@router.get("/privacy-policy", response_class=HTMLResponse)
async def privacy_policy() -> str:
    return _PRIVACY_POLICY_HTML


@router.head("/account-deletion")
async def account_deletion_head() -> Response:
    return Response(media_type="text/html")


@router.get("/account-deletion", response_class=HTMLResponse)
async def account_deletion() -> str:
    return _ACCOUNT_DELETION_HTML


_TERMS_HTML = _page(
    "Términos y condiciones",
    f"""
      <h1>Términos y condiciones de Locus</h1>
      <p class="updated">Última actualización: 12 de septiembre de 2026</p>

      <p>
        Locus es una aplicación de guía turística asistida por inteligencia artificial. Permite explorar
        puntos de interés, conversar por chat y mantener llamadas de voz con un guía virtual que documenta
        el lugar en el que te encuentras. Al usar Locus aceptas estos términos.
      </p>

      <h2>Quién presta el servicio</h2>
      <p>
        Locus lo desarrolla y opera Carlos García. Para las compras realizadas desde la web,
        <strong>Paddle.com Market Ltd actúa como vendedor autorizado (merchant of record)</strong>: es quien
        emite la factura, gestiona el cobro, aplica los impuestos que correspondan y atiende las
        devoluciones. En las compras realizadas desde la aplicación de Android, ese papel lo desempeña
        Google Play.
      </p>

      <h2>Saldo y consumo</h2>
      <ul>
        <li>Locus funciona con saldo de prepago. Recargas una cantidad y se descuenta según el uso real.</li>
        <li>Las llamadas de voz se cobran por consumo, en función de la duración y del volumen de audio
            procesado. El chat y la documentación de lugares consumen saldo de forma equivalente.</li>
        <li>El saldo no caduca, no genera intereses y no es transferible entre cuentas.</li>
        <li>Puedes consultar cada cargo, con su fecha y su importe, en el extracto de la pantalla de saldo.</li>
        <li>Si te quedas sin saldo durante una llamada, el servicio se interrumpe. No se generan deudas:
            nunca cobramos por encima del saldo disponible.</li>
      </ul>

      <h2>Qué esperar del guía</h2>
      <p>
        El guía de Locus es un sistema de inteligencia artificial, no una persona. Se le ha pedido
        expresamente que no invente datos y que reconozca lo que no sabe, pero
        <strong>puede equivocarse</strong>. La información que ofrece es orientativa y de carácter cultural:
        no debe usarse como única fuente para decisiones de seguridad, salud, desplazamiento o
        planificación que dependan de datos exactos. Comprueba horarios, precios y accesos en las fuentes
        oficiales de cada lugar.
      </p>

      <h2>Uso aceptable</h2>
      <ul>
        <li>No uses Locus para actividades ilícitas ni para obtener contenido que vulnere derechos de terceros.</li>
        <li>No compartas tu cuenta ni intentes acceder a cuentas ajenas.</li>
        <li>No intentes eludir los límites del servicio, automatizar su consumo ni revenderlo.</li>
        <li>Las fotografías que envíes al guía deben ser tuyas o tener permiso para compartirlas.</li>
      </ul>
      <p>
        Podemos suspender una cuenta que incumpla estas condiciones. Si ocurre, devolvemos el saldo no
        consumido salvo que la suspensión responda a un uso fraudulento.
      </p>

      <h2>Disponibilidad</h2>
      <p>
        Locus se presta tal cual está disponible. Dependemos de proveedores externos de inteligencia
        artificial, mapas e infraestructura, por lo que puede haber interrupciones. Si una llamada se corta
        por un fallo nuestro, no se cobra el tiempo no prestado.
      </p>

      <h2>Cambios</h2>
      <p>
        Podemos actualizar estos términos. Si el cambio es relevante, lo anunciaremos en la aplicación antes
        de que entre en vigor. La fecha de la última actualización figura al principio de esta página.
      </p>

      <h2>Ley aplicable</h2>
      <p>
        Estos términos se rigen por la legislación española. Si eres consumidor, conservas todos los
        derechos que te reconoce la normativa de consumo, incluido el acceso a los mecanismos de resolución
        de conflictos que correspondan.
      </p>

      <h2>Contacto</h2>
      <p>
        Escríbenos a <a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a>. Consulta también la
        <a href="/privacy-policy">política de privacidad</a> y la
        <a href="/refund-policy">política de reembolsos</a>.
      </p>
""",
)


_REFUND_POLICY_HTML = _page(
    "Política de reembolsos",
    f"""
      <h1>Política de reembolsos de Locus</h1>
      <p class="updated">Última actualización: 12 de septiembre de 2026</p>

      <p>
        Locus funciona con saldo de prepago: recargas una cantidad y se descuenta según lo que uses. Esta
        página explica cuándo puedes recuperar tu dinero y cómo pedirlo.
      </p>

      <h2>Resumen</h2>
      <table>
        <tr><th>Situación</th><th>Reembolso</th></tr>
        <tr><td>Saldo <strong>sin consumir</strong>, dentro de los 14 días siguientes a la recarga</td><td>Íntegro</td></tr>
        <tr><td>Saldo <strong>ya consumido</strong> en llamadas o chat</td><td>No reembolsable</td></tr>
        <tr><td>Recarga por error o duplicada</td><td>Íntegro</td></tr>
        <tr><td>Cobro sin que el saldo llegara a acreditarse</td><td>Íntegro</td></tr>
        <tr><td>Fallo nuestro que interrumpe una llamada</td><td>Se devuelve el tiempo no prestado</td></tr>
      </table>

      <h2>Derecho de desistimiento</h2>
      <p>
        Como consumidor en la Unión Europea dispones de <strong>14 días naturales</strong> para desistir de la
        compra. En un servicio digital ese derecho se pierde sobre la parte que ya has consumido, porque el
        servicio ya se ha prestado. Por eso el saldo que no has gastado se devuelve íntegro dentro de ese
        plazo, y el que ya has gastado no.
      </p>
      <p>
        Pasados los 14 días, el saldo no consumido sigue siendo tuyo y no caduca, aunque ya no exista
        derecho automático a recuperarlo en dinero. Si tienes un motivo, escríbenos: lo miramos caso por caso.
      </p>

      <h2>Cómo pedir un reembolso</h2>
      <p>
        Escribe a <a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a> desde el correo de tu cuenta,
        indicando la fecha y el importe de la recarga. Respondemos en un plazo máximo de
        <strong>5 días hábiles</strong>.
      </p>
      <p>
        En las compras hechas desde la web, el vendedor es
        <strong>Paddle.com Market Ltd</strong>, que es quien ejecuta la devolución por el mismo medio de pago
        que usaste. También puedes escribirles directamente desde el correo de confirmación de tu pedido. En
        las compras hechas desde la aplicación de Android, las devoluciones las gestiona Google Play según
        sus propias condiciones.
      </p>

      <h2>Si un cobro no acreditó saldo</h2>
      <p>
        Si el cargo aparece en tu banco pero el saldo no subió, no hace falta que pidas nada: escríbenos y lo
        corregimos. Cada pago queda registrado con un identificador único que impide que se abone dos veces,
        así que podemos comprobar exactamente qué pasó.
      </p>

      <h2>Contacto</h2>
      <p>
        <a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a> · Consulta también los
        <a href="/terms">términos y condiciones</a> y la
        <a href="/privacy-policy">política de privacidad</a>.
      </p>
""",
)


_CONTACT_HTML = _page(
    "Contacto",
    f"""
      <h1>Contacto y soporte</h1>
      <p class="updated">Última actualización: 12 de septiembre de 2026</p>

      <p>
        Locus lo desarrolla y mantiene Carlos García. Escribimos las respuestas a mano: no hay centralita ni
        formularios que se pierdan.
      </p>

      <h2>Escríbenos</h2>
      <p>
        <a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a>
      </p>
      <p>
        Respondemos en un plazo máximo de <strong>5 días hábiles</strong>, normalmente antes. Si escribes por
        un problema con un cobro, indícanos la fecha y el importe y lo miramos con el registro delante.
      </p>

      <h2>Con qué podemos ayudarte</h2>
      <ul>
        <li>Problemas con una recarga o un cobro que no cuadra.</li>
        <li>Reembolsos, según la <a href="/refund-policy">política de reembolsos</a>.</li>
        <li>Eliminar tu cuenta y tus datos, desde la <a href="/account-deletion">página de eliminación</a>.</li>
        <li>Fallos de la aplicación, del guía de voz o del chat.</li>
        <li>Cualquier consulta sobre privacidad o protección de datos.</li>
      </ul>

      <h2>Antes de escribir</h2>
      <p>
        En la pantalla de saldo de la aplicación tienes el extracto completo, con cada cargo, su fecha y su
        importe. Muchas dudas sobre consumo se resuelven ahí en un vistazo.
      </p>
""",
)


@router.head("/terms")
async def terms_head() -> Response:
    return Response(media_type="text/html")


@router.get("/terms", response_class=HTMLResponse)
async def terms() -> str:
    return _TERMS_HTML


@router.head("/refund-policy")
async def refund_policy_head() -> Response:
    return Response(media_type="text/html")


@router.get("/refund-policy", response_class=HTMLResponse)
async def refund_policy() -> str:
    return _REFUND_POLICY_HTML


@router.head("/contact")
async def contact_head() -> Response:
    return Response(media_type="text/html")


@router.get("/contact", response_class=HTMLResponse)
async def contact() -> str:
    return _CONTACT_HTML
