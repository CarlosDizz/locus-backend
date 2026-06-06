from fastapi import APIRouter
from fastapi.responses import HTMLResponse, Response


router = APIRouter(tags=["legal"])


@router.head("/privacy-policy")
async def privacy_policy_head() -> Response:
    return Response(media_type="text/html")


@router.get("/privacy-policy", response_class=HTMLResponse)
async def privacy_policy() -> str:
    return """
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


@router.head("/account-deletion")
async def account_deletion_head() -> Response:
    return Response(media_type="text/html")


@router.get("/account-deletion", response_class=HTMLResponse)
async def account_deletion() -> str:
    return """
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
