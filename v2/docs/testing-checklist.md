# Checklist de pruebas V2 — por capítulos

Vive junto a `roadmap.md`. Cada capítulo es un dominio del contrato V1. Un capítulo se marca
como probado solo cuando se ha ejercitado contra un MySQL real (`./bin/locus up`) y, cuando
aplica, contra la app Ionic real apuntando su `apiBaseUrl` a `http://localhost:8100/api`.

Estados: `construido` (código escrito, no probado en caliente) · `probado` (verificado contra
stack real) · `pendiente` (no empezado).

## Capítulo 1 — Auth (`/api/auth/*`)

Estado: **probado en caliente** (2026-09-06) contra MySQL real, salvo la app Ionic real
(pendiente por depender de un login Google interactivo real, ver más abajo).

- [x] Modelos: `UserSession` (tabla `user_sessions`, distinta de `AdminSession`).
- [x] Migración `f3a6c9d21b74_add_user_sessions`.
- [x] `MobileAuthService`: login Google (find-or-create por `provider_subject` → `email`,
      reconoce usuarios importados de V1), `authenticate` (bearer), `logout`.
- [x] Password auth: réplica exacta del comportamiento V1 cuando
      `auth_enable_password_auth=False` (mensajes y códigos HTTP idénticos); implementación
      real del password hashing queda pendiente porque V1 no lo usa en producción.
- [x] Router `/api/auth/register|login|google|me`, mismo shape que `app/schemas/auth.py`.
- [x] Separación estricta del login de admin: rol `user` únicamente, nunca `admin`; tabla y
      servicio distintos de `/admin/v2/auth/*`.
- [x] Ruff, mypy focalizado y suite de tests existente en verde.
- [x] `POST /api/auth/register` y `/login` en caliente vía curl contra la API real: 400
      "El acceso con email y contraseña está desactivado" y 401 "Usa el acceso con Google"
      respectivamente — texto y código idénticos a `app/services/auth_service.py` (comparado
      línea a línea).
- [x] `POST /api/auth/google` con un token inválido → 401 real ("No he podido verificar la
      cuenta de Google"). No se pudo probar con un token de Google real y válido: exigiría un
      login interactivo real contra una cuenta de Google, no disponible en este entorno.
- [x] Confirmado que un usuario ya importado de V1 (`carlos.garcia@ganbaru.es`, id 42,
      `legacy_v1_id=3`) se reconoce por `provider_subject` y NO crea duplicado: se instanció
      `MobileAuthService` real contra la BD real con un verificador de Google sustituido
      (solo se sustituye la llamada de red a Google — imposible de obtener un JWT firmado
      real sin ese login interactivo; toda la lógica de negocio, la BD y el token de sesión
      resultante son reales). El token real emitido se probó después contra `GET /api/auth/me`
      por HTTP real → 200 con `id=3` (el legacy id, correcto). Un segundo caso con un email
      nuevo (`zzz-auth-hot-test-new-user@example.com`, sin colisión) confirmó la rama de
      creación; se limpió el usuario de prueba al terminar.
- [x] `logout()` real revoca la sesión: tras revocar, `GET /me` con el mismo token vuelve a
      dar 401. **Hallazgo, no defecto**: no existe ruta HTTP `/api/auth/logout` ni en V1 ni en
      V2 — el cierre de sesión de la app Ionic es puramente local (`auth.service.ts::logout`
      solo limpia el storage y cierra sesión de Google en el SDK), nunca llama al backend. Es
      paridad exacta con V1, no una regresión: el método de servicio existe pero no está
      expuesto por ninguna ruta en ninguna de las dos versiones.
- [ ] Login Google real desde Postman/curl con un token válido de verdad.
- [ ] Apuntar la app Ionic real (`environment.local.ts`, `apiBaseUrl`) a V2 y loguear de verdad.
- [ ] Bono de bienvenida al crear usuario nuevo — pendiente conectar con Billing (ver TODO en
      `mobile_auth.py`).
- [ ] Decidir si merece la pena implementar password auth real o dejarlo en "desactivado" para
      siempre (hoy nadie lo usa en V1).

## Capítulo 2 — Catálogo (`/api/catalog/*`)

Estado: **pendiente**. Contrato ya inventariado (schemas de `app/schemas/catalog.py` leídos y
documentados en `roadmap.md` §11), falta escribir el router V2.

- [ ] `GET /catalog/poi-types`, `GET /catalog/cities`, `GET /catalog/pois`,
      `GET /catalog/pois/{id}` usando `legacy_v1_id` para IDs numéricos estables.
- [ ] `GET /catalog/pois/{id}/documentation` y `/access-links` (este último depende del
      Capítulo 5, afiliación GetYourGuide).
- [x] `POST /admin/v2/catalog/bootstrap-from-location` — **Fase 1 portada y probada
      (2026-09-05)**: geocodifica el punto (Nominatim), crea o reutiliza la ciudad, resuelve
      su entidad Wikidata, importa POIs cercanos vía SPARQL (consulta por entidad de ciudad,
      radio como respaldo), puntúa/filtra/deduplica con las mismas heurísticas que V1, y
      crea/actualiza `Poi` reales. Vive en `catalog/bootstrap/` (`wikidata_client.py`,
      `nominatim.py`, `poi_scoring.py`, `sparql_queries.py`, `normalize.py`, `service.py`).
      Sin credenciales nuevas — Nominatim y Wikidata son públicos.
      - Bug real encontrado y corregido durante la construcción: los constructores de SPARQL
        duplicaban el escape de llaves de f-string (`{{`/`}}` dentro de una cadena ya plana),
        generando SPARQL mal formado. Verificado imprimiendo la query generada antes y después.
      - Probado en caliente contra datos reales: geocodificación de Albacete y de Tequila,
        Jalisco (Nominatim), búsqueda y resolución de entidad Wikidata para ambas (Tequila
        resolvió a `Q2330431`), generación de SPARQL bien formado — todo confirmado por logs
        reales, no simulado.
      - El paso final (ejecutar el SPARQL contra `query.wikidata.org`) no se pudo verificar en
        vivo: el servicio está sufriendo una caída real ("active wdqs outage", limitando a 1
        petición/min) ajena a Locus. El manejo de error es correcto: se propaga como 502 con
        mensaje claro y **no deja estado parcial** — ni la ciudad reverse-geocodificada se
        guarda si el SPARQL falla después (transacción íntegra, verificado en la BD).
      - La lógica de puntuación/deduplicación/creación de POIs sí se validó completa contra una
        respuesta SPARQL simulada con datos reales (catedral real de Albacete + un candidato de
        juzgado a propósito): el juzgado se filtró correctamente por las heurísticas negativas,
        la catedral se creó con los campos y metadata correctos. Fila de prueba borrada después.
      - Interfaz en el panel: botón "Sembrar POIs desde un punto" en Ciudades y POIs, con clic
        en el mapa o campos de lat/lng, radio y límite. Probado con Playwright real (no solo
        por API): abre el panel, rellena coordenadas, lanza la petición y muestra el error real
        de Wikidata correctamente en pantalla.
      - Pendiente repetir la prueba de extremo a extremo (con POIs reales creados vía Wikidata)
        cuando `query.wikidata.org` se recupere de la caída.
      - **Fase 2 — Overpass (2026-09-05), hecha**: `overpass_client.py` (async, sin
        credenciales) + `overpass_queries.py` (consulta por radio y normalizador de
        elementos OSM). Se fusiona con los candidatos de Wikidata en el mismo ranking/dedupe
        (por `wikidata_id` cuando el elemento OSM lo trae, si no por slug). Validado con una
        respuesta Overpass simulada junto a una de Wikidata: deduplicó correctamente un
        elemento OSM que repetía la cathedral ya vista en Wikidata, y creó un candidato
        distinto exclusivo de OSM.
        - **Incidente durante esta prueba, corregido**: el candidato distinto de la prueba
          coincidió por slug con un POI real ya existente (importado de V1, "Museo de
          Albacete") y sobrescribió su descripción corta, coordenadas y metadata con datos de
          prueba. Se restauraron las coordenadas y la descripción reales consultando la misma
          entidad Wikidata que usó V1 originalmente (`Q3558939`); las 9 traducciones en
          `short_descriptions_json`/`names_json` no se habían tocado. Desde este punto, toda
          prueba usa ciudades/nombres con un marcador claramente falso para no poder chocar
          con datos reales.
      - **Fase 3 — Candidatos por IA y localización (2026-09-05), hecha**: `ai_client.py`
        (llamada estructurada con JSON schema a la Responses API, reutiliza
        `LOCUS_OPENAI_API_KEY`, sin credenciales nuevas) + `ai_candidates.py`
        (`generate_ai_candidates`, `localize_content_candidates` a 9 idiomas,
        `names_from_aliases`). Mismo control de flujo que V1: si `use_ai_candidates` es true
        y hay candidatos de IA, se usan directamente (sin tocar Wikidata/Overpass) con
        coordenadas provisionales (`source_of_truth=gpt_seed`, `import_status` en
        `seeded_gpt_coords` o `pending_wikidata`); si no se pidió IA y Wikidata/Overpass no
        encontraron nada, se usa como último recurso. Validado completo con IA y
        geocodificación simuladas (sin gasto real ni llamada externa): candidato con
        coordenadas quedó `seeded_gpt_coords`, candidato sin coordenadas quedó
        `pending_wikidata`, filas creadas y limpiadas en una ciudad de prueba con nombre
        marcado (`ZZZ-Test-City-Do-Not-Use`), cero riesgo de colisión con datos reales.
      - Interfaz en el panel: casilla "Usar IA para proponer candidatos" en el panel de
        sembrado, desactivada por defecto (gasto real de OpenAI si se activa) aunque V1
        activa IA por defecto — desviación deliberada para que un admin nuevo no dispare
        gasto sin darse cuenta la primera vez que usa el botón.
      Piezas del port original, para referencia de lo hecho vs. pendiente:
      1. `WikidataClient` (`app/clients/wikidata_client.py`) — **hecho** (versión async, ver
         arriba).
      2. Geocodificación inversa vía Nominatim — **hecho**.
      3. `create_city` + resolución de ciudad existente por slug — **hecho**.
      4. `import_city_pois` sin IA ni Overpass — **hecho**: `_resolve_city_entity_id`,
         consulta por entidad de ciudad y por radio, mapa de tipos de POI (40+ entradas),
         puntuación y filtro de ruido, dedupe por `wikidata_id`/slug. Entrega POIs reales pero
         solo en español (sin el paso de localización, ver punto 7).
      5. `OverpassClient` (`app/clients/overpass_client.py`) como respaldo cuando Wikidata no
         alcanza el mínimo de candidatos — **hecho** (ver Fase 2 arriba).
      6. Candidatos por IA (`_generate_ai_candidates`, `_upsert_ai_seed_candidates`) vía
         `OpenAIClient` — **hecho** (ver Fase 3 arriba). `_resolve_ai_candidate` SÍ se portó
         (`catalog/bootstrap/enrichment.py`) — corrección de una nota anterior de este mismo
         documento que lo daba por código muerto: en V1 la rama que lo invoca dentro de
         `import_city_pois` es efectivamente inalcanzable, pero la función se usa de verdad
         desde `enrich_city_pending_pois` (el worker de reintento, ver punto 7).
      7. Localización a 9 idiomas (`_localize_content_candidates`) — **hecho** (ver Fase 3).
         La cola de enriquecimiento en segundo plano de V1 (`start_pending_enrichment`) —
         **hecho** (2026-09-05): `CatalogEnrichmentService.enrich_city_pending_pois`,
         disparado como `BackgroundTasks` de FastAPI tras un sembrado con IA, resuelve los
         POIs en `pending_wikidata` contra Wikidata real y, si falla, contra Overpass.
      Interfaz en el panel — **hecho** (2026-09-05): botón "Sembrar POIs desde un punto" en
      Ciudades y POIs, clic en el mapa o coordenadas manuales, resultado con POIs
      creados/actualizados o error, probado con Playwright real.
- [x] CRUD de POIs desde el panel (2026-09-06): `PUT /admin/v2/catalog/pois/{id}`
      (`catalog/admin_write.py::AdminCatalogWriteService`) — nombre, nombres/descripciones
      localizadas (añadir/quitar idioma), descripción corta/larga, lat/lng, tipo (desplegable
      real desde `poi_types`), activo/inactivo, wikidata/wikipedia/google place. Marca
      `source_of_truth="manual"` y escribe un `AdminAuditEvent` con el antes/después completo.
      Sin equivalente en V1 (no tenía panel de administración). Probado en caliente con
      Playwright de principio a fin sobre un POI de prueba desechable: edición de nombre y
      tipo, verificado en la ficha, en la base de datos y en el evento de auditoría (actor,
      antes, después), y limpiado al terminar.

## Capítulo 3 — Chat (`/api/chat/*`)

Estado: **pendiente** el dominio V1-compatible; **probado** un slice mínimo interno
(2026-09-05) contra un proveedor real, solo para verificar el pipeline de uso/coste.

- [x] `ChatConfigurationResolver` mínimo (`chat/configuration.py`): resuelve
      `RoutingProfile` publicado por `service_kind=chat`, renderiza el prompt con el
      contexto del POI. Sin fallback, sin tools todavía (helpers de prompt/localización
      extraídos a `shared/prompting.py`, compartidos con Voice).
- [x] `OpenAIResponsesAdapter` (`chat/providers/openai_responses.py`): llamada real,
      no streaming, a `client.responses.create`; normaliza el usage a `NormalizedUsage`.
- [x] `ChatService.send_message`, ejercitado en su momento por un endpoint interno
      (`POST /admin/v2/dev/chat/messages`) ya retirado: superado por el botón general
      "Probar proveedor" del panel (`POST /admin/v2/configuration/models/{id}/test`,
      Capítulo 8), que prueba cualquier modelo catalogado, chat o voz, sin pasar por
      `ChatConfigurationResolver` ni una `RoutingProfile`.
- [x] Verificado en caliente contra `gpt-5.4-mini` real (routing profile `chat.poi.local`):
      respuesta correcta, `UsageEvent` creado (`interaction_type=chat_call`), recogido por
      el worker de billing y cobrado a la wallet real del usuario de prueba.
- [ ] Modelos y repositorio persistentes del dominio Chat (sesión/mensajes, hoy no hay
      historial, cada llamada es de un solo turno).
- [ ] Tools reenchufadas al chat (se resuelven en la config pero no se envían al modelo
      todavía; falta el bucle de tool-calling que sí tiene Voice).
- [ ] Fallback provider como en Voice.
- [ ] Tool `activity_referrals` (afiliación) enchufada al chat, no solo a voz.
- **Hallazgo (2026-09-06), corrige el alcance de este capítulo**: `POST /chat/setup` y
  `POST /chat/messages` de V1 NO son "chat sobre un POI" — son el chat de recomendaciones
  del mapa de la pantalla de inicio de Ionic, y dependen de `session_id`/ubicación/perfil
  (el dominio de Sesiones, Capítulo 6), no de un `RoutingProfile` con contexto de POI como
  hace hoy el slice interno de Chat. Portarlo de verdad con compatibilidad real exigía
  primero el dominio de Sesiones — que ya está hecho (ver más abajo) — así que este
  capítulo ya puede continuar: falta construir `/chat/setup`/`/chat/messages` de verdad
  sobre `MapSessionService`, no sobre `ChatConfigurationResolver`.
- [x] **Dominio de Sesiones portado (2026-09-06)**, `app/services/session_service.py` →
      `sessions/application/service.py` (`MapSessionService`) + `sessions/models.py`
      (`map_sessions`, tabla nueva vía migración `b72e9930c315`). DB-backed igual que V1
      (nunca fue el problema de arquitectura — la parte pendiente real de Capítulo 6 sigue
      siendo la sala de llamadas y el puente de protocolo, no tocados aquí).
      `POST/GET/PUT /api/sessions[/{id}]`, `/reset`, `/presence` (touch y salida),
      `/call-state`, `/call-log` — mismas rutas, formas y semántica que V1
      (`app/routes/sessions.py` + `app/schemas/session.py`), incluida la poda de
      participantes obsoletos por tiempo. Probado en caliente end-to-end por HTTP real
      sobre una sesión desechable: crear, leer, 404 real, actualizar perfil/POI activo,
      presencia con un usuario real, estado de llamada (host asignado correctamente),
      log de llamada, abandonar presencia (cierra la llamada si el host se va), y
      reiniciar conversación — limpiado al terminar. `set_nearby_pois`/`set_active_poi`/
      `set_ephemeral_map_pois` portados como métodos de servicio (igual que V1: sin
      búsqueda geográfica propia, los rellena quien construya Chat).

## Capítulo 4 — Billing (`/api/billing/*`, Google Play)

Estado: **probado en caliente** (2026-09-06) contra datos reales, salvo la verificación
real de compras de Google Play (sin credenciales de service account en este entorno).

- [x] `GET /api/billing/wallet`, `GET /ledger`, `GET /usage-events` — probados con un
      usuario real migrado de V1 (`carlos.garcia@ganbaru.es`), devolviendo su saldo y su
      historial real (incluido el asiento de "Saldo promocional de bienvenida" importado de
      V1). Forma de respuesta adaptada donde el esquema de V2 ya es mejor que el de V1: en
      vez de reconstruir a mano los campos `source`/`endpoint`/`call_id` que V1 guardaba
      sueltos en `UsageEvent`, el ledger enlaza directamente con la `VoiceSession` real
      (`billing/application/mobile_billing.py`, ver su docstring).
- [x] `POST /topups` (recarga manual) — probado el guardado (403 real cuando
      `billing_manual_topups_enabled=False`, que es el valor por defecto, igual que V1) y,
      con el flag activado solo para la prueba sobre un usuario y wallet desechables, el
      camino real de abono: wallet actualizada y asiento de ledger `credit` correctos.
      Limpiado al terminar.
- [x] `POST /billing/google-play/topups/confirm` (`billing/infrastructure/google_play.py`,
      port de `_verify_google_play_purchase`, versión async con `httpx`) — probado el
      rechazo real de un producto desconocido (400, sin efectos secundarios). La
      verificación real contra la API de Android Publisher **no se ha podido probar**: este
      entorno no tiene `GOOGLE_PLAY_SERVICE_ACCOUNT_JSON`/`_FILE` configurado (tampoco lo
      tenía V1 en `.env.example`), así que ese tramo queda construido pero sin verificar
      contra Google de verdad.
- [ ] Idempotencia de cargos y compras — la comprobación de duplicados en
      `confirm_google_play_topup` es una consulta previa a nivel de aplicación, no una
      restricción única en la base de datos (igual que V1: mismo hueco de condición de
      carrera bajo concurrencia real, no es una regresión pero tampoco está resuelto).
- [ ] Bono de bienvenida al registrar usuario (enlazar con Capítulo 1).

## Capítulo 5 — Afiliación GetYourGuide

Estado: **motor portado y probado en caliente** (2026-09-06), incluido en una llamada de
voz real de principio a fin. Falta solo la ruta HTTP pública `/catalog/pois/{id}/access-links`
(depende del API de Catálogo público, todavía sin escribir — ver Capítulo 2).

- [x] `affiliates/service.py::ReferralService` — puerto completo de
      `app/services/referral_service.py`: `poi_access_links` (enlaces curados desde
      metadatos del POI) y `activity_referrals` (búsqueda real vía `web_search` de OpenAI
      restringida a `getyourguide.es`/`.com`, con verificación de coincidencia de lugar y
      ciudad, y respaldo a un enlace de búsqueda genérico cuando no hay match fiable).
      Heurísticas de matching migradas literalmente (términos de tickets/atracciones/
      movilidad, alias de ciudades y lugares, solapamiento de tokens). Dos funciones de
      V1 genuinamente muertas (`_looks_like_guided_visit`,
      `_looks_like_non_substitutable_experience`, nunca llamadas en ningún sitio de V1)
      se dejaron fuera a propósito.
- [x] `getyourguide_referrals_enabled`, `getyourguide_partner_id` en `Settings` V2, mismos
      valores por defecto que V1.
- [x] Conectada como herramienta real: `affiliates.find_activities` en
      `voice/tools.py::VoiceToolDispatcher`, y añadida al prompt de voz publicado
      (`voice.poi.guide`) junto a `document_poi`/`plan_poi_visit`. Sin equivalente directo
      en V1 (allí solo la usaba el chat) — aquí decidimos conectarla también a voz porque
      ya se podía probar de verdad desde el nuevo panel de llamadas y encaja con el
      producto (un guía de voz que también puede ofrecer entradas).
- [x] **Bug real encontrado y corregido**: la llamada a `web_search` usaba
      `max_output_tokens=220` (heredado de V1, donde el modelo no razonaba). El modelo de
      V2 (`gpt-5-mini`) es un modelo de razonamiento que agota ese presupuesto pensando
      antes de emitir la llamada a `web_search` — la búsqueda nunca llegaba a ejecutarse
      (`status="incomplete"`, cero resultados, cero excepciones: fallaba en silencio hacia
      el enlace de respaldo). Confirmado con un script aislado y subido a 1500. Nota:
      `reasoning.effort="minimal"` (el arreglo usado en el bootstrap de catálogo) no es
      compatible con la tool `web_search` — la única vía aquí es dar más presupuesto.
- [x] **Bug de datos real encontrado y corregido**: el script de seed buscaba y parcheaba
      siempre la `PromptVersion` con `version==1`, pero el prompt de voz ya tenía una
      versión 2 publicada de una sesión anterior (vía el flujo normal de publicar desde el
      panel), que dejó la versión 1 retirada. El parche de `find_activities` estaba
      curando una fila que ya no usaba nadie. Corregido el script de seed para localizar
      siempre la versión con `status=PUBLISHED` (no una versión fija), y realineados a
      mano los perfiles de prueba que habían quedado apuntando a la versión retirada.
- [x] Probado en caliente de principio a fin con Playwright sobre una llamada real de voz
      (OpenAI Realtime, POI real "Coliseo" en Roma): el modelo decidió por sí mismo llamar
      a `find_activities`, la búsqueda real devolvió URLs de producto reales de
      GetYourGuide, y el modelo las presentó como enlaces Markdown, tal y como pide la
      política del prompt.
- [ ] Ruta HTTP pública `/catalog/pois/{id}/access-links` — pendiente del API de Catálogo
      público (Capítulo 2); `poi_access_links` ya está listo para conectarse en cuanto
      exista esa ruta.
- [ ] Idempotencia/anti-duplicado de enlaces vistos entre turnos (V1 tampoco lo tenía).

## Capítulo 6 — Sesiones, llamadas y WebSocket (`/api/sessions`, `/api/calls`, `/ws/calls/{id}`)

Estado: **pendiente** el puente hacia el protocolo V1. El protocolo V2 nativo
(`/ws/v2/live`) sí quedó verificado en caliente el 2026-09-05: sesión real vía
`voice.poi.local` con Gemini Live (proveedor primario) y, con un routing profile
temporal solo para la prueba (creado y borrado en el mismo test), con OpenAI Realtime.
Ambos casos: `session.ready` correcto, respuesta de texto real, `usage.recorded` con
tokens reales, `VoiceSession`/`VoiceTurn` persistidos, y el worker de billing cobrando
el evento contra la wallet real. La autenticación se hizo con la cookie de sesión de
admin (`voice/auth.py` ya soporta esto para depuración local); sigue sin probarse con un
token móvil real de `/api/auth`. Es el capítulo de mayor riesgo técnico en lo que falta.

- [x] Adaptador de sesiones V1 (2026-09-06) — ver Capítulo 3 para el detalle completo
      (`/api/sessions`, presencia, estado de llamada, log, probado en caliente).
- [ ] Orquestación de salas V2 equivalente a `call_room_service.py` (917 líneas en V1). Sigue
      siendo el bloque más grande y de mayor riesgo del capítulo: sesiones ya no lo era.
- [ ] Puente de protocolo: traducir `/ws/calls/{callId}` (V1, lo que habla Ionic hoy) al
      protocolo neutral ya construido en `voice/gateway.py` (`/ws/v2/live`, ver
      `docs/websocket-protocol.md`). Son protocolos distintos, esto es trabajo nuevo.
- [ ] `client-secret`, tools de realtime, análisis de fotos (`/realtime/*`).
- [ ] Pruebas de reconexión, interrupción, tools y fallback con la app real (criterio explícito
      del roadmap: el WebSocket no se considera compatible sin esto).

## Capítulo 7 — Legal y metadatos de app

Estado: **probado en caliente** (2026-09-06) contra la API real.

- [x] `/privacy-policy` y `/account-deletion` servidos desde V2, HTML idéntico a
      `app/routes/legal.py` (V1 no tiene una ruta genérica `/legal`, solo estas dos).
- [x] `GET /api/app/version` — mismo shape que V1 (`android.latest_version_code`,
      `android.update_url`, `ios.latest_build`, `ios.update_url`), settings nuevas en
      `config.py` con los mismos valores por defecto que V1.
- [x] Verificado con curl real contra la API viva: los tres endpoints devuelven 200 con
      el contenido esperado.

## Capítulo 8 — Panel de control (control-panel Angular)

Estado: parcialmente **probado** (conectado a datos reales), resto **construido** o
**pendiente**. Ver `roadmap.md` §11 para el detalle sección por sección.

- [x] Login admin, Pulso, Prompts/Proveedores, Ciudades y POIs, Consumos, Registros — conectados
      a datos reales.
- [x] Usuarios (2026-09-05): sí trae detalle completo — saldo, últimas sesiones de voz y
      movimientos de ledger, no solo la lista. El diagnóstico previo de `roadmap.md` §11
      estaba desactualizado en este punto.
- [x] Conversaciones (2026-09-05): el calendario lee las últimas 40 `VoiceSession` reales
      (`admin/infrastructure/sqlalchemy_overview.py::_read_activities`), no es una maqueta.
- [x] Auditoría (2026-09-05): nueva sección de panel (`GET /admin/v2/audit`,
      `AuditConsoleComponent`) sobre `AdminAuditEvent`, que ya se escribía en cada cambio de
      modelo/prompt/ruta pero no tenía vista. Antes/después completo por cambio.
- [x] Prueba de proveedor desde el panel (2026-09-05): botón "Probar" en Proveedores
      (`POST /admin/v2/configuration/models/{id}/test`). Llama de verdad al modelo elegido
      (chat vía Responses, voz vía el mismo `ProviderRegistry`/`LiveProvider` que usa
      `/ws/v2/live`), espera a que el worker de billing lo cobre y muestra respuesta, tokens
      y coste en el momento.
- [x] Pedido explícito (2026-09-05, hecho 2026-09-06): desde la ficha de un POI en
      Ciudades y POIs, se puede lanzar una llamada real por WebSocket sobre ese POI
      concreto (`PoiCallTestComponent`, contra `/ws/v2/live`, usando su prompt y contexto
      real vía el `routing_profile` elegido — no el prompt neutro de "Probar proveedor").
      Verificado en caliente por Playwright: sesión real, turno de texto, `VoiceSession`/
      `VoiceTurn` persistidos. Tres routing profiles de prueba nuevos (OpenAI, Gemini, Mock
      sin coste) para fijar el proveedor deliberadamente. No incluye la sala multiusuario
      (eso es Capítulo 6).
- [x] Editar un POI existente (2026-09-06) — ver Capítulo 2. Sigue sin haber alta/baja de
      ciudades ni un editor de tipos de POI desde el panel, solo edición de POIs.
- [ ] Historial de prompts navegable más allá de las versiones ya listadas, dashboard de
      salud más allá de Pulso.
- [x] Bug investigado y cerrado (2026-09-06): la sesión de Gemini Live que falló el
      2026-09-05 con `'UsageMetadata' object has no attribute 'candidates_token_count'`
      NO era un bug en `gemini_live.py` — ese código nunca referencia ese nombre; es el
      campo equivalente en la clase de uso de la API estándar de generación
      (`GenerateContentResponseUsageMetadata`), no en la del Live API. La causa real:
      `google-genai` estaba fijado como `>=1.0,<2` (sin lock), así que cada reconstrucción
      de imagen podía instalar una versión 1.x distinta sin aviso; probablemente una
      versión de tránsito de esas fechas tenía esta inconsistencia interna en el SDK. Con
      la versión actual (1.75.0) una llamada real de texto contra Gemini Live funciona de
      principio a fin sin ese error (probado con un script aislado). Corregido fijando
      `google-genai==1.75.0` exacto en `pyproject.toml` para que no vuelva a ocurrir por
      deriva de versión, y añadido el traceback completo al `context_json` de los eventos
      de error del gateway de voz (antes solo se guardaba el mensaje, lo que hizo más
      difícil diagnosticar esto la primera vez).

## Capítulo 9 — Corte a producción

Estado: **pendiente**, es el último capítulo por diseño.

- [ ] Backup completo de la base de producción antes de nada.
- [ ] `./bin/locus up` en local con datos importados, capítulos 1–7 en verde.
- [ ] Desplegar V2 en paralelo en ECS sin tráfico real.
- [ ] Cambiar `apiBaseUrl` de Ionic de `https://api.locusguide.es/api` al host V2.
- [ ] Ventana de observación con V1 disponible para rollback inmediato (revertir la URL).
- [ ] Retirar V1 solo cuando no haya regresiones ni diferencias de facturación.
