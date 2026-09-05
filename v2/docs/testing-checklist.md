# Checklist de pruebas V2 — por capítulos

Vive junto a `roadmap.md`. Cada capítulo es un dominio del contrato V1. Un capítulo se marca
como probado solo cuando se ha ejercitado contra un MySQL real (`./bin/locus up`) y, cuando
aplica, contra la app Ionic real apuntando su `apiBaseUrl` a `http://localhost:8100/api`.

Estados: `construido` (código escrito, no probado en caliente) · `probado` (verificado contra
stack real) · `pendiente` (no empezado).

## Capítulo 1 — Auth (`/api/auth/*`)

Estado: **construido**, no probado en caliente todavía.

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
- [ ] Arrancar `./bin/locus up`, aplicar `alembic upgrade head`, y hacer login Google real desde
      Postman/curl.
- [ ] Confirmar que un usuario ya importado (de los 19 de V1) se reconoce por email y no crea
      duplicado.
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
         `OpenAIClient` — **hecho** (ver Fase 3 arriba). `_resolve_ai_candidate` no se portó a
         propósito: en el V1 actual es código muerto — la rama que la invoca nunca se alcanza
         porque `import_city_pois` ya retorna antes si `ai_candidates` tiene contenido.
      7. Localización a 9 idiomas (`_localize_content_candidates`) — **hecho** (ver Fase 3).
         La cola de enriquecimiento en segundo plano (`start_pending_enrichment`) de V1 sigue
         **pendiente**: no hay todavía un worker en V2 que reintente resolver los POIs que
         quedan en `pending_wikidata` contra Wikidata más tarde.
      Interfaz en el panel — **hecho** (2026-09-05): botón "Sembrar POIs desde un punto" en
      Ciudades y POIs, clic en el mapa o coordenadas manuales, resultado con POIs
      creados/actualizados o error, probado con Playwright real.
- [ ] CRUD de ciudades/POIs para uso interno (los mismos endpoints ya cubiertos por
      `admin_catalog.py` del panel; decidir si se reutilizan o se separan). Editar POIs ya
      existentes desde el panel sigue pendiente y es independiente del sembrado — no hace
      falta esperar al sembrado completo para construirlo.

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
- [ ] `POST /chat/setup`, `POST /chat/messages` compatibles con V1 (endpoint público real,
      distinto de "Probar proveedor", que es solo de administración).
- [ ] Tool `activity_referrals` (afiliación) enchufada al chat, no solo a voz.

## Capítulo 4 — Billing (`/api/billing/*`, Google Play)

Estado: **pendiente** en la fachada pública; el motor de pricing/ledger interno ya existe
(`billing/application/processor.py`, con tests).

- [ ] `GET /billing/wallet`, `GET /ledger`, `GET /usage-events`, `POST /topups` compatibles.
- [ ] `POST /billing/google-play/topups/confirm` — no hay ninguna referencia a Google Play en
      `v2/src` todavía; es ingreso real activo en V1, máxima prioridad de negocio.
- [ ] Idempotencia de cargos y compras (clave por interacción/compra, no solo por request).
- [ ] Bono de bienvenida al registrar usuario (enlazar con Capítulo 1).

## Capítulo 5 — Afiliación GetYourGuide

Estado: **pendiente**. Hoy vive en `app/services/referral_service.py` (V1).

- [ ] Puerto neutral para "buscar experiencias reservables" reutilizable desde Catálogo y Chat.
- [ ] Migrar heurísticas de matching (`_search_getyourguide_product_links`, scoring por
      solapamiento de tokens) o simplificarlas si ya no aportan.
- [ ] Flag `getyourguide_referrals_enabled` en `Settings` V2.

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

- [ ] Adaptador de sesiones V1 (`/sessions`, presencia, estado).
- [ ] Orquestación de salas V2 equivalente a `call_room_service.py` (917 líneas en V1).
- [ ] Puente de protocolo: traducir `/ws/calls/{callId}` (V1, lo que habla Ionic hoy) al
      protocolo neutral ya construido en `voice/gateway.py` (`/ws/v2/live`, ver
      `docs/websocket-protocol.md`). Son protocolos distintos, esto es trabajo nuevo.
- [ ] `client-secret`, tools de realtime, análisis de fotos (`/realtime/*`).
- [ ] Pruebas de reconexión, interrupción, tools y fallback con la app real (criterio explícito
      del roadmap: el WebSocket no se considera compatible sin esto).

## Capítulo 7 — Legal y metadatos de app

Estado: **pendiente**, bajo riesgo técnico.

- [ ] `/privacy-policy`, `/legal` servidos desde V2.
- [ ] `/api/app/version` (control de versión mínima Android/iOS).

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
- [ ] Pedido explícito (2026-09-05): desde el mapa de Ciudades y POIs, poder lanzar una
      llamada real de chat o voz sobre un POI concreto (usando su prompt y contexto real,
      no el prompt neutro de "Probar proveedor") para ajustar prompts y tools viendo el
      comportamiento en caliente, directamente desde el panel.
- [ ] CRUD de catálogo (sigue siendo explorador de solo lectura).
- [ ] Historial de prompts navegable más allá de las versiones ya listadas, dashboard de
      salud más allá de Pulso.
- [ ] Bug encontrado (no corregido, no es de esta sesión de trabajo): una sesión de Gemini
      Live falló en producción-local con `'UsageMetadata' object has no attribute
      'candidates_token_count'` (ver Registros, 2026-09-05 16:11). Está en el mapeo de
      eventos de `voice/providers/gemini_live.py`.

## Capítulo 9 — Corte a producción

Estado: **pendiente**, es el último capítulo por diseño.

- [ ] Backup completo de la base de producción antes de nada.
- [ ] `./bin/locus up` en local con datos importados, capítulos 1–7 en verde.
- [ ] Desplegar V2 en paralelo en ECS sin tráfico real.
- [ ] Cambiar `apiBaseUrl` de Ionic de `https://api.locusguide.es/api` al host V2.
- [ ] Ventana de observación con V1 disponible para rollback inmediato (revertir la URL).
- [ ] Retirar V1 solo cuando no haya regresiones ni diferencias de facturación.
