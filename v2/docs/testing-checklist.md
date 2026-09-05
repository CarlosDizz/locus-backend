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
- [ ] `POST /catalog/cities/bootstrap-from-location` — bootstrap automático de ciudad dispersa,
      es una ruta usada en caliente por la app cuando el usuario llega a una ciudad nueva, no
      solo una herramienta de admin.
- [ ] CRUD de ciudades/POIs para uso interno (los mismos endpoints ya cubiertos por
      `admin_catalog.py` del panel; decidir si se reutilizan o se separan).

## Capítulo 3 — Chat (`/api/chat/*`)

Estado: **pendiente**. El dominio `chat` no existe todavía en `v2/src/locus_v2/` — hay que
construirlo desde cero siguiendo el patrón de `voice/` y `catalog/`, no es una adaptación.

- [ ] Modelos y repositorio del dominio Chat.
- [ ] `ChatService` orquestando prompts/tools/modelo configurados desde el panel.
- [ ] `POST /chat/setup`, `POST /chat/messages` compatibles con V1.
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

Estado: **pendiente**. Es el capítulo de mayor riesgo técnico.

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
- [ ] Confirmar en caliente si Usuarios ya trae detalle (sesiones/saldo/actividad) o solo lista.
- [ ] Confirmar en caliente si Conversaciones (calendario) lee actividad real o es maqueta.
- [ ] CRUD de catálogo, historial de prompts, prueba de proveedor desde el panel, dashboard de
      salud más allá de Pulso, auditoría de acciones.

## Capítulo 9 — Corte a producción

Estado: **pendiente**, es el último capítulo por diseño.

- [ ] Backup completo de la base de producción antes de nada.
- [ ] `./bin/locus up` en local con datos importados, capítulos 1–7 en verde.
- [ ] Desplegar V2 en paralelo en ECS sin tráfico real.
- [ ] Cambiar `apiBaseUrl` de Ionic de `https://api.locusguide.es/api` al host V2.
- [ ] Ventana de observación con V1 disponible para rollback inmediato (revertir la URL).
- [ ] Retirar V1 solo cuando no haya regresiones ni diferencias de facturación.
