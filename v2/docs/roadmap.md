# Locus Backend V2 - Hoja de ruta

## 0. Contexto de negocio y estrategia de corte

Locus V1 fue una prueba de viabilidad: demostrar que el producto funcionaba, no un backend
pensado para operarse. Está en la rama `main`, desplegado en ECS y es lo que usa la app Ionic
distribuida hoy. No se toca su contrato de cara a la app.

El objetivo de V2 no es solo paridad tecnica: es que prompts, tools, modelos, parametros de
sesion y catalogo se puedan editar **en caliente desde el panel de control**, sin redeploy ni
tocar codigo, mientras la app Ionic distribuida sigue funcionando exactamente igual.

Estrategia de corte: V2 se construye y se prueba a fondo en local hablando el mismo contrato
HTTP/WebSocket que V1. El cambio a produccion no es una reescritura de la app ni un despliegue
paralelo con riesgo: es cambiar unicamente la URL base del backend que usa Ionic (ver seccion 5),
una vez V2 haya demostrado en local paridad de datos, facturacion y comportamiento. V1 se
mantiene disponible en ECS durante la ventana de observacion por si hay que revertir el cambio
de URL sin tocar la app.

## 1. Objetivo

Construir un backend V2 mantenible, observable y configurable que pueda sustituir al backend actual sin obligarnos a modificar inicialmente la aplicacion Ionic.

La primera meta no es anadir nuevas funciones a la app, sino alcanzar paridad funcional con V1, conservar todos los datos y mejorar la operacion desde el panel de control.

## 2. Principios no negociables

- Python como base del backend.
- Arquitectura hexagonal y modular, evitando duplicacion mediante contratos, clases base y adaptadores.
- MySQL como base de datos.
- Panel de control Angular con acceso exclusivo para administradores.
- Roles `admin` y `user`; inicialmente `dizz01@gmail.com` sera el unico administrador.
- Configuracion separada para chat y voz.
- Prompts, modelos, tools y parametros de sesion configurables desde el panel.
- Soporte de varios proveedores y fallback configurable.
- Compatibilidad inicial con la app Ionic actual, sin cambios obligatorios en el frontend.
- Entornos `local` y `production`.
- Despliegue reproducible mediante Docker Compose y un ejecutable de arranque.
- No guardar audio por ahora.
- Nunca exponer secretos, claves de proveedor o credenciales en el repositorio ni en el panel.

## 3. Arquitectura objetivo

El sistema se mantendra como un monolito modular desplegado en varios procesos:

- **API HTTP:** autenticacion, catalogo, usuarios, facturacion, sesiones y administracion.
- **Realtime Gateway:** WebSockets de voz y salas, independiente del ciclo HTTP normal.
- **Worker:** importaciones, traducciones, enriquecimiento, tareas lentas y reintentos.
- **Control Panel:** aplicacion Angular para configurar y operar el sistema.
- **MySQL:** datos funcionales, configuracion, consumos, facturacion y logs persistentes.

Dominios principales:

- `Identity`: usuarios, roles, autenticacion y sesiones.
- `Catalog`: ciudades, tipos de POI, POIs, traducciones y enriquecimiento.
- `Chat`: conversaciones de texto, prompts, tools y proveedores.
- `Voice`: sesiones de voz, proveedores live, tools y control de turnos.
- `Billing`: precios, normalizacion de uso, saldo, cargos e idempotencia.
- `Operations`: configuracion, logs, metricas, alertas y auditoria.

Proveedores previstos:

- Chat: OpenAI Responses, inicialmente con la familia GPT configurable.
- Voz: OpenAI Realtime y Gemini Live.
- GPT Live: adaptador preparado pero desactivado hasta que la API sea utilizable.

Cada adaptador de proveedor sera responsable de:

- Traducir la configuracion neutral de Locus al formato del proveedor.
- Normalizar eventos, uso y errores.
- Convertir el consumo del proveedor a coste real.
- Exponer sus modelos, voces y capacidades disponibles.

## 4. Estado actual

### Completado

- Estructura V2 limpia y dockerizada.
- API, gateway realtime, worker y panel de control levantables en local.
- Login y sesiones del panel con rol administrador.
- Separacion de configuracion de chat y voz.
- Edicion por campos de modelos, voces, tokens, tools y deteccion de turnos.
- Prompts independientes para chat y voz.
- Adaptadores iniciales de OpenAI y Gemini.
- Registro persistente de eventos, warnings y errores.
- Panel de facturacion y consumo.
- Catalogo administrativo con mapa, lista, filtros y ficha de POI.
- Correccion del encuadre del mapa al cambiar de ciudad.
- Apertura de la ficha desde los marcadores.
- Importacion de datos aprovechables de V1.
- Datos importados: 19 usuarios, 19 wallets, 8 tarifas, 575 eventos de uso, 259 movimientos, 24 recargas, 17 ciudades, 6 tipos de POI, 858 POIs y 58 sesiones.
- Build Angular, Ruff, pruebas principales y comprobaciones focalizadas de Mypy superadas.

### En curso

- Inventario y formalizacion del contrato exacto que consume la app Ionic V1.
- Diseno de la fachada compatible `/api` sobre los servicios V2.

### Deuda conocida

- El tipado Mypy global conserva deuda previa, principalmente por diccionarios sin tipo y SDKs externos. No bloquea la compatibilidad, pero debe reducirse antes del lanzamiento publico.
- Faltan pruebas integrales del protocolo WebSocket usado por las salas de la app actual.

## 5. Contrato de compatibilidad con Ionic

La app actual debe poder cambiar del backend V1 al V2 modificando unicamente la URL de la API. La fachada compatible conservara rutas, cuerpos, respuestas, errores y semantica de V1.

| Area | Rutas V1 que deben conservarse | Estrategia V2 |
| --- | --- | --- |
| Autenticacion | `/auth/login`, `/auth/register`, `/auth/google`, `/auth/me` | Adaptador de identidad y tokens V1 |
| Catalogo | `/catalog/cities`, `/catalog/pois`, `/catalog/pois/{id}` | Lectura V2 usando `legacy_v1_id` |
| Catalogo dinamico | `/catalog/cities/bootstrap-from-location`, `/documentation`, `/access-links` | Servicios V2 y tareas de worker |
| Chat | `/chat/setup`, `/chat/messages` | `ChatService` configurable |
| Facturacion | `/billing/wallet`, `/ledger`, `/usage-events`, `/topups` | Adaptador sobre ledger V2 |
| Google Play | `/billing/google-play/topups/confirm` | Verificacion e idempotencia V2 |
| Sesiones | `/sessions` y operaciones de estado, presencia y llamadas | Adaptador de sesiones V1 |
| Llamadas | `/calls`, join token, leave y end | Orquestacion de salas V2 |
| Realtime | `/realtime/client-secret`, `/realtime/tool`, `/realtime/photo-insight` | Adaptadores de voz y tools |
| WebSocket | `/ws/calls/{callId}` | Puente entre el protocolo V1 y `VoiceService` V2 |

Reglas de compatibilidad:

- Las respuestas publicas utilizaran los IDs numericos originales almacenados en `legacy_v1_id`.
- Las formas JSON se validaran contra contratos copiados de las interfaces TypeScript actuales.
- Los errores conservaran codigos HTTP y campos que Ionic ya interpreta.
- La facturacion tendra claves de idempotencia para impedir cargos duplicados durante reintentos o fallback.
- El WebSocket no se considerara compatible hasta probar reconexion, turnos, tools, presencia, cierre y fallback con la app real.

## 6. Fases de ejecucion

### Fase A - Cerrar los cimientos

Estado: completada.

- Consolidar estructura modular y contratos base.
- Levantar todo el stack con Docker.
- Separar configuracion de chat y voz.
- Importar los datos V1 sin perder relaciones ni IDs historicos.
- Habilitar catalogo, facturacion, proveedores y logs en el panel.

### Fase B - Congelar el contrato V1

Estado: en curso.

- Inventariar todas las llamadas HTTP y WebSocket de Ionic.
- Documentar cuerpos, respuestas y codigos de error.
- Capturar ejemplos reales de cada flujo.
- Crear esquemas Pydantic y fixtures contractuales.
- Identificar diferencias entre IDs V1 y V2.

Resultado: especificacion verificable del contrato que no podemos romper.

### Fase C - Construir la fachada compatible

- Implementar autenticacion y renovacion de sesiones.
- Servir ciudades y POIs localizados con IDs V1.
- Adaptar wallets, ledger, recargas, consumos y compras de Google Play.
- Conectar el chat Ionic con prompts, tools, modelo y fallback configurados en V2.
- Adaptar sesiones, llamadas, presencia y estado de sala.
- Implementar `client-secret`, tools y analisis de fotos.
- Construir el puente WebSocket V1 hacia el protocolo neutral de voz V2.
- Normalizar uso y facturacion en todos los proveedores.

Resultado: Ionic funciona contra V2 sin cambios funcionales en la app.

### Fase D - Pruebas y endurecimiento

- Pruebas de contrato para cada endpoint V1.
- Pruebas de integracion con MySQL real en Docker.
- Pruebas E2E de login, mapa, chat, llamada y facturacion.
- Pruebas WebSocket de reconexion, interrupcion, tools y fallback.
- Pruebas de idempotencia de cargos y compras.
- Simulacion de caida de proveedor y recuperacion.
- Comparacion de respuestas y saldos entre V1 y V2.
- Medicion de latencia y consumo con carga concurrente.

Resultado: evidencia automatizada de paridad y seguridad economica.

### Fase E - Completar el panel operativo

- Listado y detalle de usuarios, roles, sesiones, saldo y actividad.
- CRUD y auditoria de ciudades, POIs, traducciones y tipos.
- Detalle de consumos, llamadas, costes, margen y cargos.
- Explorador de logs con filtros, correlacion y nivel.
- Historial y versionado de prompts y configuraciones.
- Validacion y prueba de proveedor desde el panel.
- Dashboard de salud, latencia, errores y gasto por periodo.
- Acciones administrativas protegidas y auditadas.

Resultado: operar Locus sin editar directamente la base de datos.

### Fase F - Despliegue en produccion

- Preparar variables y secretos fuera del repositorio.
- Ejecutar copia de seguridad completa antes de migrar.
- Aplicar migraciones de forma controlada y reversible.
- Desplegar el stack Docker en EC2.
- Configurar HTTPS, DNS, health checks y reinicio automatico.
- Añadir observabilidad y alertas de errores, latencia y gasto.
- Validar V2 en paralelo sin trafico real.
- Activar la fachada por bandera o cambio reversible de URL.
- Mantener V1 disponible durante la ventana de observacion.
- Automatizar despliegues desde la rama de produccion despues de estabilizar el corte.

Resultado: produccion en V2 con rollback documentado y probado.

### Fase G - Evolucion posterior

Solo se inicia cuando V2 haya alcanzado paridad y estabilidad:

- Nueva revision de experiencia y diseno de la app Ionic.
- Publicacion abierta en Google Play y preparacion de App Store.
- Resumenes de viajes y visitas.
- Albumes de fotos.
- Contenido compartido y grupos mejorados.
- Historial personal y recomendaciones.
- Nuevos proveedores y modelos sin modificar el dominio.

## 7. Criterios de finalizacion

V2 estara lista para sustituir V1 cuando se cumpla todo lo siguiente:

- La app Ionic completa sus flujos principales cambiando solo la URL del backend.
- Todos los endpoints V1 usados tienen pruebas de contrato.
- Login de Google, catalogo, chat, llamadas y pagos funcionan en iOS y Android.
- No existen cargos duplicados en reintentos, reconexiones o fallback.
- Los 858 POIs y el resto de datos migrados son accesibles y consistentes.
- La seleccion de proveedor, modelo, prompt, tools y fallback funciona desde el panel.
- Logs, errores, latencia, uso y coste pueden investigarse desde el panel.
- Existe copia de seguridad, procedimiento de despliegue y rollback probado.
- No hay secretos versionados.
- Las pruebas, lint y build se ejecutan correctamente en CI.

## 8. Riesgos principales y mitigacion

- **IDs incompatibles:** usar `legacy_v1_id`, restricciones unicas y pruebas de correspondencia.
- **Doble facturacion:** dedupe keys, transacciones e idempotencia por interaccion y llamada.
- **Diferencias WebSocket:** puente explicito y pruebas con eventos reales de Ionic.
- **Fallback con contexto inconsistente:** snapshot neutral de sesion y eventos normalizados.
- **Perdida de datos:** importaciones repetibles, conteos, checksums y backup previo.
- **Costes inesperados:** presupuestos por proveedor, limites, alertas y circuit breakers.
- **Secretos expuestos:** rotacion de claves, `.env.local` ignorado y secretos de produccion externos.
- **Procesos lentos:** mover importaciones, traducciones y enriquecimiento al worker.
- **Deuda de tipado:** reducir Mypy por modulos antes de ampliar funcionalidad.

## 9. Siguientes pasos inmediatos

1. Terminar el inventario del contrato Ionic y guardar ejemplos reales.
2. Implementar primero autenticacion, catalogo y facturacion compatibles.
3. Implementar chat compatible y verificar costes e idempotencia.
4. Construir el puente de sesiones, llamadas y WebSocket.
5. Ejecutar la app Ionic completa contra V2 en local.
6. Completar pruebas contractuales y E2E en Docker.
7. Cerrar las pantallas operativas pendientes del panel.
8. Preparar el despliegue paralelo y el plan de rollback en AWS.

## 10. Orden de despliegue recomendado

1. Base de datos y migraciones.
2. Worker.
3. API V2 y fachada compatible.
4. Realtime Gateway.
5. Panel de control.
6. Pruebas de humo internas.
7. Trafico de prueba controlado.
8. Cambio de la app al nuevo backend.
9. Periodo de observacion con V1 disponible.
10. Retirada de V1 cuando no haya regresiones ni diferencias de facturacion.

## 11. Diagnostico cruzado V1 / V2 / panel (2026-09-05)

Verificacion linea a linea sobre el codigo real de ambos repos, complementaria a las secciones anteriores.

### API V2 expuesta hoy

`entrypoints/api.py` unicamente monta routers de administracion: `admin`, `admin_auth`,
`admin_billing`, `admin_catalog`, `admin_configuration`, `admin_logs`, `admin_users` y `health`.
No existe todavia ningun endpoint publico para la app movil. La Fase C descrita en la seccion 6
no ha empezado en codigo, aunque el diseno este avanzado.

### El dominio Chat no existe en V2

`v2/src/locus_v2/` tiene los dominios `identity`, `catalog`, `billing`, `voice`, `observability`,
`admin` y `kernel`, pero no `chat`. Antes de poder servir `/chat/setup` y `/chat/messages` hay que
construir el dominio completo (modelos, repositorio, servicio, adaptador de proveedor) siguiendo
el mismo patron hexagonal que `voice/` y `catalog/`, no solo "adaptarlo".

### Puente de llamadas y WebSocket

`voice/gateway.py` implementa el protocolo neutral V2 (`docs/websocket-protocol.md`,
`/ws/v2/live`), distinto del protocolo que habla hoy Ionic contra V1 en `/ws/calls/{callId}`
(`app/services/call_room_service.py`, 917 lineas). Traducir un protocolo al otro es trabajo
todavia no iniciado y es el riesgo tecnico mas alto de la Fase C.

### Negocio en produccion sobre V1 que V2 no cubre todavia

- **Google Play Billing**: integrado y en uso real en V1 (`app/services/billing_service.py` +
  plugin nativo Capacitor `PlayBilling` en el frontend, `play-billing.service.ts`). No hay ninguna
  referencia a Google Play en `v2/src`. Es el gap de negocio de mayor coste si V1 se apagase sin
  puente.
- **Afiliacion GetYourGuide** (`app/services/referral_service.py`, 620 lineas; el nombre
  confunde, no es un programa de invitar amigos): genera enlaces de afiliado en dos puntos —
  `GET /catalog/pois/{id}/access-links` y la tool de IA `activity_referrals`, que el propio chat
  o la llamada de voz invoca para sugerir experiencias reservables (gated por
  `settings.getyourguide_referrals_enabled`). Es ingreso por afiliacion activo hoy y no tiene
  dominio equivalente en V2. Paridad obligatoria antes del corte: sin esto el chat/voz de V2
  pierde una fuente de ingresos que V1 ya tiene.
- **Legal y version minima de app**: `/privacy-policy` y `/legal` (`app/routes/legal.py`) y
  `/api/app/version` (`app/routes/app_info.py`) se sirven hoy solo desde V1. Impacto tecnico bajo,
  pero bloquean un corte total si no se replican antes de retirar V1.

### Panel de control: estado por seccion

Secciones de `app.component.ts`: Pulso, Conversaciones, Prompts, Proveedores, Ciudades y POIs,
Usuarios, Consumos, Registros.

- Conectadas a datos reales: Pulso (`GET /admin/v2/overview`), Prompts/Proveedores
  (`control-plane.component`), Ciudades y POIs (`catalog-explorer.component`), Consumos
  (`billing-dashboard.component`), Registros (`log-console.component`).
- Usuarios (`user-directory.component`, 69 lineas) filtra por estado pero probablemente le falta
  detalle de sesiones, saldo y actividad por usuario que pide la Fase E.
- Conversaciones (`operations-calendar.component`, 46 lineas, cargado con `@defer`) es el
  candidato mas claro a placeholder visual sin logica real confirmada todavia.
- Fase E sin confirmar en codigo: CRUD de catalogo (hoy es explorador, no editor), historial
  visible de versiones de prompts/config, prueba de proveedor desde el panel, dashboard de salud
  mas alla del resumen de Pulso, auditoria de acciones administrativas.

### Orden recomendado para reanudar Fase C

1. Cerrar el inventario de contrato (Fase B) leyendo `app/routes/*.py` junto a los
   `*.service.ts` del frontend, endpoint por endpoint.
2. Decidir paridad obligatoria vs Fase G para referidos, y confirmar fecha limite para Google
   Play y legal.
3. Auth y catalogo compatibles primero (bajo riesgo, sin dinero de por medio).
4. Construir el dominio Chat desde cero.
5. Billing (incluido Google Play) y puente WebSocket de llamadas al final, con foco en
   idempotencia.
6. En paralelo, cerrar Fase E del panel segun lo que se confirme al abrir cada componente.
