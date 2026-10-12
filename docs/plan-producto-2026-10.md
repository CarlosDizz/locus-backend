# Plan de producto — octubre 2026

Plan de trabajo acordado el 2026-10-10, después de la prueba real en Roma
(2026-10-06). Recoge lo que se decidió, en qué orden y por qué, para que
cualquiera que retome el trabajo —persona o agente— no tenga que reconstruirlo.

La migración V1 → V2 está cerrada; su historia sigue en [roadmap.md](roadmap.md).

## Orden de trabajo

1. **Diseño** de todas las pantallas nuevas, dentro de la propia app y con datos
   de ejemplo (ver [Diseño](#diseño)). Se aprueba antes de programar nada.
2. **Valoraciones** de llamadas y sitios.
3. **Planificador de viaje** por chat, con referidos de GetYourGuide.
4. **App nativa de iPhone + llamada en itinerario**, juntas.

Orden elegido así: las valoraciones son lo más pequeño y dan datos desde el
primer día para decidir todo lo demás; el planificador funciona ya en la PWA y es
lo único que genera ingresos directos; el itinerario necesita la app nativa para
funcionar con la pantalla bloqueada, y la app nativa gana sentido con él.

## Valor de cada funcionalidad

| Funcionalidad | Valor |
|---|---|
| Llamada en itinerario | La que más diferencia a Locus: ninguna audioguía de bus sabe qué tienes delante ni te contesta. Necesita app nativa. |
| Valoraciones | No atraen a nadie por sí solas, pero dicen qué sitios y qué guía funcionan, y hacen que el mapa refleje lo que gusta. |
| Planificador + GetYourGuide | La única que da dinero directo, y engancha al usuario antes de llegar a la ciudad. |

## Diseño

### Barra baja del mapa

```
[ Llamada en itinerario ]   [ Chat ]   [ Mis viajes ]
```

- El chat se queda en el centro, donde la gente ya lo busca.
- Izquierda: moverse por la ciudad (itinerario). Derecha: planificar (Mis viajes).
- Las valoraciones no tienen botón: aparecen al colgar, en la ficha del sitio y en
  el tamaño de los marcadores.
- **Pendiente de decidir:** qué hace el botón de itinerario mientras la función no
  exista. Propuesta: que no aparezca hasta que funcione (un botón muerto o un
  «Próximamente» resta).

### Pantallas a diseñar

1. Barra baja del mapa (arriba).
2. **Mis viajes**: lista de viajes y estado vacío con llamada a planificar.
3. **Chat del planificador**: chat aparte del chat del mapa.
4. **Planning de un viaje**: días y franjas, lo reservado distinguido de lo
   propuesto, botones «Reservar» de GetYourGuide, aviso de enlace de afiliado y
   botón «Editar» que reabre el chat.
5. **Valoración al colgar**: pulgar y comentario público sobre el sitio; nota y
   comentario privado sobre el guía.
6. **Ficha del sitio**: comentarios y porcentaje de «me gusta».
7. **Mapa**: marcadores en 3-4 tamaños según visitas y «me gusta».
8. **Llamada en itinerario**: qué estás viendo ahora y qué viene.

**Estado (2026-10-10):** tandas 1 (barra baja, Mis viajes, planning, chat del
planificador) y 2 (valoración al colgar, opiniones en la ficha, marcadores por
tamaño) aprobadas por el usuario; tanda 3 (llamada en itinerario) hecha.
Rutas de diseño en la app: `/trips`, `/trips/:id`, `/trips/:id/chat`,
`/call-review`, `/itinerary`. Los datos de ejemplo están marcados con
«DISEÑO (2026-10-10)» en el código. Para más adelante: añadir un mapa al
planning con las paradas de cada día.

Se hacen en el código real de la app (mismos estilos: crema/arena, terracota para
acciones, azul petróleo para etiquetas, Fraunces en títulos, Manrope en texto,
paneles translúcidos sobre el mapa) con datos de ejemplo del viaje a Roma del
5 al 9 de octubre. Lo que se apruebe es la base sobre la que luego se conectan
los datos reales. Se enseña por tandas: (1-4), (5-7), (8).

## Progreso de la construcción (2026-10-10, en local, sin desplegar)

- **Valoraciones: hechas.** Tabla `call_reviews`, moderación con el moderador de
  OpenAI, `POST /api/calls/{id}/review`, `GET /api/catalog/pois/{id}/reviews`,
  popularidad (`popularity` en `/catalog/pois`, escalones en
  `reviews/service.py::popularity_tiers`), sección «Valoraciones» en el panel. La
  app lleva a valorar al colgar una llamada de más de un minuto.
- **Planificador: hecho.** Tablas `trips` y `trip_items`, `/api/trips`, prompt
  `chat.trip.planner` editable en el panel, herramientas en `trips/service.py`,
  búsquedas de GetYourGuide en paralelo (~30 s un turno que propone un día).
- **Llamada en itinerario: hecha (PWA).** `calls/itinerary.py` (seguimiento puro:
  lado, «ahora», «próximamente», cuándo contar), modo `itinerary` en la sala,
  evento `location.update` del anfitrión, perfil `itinerary_guide` con su prompt,
  panel en la pantalla de llamada. Probada con un bus simulado por la Via dei Fori
  Imperiali: contó la Columna de Trajano y el Coliseo con su lado. Pendiente: en
  iPhone con la pantalla bloqueada solo en la app nativa (modos de fondo).
- Encontrado de paso: la tarifa de luna no tenía precio para la búsqueda web; se
  añade tarifa nueva (2026-10-10) con `tool_call_usd` 0,01.

## Valoraciones

**Qué hace.** Al colgar una llamada, una pantalla con dos partes:

- **El sitio**: pulgar arriba / abajo y comentario público para el resto de
  usuarios.
- **El guía**: nota y comentario privado, solo para nosotros.

**Piezas.**

- Backend: tabla de valoraciones (llamada, usuario, sitio, voto, comentario
  público, nota del guía, comentario privado). Agregados por sitio: número de
  visitas (llamadas) y ratio de «me gusta».
- Moderación de los comentarios públicos antes de mostrarlos (moderador de OpenAI,
  gratuito, o luna).
- Mapa: tamaño del marcador por visitas y ratio, en 3-4 escalones.
- Ficha del sitio: comentarios y porcentaje.
- Panel: valoraciones del guía y, opcional, un resumen automático de cada llamada
  (si se ciñó a datos, si repitió, si cortó) para sacar conclusiones sin
  escucharlas.

**Tamaño:** medio. Funciona en la PWA, no depende de nada.

## Planificador de viaje

**Qué hace.** Un chat aparte, sin formularios, que pregunta conversando quién va,
cuándo llega y se va, qué hay ya reservado y qué ritmo lleva el grupo, y va
proponiendo opciones y dónde colocarlas. Al final se ve como un planning, con un
botón «Editar» que reabre el chat con el plan cargado.

Ejemplo que guía el diseño: «llego a Roma el 5 de octubre por la tarde y me
vuelvo el 9 a mediodía; voy con mis padres, que son mayores; tengo entradas para
el Coliseo el 6 a las 11 y para los Museos Vaticanos el 7 a las 12».

**Piezas.**

- Chat con su propio prompt, sobre la infraestructura del chat del mapa (luna,
  herramientas).
- **El plan es datos, no texto**: el modelo lo escribe con herramientas (fijar
  fechas y viajeros, añadir reserva fija, colocar propuesta en día y hora, mover,
  quitar). Se guarda en base de datos; no depende de la memoria del modelo.
- Datos reales para proponer: catálogo de sitios, Google Places, búsqueda web
  (horarios, cierres) y tiempos de traslado (con gente mayor, nada de encadenar
  sitios a 40 minutos andando).
- Vista de planning y botón «Editar».
- Enlaces del plan a la ficha de cada sitio y a su llamada. Más adelante: el guía
  sabe lo que el grupo tiene planeado, y el itinerario del bus puede partir del
  plan.

**Referidos de GetYourGuide — lo potente de esta función.**

- Ya existe la base: `affiliates/service.py` busca páginas de producto reales de
  GetYourGuide y les añade `partner_id`; el `partner_id` está configurado en
  producción.
- El planificador propone lo reservable donde aporta (entradas sin cola, visitas
  guiadas, paseos gastronómicos), con botón «Reservar» rastreado en el chat y en
  el planning.
- Respetar lo ya reservado: no vender otra entrada del Coliseo a quien ya la
  tiene; proponer lo que encaja alrededor.
- Mezclar gratuito y reservable: un plan que es todo enlaces de pago deja de
  inspirar confianza.
- Aviso discreto de enlace de afiliado en el planning (obligatorio en la UE).
- **Solicitar la API de partners de GetYourGuide** (catálogo con precios, horarios
  y disponibilidad reales). Se empieza con la búsqueda actual y se pasa a la API
  cuando la concedan.

**Tamaño:** medio-grande, varios días; el grueso es el prompt y la vista.
Funciona en la PWA.

## App nativa de iPhone + llamada en itinerario

### Por qué nativa

Los problemas del iPhone con la PWA son del navegador: permisos que se piden en
cada llamada, el aviso «No es seguro» de Chrome, toques que no llegaban, sonido.
Y la llamada en itinerario necesita audio y ubicación con la pantalla bloqueada,
que iOS solo permite a apps nativas. El usuario pagará la cuota de Apple (99 $/año)
cuando la app esté entera.

### Requisitos de Apple que hoy no cumplimos

- **Recargas por compra dentro de la app (StoreKit).** Los créditos de IA son
  contenido digital: en la app de iPhone no se puede usar Paddle ni enlazarlo.
  Comisión del 15 % con el programa para pequeños desarrolladores (si no, 30 %).
  La PWA sigue con Paddle. **Decisión pendiente** de confirmar con el usuario.
- **Iniciar sesión con Apple**, obligatorio si se ofrece Google.
- **Borrar la cuenta desde la app.** No existe ni en la app ni en el backend;
  Google Play también lo exige.

### Estado de partida

Proyecto iOS existente (Capacitor 8 con Swift Package Manager, último cambio en
mayo de 2026), login de Google para iOS configurado, permisos de micrófono,
ubicación, cámara y fotos declarados. Xcode 26.2 y simuladores en el Mac.

### Fases

1. **Sin pagar:** compilar en simulador y arreglar lo que se haya quedado atrás;
   instalar en el iPhone del usuario con Apple ID gratuito (caduca a los 7 días);
   plugin propio de StoreKit 2 (como el de Google Play) y verificación de compras
   en el backend, probado con compras locales; borrado de cuenta; backend de
   «Iniciar sesión con Apple».
2. **Al pagar:** activar permisos de la cuenta (Sign in with Apple e In-App
   Purchase), productos en App Store Connect, compras de prueba en sandbox,
   TestFlight, ficha y revisión.

### Llamada en itinerario

Sustituir la audioguía pregrabada del bus turístico: la llamada sigue la ubicación
y el guía cuenta lo que se está viendo.

- App: posición, rumbo y velocidad cada pocos segundos durante la llamada (rumbo
  para saber qué queda a la derecha o a la izquierda).
- Backend: la llamada no está atada a un sitio. El puente cruza la posición con
  el catálogo y Google Places, decide qué merece la pena, y le pasa al guía una
  nota silenciosa («a 100 m a vuestra derecha aparece el Coliseo»); si nadie
  habla, el guía arranca solo. Sin repetir sitios y sin saturar en zonas densas.
- Coste: una hora de bus con el guía hablando mucho no es una llamada de 10
  minutos; calcular y probablemente pensar un precio para este modo.
- La lógica del backend sirve igual para Android y iOS.

**Tamaño:** grande, varios días más pruebas en la calle.

## Pendientes sueltos

- **Prompt del guía**: en la nota de cada turno, buscar nombres, fechas y cifras
  antes de decirlos (en una llamada al Pasaje de Lodares dijo el arquitecto de
  memoria, mal, y se corrigió después). Propuesto, sin respuesta todavía.
- **Probador de llamadas del panel**: usa otro camino (`voice/gateway.py`,
  `/ws/v2/live`) y otro prompt (`voice.poi.guide`), así que no refleja las llamadas
  reales. Propuesto rehacerlo sobre el camino de las llamadas.
- **Clave de Maps «My project»** (APK de Android y local) sin restringir. Se hace
  en la consola de Google Cloud con la sesión del usuario, por Playwright.
- **Sonido en algunos iPhones**: sin diagnosticar; faltan modelos, navegador y
  qué se oía. Probablemente lo resuelve la app nativa.
- **«No es seguro» en Chrome de iPhone**: sin confirmar si volvió tras el cambio
  del service worker (2026-09-27). Probablemente lo resuelve la app nativa.
- **La clave de API de Paddle caduca el 11/12/2026.**

## Hecho el 2026-10-10 (contexto)

Lo que se arregló tras la prueba de Roma, ya en producción (`bf622f9`):

- Sin compresión de contexto: cada recorte borraba la conversación entera.
- La memoria pasa de audio a texto cada 5 turnos, en la misma sesión de voz.
- Transcripción con el propio modelo de voz; fuera el transcriptor externo.
- Fotos convertidas a JPEG en la app; límite del servidor a 8 MB.
- Explicaciones largas: nota silenciosa pegada a cada turno; salida a 8192 tokens.
- Búsqueda de Google activada (el 1007 lo provocaba la compresión), cobrada a
  1,5 céntimos por consulta.
- Fuera `document_poi` en voz y la ficha de apertura con GPT.
- El guía sabe quién habla en cada turno de voz y quién entra y sale.
- Si se queda callado tras una herramienta, el puente le da pie.
- Sesiones de login renovadas con el uso; limpieza de logs y sesiones en el worker.
- Mensaje de ubicación denegada específico para iPhone (Safari y Chrome).

## Publicar en la App Store (plan del 2026-10-12)

Estado: la app de iPhone funciona en TestFlight (versión 12) contra producción.
Ficha creada en App Store Connect: «Locus», id 6821354015, `com.carlos.locusia`.

### Pagos: compras de Apple en el iPhone, Paddle en la web

- Los minutos de guía son contenido digital: en la app de iPhone hay que
  venderlos con las compras de Apple (StoreKit). El enlace a la PWA para pagar
  con Paddle no pasa la revisión en la UE.
- Encaja con cobrar como particular hasta que la app funcione y entonces montar
  la sociedad: Apple vende en nombre de Carlos como hace Paddle (cobra, gestiona
  el IVA, ingresa lo que queda). Con el Small Business Program se queda un 15 %
  sobre el neto: en 4,99 € llegan ~3,50 € frente a 3,44 € con Paddle.
- Un solo saldo: lo recargado en el iPhone vale en la web y al revés.
- Descartado el permiso de enlace externo de la UE: comisiones propias y más
  papeleo para ganar lo mismo o menos.

### Fases

| Fase | Qué | Quién | Tiempo |
|---|---|---|---|
| 0 | Acuerdo de apps de pago (banco + W-8BEN), Small Business Program, condición de comerciante UE (dirección pública; se puede usar una de domiciliación), cuenta Google de prueba con saldo | Carlos | ½ día |
| 1 | Compras de Apple: productos 4,99/9,99/19,99 €, compra en la app, verificación en servidor, abono en el saldo, pruebas sandbox | Claude | 2 días |
| 2 | Iniciar sesión con Apple (obligatorio si hay Google): botón, verificación en servidor, enlace por email con cuentas existentes | Claude | 1 día |
| 3 | Normas: denunciar comentarios y fotos, condiciones de uso, PrivacyInfo.xcprivacy, solo iPhone, quitar «ubicación siempre», repasar botones con (click) | Claude | 1 día |
| 4 | Ficha: capturas 6,9″, descripción es/en, palabras clave, Viajes, edades, privacidad, URL de soporte, notas al revisor | Claude prepara, Carlos revisa | ½ día |
| 5 | TestFlight externo opcional, envío a revisión, correcciones | Los dos | 1–3 días por vuelta |

Unas dos semanas hasta publicar, contando una ronda de rechazo.
