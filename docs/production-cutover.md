# Corte de produccion V1 a V2

La app conserva `https://api.locusguide.es/api`; no se cambia su configuracion. El corte
sustituye los contenedores del proyecto Compose `locus-backend`, conserva el volumen MySQL
`locus-backend_mysql_data` y crea una base separada llamada `locus_v2`.

## Garantias previas

- V1 esta congelada en la etiqueta `v1-production-before-v2-cutover-20260908`.
- La base V1 `locus` no se modifica ni se elimina durante el corte.
- El dump debe superar `gzip -t` antes de parar la API vieja.
- `.env.production` vive solo en EC2, con permisos `600`, y nunca se versiona.
- La primera ejecucion debe hacerse con una copia nueva de los datos; no se cobra ni se
  corrigen movimientos historicos durante la importacion.

## Secuencia

1. Copiar esta revision a `/home/ec2-user/locus-backend-v2` y crear `.env.production` con
   `bin/build-production-env.py`; el script traduce los nombres sin mostrar secretos.
2. Ejecutar `./bin/production build` mientras V1 sigue atendiendo trafico.
3. Ejecutar `./bin/production backup` y conservar el nombre del dump verificado.
4. Ejecutar `./bin/production prepare-db` para crear/migrar/sembrar `locus_v2`.
5. Ejecutar `./bin/production import-v1` y comprobar los conteos impresos.
6. Ejecutar `./bin/production up`; Compose usa el nombre de proyecto `locus-backend`, por lo
   que sustituye API y Caddy, conserva MySQL y anade realtime, worker, panel y Valkey.
7. Ejecutar `./bin/production smoke` y revisar `./bin/production logs api`.
8. Probar en un movil login, mapa, POI, chat, llamada y saldo antes de dar el corte por cerrado.

## Rollback

La base `locus` y el directorio `/home/ec2-user/locus-backend` permanecen intactos. Para volver:

1. Desde V2, parar `api`, `realtime`, `worker`, `control-panel`, `caddy` y `valkey`, sin usar
   `down -v` y sin tocar `mysql`.
2. En `/home/ec2-user/locus-backend`, levantar `api`, `mysql` y `caddy` con el Compose V1.
3. Verificar `https://api.locusguide.es/api/health` y una sesión real de la app.

No se elimina `locus_v2`, `locus`, ningún dump ni ningún volumen hasta terminar la ventana de
observacion. Está prohibido usar `docker compose down -v` durante el corte o el rollback.
