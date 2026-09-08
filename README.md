# Locus Backend V2

Locus V2 is a greenfield modular monolith. It intentionally does not import code from the
legacy `app/` package.

## Runtime processes

- `api`: mobile and control-panel REST APIs.
- `realtime`: Locus WebSocket protocol and live-provider sessions.
- `worker`: durable asynchronous jobs.

All processes use the same domain and application packages. They can run in Docker Compose
on EC2 now and move independently to ECS later.

## Local development

```bash
cp .env.example .env.local
./bin/locus up
```

Docker Compose reads `.env.local` by default for every service. Set
`LOCUS_ENV_FILE=.env.production` when a different environment file is required.

`./bin/locus` is the single local operations entry point. It supports `up`, `down`, `rebuild`,
`logs`, `status`, `migrate`, `seed`, `test`, and the idempotent `import-v1` data migration.

REST health: `http://localhost:8100/api/v2/health`

Realtime health: `http://localhost:8101/ws/v2/health`

## Non-negotiable boundaries

- Locus owns conversation state and the client protocol.
- Provider adapters never contain product or billing policy.
- Provider usage is normalized before cost and customer charging are calculated.
- Published prompts and routing profiles are immutable snapshots for active sessions.
- Audio is transient and is not persisted.
- Important searchable content is relational, not hidden in JSON fields.

See `docs/project-structure.md` for the hexagonal package convention.

## Production

`docker-compose.prod.yml` replaces V1 on the existing EC2 host while retaining its named MySQL
volume. V2 uses a separate `locus_v2` database in that same MySQL process, which avoids running
a second database on the 1 GiB instance and keeps the V1 data intact for rollback.

Create `.env.production` from the example, then use `./bin/production`. The intended cutover
order is `build`, `backup`, `prepare-db`, `import-v1`, `up`, and `smoke`. Never run `up` before
the verified backup and import have completed.
