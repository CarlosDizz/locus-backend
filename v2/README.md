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
cp .env.example .env
./bin/locus up
```

`./bin/locus` is the single local operations entry point. It supports `up`, `down`, `rebuild`,
`logs`, `status`, `migrate`, `seed`, and the idempotent `import-v1` data migration.

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
