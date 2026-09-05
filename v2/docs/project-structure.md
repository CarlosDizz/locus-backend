# Project structure

Each Locus bounded context follows the same small-framework convention:

- `models.py`: persistence models. Shared identity and timestamps inherit from the database base.
- `repository.py`: outbound persistence adapters inheriting from `SQLAlchemyRepository`.
- `service.py`: application use cases inheriting common CRUD orchestration where appropriate.
- `providers/`: outbound adapters for external AI services.
- `api/` and `entrypoints/`: inbound HTTP/WebSocket adapters only.
- `kernel/`: dependency-light reusable ports and bases; it contains no Locus product rules.

Dependencies point inward: presentation and infrastructure may depend on application/domain code,
but provider SDKs and FastAPI never leak into business policy. Billing, routing, and migration use
explicit use cases instead of generic CRUD whenever an operation has meaningful invariants.

The Angular control panel wraps Google Maps, ECharts, FullCalendar, and Lucide behind feature
components. Map, analytics, and calendar sections are lazy loaded so dashboard integrations do not
inflate the initial control-panel bundle.
