# Architecture decision record: V2 foundation

Status: accepted

## Shape

Locus V2 is a modular monolith with three runtime processes: API, realtime gateway, and
worker. The code is divided by business domain rather than technical layer at repository
level.

## Service boundaries

- `ChatService` orchestrates text conversations.
- `VoiceService` orchestrates live voice sessions.
- `LiveProvider` adapters translate between Locus events and provider protocols.
- `UsageNormalizer` translates provider events into neutral usage measurements.
- `ProviderCostCalculator` prices normalized measurements using an immutable snapshot.
- `BillingService` applies exchange rate, margin, rounding, and customer ledger policy.

## Providers at launch

- OpenAI Responses for chat.
- OpenAI Realtime for voice.
- Gemini Live for voice.
- OpenAI GPT Live is catalogued but disabled until the public API is available.

## Environments

Only `local` and `production` exist. V2 runs privately in production behind a feature flag
until rollout. It uses its own database schema, configuration, and endpoints.

## Identity

Users can hold multiple roles. Initial roles are `user` and `admin`. Admin users can access
both the mobile application and control panel. The first administrator is bootstrapped from
an environment setting and then represented normally in the role tables.

## Data retention

Audio is never stored. Audio buffers live only for the active transport operation and are
discarded when a session closes. Transcripts, tool results, configuration snapshots, usage,
and billing records are persisted and removed under account-deletion policy where legally
possible.
