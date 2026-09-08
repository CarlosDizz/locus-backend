# Locus Voice Protocol V2

Status: draft 1

Path: `/ws/v2/voice/{session_id}`

JSON is used for control events. Binary frames contain raw audio and are associated with the
most recent `audio.start` event. Clients never speak a provider-specific protocol.

## Client events

- `session.start`: starts or resumes a Locus voice session.
- `audio.start`: declares format and begins an input stream.
- binary frame: an audio chunk.
- `audio.commit`: completes the current input turn.
- `response.cancel`: interrupts current assistant output.
- `text.send`: sends a typed contribution.
- `tool.approval`: approves a tool requiring explicit user consent.
- `session.close`: closes the session intentionally.

## Server events

- `session.ready`: includes the immutable routing and prompt snapshot IDs.
- `input.transcript.delta` and `input.transcript.done`.
- `assistant.text.delta` and `assistant.text.done`.
- `assistant.audio.start`, binary audio frames, and `assistant.audio.done`.
- `tool.started` and `tool.completed`.
- `provider.changed`: emitted after fallback between turns.
- `usage.updated`: normalized provisional session usage.
- `error`: stable Locus error code plus retryability.
- `session.closed`.

## Invariants

- A session is pinned to a published routing profile and prompt release.
- Provider changes happen between turns unless the failed response is discarded.
- Audio is not persisted.
- Every event carries a sequence number and trace ID.
- Duplicate client event IDs are idempotent.
