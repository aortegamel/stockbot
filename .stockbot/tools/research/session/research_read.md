# research_read

Domain: research
Family: session
Intent: read_research_resource
Output kind: current_snapshot
Source: local
Entity scope: single_session
Time mode: latest_or_as_of

Read one persisted research resource: evidence, freeze, dossier, job, or session record.

## Choose when

- Reading a single evidence, freeze, dossier, job, or session record.

## Reject when

- Not for session state overviews (research_status).
- Coverage artifacts are inspection only, never citable as raw evidence.

## Conflicts with

None

## Related tools

- research_status

## Prerequisites

None

## Required arguments

- `kind` (string): Resource store to read.
- `resource_id` (string): Evidence, freeze, dossier, coverage-artifact, job, or session ID.
- `session_id` (string): Research session ID.

## Optional arguments

- `freeze_id` (string): Optional freeze scope: an evidence read must be a member of that freeze's evidence set.
