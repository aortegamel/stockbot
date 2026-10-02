# research_add_evidence

Domain: research
Family: session
Intent: add_research_evidence
Output kind: governed_action
Source: local
Entity scope: single_session
Time mode: current

Record one finding on a research job; provenance, point-in-time, and IDs are kernel-validated.

## Choose when

- Recording a finding from a dispatched research job.

## Reject when

- Not for session state overviews (research_status).

## Conflicts with

None

## Related tools

- research_status

## Prerequisites

None

## Required arguments

- `item` (object): Finding whose provenance must match its claim_kind and the owning job's domain: SEC jobs cite the get_sec_document source_handle + cited passage, or the persisted tool_result_id of an SEC structured response they read plus the cited record values, for observed_fact; FINRA/WEB jobs cite the persisted tool_result_id of the FINRA/search_web response they read plus the cited record values/highlight (the kernel replays the persisted result itself); search scope for absence_observation.
- `job_id` (string): Running job ID the finding belongs to.
- `session_id` (string): Research session ID.

## Optional arguments

None
