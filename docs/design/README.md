# Design documents

The design record for SourceWeave: why it was built, what it had to do, how it
was structured, which decisions were taken, and which assumptions were tested.

| Document | Answers |
|---|---|
| [Project Charter](charter.md) | Why the project exists, scope, and success criteria |
| [Requirements](requirements.md) | What Release 0.1 must do — user, functional, quality and evaluation requirements, prioritised with MoSCoW |
| [Architecture Overview](architecture-overview.md) | How the system is structured: layers, agents, state, retrieval, failure and security boundaries |
| [Decision Register](decision-register.md) | 22 architecture decisions with context, status and consequences |
| [Architecture Spikes](spikes.md) | Five time-boxed experiments used to test risky assumptions before committing to them |

## How to read these

Start with the **Decision Register** if you want the short version — each entry
states what was decided and why, and the summary table shows current status at a
glance.

Read the **Spikes** if you want the evidence. Four of the five have been run and
carry recorded outcomes, including one where the hypothesis was only partly
supported and that result was kept rather than quietly dropped.

**Architecture Overview section 23** is the honest summary: what the
implementation actually does versus what the original design intended, with
section 23.1 listing what remains outstanding.

## Relationship to the Word originals

These Markdown files are derived from `.docx` originals, which are kept
unchanged as the initial drafts and held outside this repository. The Markdown
copies have since been updated to match the implementation and are the current
version — read these, not the Word files.

`scripts/docx2md.py` performed the original conversion. Re-running it regenerates
from the Word files and would discard the updates, so edit the Markdown directly.

## What changed after implementation

| Document | Update |
|---|---|
| Decision Register | ADR-003, 004, 005, 008, 010, 011 moved from Experimental/Proposed to Accepted; ADR-021 (weighted fusion) and ADR-022 (answered vs verified) added |
| Spikes | Outcomes recorded for all five; scorecard completed |
| Architecture Overview | Section 23 rewritten against the implementation; 23.1 added for outstanding gaps |
| Charter, Requirements | Project name filled in |
