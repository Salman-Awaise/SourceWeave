# Architecture Decision Register

> **Derived from `docx/Architecture Decision Register.docx`, then updated to match the implementation.**
>
> The Word file is the original draft, kept unchanged for history and held
> outside this repository.
> This Markdown copy is the current version and has since diverged:
> Statuses updated for ADR-003, 004, 005, 008, 010 and 011; ADR-021 and ADR-022 added.
>
> Re-running `scripts/docx2md.py` regenerates from the Word file and will
> discard these edits. Edit this file directly instead.

Project: SourceWeave
Version: 0.1
Status: Active
Purpose: Track significant architectural decisions and the evidence supporting them.

---

### Decision Statuses

Proposed — decision requires review or validation.

Accepted — current architectural direction.

Experimental — intentionally being tested and may be reversed.

Superseded — replaced by a later decision.

Rejected — considered but deliberately not selected.

---

## Decision Register

| ADR | Decision | Status | Risk | Validation |
|---|---|---|---|---|
| ADR-001 | Use LangGraph for workflow orchestration | Accepted / Experimental | Medium | Baseline comparison and maintainability |
| ADR-002 | Use explicit shared state as agent contract | Accepted | Low | Integration tests |
| ADR-003 | Separate Planner, Retriever, Generator and Verifier responsibilities | Accepted — validated | High | SPIKE-ARCH-001 complete; benefit is corpus-dependent |
| ADR-004 | Use hybrid vector + web retrieval | Accepted — implemented | Medium | SPIKE-ARCH-002 complete |
| ADR-005 | Use Reciprocal Rank Fusion as baseline fusion strategy | Accepted — extended by ADR-021 | Medium | Fusion mix measured |
| ADR-006 | Use LiteLLM as model-provider abstraction | Accepted | Low | Provider portability tests |
| ADR-007 | Separate model abstraction from model-routing policy | Proposed | Medium | Cost/quality benchmark |
| ADR-008 | Use independent verification before accepting answers | Accepted — implemented | High | SPIKE-ARCH-003 still open |
| ADR-009 | Bound verification retries | Accepted | Low | Retry/termination tests |
| ADR-010 | Refine retrieval based on failed verification | Accepted — implemented | Medium | SPIKE-ARCH-004 complete; retry rate 0.50 -> 0.17 |
| ADR-011 | Keep memory separate from factual evidence | Accepted — implemented | High | SPIKE-ARCH-005 complete; enforced by test |
| ADR-012 | Use Qdrant as v0.1 vector store | Accepted | Medium | Retrieval benchmark |
| ADR-013 | Use Tavily as v0.1 web retrieval adapter | Accepted | Medium | Retrieval/provider tests |
| ADR-014 | Require stable evidence identifiers | Proposed | Medium | Citation integrity tests |
| ADR-015 | Use LangSmith as initial tracing solution | Accepted | Low | Observability review |
| ADR-016 | Use Ragas as initial RAG evaluation framework | Accepted / Experimental | Medium | Human-vs-automated correlation |
| ADR-017 | Keep evaluation isolated from runtime memory | Accepted | Low | Evaluation test |
| ADR-018 | Treat runtime verification threshold separately from release quality gates | Accepted | Low | Scorecard review |
| ADR-019 | Add explicit terminal result states | Proposed | Low | Pipeline contract tests |
| ADR-020 | Introduce provider interfaces only where they improve testability/replaceability | Proposed | Low | Architecture review |
| ADR-021 | Weight rank fusion by source type, with a per-type floor | Accepted — implemented | Medium | Fusion mix measured before and after |
| ADR-022 | Track whether the Generator answered, separately from verification | Accepted — implemented | Low | Pipeline contract tests |

---

## ADR-001 — LangGraph Orchestration

### Context

SourceWeave requires:

- sequential agent execution;
- shared state;
- conditional routing;
- retry loops;
- observable execution.

### Decision

Use LangGraph as the Release 0.1 orchestration mechanism.

Baseline flow:

```text
Planner → Retriever → Generator → Verifier
                  ↑                    │
                  └──── retry ─────────┘
```

### Rationale

LangGraph provides explicit state-machine behavior and conditional edges rather than hiding orchestration inside prompts.

### Alternatives

- custom Python orchestration;
- queue/event-driven agents;
- single ReAct-style agent;
- another agent framework.

### Consequences

#### Positive

- graph behavior is explicit;
- retry logic is visible;
- state transitions are testable;
- topology can evolve.

#### Negative

- framework dependency;
- added complexity compared with a simple Python pipeline;
- developers must understand graph semantics.

### Validation

Compare the multi-agent graph against a simpler baseline.

If the graph provides no meaningful quality, maintainability, or observability benefit, simplification remains acceptable.

---

## ADR-002 — Explicit Shared State

### Decision

Use a typed shared pipeline state as the primary contract between agents.

Agents should return state updates rather than directly invoking downstream agents.

### Rationale

Explicit state improves:

- debugging;
- tracing;
- unit testing;
- retry behavior;
- reproducibility.

### Constraint

New state fields must have clear ownership and purpose.

Avoid using state as an uncontrolled dumping ground.

---

## ADR-003 — Four-Agent Responsibility Model

### Decision

Begin with four logical responsibilities:

```text
Planner
Retriever
Generator
Verifier
```

### Status

Accepted — validated, with a corpus-dependent caveat.

SPIKE-ARCH-001 compared the four-agent pipeline against a single-agent baseline
holding model, temperature, generation rules and retrieval backends constant.

On a corpus that can answer the questions: faithfulness +0.084, context
precision +0.037, at roughly +2.3 s per question.

On a corpus that cannot: the baseline wins three of four metrics. Decomposition
retrieves more documents, raising recall and lowering precision.

The topology is retained, with the understanding that its benefit depends on
corpus fit rather than being unconditional.


### Rationale

Each component represents a different concern and can be independently evaluated.

### Key Question

Does the separation create enough quality and maintainability benefit to justify additional model calls and latency?

### Validation

Compare against:

```text
Retriever → Generator
```

and/or:

```text
Single-Agent Baseline
```

The four-agent topology is not considered permanently justified until the benchmark supports it.

---

## ADR-004 — Hybrid Retrieval

### Decision

Support vector, web and hybrid retrieval.

### Status

Accepted — implemented.

Both backends contribute per sub-query and are fused by rank. Hybrid degrades to
whichever backend is available, and broadens to an unused backend when the
selected strategy returns no evidence at all.


### Rationale

Different questions require different knowledge sources:

```text
indexed/private knowledge → vector
current/public knowledge  → web
mixed research            → hybrid
```

### Validation

Evaluate each mode independently.

Measure:

- context precision;
- context recall;
- answer quality;
- latency;
- cost.

Hybrid retrieval should not automatically be assumed to outperform simpler retrieval.

---

## ADR-005 — Reciprocal Rank Fusion

### Decision

Use RRF as the initial hybrid-result fusion strategy.

### Rationale

RRF:

- is simple;
- does not require score normalization;
- combines heterogeneous ranked lists;
- provides a strong baseline.

### Status

Accepted — extended by ADR-021.

RRF remains the base strategy. Validation showed it under-determined: with each
result appearing in a single ranked list, every score collapses to about 1/(k+1)
and ordering between a primary source and a web summary is arbitrary.
Source-type weighting was added on that evidence.


### Validation

Compare against:

- concatenation;
- weighted score fusion;
- reranking.

Do not add advanced reranking unless the benchmark demonstrates a meaningful gap.

---

## ADR-006 — LiteLLM Provider Abstraction

### Decision

Use LiteLLM as the primary model-provider abstraction.

### Rationale

Agents should depend on a common invocation layer rather than directly coupling to one model vendor.

### Important Distinction

LiteLLM answers:

> How can SourceWeave invoke different providers?

It does not itself define:

> Which model should execute each SourceWeave role?

That is a separate architecture decision.

---

## ADR-007 — Model Routing Policy

### Decision

Keep model selection configuration-driven.

Release 0.1 may initially use one model across agents.

The architecture must permit future configuration such as:

```text
planner_model
generator_model
verifier_model
```

### Status

Proposed.

### Trigger

Introduce role-specific routing only if evaluation shows meaningful cost, latency, or quality improvements.

---

## ADR-008 — Independent Verification

### Decision

Generation and verification remain separate responsibilities.

The same component that generates an answer should not be treated as sufficient proof that the answer is grounded.

### Status

Accepted — implemented.

Verification runs on every query, and the faithfulness score is recomputed in
code from the claim lists rather than taken from the model.

SPIKE-ARCH-003, which would isolate the value of independent verification
against a no-verifier variant, has not yet been run.


### Key Hypothesis

Independent verification reduces unsupported factual claims enough to justify:

- another LLM call;
- additional latency;
- increased inference cost.

### Validation

Benchmark verified and non-verified variants against labeled factual questions.

---

## ADR-009 — Bounded Verification Retries

### Decision

Verification retries must always have a configured upper bound.

### Rationale

Unbounded agent loops create unacceptable:

- latency;
- cost;
- reliability risk.

### Required Behavior

After retry exhaustion, return an explicit qualified state rather than continuing indefinitely.

---

## ADR-010 — Evidence-Driven Retry Queries

### Context

Repeating the same retrieval after verification failure may simply reproduce the same evidence.

### Decision

The target architecture will construct retry retrieval queries using unsupported claims or identified evidence gaps.

Conceptually:

```text
Unsupported Claims
        ↓
Retry Query Builder
        ↓
Retriever
```

### Status

Accepted — implemented.

Option A was taken: the Verifier supplies refined queries directly, built from
the specific claims that failed. Prior evidence is carried into the retry and
fused with the new results rather than discarded.

Measured on a six-question set: retry rate fell from 0.50 to 0.17 and the number
of questions producing scorable answers rose from 3/6 to 6/6.

Refinement is skipped when the retry budget is already exhausted, so no model
call is spent on a result that cannot be used.


### Alternatives

#### Option A

Verifier directly supplies refined queries.

#### Option B

Route failure information back through Planner.

#### Option C

Use a deterministic query builder.

### Initial Preference

Start with the simplest measurable approach.

A deterministic or structured refinement mechanism is preferable before adding another unconstrained agent loop.

---

## ADR-011 — Memory Is Context, Not Evidence

### Decision

Persistent conversational memory must remain logically separate from retrieved factual evidence.

Target state should distinguish:

```text
original_query
chat_history
memory_context
retrieved_documents
```

### Status

Accepted — implemented.

`memory_context` is a distinct state field and reaches the Planner only. It is
never passed to the Generator, so a remembered statement cannot become a cited
source. A unit test asserts memory text cannot appear in the evidence prompt.

Consequence found during validation: memory can steer what is searched but can
never itself be the answer. A question whose answer exists only in memory
returns NO ANSWER. That is the intended trade-off, since mem0 stores a
model-written paraphrase and citing it would break the guarantee that every
claim traces to a retrievable source. The response now states this explicitly.


### Rationale

Combining memory directly with the original research query creates risks involving:

- evaluation contamination;
- loss of provenance;
- unexpected retrieval changes;
- user-specific facts being mistaken for evidence.

### Evaluation Rule

Standard benchmarks execute with memory disabled unless memory behavior is specifically under test.

---

## ADR-012 — Qdrant Vector Store

### Decision

Use Qdrant as the Release 0.1 vector-store implementation.

### Status

Accepted for v0.1.

### Consequence

Agent-level code should still depend on a retrieval contract rather than Qdrant-specific objects.

This preserves future replaceability.

---

## ADR-013 — Tavily Web Retrieval

### Decision

Use Tavily as the initial web-search adapter.

### Status

Accepted for v0.1.

### Consequence

Tavily-specific result structures must be normalized into SourceWeave's common retrieval domain model.

---

## ADR-014 — Stable Evidence Identity

### Decision

Every retrieved chunk should eventually carry stable identity.

Recommended fields:

```text
document_id
chunk_id
source
source_type
```

### Rationale

Stable identity enables:

- reliable deduplication;
- citation validation;
- benchmark analysis;
- tracing;
- retrieval caching.

Content prefixes should not be the long-term identity mechanism.

---

## ADR-015 — LangSmith Observability

### Decision

Use LangSmith as the initial trace platform.

### Architecture Rule

SourceWeave's functional correctness should not require LangSmith availability.

Tracing failure should generally degrade observability, not prevent answering a query.

---

## ADR-016 — Ragas Evaluation

### Decision

Use Ragas as the Release 0.1 RAG-quality evaluation framework.

Initial metrics include:

- Faithfulness;
- Answer Relevancy;
- Context Precision;
- Context Recall.

### Constraint

Ragas scores must not become the only source of product truth.

Important changes may also require labeled or human-reviewed evaluation.

---

## ADR-017 — Evaluation Isolation

### Decision

Benchmark execution must be reproducible and isolated from uncontrolled persistent conversational memory.

### Required Benchmark Metadata

Capture:

- dataset version;
- code revision;
- model configuration;
- retrieval configuration;
- threshold;
- retry policy.

---

## ADR-018 — Runtime Threshold vs Release Threshold

### Decision

Runtime verification thresholds and product/release quality thresholds are different concepts.

Example:

```text
Runtime:
faithfulness threshold used for routing

Release:
aggregate Ragas score required for release
```

Changing one does not automatically imply changing the other.

---

## ADR-019 — Explicit Terminal Status

### Decision

Move toward an explicit result status:

```text
VERIFIED
BEST_EFFORT
FAILED
```

instead of relying only on:

```text
is_verified = True / False
```

### Rationale

A Boolean cannot adequately distinguish:

- verified success;
- retry-exhausted answer;
- infrastructure failure.

---

## ADR-020 — Pragmatic Provider Abstraction

### Decision

Do not perform a large architecture rewrite purely to introduce interfaces.

Add provider boundaries where they provide measurable value through:

- testability;
- mocking;
- fallback behavior;
- provider replacement.

### Principle

Prefer sufficient abstraction over speculative abstraction.

---

## ADR Governance

A new ADR is required when a decision:

- materially changes graph topology;
- introduces a new external platform dependency;
- changes persistence architecture;
- changes verification behavior;
- changes evidence provenance;
- materially affects security boundaries;
- changes evaluation methodology;
- creates meaningful long-term switching cost.

Small implementation choices do not require ADRs.

---

## ADR Review in Agile Delivery

Architecture decisions should participate in normal backlog refinement.

A proposed ADR may generate:

```text
Architecture question
       ↓
Spike / experiment
       ↓
Evaluation
       ↓
ADR decision
       ↓
Implementation stories
```

This prevents architecture from becoming a large up-front design phase.

Architecture work remains iterative and evidence-driven.

---

## ADR-021 — Source-Type Weighted Fusion

### Context

Architecture Overview section 8 listed weighted fusion and source-aware ranking
as candidate future strategies, not to be added until evaluation demonstrated a
need. Evaluation demonstrated the need.

Plain Reciprocal Rank Fusion only distinguishes documents that appear in several
ranked lists. When each result appears in exactly one list, every score collapses
to approximately `1 / (k + 1)` and the ordering between a primary source and a
web summary is close to arbitrary.

Observed: a question about an indexed paper produced three web citations,
including a video link, while the paper itself sat unused at rank 1.

### Decision

Apply a per-source-type multiplier during fusion, and reserve a minimum number
of result slots for each source type that returned results.

```text
DOCUMENT_WEIGHT      1.2    favour indexed primary sources
WEB_WEIGHT           1.0
MIN_PER_SOURCE_TYPE  2      slots reserved per backend
```

### Status

Accepted — implemented.

### Consequences

Weighting is decisive rather than marginal. The `1 / (k + rank)` curve is nearly
flat, so at 1.2 a document outranks a top web result from roughly twelve places
below. Since a backend returns only `SIMILARITY_TOP_K` results, any weight above
1.0 means documents outrank web across the whole realistic range.

Applied alone, weighting removed web results entirely from a question that asked
for recent developments, which defeats the purpose of hybrid retrieval. The
per-type floor exists to prevent that: weighting decides order, the floor
protects representation.

The floor has no quality bar, so a poor web result can occupy a reserved slot. A
relevance-score threshold was tested and rejected: the irrelevant result scored
0.80, higher than several legitimate sources.

Measured outcome on the original failing question: zero primary-source citations
became six, with two web sources retained for the part of the question that
needed current information.

---

## ADR-022 — Answered Separated From Verified

### Context

A response that declines for lack of evidence is truthful, so its claims are
supported and it scores 1.0 faithfulness — correctly. It therefore displayed as
`VERIFIED  confidence 0%`, which reads to a user as a successful answer.

Related to ADR-019, which proposes explicit terminal result states.

### Decision

Track whether the Generator produced an answer separately from whether
verification passed, and surface three states rather than a boolean.

```text
VERIFIED     answered, and every claim held up
UNVERIFIED   answered, but claims were not supported
NO ANSWER    declined for lack of evidence
```

### Status

Accepted — implemented.

### Consequences

`answered` and `documents_retrieved` are part of the public result, so a caller
reading the result programmatically can make the same distinction as the CLI.

A declined response now cites nothing. Previously the retrieved documents were
listed as sources, which implied they supported an answer that did not exist.
The retrieved count is reported separately instead.

This is a partial implementation of ADR-019. The named `VERIFIED / BEST_EFFORT /
FAILED` taxonomy is not yet a formal enum in the state contract.
