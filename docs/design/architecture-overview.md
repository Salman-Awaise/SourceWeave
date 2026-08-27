# Architecture Overview

> **Derived from `docx/Architecture Overview.docx`, then updated to match the implementation.**
>
> The Word file is the original draft, kept unchanged for history and held
> outside this repository.
> This Markdown copy is the current version and has since diverged:
> Section 23 rewritten against the implementation; section 23.1 added listing what remains outstanding.
>
> Re-running `scripts/docx2md.py` regenerates from the Word file and will
> discard these edits. Edit this file directly instead.

Project: SourceWeave
Version: 0.1
Status: Draft
Target Release: v0.1
Architecture Style: Stateful multi-agent pipeline with retrieval-augmented generation and verification

---

### 1. Purpose

This document describes the software architecture for SourceWeave Release 0.1.

It defines:

- system boundaries;
- major components;
- responsibilities;
- data flow;
- integration boundaries;
- runtime behavior;
- architectural constraints;
- quality attributes;
- known architectural gaps;
- target-state improvements.

The architecture is expected to evolve through Agile delivery and validated engineering experiments.

---

## 2. Architectural Goals

SourceWeave architecture should optimize for:

- Grounded answers — generated factual claims should be based on retrieved evidence.
- Verifiability — generated answers should be checked before acceptance.
- Experimentability — retrieval, model, prompt, verification, and routing strategies should be replaceable and measurable.
- Observability — developers should understand how each query moves through the system.
- Reproducibility — behavior should be sufficiently configurable to support repeatable evaluation.
- Provider portability — external providers should be replaceable without rewriting orchestration logic.
- Controlled failure — external-service failures or verification failures should not create uncontrolled execution.
- Simplicity — additional agents and infrastructure should only be introduced when evidence justifies the complexity.

---

## 3. System Context

SourceWeave sits between a user or calling application and several AI/data providers.

```text
                         ┌──────────────────────┐
                         │       User / CLI      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     SourceWeave API      │
                         │   / Pipeline Runner   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                   ┌────────────────────────────────┐
                   │      LangGraph Orchestrator     │
                   │                                │
                   │ Planner → Retriever → Generator│
                   │                    ↓           │
                   │                Verifier        │
                   │                    │           │
                   │              retry / respond   │
                   └────────────────────────────────┘
                         │       │        │
               ┌─────────┘       │        └──────────┐
               ▼                 ▼                   ▼
          Qdrant Store       Web Search          LLM Provider
                             Provider            via LiteLLM

                    Cross-Cutting Capabilities
                ┌──────────────┬───────────────┐
                │ Memory       │ Observability │
                │ mem0         │ LangSmith     │
                └──────────────┴───────────────┘
```

---

## 4. Primary Runtime Flow

The baseline runtime flow is:

```text
User Query
    │
    ▼
Memory Context
(optional)
    │
    ▼
Planner
    │
    ├── decompose query
    └── select retrieval strategy
    │
    ▼
Retriever
    │
    ├── vector retrieval
    ├── web retrieval
    └── hybrid fusion
    │
    ▼
Generator
    │
    ├── synthesize evidence
    ├── attribute sources
    └── estimate confidence
    │
    ▼
Verifier
    │
    ├── extract claims
    ├── compare against evidence
    └── calculate faithfulness
    │
    ├──── verified ──────────────► Response
    │
    └──── not verified
              │
              ▼
          Retry Policy
              │
              ├── retries available → Retriever
              └── retries exhausted → Best Attempt
```

---

## 5. Architectural Layers

### 5.1 Interface Layer

Responsibilities:

- receive user queries;
- validate basic inputs;
- invoke the SourceWeave pipeline;
- present results.

Release 0.1 may use a CLI or Python interface.

A web/API interface is not required to define the internal architecture.

---

### 5.2 Application / Pipeline Layer

Primary location:

core/

Responsibilities:

- initialize execution state;
- invoke the orchestration graph;
- integrate cross-cutting services;
- expose the high-level SourceWeave interface;
- build the final result.

The application layer should not contain provider-specific retrieval or model logic.

---

### 5.3 Orchestration Layer

Primary technology:

LangGraph

Responsibilities:

- define workflow nodes;
- define execution order;
- enforce conditional routing;
- control retry loops;
- guarantee termination.

Baseline topology:

```text
START
  ↓
Planner
  ↓
Retriever
  ↓
Generator
  ↓
Verifier
  ↓
┌──────────────────────────────┐
│ verified?                    │
├───────────┬──────────────────┤
│ yes       │ no               │
▼           ▼                  │
END       retries available?   │
              │                │
          yes │ no             │
              │                │
        Retriever             END
```

The orchestration graph owns workflow behavior.

Individual agents should not independently orchestrate other agents.

---

## 6. Shared State Architecture

SourceWeave uses an explicit state object as the contract between graph nodes.

Major state categories include:

#### Input

- original query;
- conversation history;
- execution context.

#### Planning

- decomposed sub-queries;
- retrieval strategy.

#### Retrieval

- retrieved documents;
- retrieval queries used.

#### Generation

- response;
- source references;
- confidence.

#### Verification

- claims;
- supported claims;
- unsupported claims;
- faithfulness;
- verification status.

#### Execution Metadata

- retry count;
- errors;
- trace identifiers.

### Architectural Principle

Agents communicate through the defined pipeline state rather than hidden global state.

This provides:

- debuggability;
- testability;
- traceability;
- deterministic workflow contracts.

---

## 7. Agent Architecture

### 7.1 Planner

The Planner is responsible for deciding how research should be performed, not answering the question.

Responsibilities:

- analyze query intent;
- decompose complex questions;
- choose retrieval strategy;
- produce structured planning output.

Supported strategies:

```text
vector
web
hybrid
```

The Planner should not directly call retrieval providers.

---

### 7.2 Retriever

The Retriever owns evidence acquisition.

Responsibilities:

- execute the selected retrieval strategy;
- query internal vector sources;
- query external web sources;
- normalize retrieved results;
- combine result sets;
- return bounded evidence.

Retrieval must produce a common domain model independent of provider.

Conceptually:

```text
RetrievalProvider
    │
    ├── VectorRetriever
    │       └── Qdrant
    │
    └── WebRetriever
            └── Tavily
```

This provider boundary should become more explicit over time.

---

## 8. Hybrid Retrieval Architecture

Hybrid retrieval combines:

```text
Vector Results
      │
      ├───────────┐
      │           │
      ▼           │
                  ▼
               Fusion
                  ▲
      ▲           │
      │           │
Web Results ──────┘
```

Release 0.1 uses Reciprocal Rank Fusion as the baseline strategy.

The fusion strategy should remain independently replaceable so alternatives can be evaluated.

Candidate future strategies include:

- weighted fusion;
- learned reranking;
- cross-encoder reranking;
- source-aware ranking.

These should not be added until evaluation demonstrates a need.

---

## 9. Retrieval Domain Model

Every retrieval provider should normalize output to a common representation.

Target conceptual model:

```text
RetrievedDocument
├── document_id
├── chunk_id
├── content
├── source
├── source_type
├── retrieval_score
├── retrieval_method
├── timestamp
└── metadata
```

Stable document and chunk identifiers are important for:

- deduplication;
- source attribution;
- evaluation;
- tracing;
- retrieval comparison.

Release 0.1 should move away from content-based identity where practical.

---

## 10. Generator Architecture

The Generator is responsible for transforming evidence into an answer.

It should:

- consume the original research question;
- consume retrieved evidence;
- produce a structured answer;
- identify evidence used;
- communicate uncertainty.

The Generator must not decide whether its answer is ultimately trustworthy.

That responsibility belongs to the Verifier.

This separation reduces the likelihood that generation confidence is treated as independent proof of correctness.

---

## 11. Source Attribution Architecture

Source attribution should be system-controlled where possible rather than exclusively model-controlled.

Target flow:

```text
RetrievedDocument
      │
      ├── stable source ID
      │
      ▼
Generator Context
      │
      ▼
Generated Citation Reference
      │
      ▼
Citation Validator
      │
      ▼
Final Source Metadata
```

The system should reject or ignore references to sources that were not part of the retrieved evidence.

---

## 12. Verification Architecture

Verification is a distinct architectural capability.

The Verifier:

- receives generated answer;
- receives evidence;
- extracts factual claims;
- evaluates support for each claim;
- produces structured verification results;
- calculates a faithfulness result;
- informs routing.

Conceptual result:

```text
VerificationResult
├── claims
├── supported_claims
├── unsupported_claims
├── faithfulness_score
└── reasoning
```

The Verifier does not directly control the graph.

It produces state.

The orchestration layer decides what happens next.

---

## 13. Verification Retry Architecture

A failed verification may create a new retrieval attempt.

Target flow:

```text
Verifier
   │
   ▼
Unsupported Claims
   │
   ▼
Query Refinement
   │
   ▼
Retriever
   │
   ▼
Generator
   │
   ▼
Verifier
```

#### Important Target-State Decision

Retry retrieval should eventually use information from the failed verification.

Simply rerunning identical retrieval queries provides limited value.

Therefore the target architecture introduces explicit:

```text
unsupported_claims
       ↓
retry_queries
```

into state.

The first implementation may use a deterministic query-refinement function or a Planner refinement step.

This behavior must be evaluated before increasing architectural complexity.

---

## 14. Retry and Termination Policy

All agent loops must be bounded.

Release 0.1 baseline:

```text
verification threshold = configurable
maximum verification retries = configurable
```

Possible terminal states:

```text
VERIFIED
BEST_EFFORT
FAILED
```

These should eventually become explicit result statuses rather than relying exclusively on a Boolean verification flag.

---

## 15. LLM Architecture

LiteLLM provides a common model invocation boundary.

Target architecture:

```text
Agent
  │
  ▼
Model Policy
  │
  ▼
LiteLLM
  │
  ├── OpenAI
  ├── Anthropic
  └── other supported provider
```

Release 0.1 should distinguish:

#### Provider Abstraction

“How do we call different model providers?”

from:

#### Model Routing Policy

“Which model should execute which workload?”

The two concepts should not be conflated.

Initial configuration may use one default model.

Future evaluation may justify:

```text
Planner     → lower-cost model
Generator   → high-quality generation model
Verifier    → deterministic verification model
```

Model selection should be configuration-driven and benchmarked.

---

## 16. Memory Architecture

Memory is a cross-cutting capability, not part of factual evidence retrieval.

Conceptually:

```text
Conversation
     │
     ▼
Memory Retrieval
     │
     ▼
Context
     │
     ├────► Planner / Generator
     │
     └────► personalization
```

Memory must remain distinguishable from evidence.

#### Architectural Rule

User memory should not automatically be treated as factual supporting evidence.

The system should retain separate concepts for:

```text
original_query
memory_context
retrieved_evidence
```

rather than permanently combining them into one string.

This separation improves:

- evaluation isolation;
- tracing;
- prompt control;
- provenance;
- security.

Memory should be disabled during standard benchmark runs unless memory itself is the feature under test.

---

## 17. External Integration Architecture

External dependencies should be accessed through controlled adapters.

Target structure:

```text
agents/
    │
    ▼
ports / interfaces
    │
    ├── RetrievalPort
    ├── ModelPort
    ├── MemoryPort
    └── TracePort
            │
            ▼
        adapters/
        ├── qdrant
        ├── tavily
        ├── litellm
        ├── mem0
        └── langsmith
```

Release 0.1 does not require a formal hexagonal architecture rewrite.

The objective is simply to prevent provider-specific behavior from spreading into agent orchestration.

---

## 18. Configuration Architecture

Configuration should remain external to business logic.

Configuration categories include:

- models;
- embeddings;
- vector store;
- web provider;
- retrieval limits;
- chunking;
- verification threshold;
- retry count;
- generation temperature;
- tracing;
- memory.

Configuration affecting evaluation must be recorded with benchmark results.

---

## 19. Observability Architecture

Observability is cross-cutting.

Every query should eventually support traceability across:

```text
Query
 ↓
Planning
 ↓
Retrieval
 ↓
Generation
 ↓
Verification
 ↓
Retry
 ↓
Result
```

Useful telemetry includes:

- trace ID;
- node;
- model;
- latency;
- token usage;
- retrieval strategy;
- document count;
- verification result;
- retry count;
- provider error.

Tracing must never intentionally expose secrets.

---

## 20. Evaluation Architecture

Evaluation remains separate from production execution.

```text
Benchmark Dataset
       │
       ├──────────────┐
       ▼              ▼
SourceWeave        Baseline
       │              │
       └──────┬───────┘
              ▼
          Evaluation
              │
              ▼
           Scorecard
```

Evaluation should capture:

- code version;
- dataset version;
- model configuration;
- retrieval configuration;
- prompt/config version;
- quality metrics;
- latency;
- retry rate;
- cost where measurable.

---

## 21. Failure Architecture

External dependencies are failure boundaries.

SourceWeave must expect failures from:

- model providers;
- embeddings;
- Qdrant;
- Tavily;
- mem0;
- tracing.

Target error taxonomy:

```text
ConfigurationError
RetrievalError
ModelError
VerificationError
MemoryError
ProviderTimeout
PipelineError
```

The orchestration layer should determine whether failures are:

```text
retryable
degradable
terminal
```

Observability failures should generally not prevent query completion.

Memory failures should generally permit stateless operation.

Evidence retrieval failure requires more conservative handling because it directly affects answer trustworthiness.

---

## 22. Security Boundaries

The architecture must treat all external content as untrusted.

Primary boundaries include:

- user input;
- retrieved web content;
- indexed documents;
- model output;
- memory content.

Prompt instructions contained in retrieved content must not automatically override SourceWeave system behavior.

Secrets must remain outside agent prompts and persistent state wherever possible.

---

## 23. Current-to-Target Architecture Gaps

Updated after the Release 0.1 implementation and the architecture spikes. Rows
marked **done** were target direction in the original draft and are now current
behaviour.

| Area | Current Baseline | Target Direction | State |
|---|---|---|---|
| Shared state | Explicit typed state | Keep | current |
| Orchestration | LangGraph | Keep pending evaluation | current |
| Agents | Planner/Retriever/Generator/Verifier | Keep pending baseline comparison | current — SPIKE-ARCH-001 complete, benefit is corpus-dependent |
| Retry | Verifier → Retriever | Add evidence-driven query refinement | **done** — refined queries built from unsupported claims (ADR-010) |
| Memory | Added to query context | Separate query and memory provenance | **done** — `memory_context` reaches the Planner only (ADR-011) |
| Source attribution | LLM-assisted indexes | Validate against retrieved source registry | **done** — citation indices validated; invalid ones dropped |
| Failure handling | Basic | Typed failures + degraded modes | **done** — typed errors, hybrid degradation, empty-result broadening |
| Fusion | Reciprocal Rank Fusion | Source-aware ranking | **done** — source-type weights plus a per-type floor (ADR-021) |
| Result state | Boolean verification | Explicit VERIFIED/BEST_EFFORT/FAILED | partial — `answered` split from `is_verified`; no formal enum (ADR-019, ADR-022) |
| Retrieval identity | Content-oriented | Stable document/chunk IDs | **outstanding** — dedup is still content-based (ADR-014) |
| Model configuration | Shared default model | Per-role model policy if justified | **outstanding** — single `DEFAULT_LLM` (ADR-007) |
| Provider boundary | Tool wrappers | Explicit interfaces where useful | partial — protocols exist per adapter; no `ports/` module or `TracePort` (ADR-020) |
| Evaluation | Separate evaluator | Versioned reproducible benchmark | **done** — dataset SHA-256, dependency versions and model config recorded per run |

### 23.1 Outstanding Gaps

The following remain unimplemented and are recorded here so the architecture
document does not overstate the system.

**Stable evidence identifiers (ADR-014, section 9).** `RetrievedDocument` carries
`content`, `source`, `score` and `metadata`. There is no first-class
`document_id`, `chunk_id`, `retrieval_method` or `timestamp`, and deduplication
is a hash of normalised content plus canonical source. Section 9 calls for moving
away from content-based identity; that has not happened.

**Explicit terminal result states (ADR-019, section 14).** The pipeline exposes
`is_verified` and `answered` as booleans, and the CLI renders three labels from
them. A formal `VERIFIED / BEST_EFFORT / FAILED` value in the state contract does
not exist.

**Per-role model policy (ADR-007, section 15).** One `DEFAULT_LLM` serves every
agent. The distinction between provider abstraction and routing policy is
understood but not implemented; there is no `planner_model` or `verifier_model`.

**Ports layer (ADR-020, section 17).** Protocols exist for the LLM, embeddings,
vector store, web search and memory, which is what allows the offline test suite
to run without network access. They live inside their adapter modules rather than
a dedicated interface layer, and there is no `TracePort`.

**Error taxonomy naming (section 21).** The implemented hierarchy is
`ResearchSystemError` with `ConfigurationError`, `RetrievalError`,
`VectorStoreError`, `WebSearchError`, `EmbeddingError`, `LLMError`,
`StructuredOutputError`, `SchemaValidationError`, `IngestionError` and
`EvaluationError`. The document names `ModelError`, `VerificationError`,
`MemoryError`, `ProviderTimeout` and `PipelineError`. Coverage is broadly
equivalent — `LLMError` fills the `ModelError` role — but the names differ, and
verification, memory and timeout failures are surfaced as warnings and degraded
behaviour rather than distinct exception types.

**Cost and token telemetry (section 19, NFR-COST-001/002).** Not implemented.
Token usage is not captured from provider responses and cost per query is not
estimated.

**Per-stage latency (section 19, NFR-PERF-002).** End-to-end latency is recorded
per query. Planning, retrieval, generation and verification are not timed
individually.

## 24. Architectural Fitness Functions

Architecture should be validated continuously.

Candidate fitness functions:

#### AF-01 — Termination

No query may execute beyond the configured retry limit.

#### AF-02 — Source Integrity

Final source references must correspond to retrieved evidence.

#### AF-03 — Evaluation Isolation

Benchmark execution must not consume uncontrolled persistent memory.

#### AF-04 — Provider Replaceability

Changing the configured LLM provider must not require modifying graph topology.

#### AF-05 — Traceability

Every pipeline execution should expose enough information to identify its major stages.

#### AF-06 — Quality Regression

Architectural changes affecting AI behavior must pass defined regression thresholds.

#### AF-07 — Complexity Justification

New agents or pipeline stages require a measurable problem and evaluation plan.

---

## 25. Architectural Principle

SourceWeave will favor:

explicit state, bounded workflows, replaceable integrations, measurable behavior, and evidence-driven complexity.

Architecture is not considered successful because it contains more agents.

Architecture is successful when it makes the system easier to trust, evaluate, change, and operate.
