# Project Requirements

> **Derived from `docx/Project Requirements.docx`, then updated to match the implementation.**
>
> The Word file is the original draft, kept unchanged for history and held
> outside this repository.
> This Markdown copy is the current version and has since diverged:
> Project name filled in.
>
> Re-running `scripts/docx2md.py` regenerates from the Word file and will
> discard these edits. Edit this file directly instead.

Project: SourceWeave
Document Version: 0.1
Status: Draft
Target Release: v0.1
Delivery Model: Agile — one-week sprints
Requirements Approach: Iterative / backlog-driven
Priority Method: MoSCoW

---

### 1. Purpose

This document defines the initial project, product, functional, non-functional, quality, operational, and delivery requirements for SourceWeave Release 0.1.

The requirements establish the expected outcomes for the release while allowing implementation details to evolve through Agile delivery, experimentation, evaluation, and backlog refinement.

Requirements may be changed as evidence is gathered, provided changes are documented and the Product Goal remains protected.

---

## 2. Product Goal

SourceWeave will provide a reproducible multi-agent research pipeline capable of:

- accepting a research query;
- planning how the query should be answered;
- retrieving appropriate supporting evidence;
- generating an evidence-grounded response;
- verifying claims against retrieved evidence;
- retrying retrieval when verification fails;
- returning the best available response with sources and confidence information;
- measuring system quality through repeatable evaluation.

The Release 0.1 goal is not merely to demonstrate that the pipeline runs.

The release must provide measurable evidence that the system behaves reliably and that its agentic architecture can be evaluated against a simpler baseline.

---

## 3. Requirement Priorities

Requirements use the following priorities:

| Priority | Meaning |
|---|---|
| Must | Required for Release 0.1 |
| Should | Important but release can proceed if formally deferred |
| Could | Valuable enhancement if capacity permits |
| Won't Yet | Explicitly outside Release 0.1 |

A Must requirement cannot be removed from the release without revisiting the Release Goal.

---

## 4. User Requirements

### UR-001 — Submit Research Query

Priority: Must

A user must be able to submit a natural-language research question to SourceWeave.

#### Acceptance Criteria

- A valid text query can be submitted.
- Empty or invalid queries are handled gracefully.
- The query enters the SourceWeave orchestration pipeline.
- A response is eventually returned or an understandable failure is reported.

---

### UR-002 — Receive Research Answer

Priority: Must

The user must receive a synthesized response relevant to the submitted research question.

#### Acceptance Criteria

The final output contains, where applicable:

- generated answer;
- supporting sources;
- confidence information;
- verification status;
- retry information.

---

### UR-003 — Understand Supporting Evidence

Priority: Must

The user must be able to identify the evidence used to construct factual responses.

#### Acceptance Criteria

- Retrieved sources are retained through the pipeline.
- Sources used by the generated response can be identified.
- Source metadata is sufficient to determine origin where available.
- Unsupported source references must not be fabricated.

---

### UR-004 — Receive Appropriate Uncertainty

Priority: Must

SourceWeave must communicate when available evidence is insufficient to confidently support an answer.

#### Acceptance Criteria

The system must not silently represent failed verification as successful verification.

When quality thresholds cannot be achieved, the response must provide an appropriate confidence or verification indication.

---

## 5. Functional Requirements

## 5.1 Query Planning

### FR-PLN-001 — Query Analysis

Priority: Must

The Planner must analyze incoming research queries before retrieval.

#### Acceptance Criteria

The Planner produces information necessary for downstream retrieval.

---

### FR-PLN-002 — Retrieval Strategy Selection

Priority: Must

The Planner must support selection among:

- vector retrieval;
- web retrieval;
- hybrid retrieval.

#### Acceptance Criteria

- A retrieval strategy is recorded in pipeline state.
- The Retriever can execute the selected strategy.
- Invalid strategies fail safely or fall back to an approved default.

---

### FR-PLN-003 — Query Decomposition

Priority: Should

The Planner should be capable of decomposing complex research questions into useful sub-questions or search objectives.

#### Acceptance Criteria

- Complex queries may produce multiple retrieval objectives.
- Sub-questions remain traceable to the original user query.
- Decomposition does not cause infinite processing loops.

---

## 5.2 Retrieval

### FR-RET-001 — Vector Retrieval

Priority: Must

SourceWeave must support semantic retrieval from indexed documents using Qdrant.

#### Acceptance Criteria

- A query can be embedded.
- Relevant documents can be retrieved from a configured Qdrant collection.
- Retrieved results include document content and available metadata.
- No-results conditions are handled gracefully.

---

### FR-RET-002 — Web Retrieval

Priority: Must

SourceWeave must support retrieval of external information through web search.

#### Acceptance Criteria

- SourceWeave can execute a web search for an approved query.
- Results contain usable content and source information.
- Provider errors do not crash the entire application without explanation.

---

### FR-RET-003 — Hybrid Retrieval

Priority: Must

SourceWeave must support retrieval combining vector-store and web-search results.

#### Acceptance Criteria

- Both retrieval channels can contribute evidence.
- Results are combined into a common context.
- Duplicate or substantially overlapping results are handled appropriately.
- Result ranking is reproducible enough to support evaluation.

---

### FR-RET-004 — Result Fusion

Priority: Must

Hybrid retrieval must apply a defined ranking/fusion strategy.

The current implementation uses Reciprocal Rank Fusion.

#### Acceptance Criteria

- Fusion behavior is documented.
- Fusion can be evaluated independently.
- Ranking configuration is explicit rather than hidden in prompts.

---

### FR-RET-005 — Retrieval Limits

Priority: Must

The system must apply configurable limits to retrieval.

Examples include:

- number of returned results;
- maximum context size;
- provider timeout;
- retrieval strategy.

#### Acceptance Criteria

Configuration can be modified without editing core orchestration code.

---

## 5.3 Document Ingestion

### FR-ING-001 — Document Ingestion

Priority: Must

Developers must be able to ingest supported documents into the vector store.

#### Acceptance Criteria

The ingestion process can:

- load source documents;
- divide documents into chunks;
- generate embeddings;
- store searchable records in Qdrant.

---

### FR-ING-002 — Metadata Preservation

Priority: Must

Important source metadata must be retained during ingestion.

At minimum, where available:

- source name;
- source identifier/path;
- chunk identifier;
- document metadata needed for attribution.

---

### FR-ING-003 — Repeatable Ingestion

Priority: Should

Developers should be able to repeat ingestion without unintentionally creating uncontrolled duplication.

---

## 5.4 Response Generation

### FR-GEN-001 — Evidence-Grounded Generation

Priority: Must

The Generator must use retrieved context when producing research responses.

#### Acceptance Criteria

- Retrieved context is provided to the Generator.
- Generator instructions emphasize evidence-grounded responses.
- Evaluation can determine whether generated claims are supported by retrieved context.

---

### FR-GEN-002 — Source Attribution

Priority: Must

Generated factual responses must support source attribution where evidence is available.

#### Acceptance Criteria

- Source identifiers correspond to actual retrieved sources.
- The Generator must not invent unavailable source identifiers.
- Attribution behavior is included in evaluation.

---

### FR-GEN-003 — LLM Abstraction

Priority: Must

SourceWeave must support model invocation through the project's model abstraction layer.

#### Acceptance Criteria

Model configuration can be changed without rewriting individual agent workflows.

---

### FR-GEN-004 — Response Structure

Priority: Must

The generated result must contain a consistent machine-readable structure sufficient for downstream verification and final pipeline output.

---

## 5.5 Verification

### FR-VER-001 — Claim Verification

Priority: Must

Generated responses must be evaluated against retrieved evidence before they are accepted as verified.

#### Acceptance Criteria

- Generated claims can be compared with retrieved evidence.
- Verification produces a measurable result.
- Verification results become part of pipeline state.

---

### FR-VER-002 — Verification Threshold

Priority: Must

The minimum acceptable verification threshold must be configurable.

#### Acceptance Criteria

- Threshold configuration is explicit.
- Changes can be tracked.
- Evaluation can compare different threshold settings.

The runtime threshold must be treated separately from the project's release-level Ragas targets.

---

### FR-VER-003 — Verification Failure Routing

Priority: Must

A failed verification attempt must be capable of triggering additional retrieval.

#### Acceptance Criteria

- A below-threshold result can route back to retrieval.
- The reason for retry is retained where practical.
- Retry behavior can be observed through pipeline state or tracing.

---

### FR-VER-004 — Retry Limit

Priority: Must

Verification retries must be bounded.

#### Acceptance Criteria

- Maximum retry count is configurable.
- Processing cannot loop indefinitely.
- Exhausting retries produces a controlled result.

---

### FR-VER-005 — Best-Attempt Handling

Priority: Must

If verification cannot meet the desired threshold after the permitted retries, SourceWeave must return an appropriate best-attempt response or controlled failure.

The result must not falsely claim successful verification.

---

## 5.6 Memory

### FR-MEM-001 — Conversation Memory

Priority: Should

SourceWeave should support persistent contextual memory for appropriate user interactions.

---

### FR-MEM-002 — Evaluation Isolation

Priority: Must

Persistent memory must be disabled, isolated, or reset during controlled benchmark evaluation unless memory is specifically being tested.

#### Rationale

Previous interactions must not unintentionally contaminate benchmark results.

---

### FR-MEM-003 — Memory Failure Isolation

Priority: Should

Failure of the memory provider should not unnecessarily prevent stateless research queries from executing.

---

## 5.7 Pipeline Orchestration

### FR-ORC-001 — Explicit Pipeline State

Priority: Must

Relevant information passed between agents must be maintained through an explicit shared pipeline state.

---

### FR-ORC-002 — Agent Sequence

Priority: Must

The baseline pipeline must support the logical workflow:

Planner → Retriever → Generator → Verifier

---

### FR-ORC-003 — Conditional Routing

Priority: Must

The orchestration layer must support conditional transitions based on verification results.

---

### FR-ORC-004 — Controlled Termination

Priority: Must

Every query execution must eventually:

- complete successfully;
- complete with an appropriately qualified response; or
- terminate with a controlled error.

Infinite agent execution is prohibited.

---

## 6. Evaluation Requirements

Evaluation is a core product capability, not an optional testing activity.

### ER-001 — Benchmark Dataset

Priority: Must

SourceWeave must have a committed or reproducibly generated benchmark dataset.

#### Dataset records should include where applicable:

- question;
- expected/reference answer;
- expected supporting context;
- category or difficulty;
- metadata useful for analysis.

---

### ER-002 — Reproducible Evaluation

Priority: Must

A developer must be able to execute the evaluation process using documented commands.

---

### ER-003 — RAG Quality Metrics

Priority: Must

The evaluation system must measure at least:

- Faithfulness;
- Answer Relevancy;
- Context Precision;
- Context Recall.

---

### ER-004 — Operational Metrics

Priority: Must

Evaluation must collect or derive:

- response latency;
- verification pass rate;
- retry rate.

---

### ER-005 — Baseline Comparison

Priority: Must

SourceWeave must be compared against a defined simpler baseline.

#### Acceptance Criteria

- Baseline architecture is documented.
- Both systems use an equivalent evaluation dataset.
- Results are recorded consistently.
- Comparison can be reproduced.

---

### ER-006 — Regression Detection

Priority: Must

Changes affecting AI behavior must be evaluated for regression.

Examples include changes to:

- prompts;
- models;
- embeddings;
- chunking;
- retrieval;
- ranking;
- agent routing;
- verification;
- retries;
- memory.

---

### ER-007 — Benchmark Versioning

Priority: Should

Evaluation results should identify:

- code/version under test;
- model configuration;
- dataset version;
- relevant runtime configuration.

---

## 7. Quality Requirements

The following are initial Release 0.1 quality targets.

They are targets to validate, not assumed current performance.

| ID | Quality Measure | Target | Priority |
|---|---|---|---|
| QR-001 | Faithfulness | ≥ 0.80 | Must |
| QR-002 | Answer Relevancy | ≥ 0.80 | Must |
| QR-003 | Context Precision | ≥ 0.75 | Must |
| QR-004 | Context Recall | ≥ 0.75 | Must |
| QR-005 | Verification Pass Rate | ≥ 85% | Should |
| QR-006 | Source Attribution for evidence-supported factual answers | 100% | Must |
| QR-007 | Unhandled Pipeline Failures | < 1% | Must |
| QR-008 | Relative regression on key quality metrics | < 3% | Must |

Performance and cost thresholds will be baselined before final targets are committed.

---

## 8. Performance Requirements

### NFR-PERF-001 — Latency Measurement

Priority: Must

SourceWeave must measure end-to-end query latency during evaluation.

---

### NFR-PERF-002 — Component Latency

Priority: Should

Latency should be observable for major stages such as:

- planning;
- retrieval;
- generation;
- verification.

---

### NFR-PERF-003 — Retrieval Performance

Priority: Should

Retrieval latency must be benchmarked under a defined local/test environment.

A final Release 0.1 target will be established after the baseline sprint.

---

### NFR-PERF-004 — Retry Cost

Priority: Must

The project must measure the latency impact of verification retries.

---

## 9. Cost Requirements

### NFR-COST-001 — Token Usage

Priority: Should

LLM token usage should be measurable per query or per major model invocation.

---

### NFR-COST-002 — Cost Per Query

Priority: Should

The evaluation process should estimate cost per query for supported paid model configurations.

---

### NFR-COST-003 — Quality/Cost Trade-Off

Priority: Could

The project should eventually support comparison of quality improvements against additional inference cost.

---

## 10. Reliability Requirements

### NFR-REL-001 — External Provider Failure

Priority: Must

Failures from external providers must produce controlled errors or approved fallback behavior.

Relevant providers include services used for:

- LLM inference;
- embeddings;
- vector search;
- web search;
- memory;
- tracing.

---

### NFR-REL-002 — Timeouts

Priority: Must

External network operations must have bounded execution or timeout behavior.

---

### NFR-REL-003 — Malformed Model Output

Priority: Must

Unexpected or malformed LLM responses must not cause uncontrolled pipeline crashes.

---

### NFR-REL-004 — Empty Retrieval

Priority: Must

The system must safely handle retrieval operations that return no useful context.

---

## 11. Observability Requirements

### NFR-OBS-001 — Pipeline Tracing

Priority: Must

Developers must be able to inspect major pipeline execution steps.

---

### NFR-OBS-002 — Agent-Level Visibility

Priority: Should

Tracing should expose relevant information about:

- agent execution;
- latency;
- model invocation;
- retrieval;
- verification;
- retries.

---

### NFR-OBS-003 — Failure Diagnosis

Priority: Must

A failed query should provide sufficient diagnostic information for a developer to determine which major pipeline stage failed.

---

### NFR-OBS-004 — Sensitive Data Protection

Priority: Must

Tracing and logs must not intentionally expose API credentials or secrets.

---

## 12. Security Requirements

### NFR-SEC-001 — Secret Management

Priority: Must

API keys and credentials must be supplied through approved environment/configuration mechanisms.

Secrets must not be committed to source control.

---

### NFR-SEC-002 — Environment Template

Priority: Must

A non-secret environment configuration template must document required configuration.

---

### NFR-SEC-003 — Dependency Awareness

Priority: Should

Project dependencies should support automated vulnerability/dependency scanning where practical.

---

### NFR-SEC-004 — Input Handling

Priority: Should

External input should be treated as untrusted data and handled without enabling unintended code execution.

---

## 13. Developer Experience Requirements

### DX-001 — Clone-to-Run Documentation

Priority: Must

A new developer must have documented steps for:

- cloning the repository;
- installing dependencies;
- configuring required services;
- ingesting sample data;
- running a query;
- running tests;
- running evaluation.

---

### DX-002 — Reproducible Dependencies

Priority: Must

The project must provide a reproducible dependency strategy.

---

### DX-003 — Automated Tests

Priority: Must

The codebase must provide automated tests for critical pipeline behavior.

At minimum:

- pipeline construction;
- agent routing;
- retrieval behavior;
- verification pass;
- verification retry;
- retry exhaustion;
- provider/error handling.

---

### DX-004 — Continuous Integration

Priority: Must

Pull requests must run automated quality checks.

Initial checks should include:

- install/build validation;
- import validation;
- tests;
- appropriate static checks.

---

### DX-005 — Documentation Accuracy

Priority: Must

Repository documentation must accurately represent implemented and available functionality.

Documented commands, files, benchmarks, and capabilities must exist or be clearly identified as planned.

---

## 14. Agile Delivery Requirements

### PM-001 — Product Backlog

Priority: Must

All planned implementation work must be represented in a prioritized Product Backlog.

---

### PM-002 — Sprint Goal

Priority: Must

Every sprint must have a clearly stated outcome-oriented Sprint Goal.

---

### PM-003 — Acceptance Criteria

Priority: Must

Stories entering a sprint must contain testable acceptance criteria.

---

### PM-004 — Definition of Done

Priority: Must

Completed backlog items must satisfy the SourceWeave Definition of Done.

---

### PM-005 — Evaluation Gate

Priority: Must

A backlog item that materially changes AI behavior cannot be considered Done until relevant evaluation has been completed or an explicit exception has been recorded.

---

### PM-006 — Sprint Review

Priority: Must

Each sprint must conclude with review of:

- working increment;
- Sprint Goal;
- relevant product metrics;
- learning or experiment results.

---

### PM-007 — Retrospective

Priority: Must

Each sprint must include a retrospective identifying process improvements.

---

### PM-008 — Backlog Adaptation

Priority: Must

Evaluation results, failures, technical discoveries, and user feedback may change backlog priority.

The roadmap must not prevent evidence-based reprioritization.

---

## 15. Release Requirements

SourceWeave v0.1 will be considered release-ready when the following Release Gate is satisfied.

### RG-001 — Reproducibility

A clean environment can:

- install SourceWeave;
- configure required services;
- ingest test content;
- run the pipeline;
- run tests;
- execute evaluation.

---

### RG-002 — End-to-End Capability

The complete research workflow operates successfully:

Query → Planner → Retriever → Generator → Verifier → Result

including verification retry behavior.

---

### RG-003 — Automated Tests

All required automated tests pass.

No known release-blocking defects remain.

---

### RG-004 — Evaluation

The approved benchmark dataset executes successfully and produces required quality metrics.

---

### RG-005 — Baseline Comparison

SourceWeave has been compared against the approved simpler baseline.

Results are documented whether they support or challenge the product hypotheses.

---

### RG-006 — Quality Gate

Required Release 0.1 quality thresholds are achieved or explicitly revised through an approved product decision.

---

### RG-007 — Documentation

Documentation accurately describes:

- setup;
- execution;
- ingestion;
- configuration;
- evaluation;
- known limitations.

---

### RG-008 — Observability

Major pipeline stages and failures can be diagnosed through approved logging/tracing mechanisms.

---

### RG-009 — Security

No known secrets are committed, and basic security/dependency checks pass.

---

## 16. Out-of-Scope Requirements for Release 0.1

The following are classified as Won't Yet unless the Product Backlog is deliberately reprioritized:

- production web UI;
- mobile client;
- enterprise SSO;
- complex RBAC;
- multi-tenancy;
- billing;
- subscription management;
- production Kubernetes deployment;
- advanced analytics dashboard;
- marketplace integrations;
- high-scale distributed architecture;
- formal enterprise SLA;
- extensive end-user customization.

---

## 17. Initial Requirement Dependencies

| Requirement Area | Key Dependencies |
|---|---|
| Vector Retrieval | Qdrant + embedding service |
| Web Retrieval | Tavily |
| Generation | LiteLLM-supported model provider |
| Verification | LLM + retrieved evidence |
| Memory | mem0 |
| Evaluation | Ragas + benchmark dataset |
| Observability | LangSmith or approved tracing alternative |
| CI | GitHub Actions or approved equivalent |

Failures or changes to these dependencies must be captured through the project RAID process.

---

## 18. Requirements Traceability

| Charter Objective | Supporting Requirement Areas |
|---|---|
| Reproducibility | DX, ING, RG |
| Retrieval Quality | PLN, RET, ER |
| Grounded Generation | GEN, RET, QR |
| Verification Effectiveness | VER, ER |
| Measurable Quality | ER, QR, PM |
| Operational Visibility | OBS, REL |

Every Must requirement should eventually trace to at least one:

- Epic;
- User Story;
- automated test;
- evaluation;
- release criterion.

---

## 19. Requirements Change Process

Because SourceWeave is being built using Agile delivery, these requirements are not intended to function as a fixed waterfall specification.

A requirement may be:

Added → Refined → Reprioritized → Split → Replaced → Removed

based on:

- evaluation results;
- technical discoveries;
- user feedback;
- Sprint Reviews;
- risk findings;
- architectural decisions.

Changes to a Must requirement should include:

- reason for change;
- impact on Product Goal;
- impact on Release 0.1;
- associated backlog change;
- decision owner;
- date of decision.

---

## 20. Initial Release 0.1 Requirement Summary

SourceWeave v0.1 must demonstrate four things:

#### 1. It works.

A developer can reproducibly execute the complete multi-agent pipeline.

#### 2. It is measurable.

The pipeline can be evaluated through a repeatable benchmark.

#### 3. It is trustworthy enough to assess.

Sources, verification outcomes, retries, failures, and major pipeline decisions are observable.

#### 4. Its complexity is justified—or challenged—by evidence.

SourceWeave can be objectively compared against a simpler baseline.

The project will prioritize measurable product learning over simply implementing additional agents or features.
