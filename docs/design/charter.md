# Project Charter

> **Derived from `docx/Project Charter.docx`, then updated to match the implementation.**
>
> The Word file is the original draft, kept unchanged for history and held
> outside this repository.
> This Markdown copy is the current version and has since diverged:
> Project name filled in.
>
> Re-running `scripts/docx2md.py` regenerates from the Word file and will
> discard these edits. Edit this file directly instead.

Project: SourceWeave
Charter Version: 0.1
Status: Draft
Delivery Approach: Agile / iterative product development
Repository: SourceWeave

### 1. Project Vision

Build SourceWeave into a reliable, measurable multi-agent research system that can retrieve information from multiple sources, synthesize evidence, verify generated claims, and produce trustworthy answers with traceable supporting evidence.

The project will evolve through short Agile iterations, with each sprint producing a working, testable increment and measurable evidence about whether the system is improving.

### 2. Problem Statement

Traditional LLM-based research workflows can produce relevant-looking answers that contain unsupported claims, weak source grounding, or hallucinated information.

SourceWeave aims to address this problem by combining:

- query planning;
- hybrid retrieval;
- evidence-grounded generation;
- automated verification;
- controlled retry behavior;
- evaluation and observability.

The project must demonstrate that this added agentic complexity produces meaningful quality improvements compared with simpler approaches.

### 3. Product Goal

Create a reproducible research pipeline that produces evidence-grounded responses and measurably improves answer quality and faithfulness compared with a defined baseline.

The initial product goal is achieved when SourceWeave can:

- accept a research question;
- determine an appropriate retrieval strategy;
- retrieve relevant internal and/or external information;
- generate an evidence-grounded response;
- verify generated claims against retrieved evidence;
- retry when verification fails;
- provide sources and verification results;
- measure performance through a repeatable evaluation framework.

### 4. Project Objectives

#### Objective 1 — Reproducibility

A developer should be able to clone, configure, run, test, and evaluate SourceWeave using documented steps.

#### Objective 2 — Retrieval Quality

SourceWeave should retrieve sufficiently relevant evidence from vector, web, or hybrid retrieval strategies to support reliable answer generation.

#### Objective 3 — Grounded Generation

Generated responses should be based on retrieved evidence and provide clear source attribution where factual claims are made.

#### Objective 4 — Verification Effectiveness

The verification stage should identify unsupported claims and prevent low-confidence responses from being accepted without additional retrieval or controlled termination.

#### Objective 5 — Measurable Quality

Changes to prompts, retrieval, models, verification logic, memory, or orchestration should be evaluated against a repeatable benchmark.

#### Objective 6 — Operational Visibility

Developers should be able to understand how a query moved through the system, where failures occurred, and why retries or routing decisions were made.

### 5. Initial Scope

The initial scope includes:

- LangGraph-based agent orchestration;
- Planner agent;
- Retriever agent;
- Generator agent;
- Verifier agent;
- vector retrieval using Qdrant;
- web retrieval using Tavily;
- hybrid retrieval and result fusion;
- LiteLLM-based model access;
- memory integration;
- evaluation using Ragas;
- tracing and observability;
- automated tests;
- benchmark datasets;
- developer documentation;
- CI checks;
- release-quality gates.

### 6. Out of Scope for Initial Release

The following are not required for the first validated release unless prioritized through the product backlog:

- full production web application;
- mobile application;
- enterprise identity and access management;
- multi-tenant architecture;
- billing and subscription management;
- large-scale production infrastructure;
- advanced administrative dashboards;
- broad third-party integration marketplace;
- highly customized end-user interfaces.

These items may enter future releases based on evidence, user feedback, and product priorities.

### 7. Target Users

#### Primary

Developers, AI engineers, researchers, and technical users who need research responses grounded in identifiable evidence.

#### Secondary

Teams exploring multi-agent research workflows, retrieval-augmented generation, automated verification, or agentic AI architectures.

Target-user assumptions will be refined as the product evolves.

### 8. Core Value Proposition

SourceWeave should provide more trustworthy research responses by combining evidence retrieval and explicit verification rather than relying exclusively on a single generation step.

Its value will be demonstrated through measurable improvements in:

- faithfulness;
- answer relevancy;
- retrieval quality;
- unsupported-claim detection;
- source grounding;
- reliability.

### 9. Initial Success Measures

The following are initial targets and may be revised as baseline data becomes available.

| Measure | Initial Target |
|---|---|
| Faithfulness | ≥ 0.80 |
| Answer relevancy | ≥ 0.80 |
| Context precision | ≥ 0.75 |
| Context recall | ≥ 0.75 |
| Verification pass rate | ≥ 85% |
| Source attribution for evidence-based factual answers | 100% |
| Unhandled pipeline failure rate | < 1% |
| Regression tolerance for key quality metrics | < 3% relative decline |

Latency and cost-per-query targets will be established after a reproducible baseline is measured.

Success will be evaluated against both absolute quality thresholds and comparison with a simpler baseline implementation.

### 10. Key Product Hypotheses

The project will treat the following as hypotheses to test rather than assumptions to accept:

#### H1 — Hybrid Retrieval

Hybrid vector and web retrieval produces better supporting context than either retrieval strategy alone for appropriate research queries.

#### H2 — Verification

Adding a verification step reduces unsupported or unfaithful answers compared with generation without verification.

#### H3 — Retry Loop

Targeted retrieval after verification failure improves answer quality enough to justify the additional latency and cost.

#### H4 — Multi-Agent Architecture

The Planner → Retriever → Generator → Verifier architecture produces sufficient quality or maintainability benefits to justify its additional complexity.

#### H5 — Memory

Memory improves relevant user interactions without contaminating factual accuracy or benchmark evaluation.

Each hypothesis should eventually have measurable evidence supporting, rejecting, or modifying it.

### 11. Agile Delivery Approach

SourceWeave will be developed using short, iterative delivery cycles.

The initial cadence will be one-week sprints.

Each sprint will have:

- one clear Sprint Goal;
- a prioritized Sprint Backlog;
- small, testable stories;
- explicit acceptance criteria;
- a working increment;
- relevant automated evaluation;
- a Sprint Review;
- a retrospective;
- backlog reprioritization based on results.

The team will optimize for achieving the Sprint Goal rather than maximizing story-point completion.

### 12. Agile Principles for SourceWeave

#### Working software over speculative architecture

Architecture should evolve in response to validated product and engineering needs.

#### Evaluation over intuition

AI-related changes should be measured whenever practical.

#### Small experiments over large implementations

Uncertain approaches should first be tested through the smallest useful experiment.

#### Outcomes over output

Completing an agent, prompt, or feature is not sufficient if it does not improve the intended product outcome.

#### Adaptation over fixed long-term plans

Roadmaps provide direction but may change as benchmark results, technical findings, and user feedback emerge.

### 13. Definition of a Product Increment

A sprint increment is considered potentially releasable when the relevant work:

- functions end to end;
- meets its acceptance criteria;
- passes applicable automated tests;
- has been reviewed;
- has been evaluated when it changes AI behavior;
- introduces no unacceptable regression;
- includes appropriate observability;
- includes updated documentation where needed.

### 14. Initial Definition of Done

A backlog item is Done when:

- implementation is complete;
- acceptance criteria are satisfied;
- automated tests pass;
- code review is complete;
- relevant evaluation has been executed;
- quality regressions are within accepted tolerance;
- documentation is updated;
- observability is included where appropriate;
- no secrets or credentials are exposed;
- the work can be demonstrated during the Sprint Review.

Changes affecting prompts, models, retrieval, ranking, chunking, verification thresholds, retry behavior, or memory require before-and-after evaluation unless explicitly exempted.

### 15. Initial Product Backlog Themes

The Product Backlog will initially be organized around these themes:

- Foundation and reproducibility
- Agent orchestration
- Retrieval and ingestion
- Generation and source attribution
- Verification and retry behavior
- Evaluation and benchmarking
- Observability and reliability
- Performance and cost optimization
- Release readiness

These themes are organizational aids rather than fixed phases. Work may occur across multiple themes within a sprint when necessary to achieve the Sprint Goal.

### 16. Initial Release Strategy

The project will evolve through validated increments rather than a fixed waterfall sequence.

The expected progression is:

Reproducible baseline → working vertical slice → retrieval validation → verification validation → reliability → optimization → release candidate

A release should be driven by evidence that quality and reliability gates have been met rather than simply by completion of a predetermined feature list.

### 17. Initial Risks

| Risk | Impact | Initial Response |
|---|---|---|
| Multi-agent complexity does not materially improve quality | High | Compare against simpler baseline |
| Verification model produces incorrect judgments | High | Create labeled verification test cases |
| External provider dependencies create instability | High | Add failure handling and provider abstraction |
| Model/provider updates alter results | High | Pin configurations and run regression evaluation |
| Verification retries create excessive cost or latency | High | Measure retry effectiveness and cost |
| Retrieval returns irrelevant context | High | Benchmark retrieval strategies independently |
| Memory contaminates evaluation | Medium | Disable or isolate memory during benchmark runs |
| Documentation and implementation drift apart | Medium | Include documentation in Definition of Done |

### 18. Assumptions

Initial assumptions include:

- research-answer quality can be meaningfully evaluated through a combination of automated metrics and targeted human review;
- sufficient benchmark questions can be created to detect regressions;
- external retrieval and LLM providers remain available during early development;
- SourceWeave will initially optimize for developer and research use cases rather than mass-market consumer deployment;
- the existing multi-agent architecture is a starting hypothesis and may be simplified if evidence supports doing so.

### 19. Constraints

Potential constraints include:

- LLM API cost;
- external API availability and rate limits;
- retrieval latency;
- model nondeterminism;
- evaluation cost;
- dependency/version drift;
- limited initial benchmark data;
- development capacity.

Constraints should be reviewed during backlog refinement and sprint planning.

### 20. Governance and Decision Making

The Product Backlog will represent the authoritative list of planned work.

Backlog priority should be influenced by:

- product value;
- risk reduction;
- learning value;
- dependencies;
- engineering effort;
- measurable quality impact.

Significant architectural decisions should be captured using lightweight Architecture Decision Records.

Sprint results may cause planned work to be reprioritized, modified, or removed.

### 21. Initial Sprint Goal

Sprint 1 Goal: Establish a reproducible SourceWeave baseline that can be executed and evaluated consistently.

Expected outcomes include:

- reproducible local setup;
- automated smoke testing;
- initial benchmark dataset;
- functioning evaluation workflow;
- documented baseline quality metrics;
- resolved critical documentation/repository inconsistencies.

This baseline will provide the reference point against which subsequent Agile increments are measured.

### 22. Charter Review

This charter is intentionally lightweight and should evolve with the project.

It should be reviewed when:

- the primary user or product direction changes;
- major scope decisions are made;
- evidence challenges a core product hypothesis;
- a significant release is planned;
- major architectural changes alter the nature of the product.

Changes should preserve the project's core principle:

> Build small, evaluate objectively, learn quickly, and adapt the backlog based on evidence.
