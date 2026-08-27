# Architecture Spikes

> **Derived from `docx/Architecture Spikes.docx`, then updated to match the implementation.**
>
> The Word file is the original draft, kept unchanged for history and held
> outside this repository.
> This Markdown copy is the current version and has since diverged:
> Outcomes recorded for all five spikes; scorecard completed.
>
> Re-running `scripts/docx2md.py` regenerates from the Word file and will
> discard these edits. Edit this file directly instead.

Project: SourceWeave
Version: 0.1
Status: Proposed
Delivery Model: Agile
Purpose: Resolve high-risk architectural uncertainty through time-boxed experiments before committing to larger implementation work.

---

## 1. Architecture Spike Principles

Architecture spikes are temporary investigative backlog items.

A spike should:

- answer a specific architectural question;
- test an explicit hypothesis;
- be time-boxed;
- produce evidence;
- avoid unnecessary production implementation;
- conclude with a decision;
- generate follow-up backlog items when needed.

A spike is Done when the architectural question has been answered sufficiently to make a decision.

The outcome does not need to confirm the original hypothesis.

A disproven hypothesis is a successful spike if it prevents unnecessary implementation.

---

## 2. Standard Spike Output

Every SourceWeave architecture spike should produce:

- experiment description;
- test configuration;
- dataset or test cases used;
- quantitative results;
- qualitative findings;
- architecture recommendation;
- ADR status recommendation;
- follow-up stories;
- identified risks or unanswered questions.

---

## SPIKE-ARCH-001 — Validate Multi-Agent Topology

Related ADR: ADR-003 — Four-Agent Responsibility Model
Priority: High
Suggested Timebox: 1–2 engineering days
Risk Addressed: Architectural complexity may not produce sufficient value.

### Architectural Question

Does the:

Planner → Retriever → Generator → Verifier

architecture provide enough measurable benefit over a simpler pipeline to justify its additional:

- complexity;
- latency;
- model calls;
- operating cost;
- failure modes?

---

### Hypothesis

> Separating planning, retrieval, generation, and verification into distinct responsibilities produces measurably better research quality and diagnosability than a simpler retrieval-generation pipeline, with an acceptable increase in latency and cost.

---

### Variants to Compare

#### Variant A — Full SourceWeave

```text
Planner
   ↓
Retriever
   ↓
Generator
   ↓
Verifier
```

#### Variant B — Simplified RAG

```text
Retriever
   ↓
Generator
```

#### Optional Variant C — Single-Agent

```text
Single Agent
   ↓
tools/retrieval as needed
   ↓
Response
```

Variant C is optional for the first spike if implementing it substantially increases the spike scope.

---

### Experiment

Run the same evaluation dataset through Variant A and Variant B.

Use identical where practical:

- underlying LLM;
- embedding model;
- document corpus;
- retrieval limits;
- generation settings;
- evaluation dataset.

Capture results per question as well as aggregate results.

---

### Evaluation Metrics

#### Quality

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall
- unsupported claim rate

#### Operational

- average latency;
- p95 latency;
- model calls/query;
- token usage/query;
- estimated cost/query;
- failure rate.

#### Architecture

Qualitatively assess:

- debuggability;
- traceability;
- ease of testing;
- isolation of responsibilities;
- complexity of failure handling.

---

### Acceptance Criteria

The spike is complete when:

- At least two architecture variants are implemented sufficiently for comparison.
- Both variants execute against the same benchmark dataset.
- Quality results are captured for both variants.
- Latency and model-call counts are captured.
- Cost/token differences are estimated where possible.
- Major failure modes are documented.
- Results identify where the multi-agent architecture improves or degrades performance.
- A recommendation is documented.

---

### Decision Criteria

#### Accept ADR-003

If the four-agent architecture demonstrates a meaningful combination of:

- improved faithfulness;
- improved answer quality;
- better failure visibility;
- improved controllability;

without unacceptable cost or latency.

#### Modify ADR-003

If only some responsibilities justify separation.

Example:

```text
Retriever → Generator → Verifier
```

may outperform the architecture sufficiently without requiring Planner execution for every query.

#### Reject ADR-003

If simpler architecture performs equivalently or better and the additional agent boundaries provide no meaningful operational advantage.

---

### Expected Output

Architecture Decision:

ACCEPT / MODIFY / REJECT / MORE EVIDENCE REQUIRED

Include a short evidence statement.

Example:

> Verification improved faithfulness by 9%, while Planner routing provided no measurable benefit on the initial dataset. Retain Verifier but make Planner conditional.

---

### Outcome — complete

**Status:** Complete. Hypothesis partly supported.

Compared the four-agent pipeline against a single-agent baseline holding model,
temperature, generation rules and retrieval backends constant. The baseline
skips decomposition and has no verification loop; nothing else differs.

On a ten-question set the corpus can answer:

| Metric | Multi-agent | Single-agent | Delta |
|---|---|---|---|
| faithfulness | 0.9833 | 0.8994 | +0.0839 |
| answer relevancy | 0.9425 | 0.9422 | +0.0003 |
| context precision | 0.8126 | 0.7760 | +0.0366 |
| context recall | 1.0000 | 1.0000 | 0.0000 |

On a six-question set the corpus **cannot** answer, the baseline wins three of
four metrics, losing only on recall. The mechanism is not noise: decomposition
into four sub-queries retrieves ten documents where one query retrieves three.
More retrieval raises recall and lowers precision, and precision-weighted
metrics then favour the simpler system.

**Finding:** query decomposition helps when the corpus can answer the question
and hurts when it cannot.

**Decision:** keep the topology, and record that its benefit is conditional on
corpus fit rather than unconditional. See ADR-003.

**Caveat:** ten and six questions, single runs, self-authored evaluation sets.
Faithfulness moved 0.9459 → 0.9833 between two identical invocations, a spread
wider than several of the reported deltas.

---

## SPIKE-ARCH-002 — Validate Hybrid Retrieval

Related ADR: ADR-004 — Hybrid Retrieval
Related ADR: ADR-005 — Reciprocal Rank Fusion
Priority: High
Suggested Timebox: 1–2 engineering days
Risk Addressed: Hybrid retrieval may increase latency without improving evidence quality.

---

### Architectural Question

Does combining vector and web retrieval provide measurably better evidence than either retrieval channel independently?

---

### Hypothesis

> Hybrid retrieval improves context recall and downstream answer quality for research questions requiring both indexed knowledge and current/public information, while maintaining acceptable context precision and latency.

---

### Variants

#### Variant A — Vector Only

```text
Query
 ↓
Qdrant
 ↓
Context
```

#### Variant B — Web Only

```text
Query
 ↓
Web Search
 ↓
Context
```

#### Variant C — Hybrid

```text
        Qdrant
          ↓
Query → Fusion → Context
          ↑
       Web Search
```

---

### Dataset Segmentation

The benchmark should contain identifiable question types.

#### Category 1 — Internal / Indexed

Expected to favor vector retrieval.

#### Category 2 — Current / External

Expected to favor web retrieval.

#### Category 3 — Mixed

Requires evidence from both sources.

#### Category 4 — Ambiguous

Used to test Planner strategy selection.

---

### Experiment

Run every query against:

- vector-only;
- web-only;
- hybrid.

Where possible, evaluate retrieval independently from answer generation.

Then evaluate downstream answer quality.

---

### Evaluation Metrics

#### Retrieval

- Context Precision
- Context Recall
- relevant documents @ K
- duplicate-result rate
- empty-result rate

#### Generation

- Faithfulness
- Answer Relevancy

#### Operational

- retrieval latency;
- total query latency;
- external API calls;
- cost/query.

---

### RRF Evaluation

For hybrid retrieval, separately compare:

#### Baseline 1

Simple concatenation.

#### Baseline 2

Reciprocal Rank Fusion.

Optional future comparisons:

- weighted fusion;
- learned reranking;
- cross-encoder reranking.

These are not required unless RRF performs poorly.

---

### Acceptance Criteria

- Vector-only retrieval can be independently executed.
- Web-only retrieval can be independently executed.
- Hybrid retrieval can be independently executed.
- All three are evaluated against the same classified dataset.
- Context Precision and Recall are calculated.
- Latency is measured.
- Mixed-query performance is analyzed separately.
- RRF is compared against at least one simpler fusion baseline.
- A retrieval-strategy recommendation is produced.

---

### Decision Criteria

#### Accept Hybrid as Default for Mixed Queries

If it improves recall/answer quality materially without unacceptable precision loss or latency.

#### Keep Hybrid but Make It Conditional

If hybrid helps only specific query categories.

This is likely preferable to automatically using hybrid retrieval everywhere.

#### Reject Hybrid

If the quality improvement is negligible or negative relative to operational cost.

---

### Key Evaluation Question

Do not ask only:

> Is hybrid retrieval better?

Ask:

> For which query classes is hybrid retrieval better?

The architecture should allow routing based on that answer.

---

### Outcome — complete

**Status:** Complete. Hypothesis supported, with a correction.

Hybrid retrieval works, but validation exposed a flaw in the fusion strategy
rather than in the hybrid approach. With each result appearing in exactly one
ranked list, every RRF score collapsed to approximately `1 / (k + 1)`, leaving
the ordering between a primary source and a web summary effectively arbitrary.

Observed: a question about an indexed paper produced three web citations,
including a video link, while the paper sat unused at rank 1.

Measured fusion mixes for the same question:

```text
weights 1.0 / 1.0, floor 0    DwDwDwDwDw    5 documents, 5 web
weights 1.2 / 1.0, floor 0    DDDDDDDDDD   10 documents, 0 web
weights 1.2 / 1.0, floor 2    DDDDDDDDww    8 documents, 2 web
```

Weighting alone removed web entirely from a question that asked for recent
developments, which is why the per-type floor exists.

**Decision:** keep hybrid retrieval; extend fusion per ADR-021.

---

## SPIKE-ARCH-003 — Validate Independent Verification

Related ADR: ADR-008 — Independent Verification
Priority: Critical
Suggested Timebox: 2 engineering days
Risk Addressed: Verification may add cost without reliably detecting unsupported claims.

---

### Architectural Question

Does an independent Verifier materially reduce unsupported factual responses?

---

### Hypothesis

> Evaluating generated answers against retrieved evidence using an independent verification step significantly improves faithfulness and detects unsupported claims that would otherwise be returned to users.

---

### Experiment Variants

#### Variant A — No Verification

```text
Retriever
   ↓
Generator
   ↓
Response
```

#### Variant B — Verification

```text
Retriever
   ↓
Generator
   ↓
Verifier
   ↓
Response
```

#### Variant C — Verification + Retry

```text
Retriever
   ↓
Generator
   ↓
Verifier
   │
   ├── pass → Response
   │
   └── fail → Retry
```

Variant C may overlap with SPIKE-ARCH-004 and can be separated if needed.

---

### Verification Test Dataset

Create intentionally varied examples.

Include:

- fully supported answers;
- partially supported answers;
- clearly unsupported factual claims;
- conflicting evidence;
- insufficient evidence;
- plausible but fabricated facts;
- numerical claims;
- multi-claim responses.

Where practical, label the claims manually.

---

### Metrics

#### Detection Quality

- True Positives;
- False Positives;
- True Negatives;
- False Negatives.

Calculate where dataset size permits:

- precision;
- recall;
- F1 score.

#### Product Quality

- Faithfulness;
- unsupported claim rate;
- answer relevancy.

#### Operational

- verification latency;
- additional model tokens;
- cost/query.

---

### Important Evaluation

Measure Verifier false negatives carefully.

A Verifier that regularly approves unsupported answers provides false assurance and may be worse than having no verification mechanism.

Also measure false positives:

> Does the Verifier reject well-grounded answers too often?

---

### Acceptance Criteria

- A labeled set of verification scenarios exists.
- Supported and unsupported answers are represented.
- Verification results can be compared against labels.
- False positive and false negative behavior is measured.
- End-to-end faithfulness is measured with and without verification.
- Added latency is measured.
- Added model calls/cost are measured.
- At least one verification threshold comparison is performed.
- Recommendation on retaining/modifying Verifier is documented.

---

### Decision Criteria

#### Accept ADR-008

If verification materially improves unsupported-claim detection and faithfulness with acceptable cost/latency.

#### Modify ADR-008

Possible modifications include:

- use Verifier only for factual queries;
- verify only high-risk claims;
- use deterministic validation where possible;
- use a different model for verification;
- modify verification threshold.

#### Reject ADR-008

If:

- detection is unreliable;
- false confidence is excessive;
- quality gains are negligible;
- operational cost outweighs benefit.

---

### Outcome — not yet run

**Status:** Open.

Verification is implemented and running in the production path, but no
experiment has isolated its *value* against a no-verifier variant on an
identical dataset. The single-agent baseline used in SPIKE-ARCH-001 differs in
two ways at once — no decomposition and no verification — so it cannot attribute
the effect to verification alone.

A dedicated variant holding decomposition constant and toggling only
verification remains outstanding.

---

## SPIKE-ARCH-004 — Validate Evidence-Driven Retry

Related ADR: ADR-010 — Evidence-Driven Retry Queries
Priority: High
Suggested Timebox: 1–2 engineering days
Risk Addressed: Verification retries may repeat the same failure while doubling cost.

---

### Architectural Question

When verification fails, how should SourceWeave obtain better evidence?

---

### Hypothesis

> Retrieval queries derived from unsupported claims or identified evidence gaps produce more useful retry evidence and higher verification-success rates than simply rerunning the original retrieval queries.

---

### Variants

#### Variant A — Existing Retry

```text
Verification Failure
      ↓
Reuse Previous Query
      ↓
Retriever
```

#### Variant B — Unsupported-Claim Query

```text
Verification Failure
      ↓
Unsupported Claims
      ↓
Retry Query Builder
      ↓
Retriever
```

#### Variant C — Re-Planning

```text
Verification Failure
      ↓
Failure Evidence
      ↓
Planner
      ↓
New Retrieval Plan
      ↓
Retriever
```

Variant C should be included only if implementation cost remains within the spike timebox.

---

### Example

Initial query:

> Compare the long-term memory approaches used by Agent A and Agent B.

Generated answer contains:

> Agent B stores episodic memory in a graph database.

Verifier determines that claim is unsupported.

Instead of rerunning:

> Compare the long-term memory approaches used by Agent A and Agent B.

retry with something closer to:

> What storage system does Agent B use for episodic memory?

This creates a targeted research loop.

---

### Evaluation Metrics

#### Retry Effectiveness

- retry success rate;
- percentage of failed answers becoming verified;
- new relevant evidence retrieved;
- duplication of original results;
- change in faithfulness score.

#### Efficiency

- additional retrieval latency;
- additional model calls;
- additional token usage;
- cost per successful recovery.

#### Termination

- retries exhausted rate;
- average retries/query.

---

### Acceptance Criteria

- Existing retry behavior is measured.
- At least one evidence-driven retry strategy is implemented.
- Identical failed scenarios are run through both approaches.
- New-document retrieval rate is measured.
- Verification recovery rate is measured.
- Additional latency is measured.
- Additional cost/model usage is measured.
- Retry loops remain bounded.
- Recommendation identifies the preferred retry strategy.

---

### Decision Criteria

#### Accept Evidence-Driven Retry

If targeted retries materially increase successful recovery.

Example decision threshold:

> At least 15–20% relative improvement in verification recovery without disproportionate latency/cost.

The exact threshold should be adjusted after baseline measurement.

#### Prefer Deterministic Refinement

If simple unsupported-claim transformations perform nearly as well as calling Planner again.

#### Re-Plan

If Planner-based refinement clearly produces better evidence.

#### Reject Retry

If retries rarely improve outcomes.

In that case, returning:

BEST_EFFORT / INSUFFICIENT_EVIDENCE

may be more appropriate than consuming additional model calls.

---

### Outcome — complete

**Status:** Complete. Hypothesis supported.

The Verifier now derives refined queries from the specific unsupported claims,
and the graph loops back to Retrieval with prior evidence carried forward and
fused with the new results.

| Measure | Before | After |
|---|---|---|
| Questions producing scorable answers | 3 of 6 | 6 of 6 |
| Retry rate | 0.50 | 0.17 |
| Verification pass rate | 0.67 | 0.83 |

Retry rate fell because the pipeline stopped spending retries re-running
searches that could not succeed.

A related inefficiency was found and fixed: refinement was being requested even
when the retry budget was already exhausted, spending a model call whose result
was discarded.

**Decision:** adopt. See ADR-010.

---

## SPIKE-ARCH-005 — Validate Memory Separation

Related ADR: ADR-011 — Memory Is Context, Not Evidence
Priority: High
Suggested Timebox: 1 engineering day
Risk Addressed: Persistent memory may contaminate retrieval, evaluation, and evidence provenance.

---

### Architectural Question

Should memory remain explicitly separate from the original research query and retrieved evidence?

---

### Hypothesis

> Keeping memory, user query, and retrieved evidence as separate state objects improves provenance, benchmark reproducibility, and factual grounding without materially reducing useful conversational personalization.

---

### Architecture Variants

#### Variant A — Concatenated Context

```text
Memory
  +
User Query
   ↓
Combined Query
```

#### Variant B — Explicit Separation

```text
original_query
memory_context
retrieved_evidence
```

Agents receive each field intentionally based on responsibility.

---

### Experiment

Create a small set of multi-turn scenarios.

#### Scenario A — Useful Memory

User establishes a legitimate preference or context.

Example:

> Focus on Python examples.

Follow-up:

> Show me how to implement this.

Expected:

Memory improves response relevance.

#### Scenario B — Incorrect Historical Memory

Memory contains outdated or incorrect factual information.

Expected:

Memory must not override retrieved evidence.

#### Scenario C — Benchmark Contamination

Run the same benchmark:

- clean memory;
- polluted memory.

Expected:

Standard evaluation results should remain stable when memory is disabled/isolated.

#### Scenario D — Conflicting Evidence

Memory says X.

Retrieved evidence says Y.

Expected:

Factual response follows evidence or clearly represents uncertainty.

---

### Evaluation Metrics

- benchmark score variance;
- retrieval-query variance;
- unsupported claim rate;
- correct evidence preference;
- personalization preservation;
- trace/provenance clarity.

---

### Security / Privacy Evaluation

Also determine whether memory content can unnecessarily propagate into:

- web-search queries;
- traces;
- source attribution;
- benchmark datasets.

---

### Acceptance Criteria

- Memory and no-memory scenarios are tested.
- A conflicting-memory scenario is tested.
- Benchmark contamination is tested.
- Retrieved evidence remains identifiable separately from memory.
- Agent behavior is documented when memory conflicts with evidence.
- Standard evaluation can disable or isolate memory.
- Trace output preserves provenance.
- Recommendation on state separation is documented.

---

### Decision Criteria

#### Accept ADR-011

If separation improves:

- provenance;
- evaluation reproducibility;
- security;
- factual behavior;

with limited implementation overhead.

#### Modify

If only specific agents need explicit memory separation.

For example:

```text
Planner     → query + limited memory
Retriever   → query only
Generator   → query + evidence + relevant memory
Verifier    → evidence + response only
```

This may become the preferred architecture.

#### Reject

Only if separation creates substantial complexity without observable benefit.

---

### Outcome — complete

**Status:** Complete. Hypothesis supported, with a limitation worth recording.

`memory_context` is a separate state field, reaches the Planner only, and never
enters the Generator's evidence prompt. A unit test asserts memory text cannot
appear there. Benchmark runs execute with memory disabled.

Verified live: memory written in one session was recalled in a separate process
with no chat history, and a different user retrieved nothing.

**Limitation found:** memory can steer *what is searched* but can never itself
be the answer. Asked "which paper did I say I care about", the system returned
NO ANSWER while memory held exactly that fact.

This is the intended trade-off. mem0 stores a model-written paraphrase of the
conversation, so treating it as citable evidence would mean asserting facts from
a summary of a summary and would break the guarantee that every claim traces to
a retrievable source.

Two alternatives were rejected. Showing the Generator memory without citation
breaks verification, since the resulting claims are ungroundable and trigger
pointless retries. Making memory a citable source lets an incorrect memory
become a *verified* fact.

**Decision:** keep the separation; state the reason in the response when
declining. See ADR-011.

---

## 3. Recommended Execution Order

The spikes should not all run simultaneously.

Recommended order:

```text
SPIKE-001
Multi-Agent Topology
       │
       ▼
SPIKE-002
Hybrid Retrieval
       │
       ▼
SPIKE-003
Independent Verification
       │
       ▼
SPIKE-004
Evidence-Driven Retry
       │
       ▼
SPIKE-005
Memory Separation
```

However, SPIKE-002 and SPIKE-003 can run independently if multiple contributors are available.

---

## 4. Suggested Sprint Placement

### Sprint 1 — Baseline

Primary objective:

Make evaluation reproducible.

Architecture spikes should not begin in earnest until the benchmark harness is trustworthy.

---

### Sprint 2 — Retrieval Architecture

Run:

- SPIKE-ARCH-002 — Hybrid Retrieval

Potential follow-up ADR decisions:

- hybrid retrieval;
- RRF;
- Planner routing.

---

### Sprint 3 — Agent & Verification Architecture

Run:

- SPIKE-ARCH-001 — Multi-Agent Topology;
- SPIKE-ARCH-003 — Independent Verification.

Primary question:

> Does SourceWeave's agentic architecture produce enough measurable value?

---

### Sprint 4 — Feedback Loop Architecture

Run:

- SPIKE-ARCH-004 — Evidence-Driven Retry.

Primary question:

> Can failed verification actually drive better research?

---

### Sprint 5 — Context Architecture

Run:

- SPIKE-ARCH-005 — Memory Separation.

Primary question:

> Can personalization remain useful without compromising evidence provenance?

---

## 5. Spike Definition of Done

An architecture spike is Done when:

- The architectural question is explicitly answered.
- The hypothesis was tested.
- Experiment configuration is documented.
- Relevant metrics were captured.
- Results are reproducible enough for review.
- Trade-offs are documented.
- A decision recommendation exists.
- Associated ADR is updated.
- Follow-up implementation stories are identified.
- Remaining uncertainty is explicitly documented.

A spike is not Done simply because a prototype was created.

---

## 6. Architecture Decision Review Format

At Sprint Review, present each completed spike using five questions:

#### 1. What did we believe?

State the hypothesis.

#### 2. What did we test?

Describe the experiment.

#### 3. What happened?

Present quantitative and qualitative evidence.

#### 4. What did we learn?

Explain why the result matters architecturally.

#### 5. What are we deciding?

Choose:

ACCEPT

MODIFY

REJECT

DEFER

RUN FOLLOW-UP SPIKE

---

## 7. Spike Scorecard

Use this summary table during Sprint Review.

| Spike | Hypothesis Supported? | Quality Impact | Latency Impact | Cost Impact | Decision |
|---|---|---|---|---|---|
| ARCH-001 Multi-Agent | Partly — corpus-dependent | +0.084 faithfulness, +0.037 context precision on a matched corpus; baseline wins 3 of 4 metrics on a mismatched one | +2.3 s/question | 3–6 model calls vs 1 | Keep, with the caveat recorded (ADR-003) |
| ARCH-002 Hybrid Retrieval | Yes, with a correction | Plain RRF under-determined the ranking; source-type weighting added | negligible | none | Keep; extended by ADR-021 |
| ARCH-003 Verification | Not yet run | — | — | — | Open |
| ARCH-004 Retry Refinement | Yes | Scorable answers 3/6 → 6/6 | retry rate 0.50 → 0.17, so lower in aggregate | fewer wasted retries | Adopt (ADR-010) |
| ARCH-005 Memory Separation | Yes, with a limitation | No contamination; memory cannot become a citation | none | none | Keep (ADR-011) |

---

## 8. Guiding Rule

The purpose of these spikes is not to prove that SourceWeave's current architecture is correct.

The purpose is to discover the simplest architecture that achieves the required quality, trustworthiness, observability, and performance.

Every significant architectural component should earn its place through evidence.
