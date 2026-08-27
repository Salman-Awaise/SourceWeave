# Developer guide

How to work on SourceWeave: where the seams are, how to extend it, and the
things that will bite you.

For *why* the system is shaped this way, read [`docs/design/`](design/). This
document is the *how do I change it* layer.

---

## Setup

```bash
uv sync --all-extras          # runtime + eval + memory extras + dev tools
uv run pytest                 # 270 tests, ~3 seconds, no credentials needed
```

That second command should pass on a clean machine with no `.env` and no
network. If it doesn't, something is wrong with the environment, not your
changes.

For anything that touches a real service:

```bash
cp .env.example .env          # OPENAI_API_KEY is the only one really required
docker compose up -d qdrant   # or point QDRANT_URL at Qdrant Cloud
uv run research-system config # shows which backends are ready, redacts secrets
```

### The checks that must pass

```bash
uv run ruff check .           # lint
uv run ruff format .          # formatting
uv run mypy                   # types, 27 source files
uv run pytest                 # offline suite
```

---

## The one idea that explains the codebase

**Every external service sits behind a Protocol, and nodes receive their
collaborators as an argument.**

A LangGraph node's signature is fixed — it takes state and returns state. There
is nowhere to inject a database client. So each agent takes a second argument:

```python
def planner_node(state: AgentState, deps: Dependencies) -> AgentState: ...
```

and `graph.py` binds `deps` when it builds the graph:

```python
builder.add_node("plan", _safe_node("plan", planner_node, deps))
```

`Dependencies` is just a box holding six things:

```python
settings  llm  embeddings  vector_store  web_search  memory
```

Each of the five adapters is a `Protocol` with a real implementation and a fake
one:

| Protocol | Real | Fake |
|---|---|---|
| `LLMClient` | `LiteLLMClient` | `FakeLLMClient` |
| `EmbeddingProvider` | `OpenAIEmbeddings` | `FakeEmbeddings` |
| `VectorStore` | `QdrantVectorStore` | `InMemoryVectorStore` |
| `WebSearchProvider` | `TavilySearch` | `FakeWebSearch` |
| `MemoryStore` | `Mem0MemoryStore` | `InMemoryMemoryStore`, `NullMemoryStore` |

This is why the whole suite — including full graph runs with retry loops — needs
no API keys and no network. It is also why swapping model providers is a line in
`.env` rather than a code change.

**If you add an external dependency, it goes behind a Protocol with a fake.**
Otherwise the offline suite stops being offline, and that property is worth more
than the convenience of calling a client directly.

---

## Writing tests

`tests/conftest.py` gives you a wired-up offline world:

```python
def test_something(deps):          # Dependencies with every adapter faked
    deps.llm = FakeLLMClient([...]) # script the model responses in order
    update = planner_node(create_initial_state("q"), deps)
    assert update["retrieval_strategy"] == "web"
```

Fixtures: `deps`, `memory_deps`, `settings`, `llm`, `embeddings`,
`vector_store`, `web_search`, `doc_factory`.

### FakeLLMClient

Responses are consumed in order and may be a Pydantic model, a dict, a JSON
string, or an `Exception` to raise:

```python
deps.llm = FakeLLMClient([
    PlannerOutput(sub_queries=["a"], retrieval_strategy="web"),
    GeneratorOutput(answer="A [Source 1]", confidence=0.9, sources_used=[1]),
    VerifierOutput(claims=["c"], supported_claims=["c"], unsupported_claims=[]),
])
```

Running out of responses raises `AssertionError` rather than returning
something plausible, so a test that triggers an unexpected extra model call
fails loudly instead of passing by accident.

`deps.llm.calls` records what was sent, which is how the prompt-content tests
work.

### Disabling a backend

Set it to `None`:

```python
deps.vector_store = None   # hybrid now degrades to web-only
deps.web_search = None     # and vice versa
```

### Test the behaviour, not the mock

Assert on returned state, warnings, and citations. A test that only checks a
fake was called proves nothing about whether the system works.

---

## Common tasks

### Add a configuration setting

1. Add the field to `Settings` in `config.py` with a default and bounds:
   ```python
   my_setting: int = Field(default=5, ge=1, le=100)
   ```
2. Document it in `.env.example` and the README's config table.
3. If it is a secret, the redaction in `Settings.redacted()` keys off suffixes
   (`_api_key`, `_key`, `_token`, `_secret`) — name it accordingly and it is
   hidden automatically.

There are 33 settings. Anything tunable belongs here, not hard-coded in an
agent.

### Add a retrieval backend

1. Write an adapter implementing `WebSearchProvider` or `VectorStore`.
2. Write a fake alongside it in the same module.
3. Add it to `Dependencies` and `build_dependencies()`.
4. Handle it in `retriever.py`: `_resolve_strategy()` decides which backends
   run, `gather()` collects ranked lists.
5. Give its documents a `source_type` in metadata — fusion weighting and the
   per-type floor key off that.

### Add or change an agent

Agents are plain functions `(state, deps) -> AgentState`. They return **partial**
updates; LangGraph merges them into state. Only return the keys you changed.

Register the node in `graph.py` and wire its edges. Note that `plan`,
`retrieve` and `generate` use conditional edges that short-circuit to `END` when
`error` is set, so a hard failure stops immediately instead of flowing onward
and having its cause overwritten by a downstream symptom.

### Change a prompt

Prompts live with their agent. Evidence formatting is shared in `prompts.py` on
purpose: the generator and verifier **must** see identically numbered sources,
or a `[Source 3]` citation means different things in the two stages. Don't
duplicate that formatting.

### Add structured model output

Define a Pydantic model in `schemas.py`, then:

```python
output = deps.require_llm().complete_structured(
    system=PROMPT, user=user_prompt, schema=MySchema,
)
```

Always have a fallback for `StructuredOutputError`. Models return invalid output
often enough that treating it as exceptional is wrong.

---

## Things that will bite you

### `ruff` saying "No fixes available" is not "passed"

**In short:** the linter can print something that looks like success when it
actually failed. Read the last line, not the shape of the output.

```
Found 1 error.
No fixes available (1 hidden fix can be enabled with the `--unsafe-fixes` option).
```

That is a **failure**. A clean run says `All checks passed!`. This has already
caused one lint error to be committed.

### A skipped test is not a passing test

**In short:** a test that never ran still shows up as "not failed". A green
summary can mean nothing was actually checked.

```
RUN_INTEGRATION=1 uv run pytest
270 passed, 11 skipped    <- the 11 tested nothing
```

Integration tests skip when Qdrant is unreachable, and the summary line reads
as success. That once hid a real bug for an entire session: the credential-
scrubbing fixture also deleted `QDRANT_URL`, so the suite silently fell back to
`localhost` and reported clean while testing nothing at all. Check the count.

To run them against Qdrant Cloud rather than a local container:

```bash
eval "$(grep -E '^(QDRANT_URL|QDRANT_API_KEY)=' .env | sed 's/^/export /')"
RUN_INTEGRATION=1 uv run pytest      # expect 281 passed, 0 skipped
```

### The test environment is deliberately scrubbed

**In short:** tests cannot see your real API keys, on purpose. If a test says a
setting is missing when you know it is in `.env`, this is why.

`conftest.isolated_env` deletes `OPENAI_API_KEY`, `QDRANT_URL` and friends from
the environment for every test, so a developer's real `.env` can never influence
a result. Integration tests are exempted for the Qdrant variables only, and only
when `RUN_INTEGRATION=1`.

If a test mysteriously cannot see an environment variable, this is why.

### Faithfulness is recomputed, not reported

**In short:** we do not trust the model's own grade for its own answer. We count
the supported claims ourselves.

`verifier.py` discards the model's score and recalculates `supported / total`.
If you change the verifier, keep that. A model that reports 0.95 while listing
three of ten claims unsupported must not set the number.

---

## Debugging a query

```bash
uv run research-system demo --user-id you -v    # -v enables debug logging
```

To inspect a single stage without running the whole graph:

```python
from research_system.core.deps import build_dependencies
from research_system.core.state import create_initial_state
from research_system.agents.planner import planner_node
from research_system.agents.retriever import retriever_node

deps = build_dependencies()
state = create_initial_state("your question")
state.update(planner_node(state, deps))
print(state["sub_queries"], state["retrieval_strategy"])

docs = retriever_node(state, deps)["retrieved_documents"]
for d in docs:
    print(d.source_type, d.source, d.score, d.metadata.get("rrf_weight"))
```

That is how the fusion and decomposition problems were found. Most retrieval
bugs are visible at the sub-query level before any model writes an answer.

Set `LANGCHAIN_API_KEY` for LangSmith traces of the whole graph. Document
content is redacted unless `TRACE_CONTENT=true`.

---

## Evaluation

```bash
uv run research-system benchmark eval/datasets/your_qa.json --strategy hybrid
```

Datasets are JSON arrays of `{"question": ..., "ground_truth": ...}`. Output
records model names, dependency versions and a SHA-256 of the dataset, so a
score can always be traced to what produced it.

Two things worth knowing before you read any number:

- **Match the dataset to the corpus.** Questions the corpus cannot answer
  measure web search, not your system.
- **Run it more than once.** Faithfulness moved `0.9459 → 0.9833` between two
  identical invocations. A single run cannot distinguish a real change from
  noise.

---

## Design documents

`.docx` originals live outside the repository. The Markdown in
[`docs/design/`](design/) is the current version and has been updated to match
the implementation.

`scripts/docx2md.py` did the original conversion. It skips existing files unless
you pass `--force`, because regenerating would discard those updates.
