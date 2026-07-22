# mm × pydantic-ai

Five ways to give a pydantic-ai agent multimodal senses with `mm`, ordered
from "works with any model" to "needs a capable agentic model".

The through-line: **LLMs can't read video, audio, or PDFs — mm turns them
into something they can.** Where you put mm in the loop is the design choice.

## Setup

```bash
uv pip install -e ".[dev,mcp]"
uv pip install "pydantic-ai-slim[openai]" "pydantic-ai-harness[code-mode]"
```

All examples read the model from mm's active profile, so one command
repoints every one of them:

```bash
mm profile use ollama        # fully local
mm profile use gateway       # self-hosted / remote gateway
```

Check what your model can actually do before picking an example:

```bash
uv run python examples/pydantic-ai-harness/check_model.py
```

```
  vision (image input)  : yes
  tool calling          : no
```

That distinction matters. Small local models often handle vision fine but
**cannot emit tool calls at all** — with those, variants 3–5 produce
confident prose instead of calling mm. Variants 1–2 have no such
requirement, which is exactly why they exist.

## The variants

| # | Example | mm's role | Tool calling? | Vision? |
|---|---------|-----------|---------------|---------|
| 1 | `01_context_builder.py` | Pre-extracts media → text into the prompt | no | no |
| 2 | `02_vlm_encoder.py` | Compresses media → images a VLM can ingest | no | yes |
| 3 | `03_direct_tools.py` | In-process agent tools | yes | no |
| 4 | `04_mcp_server.py` | Same tools over MCP | yes | no |
| 5 | `05_code_mode.py` | MCP tools batched inside one sandboxed program | yes | no |

### 1. Context builder — mm as a processor

mm extracts every file *before* the agent runs. The model never decides to
call anything; the directory simply arrives pre-digested. This is the
highest-leverage option for small local models.

```python
extractions = mm.cat_many(files)
agent.run_sync([question, *(f"## {e.path.name}\n{e.content}" for e in extractions)])
```

Measured on `mmbench-tiny` (5 files: image, video, audio, PDF):

```
mm compressed 44.2 MB of media → 34,641 chars of text  (1,277x reduction)
```

### 2. VLM encoder — mm as a compressor

Variant 1 hands over mm's *text*. This one keeps the pixels: mm's `mosaic`
encoder samples a video into JPEG contact sheets, and pydantic-ai passes
them as native image content. The VLM never sees a video container.

```
mm encoded 29.3 MB of video → 5 mosaic image(s), 0.3 MB  (101x smaller)
```

`mm_encode.py` holds the ~20 lines that convert mm encoder messages into
pydantic-ai `BinaryContent` — copy it into your own project.

### 3. Direct tools — mm as an agentic toolset

Now the agent chooses: list the directory, peek at metadata, read only what
the question needs. This is what you want once a directory is too big to
extract wholesale — the agent skips the 2 GB video it doesn't care about.

```python
agent = Agent(model, capabilities=[Toolset(mm_toolset())])
```

`mm_tools.py` exposes `cat`, `cat_many`, `peek`, `find`, and `sql`. Each is
a thin forward to mm's public API — `mm.cat()` resolves encoders, pipelines,
and prompts from the file's type.

### 4. MCP server — the same tools, out of process

```bash
mm mcp serve                    # http://127.0.0.1:8765/mcp
```

```python
agent = Agent(model, capabilities=[MCP('http://127.0.0.1:8765/mcp', native=False)])
```

The server ships with mm (`mm/mcp/server.py`) and exposes `cat`, `cat_many`,
`peek`, `find`, `grep`, `sql`. One running server serves pydantic-ai, Claude
Code, an IDE — anything that speaks MCP — and keeps heavy media work off the
agent process.

`native=False` forces the local MCP toolset so tools execute in your
process, which is what lets CodeMode wrap them.

### 5. CodeMode — batch it all in one round-trip

CodeMode collapses every tool into a single sandboxed `run_code`. The model
writes Python that loops, filters, and calls `cat_many` — N files in one
model round-trip instead of N.

```python
agent = Agent(model, capabilities=[CodeMode(), MCP(MCP_URL, native=False)])
```

This pairs unusually well with mm: extraction is slow and parallel, so
batching it inside one program beats a chatty tool loop.

## Notes

- **Caching is free.** mm caches extractions by content hash + profile +
  model, so re-reading a file across runs costs nothing. The second run of
  any example is near-instant.
- **`fast` vs `accurate`.** `fast` is a quick caption or page text;
  `accurate` runs the full LLM pipeline. Tools default to `fast` — the
  agent opts into `accurate` when the question needs detail.
- **Sandbox convention.** Code written for CodeMode returns via `print()`,
  not a top-level `return`.
