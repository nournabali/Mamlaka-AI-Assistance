# Mamlaka AI

A bilingual Arabic/English RAG chatbot for the Almamlaka TV project. It answers only from the three approved PDFs, cites supporting pages, reports conflicting revisions, and refuses unsupported requests.

## Live demo

Deployment URL: _Add the deployed application link here before submission._

If the live service is unavailable during review, use the submitted sample Q&A screenshots or
demo video as a fallback.

## Architecture

```text
Approved PDFs
  → PyMuPDF extraction
  → section-aware chunks
  → multilingual E5 embeddings + FAISS
  → dense/BM25 hybrid retrieval
  → grounding and conflict checks
  → Groq or Ollama generation
  → citation validation
  → Streamlit UI
```

## Approach

- Chunking keeps page, document, and section metadata. Long sections split near sentence boundaries with a 900-character limit and 150-character overlap by default.
- `intfloat/multilingual-e5-small` embeds queries and passages with the required E5 prefixes. This allows Arabic questions to retrieve English PDF text while keeping deployment memory usage practical.
- FAISS is the local vector store and index. It stores normalized vectors for cosine search. BM25 adds exact-name and exact-number recall, and reciprocal-rank fusion combines both rankings.
- Every user turn runs retrieval again. When conversation history exists, the rewrite step distinguishes context-dependent follow-ups from standalone topic changes, preserves requested formats such as bullet lists, selects the requested Arabic or English response language, and can add an English query for cross-language retrieval.
- The LLM receives only retrieved excerpts as evidence. A similarity gate, prompt rules, conflict detection, and prompt-injection screening prevent unsupported answers.
- Common unsupported creative requests, unavailable document facts, prompt injection, forced guessing, and full-document extraction use distinct localized refusal paths in Arabic and English.
- Generated citations are accepted only when their document and page match evidence retrieved for that turn. A section name, when included, must also match. The UI shows validated sources as compact badges.
- All interface images are local assets; the browser does not contact a third-party logo host.

## User interface

- Three suggested questions help users quickly explore the approved project documents. They are selected from a bilingual pool of distinct topics, always include Arabic and English, and remain stable during the conversation.
- Each assistant response includes a copy button for copying the answer text.
- Validated citations are displayed as compact source badges beneath each grounded answer.
- Clearing the conversation requires confirmation to prevent messages from being removed accidentally. A confirmed clear also selects a new random set of three suggested questions.
- The colors, theme, logo, and local avatar assets are styled to match Almamlaka TV branding.
- Arabic messages use a right-to-left layout, while English messages use a left-to-right layout.
- Loading indicators and clear error messages communicate index or LLM-provider availability.
- The responsive layout supports desktop and smaller screens.

## Local setup

Requirements: Python 3.11+ and either a Groq API key or a local Ollama installation.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
cp .env.example .env
```

The included index is ready to use. Inspect it, or rebuild it after changing a PDF, chunk settings,
or the embedding model:

```bash
python scripts/build_index.py --inspect
python scripts/build_index.py --force
```

The inspection command should report 3 documents, 6 pages, and 19 chunks. The first rebuild may
download the embedding model.

## LLM setup

Groq is the default hosted provider. Put the key in `.env`:

```ini
LLM_PROVIDER=groq
LLM_MODEL=qwen/qwen3.6-27b
GROQ_API_KEY=your_groq_key
```

For local Ollama:

```ini
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
```

```bash
ollama pull qwen3:8b
ollama serve
```

Switching LLM providers does not require rebuilding the index. See `.env.example` for retrieval and runtime settings.

## Run

```bash
streamlit run src/mamlaka_ai/ui/streamlit_app.py
```

Open <http://localhost:8501>. Arabic messages use RTL layout and English messages use LTR layout.

## Example questions

- What is the project's main goal?
- Who is the project lead, and which department is she from?
- What is the revised project budget?
- When is the revised public launch date?
- كم عدد اللغات التي سيدعمها تطبيق الهاتف؟
- من يوافق على تغييرات الميزانية التي تزيد عن 50,000 دولار؟

## Test

Fast tests do not require an LLM or embedding-model download:

```bash
pytest
```

Run the multilingual embedding and FAISS checks:

```bash
RUN_EMBEDDING_TESTS=1 pytest
```

The complete embedding-enabled automated suite currently contains 89 tests covering ingestion,
retrieval, conversation rewriting, bilingual behavior, safety routes, citations, UI assets and
suggestions, provider transports, evaluation coverage, and the deployment gate.

Run the acceptance prompts against the configured live LLM:

```bash
python scripts/run_evaluation.py
```

The live suite covers Arabic questions, paraphrases, ambiguous follow-ups, adversarial prompts,
grounded answers, and expected refusals. Run the complete deployment quality gate with:

```bash
python scripts/pre_deploy_check.py
```

This command runs the full embedding/FAISS test suite followed by the live-provider evaluation.
It exits with a failure status if any check fails. The Groq evaluation deliberately pauses between
model-backed cases to stay within token-per-minute limits, so the full gate takes several minutes.

## Deployment

### Prerequisites

Container deployment requires:

- Docker Engine
- Docker Compose v2 (`docker compose`)
- A configured Groq API key, or Ollama with the required model

Use a deployment secret for `GROQ_API_KEY`, then deploy through the guarded wrapper:

```bash
python scripts/deploy.py
```

The wrapper automatically runs `scripts/pre_deploy_check.py` and stops before Docker Compose if
any deterministic or live evaluation fails.

The app listens on port `8501`. For the optional containerized Ollama profile:

```bash
LLM_PROVIDER=ollama python scripts/pre_deploy_check.py
LLM_PROVIDER=ollama docker compose --profile ollama up --build -d
```

## Troubleshooting

- **Missing index:** Run `python scripts/build_index.py --force`.
- **Embedding-model error:** Confirm internet access for the first model download.
- **Groq authentication error:** Check `GROQ_API_KEY` and `LLM_MODEL`.
- **Ollama unavailable:** Run `ollama serve` and confirm the model appears in `ollama list`.
- **Port already in use:** Start Streamlit on another port, for example: `streamlit run src/mamlaka_ai/ui/streamlit_app.py --server.port 8502`.

## Security

Never commit or distribute `.env`, API keys, model caches, or platform credentials. Store
`GROQ_API_KEY` in deployment secrets and keep `DEBUG_RETRIEVAL=false` in public environments,
because retrieval diagnostics can expose document excerpts.
