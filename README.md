# Jarvis

Local document question-answering with FastAPI, PostgreSQL/pgvector, and Ollama.

## Project map

- `app/main.py` — FastAPI routes for chat, health, and document management.
- `app/ingestion.py` — reads uploaded files and splits them into chunks.
- `app/rag.py` — retrieves document chunks, asks Ollama, and checks citations.
- `app/database.py` — PostgreSQL connections, vector storage, and setup.
- `app/config.py` — reads settings from `.env`.
- `static/` — the web page, styling, JavaScript, and favicon.
- `tests/` — automated checks for the API, ingestion, and RAG behavior.
- `docker-compose.yml` — starts the local PostgreSQL/pgvector database.

Start with `app/main.py` for API behavior, `app/ingestion.py` for uploads, or
`app/rag.py` for answers. Keep local secrets in `.env`; only share
`.env.example`.

## Run locally

1. Create and activate a virtual environment, then install dependencies:

   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env`. Set `POSTGRES_PASSWORD` for Docker Compose
   and use the same password in `DATABASE_URL`.
3. Start PostgreSQL:

   ```sh
   docker compose up -d db
   ```

4. Install and start Ollama, then download the models named by `CHAT_MODEL` and
   `EMBEDDING_MODEL` in `.env`:

   ```sh
   ollama pull qwen2.5:3b
   ollama pull embeddinggemma
   ```

5. Initialize the vector table and start the API:

   ```sh
   python init_db.py
   uvicorn app.main:app --reload
   ```

   Keep Ollama running locally while using Jarvis. `/health` checks PostgreSQL,
   Ollama, and both configured models; it returns HTTP 503 if any dependency is
   unavailable.

## Data behavior

Each upload receives its own document ID, so uploading two files with the same
filename keeps them distinct and allows deleting either one. Documents created
before IDs were introduced remain listed and can still be deleted by their
legacy filename grouping.

Changing `EMBEDDING_MODEL` may change vector dimensions. Run `python init_db.py`
after changing it; initialization refuses to use a table whose vector dimension
does not match the configured model. Rebuilding the table removes existing
vectors, so back up or re-upload documents before doing so.

## Checks

Run the tests using the project virtual environment, then compile-check the
application modules:

```sh
.venv/bin/python -m pytest
python -m compileall -q app init_db.py
```
