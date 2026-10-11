
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.config import CHAT_MODEL, EMBEDDING_MODEL, OLLAMA_BASE_URL
from app.database import get_sql_engine
from app.ingestion import ingest_document
from app.rag import answer_question

# --------------------------------------------------
# Configuration
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"


# --------------------------------------------------
# Application lifespan
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Jarvis RAG Chatbot")
    yield
    logger.info("Shutting down Jarvis RAG Chatbot")


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Jarvis RAG Chatbot",
    description="Ask questions about your uploaded documents.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# Frontend static files
# --------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR), check_dir=True),
    name="static",
)


# --------------------------------------------------
# Request schemas
# --------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
    )


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.get("/", include_in_schema=False)
def home():
    index_file = STATIC_DIR / "index.html"

    if not index_file.is_file():
        raise HTTPException(
            status_code=500,
            detail="Frontend index.html was not found.",
        )

    return FileResponse(index_file)


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
async def health():
    dependencies = {
        "database": "ok",
        "ollama": "ok",
        "chat_model": "ok",
        "embedding_model": "ok",
    }

    try:
        async with get_sql_engine().connect() as connection:
            await connection.execute(text("SELECT 1"))
    except Exception:
        logger.exception("Database health check failed")
        dependencies["database"] = "unavailable"

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            model_names = {
                model.get("name")
                for model in response.json().get("models", [])
            }
            for key, model_name in (
                ("chat_model", CHAT_MODEL),
                ("embedding_model", EMBEDDING_MODEL),
            ):
                if (
                    model_name not in model_names
                    and f"{model_name}:latest" not in model_names
                ):
                    dependencies[key] = "unavailable"
    except Exception:
        logger.exception("Ollama health check failed")
        dependencies["ollama"] = "unavailable"
        dependencies["chat_model"] = "unavailable"
        dependencies["embedding_model"] = "unavailable"

    healthy = all(status == "ok" for status in dependencies.values())
    if not healthy:
        raise HTTPException(
            status_code=503,
            detail={"status": "unhealthy", "dependencies": dependencies},
        )

    return {"status": "ok", "dependencies": dependencies}


# --------------------------------------------------
# Document upload
# --------------------------------------------------

@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    try:
        if not file.filename or not file.filename.strip():
            raise HTTPException(
                status_code=400,
                detail="A filename is required.",
            )

        filename = Path(file.filename).name

        if not filename:
            raise HTTPException(
                status_code=400,
                detail="A valid filename is required.",
            )

        content = await file.read(MAX_FILE_SIZE + 1)

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File exceeds the 10 MB limit.",
            )

        logger.info("Processing uploaded document: %s", filename)
        document_id = str(uuid4())

        result = await run_in_threadpool(
            ingest_document,
            filename,
            content,
            document_id,
        )

        logger.info("Document processed successfully: %s", filename)

        return result

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception("Document processing failed")

        raise HTTPException(
            status_code=500,
            detail="Document processing failed. Check server logs.",
        ) from error

    finally:
        await file.close()


# --------------------------------------------------
# Chat
# --------------------------------------------------

@app.post("/chat")
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        return answer_question(question)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception("Chat request failed")

        raise HTTPException(
            status_code=500,
            detail="Chat failed. Check server logs.",
        ) from error


@app.get("/documents")
async def list_documents():
    query = text(
        """
        SELECT
            COALESCE(
                langchain_metadata->>'document_id',
                'legacy:' || (langchain_metadata->>'source')
            ) AS id,
            langchain_metadata->>'source' AS filename,
            COUNT(*) AS chunks
        FROM jarvis_data
        WHERE langchain_metadata->>'source' IS NOT NULL
        GROUP BY id, filename
        ORDER BY filename, id
        """
    )

    try:
        async with get_sql_engine().connect() as connection:
            result = await connection.execute(query)
            rows = result.mappings().all()

        return {
            "documents": [
                {
                    "id": row["id"],
                    "filename": row["filename"],
                    "chunks": row["chunks"],
                }
                for row in rows
            ],
            "total": len(rows),
        }

    except Exception as error:
        logger.exception("Document listing failed")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve documents.",
        ) from error

@app.delete("/documents")
async def delete_document(document_id: str):
    """Delete one uploaded document and its associated chunks."""
    document_id = document_id.strip()

    if not document_id:
        raise HTTPException(
            status_code=400,
            detail="Document ID cannot be empty.",
        )

    try:
        legacy_filename = (
            document_id.removeprefix("legacy:")
            if document_id.startswith("legacy:")
            else None
        )
        query = text(
            """
            DELETE FROM jarvis_data
            WHERE langchain_metadata->>'document_id' = :document_id
                OR (
                    :legacy_filename IS NOT NULL
                    AND langchain_metadata->>'source' = :legacy_filename
                    AND langchain_metadata->>'document_id' IS NULL
                )
            RETURNING langchain_metadata->>'source' AS filename
            """
        )
        async with get_sql_engine().begin() as connection:
            result = await connection.execute(
                query,
                {
                    "document_id": document_id,
                    "legacy_filename": legacy_filename,
                },
            )
            row = result.mappings().first()

        if row is None:
            raise HTTPException(
                status_code=404,
                detail="Document was not found.",
            )

        logger.info(
            "Deleted document %s (%s)", document_id, row["filename"]
        )

        return {
            "message": "Document deleted successfully.",
            "document_id": document_id,
            "filename": row["filename"],
        }
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Document deletion failed")
        raise HTTPException(
            status_code=500,
            detail="Could not delete the document.",
        ) from error
