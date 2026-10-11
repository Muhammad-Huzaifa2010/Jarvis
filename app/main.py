
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from app.database import get_vector_store
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.ingestion import ingest_document
from app.rag import answer_question

from sqlalchemy import text

from app.database import get_pg_engine




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
def health():
    return {"status": "ok"}


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

        result = await run_in_threadpool(
            ingest_document,
            filename,
            content,
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
def list_documents():
    query = text(
        """
        SELECT
            langchain_metadata->>'source' AS filename,
            COUNT(*) AS chunks
        FROM jarvis_data
        WHERE langchain_metadata->>'source' IS NOT NULL
        GROUP BY langchain_metadata->>'source'
        ORDER BY filename
        """
    )

    try:
        engine = get_pg_engine()

        async def fetch_documents():
            async with engine._pool.connect() as connection:
                result = await connection.execute(query)
                return result.mappings().all()

        import asyncio

        rows = asyncio.run(fetch_documents())

        return {
            "documents": [
                {
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
        

@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """Delete one uploaded document and its associated chunks."""
    filename = filename.strip()

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Filename cannot be empty.",
        )

    try:
        engine = get_pg_engine()

        async def document_exists():
            async with engine._pool.connect() as connection:
                result = await connection.execute(
                    text(
                        """
                        SELECT 1
                        FROM jarvis_data
                        WHERE langchain_metadata->>'source' = :filename
                        LIMIT 1
                        """
                    ),
                    {"filename": filename},
                )
                return result.scalar() is not None

        import asyncio

        if not asyncio.run(document_exists()):
            raise HTTPException(
                status_code=404,
                detail=f"Document '{filename}' was not found.",
            )

        store = get_vector_store()
        deleted = store.delete(
            filter={"source": filename}
        )

        if deleted is False:
            raise HTTPException(
                status_code=500,
                detail="The vector store could not confirm deletion.",
            )

        logger.info("Deleted document: %s", filename)

        return {
            "message": "Document deleted successfully.",
            "filename": filename,
        }

    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Document deletion failed")
        raise HTTPException(
            status_code=500,
            detail="Could not delete the document.",
        ) from error
