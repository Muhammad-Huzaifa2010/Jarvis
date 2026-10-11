import os
import tempfile
from pathlib import Path

from langchain_community.document_loaders import (
PyPDFLoader,
TextLoader,
Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.database import get_vector_store

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

def ingest_document(filename: str, content: bytes, document_id: str) -> dict:
    extension = Path(filename).suffix.lower()


    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Only DOCX, PDF, TXT, and MD files are supported."
        )

    if not content:
        raise ValueError("The uploaded file is empty.")

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        if extension == ".pdf":
            loader = PyPDFLoader(temp_path)
        elif extension == ".docx":
            loader = Docx2txtLoader(temp_path)
        else:
            loader = TextLoader(
                temp_path,
                autodetect_encoding=True,
            )

        documents = loader.load()

        if not documents:
            raise ValueError(
                "No readable text was found in this file."
            )

        for document in documents:
            document.metadata["source"] = filename
            document.metadata["document_id"] = document_id

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError(
                "The document did not produce any text chunks."
            )

        get_vector_store().add_documents(chunks)

        return {
            "document_id": document_id,
            "filename": filename,
            "chunks_added": len(chunks),
            "status": "success",
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
