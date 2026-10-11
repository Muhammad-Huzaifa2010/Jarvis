from functools import lru_cache
import asyncio

from sqlalchemy import text

from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGEngine, PGVectorStore

from app.config import (
    DATABASE_URL,
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
)

TABLE_NAME = "jarvis_data"


@lru_cache(maxsize=1)
def get_embeddings():
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


@lru_cache(maxsize=1)
def get_pg_engine():
    return PGEngine.from_connection_string(
        url=DATABASE_URL
    )
async def _table_exists(engine) -> bool:
    async with engine._pool.connect() as connection:
        result = await connection.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = :table_name
                    AND table_schema = :schema_name
                )
                """
            ),
            {
                "table_name": TABLE_NAME,
                "schema_name": "public",
            },
        )
        return bool(result.scalar_one())



def initialize_database():
    embeddings = get_embeddings()

    # Determine the embedding model's vector dimensions.
    dimension = len(
        embeddings.embed_query("Jarvis initialization test")
    )

    engine = get_pg_engine()

    table_exists = asyncio.run(_table_exists(engine))
    if table_exists:
        # Do not recreate or overwrite an existing table.
        print(
            f"Table '{TABLE_NAME}' already exists. "
            f"Keeping existing data. "
            f"Embedding dimensions: {dimension}"
        )
        return

    # Create the table only when it does not exist.
    engine.init_vectorstore_table(
        table_name=TABLE_NAME,
        vector_size=dimension,
    )

    print(
        f"Jarvis database initialized. "
        f"Vector dimensions: {dimension}"
    )



@lru_cache(maxsize=1)
def get_vector_store():
    return PGVectorStore.create_sync(
        engine=get_pg_engine(),
        table_name=TABLE_NAME,
        embedding_service=get_embeddings(),
    )