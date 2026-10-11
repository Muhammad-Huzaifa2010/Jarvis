import asyncio
import re
from functools import lru_cache

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

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


@lru_cache(maxsize=1)
def get_sql_engine():
    return create_async_engine(DATABASE_URL)


async def _existing_vector_dimension() -> int | None:
    engine = create_async_engine(DATABASE_URL)
    try:
        async with engine.connect() as connection:
            result = await connection.execute(
                text(
                    """
                    SELECT
                        to_regclass(:qualified_table)::text AS table_name,
                        (
                            SELECT format_type(
                                attribute.atttypid,
                                attribute.atttypmod
                            )
                            FROM pg_attribute AS attribute
                            WHERE attribute.attrelid = to_regclass(:qualified_table)
                                AND attribute.attname = 'embedding'
                                AND NOT attribute.attisdropped
                        ) AS vector_type
                    """
                ),
                {"qualified_table": f"public.{TABLE_NAME}"},
            )
            row = result.mappings().one()
    finally:
        await engine.dispose()

    if row["table_name"] is None:
        return None

    vector_type = row["vector_type"]
    match = re.fullmatch(r"vector\((\d+)\)", vector_type or "")
    if match is None:
        raise RuntimeError(
            f"Table '{TABLE_NAME}' exists but its embedding column "
            f"has unexpected type {vector_type!r}."
        )

    return int(match.group(1))


def initialize_database() -> None:
    embeddings = get_embeddings()

    # Determine the embedding model's vector dimensions.
    dimension = len(
        embeddings.embed_query("Jarvis initialization test")
    )

    existing_dimension = asyncio.run(_existing_vector_dimension())
    if existing_dimension is not None:
        if existing_dimension != dimension:
            raise RuntimeError(
                f"Table '{TABLE_NAME}' uses {existing_dimension}-dimensional "
                f"embeddings, but model '{EMBEDDING_MODEL}' produces "
                f"{dimension}. Rebuild or migrate the vector table before "
                "using this embedding model."
            )
        print(
            f"Table '{TABLE_NAME}' already exists. "
            f"Keeping existing data. Embedding dimensions: "
            f"{existing_dimension}"
        )
        return

    # Create the table only when it does not exist.
    get_pg_engine().init_vectorstore_table(
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