
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(
            f"Missing required environment variable: {name}"
        )
    return value


def positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an integer."
        ) from exc

    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")

    return value


DATABASE_URL = required_env("DATABASE_URL")
OLLAMA_BASE_URL = required_env("OLLAMA_BASE_URL").rstrip("/")
CHAT_MODEL = required_env("CHAT_MODEL")
EMBEDDING_MODEL = required_env("EMBEDDING_MODEL")

CHUNK_SIZE = positive_int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVAL_K = positive_int("RETRIEVAL_K", 6)

if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
    raise ValueError(
        "CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE."
    )