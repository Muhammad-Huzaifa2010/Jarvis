
import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.config import CHAT_MODEL, OLLAMA_BASE_URL, RETRIEVAL_K
from app.database import get_vector_store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Main answer prompt
# ---------------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are Jarvis, a helpful AI assistant.

Answer the user's question using the provided document context
when it is relevant.

Instructions:
- Give a clear, natural, useful answer.
- Do not include citations or source labels such as [S1] or [S2].
- If the context does not contain enough information, say so.
- Do not invent facts about the uploaded documents.
- If the question is unrelated to the documents, answer normally
  when you can do so reliably.
- Return only the answer intended for the user.

Document context:
{context}

Question:
{question}

Answer:
"""
)


# ---------------------------------------------------------
# Local Ollama model and answer chain
# ---------------------------------------------------------

llm = ChatOllama(
    model=CHAT_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
)

answer_chain = prompt | llm | StrOutputParser()


# ---------------------------------------------------------
# Main RAG function
# ---------------------------------------------------------


def answer_question(question: str) -> dict:
    """Retrieve relevant documents and answer without citations."""

    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    try:
        vector_store = get_vector_store()

        docs = vector_store.similarity_search(
            question.strip(),
            k=RETRIEVAL_K,
        )

        if not docs:
            return {
                "answer": (
                    "I couldn't find relevant information "
                    "in your uploaded documents."
                ),
                "sources": [],
            }

        context = "\n\n".join(
            doc.page_content
            for doc in docs
            if doc.page_content and doc.page_content.strip()
        )

        if not context.strip():
            return {
                "answer": (
                    "I couldn't find useful text in the "
                    "retrieved documents."
                ),
                "sources": [],
            }

        answer = answer_chain.invoke({
            "context": context,
            "question": question.strip(),
        }).strip()

        sources = list(dict.fromkeys(
            doc.metadata.get("source", "Unknown document")
            for doc in docs
        ))

        return {
            "answer": answer,
            "sources": sources,
        }

    except ValueError:
        raise
    except Exception:
        logger.exception("Failed to answer question")
        raise
