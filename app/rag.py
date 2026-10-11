
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.config import CHAT_MODEL, OLLAMA_BASE_URL, RETRIEVAL_K
from app.database import get_vector_store


prompt = ChatPromptTemplate.from_template(
    """
You are Jarvis, a document-grounded question-answering assistant.

STRICT RULES:
1. Answer the question using only the supplied document context.
2. Every factual bullet or paragraph must include at least one source reference.
3. Copy source references exactly as they appear in the context: [S1], [S2], etc.
4. Put citations at the end of the sentence or bullet they support.
5. Use ONLY references listed under VALID SOURCE REFERENCES.
6. Never invent a source reference.
7. If the context does not support the answer, clearly say so.
8. Keep the answer clear, relevant, and concise.
9. Do not claim an exact quotation unless it matches the context.

CITATION EXAMPLE:
- DNS translates domain names into IP addresses. [S1]
- The documentation covers Docker networking and WebSockets. [S2]

Before returning your answer, check that every factual bullet or paragraph has a valid citation.
VALID SOURCE REFERENCES:
{valid_references}

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
)

llm = ChatOllama(
    model=CHAT_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
)

answer_chain = prompt | llm | StrOutputParser()


def answer_question(question: str) -> dict:
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    vector_store = get_vector_store()

    documents = vector_store.similarity_search(
        question,
        k=RETRIEVAL_K,
    )

    if not documents:
        return {
            "answer": "I couldn't find relevant uploaded document passages.",
            "sources": [],
            "citations": [],
        }

    context_sections = []
    citations = []
    seen_chunks = set()

    for document in documents:
        content = document.page_content.strip()
        source = document.metadata.get("source", "Unknown")
        page = document.metadata.get("page")

        if not content:
            continue

        chunk_key = (
            document.metadata.get("document_id"),
            source,
            page,
            content,
        )
        if chunk_key in seen_chunks:
            continue

        seen_chunks.add(chunk_key)
        reference = f"S{len(citations) + 1}"

        display_page = (
            page + 1
            if isinstance(page, int) and page >= 0
            else page
        )

        citations.append({
            "reference": reference,
            "document_id": document.metadata.get("document_id"),
            "filename": source,
            "page": display_page,
            "excerpt": content[:500],
        })

        page_label = (
            f"\nPage: {display_page}"
            if display_page is not None
            else ""
        )

        context_sections.append(
            f"[{reference}]\n"
            f"Source: {source}{page_label}\n"
            f"Content:\n{content}"
        )

    if not context_sections:
        return {
            "answer": "No readable document passages were found.",
            "sources": [],
            "citations": [],
        }

    valid_references = [
        citation["reference"] for citation in citations
    ]

    context = "\n\n---\n\n".join(context_sections)

    answer = answer_chain.invoke({
        "context": context,
        "question": question,
        "valid_references": ", ".join(
            f"[{reference}]" for reference in valid_references
        ),
    }).strip()

    # Normalize numeric-only references when they map to an
    # existing reference in the supplied context.
    def normalize_reference(match):
        number = match.group(1)

        if f"S{number}" in valid_references:
            return f"[S{number}]"

        return match.group(0)

    answer = re.sub(
        r"(?<![A-Za-z0-9])\[(\d+)\]",
        normalize_reference,
        answer,
    )

    # Keep only source cards referenced by the answer.
    used_references = set(
        re.findall(r"\[(S\d+)\]", answer)
    )

    citation_lookup = {
        citation["reference"]: citation
        for citation in citations
    }

    used_citations = [
        citation_lookup[reference]
        for reference in valid_references
        if reference in used_references
    ]

    if not used_citations:
        return {
            "answer": (
                "I couldn't produce an answer with verifiable "
                "source references. Please try rephrasing your "
                "question."
            ),
            "sources": [],
            "citations": [],
        }

    used_sources = list(dict.fromkeys(
        citation["filename"] for citation in used_citations
    ))

    return {
        "answer": answer,
        "sources": used_sources,
        "citations": used_citations,
    }
