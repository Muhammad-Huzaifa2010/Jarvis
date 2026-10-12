
import logging
import re

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
You are Jarvis, a document-grounded question-answering assistant.

STRICT RULES:
1. Answer using only the supplied document context.
2. Every factual paragraph or bullet must include a source reference.
3. Use only references listed under VALID SOURCE REFERENCES.
4. Never invent a source reference.
5. If the context does not support the answer, say so clearly.
6. Keep the answer clear, relevant, and concise.
7. Do not claim an exact quotation unless it matches the context.
8. Start directly with the answer.
9. Never describe your reasoning or citation process.
10. Never say "Here is the revised answer".
11. Never add a conclusion about following rules or adding citations.
12. Use natural wording, correct spelling, and grammar.
13. Do not invent facts about the project or its documents.

CITATION EXAMPLES:
- DNS translates domain names into IP addresses. [S1]
- The documentation covers Docker networking. [S2]
- LangChain connects the model and retrieval components. [S1] [S2]

Before returning your answer, verify that every factual paragraph
or bullet has a valid citation.

VALID SOURCE REFERENCES:
{valid_references}

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

Return only the final answer.
"""
)


# ---------------------------------------------------------
# Citation repair prompt
# ---------------------------------------------------------

citation_repair_prompt = ChatPromptTemplate.from_template(
    """
You are correcting a document-grounded answer.

QUESTION:
{question}

VALID SOURCE REFERENCES:
{valid_references}

DOCUMENT CONTEXT:
{context}

EXISTING ANSWER:
{answer}

RULES:
1. Use only facts supported by the supplied context.
2. Add valid citations to factual claims.
3. Use only the exact references listed above.
4. Never invent facts or references.
5. Remove unsupported claims rather than guessing.
6. Preserve supported information from the existing answer.
7. Return only the final answer intended for the user.
8. Do not include an introduction about revising or correcting.
9. Do not explain your citation process.
10. Do not add horizontal separators or closing commentary.
11. Use natural wording and correct spelling.
12. If the context does not answer the question, say so briefly.

Return only the corrected answer.
"""
)


# ---------------------------------------------------------
# Local Ollama model and chains
# ---------------------------------------------------------

llm = ChatOllama(
    model=CHAT_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
)

answer_chain = prompt | llm | StrOutputParser()

citation_repair_chain = (
    citation_repair_prompt | llm | StrOutputParser()
)


# ---------------------------------------------------------
# Citation normalization
# ---------------------------------------------------------

def normalize_answer_references(
    text: str,
    valid_references: list[str],
) -> str:
    """Normalize citation formats without inventing references."""

    # Convert (S1) and (S1 and S3) to [S1] and [S3].
    def normalize_parenthetical(match):
        references = re.findall(
            r"S\d+",
            match.group(1),
        )

        if references and all(
            reference in valid_references
            for reference in references
        ):
            return " ".join(
                f"[{reference}]"
                for reference in references
            )

        return match.group(0)

    text = re.sub(
        r"\((S\d+(?:\s+and\s+S\d+)*)\)",
        normalize_parenthetical,
        text,
    )

    # Convert [1] to [S1] only when S1 is a valid reference.
    def normalize_numeric(match):
        reference = f"S{match.group(1)}"

        if reference in valid_references:
            return f"[{reference}]"

        return match.group(0)

    text = re.sub(
        r"(?<![A-Za-z0-9])\[(\d+)\]",
        normalize_numeric,
        text,
    )

    return text


# ---------------------------------------------------------
# Answer formatting
# ---------------------------------------------------------

def clean_answer_format(text: str) -> str:
    """Remove common model-generated preambles and commentary."""

    text = text.strip()

    # Remove common preambles.
    text = re.sub(
        r"(?im)^\s*(?:here is|here's) "
        r"(?:the )?(?:revised|corrected|final) answer"
        r"(?:\s+with citations added[^:\n]*)?:?\s*",
        "",
        text,
    )

    # Remove standalone Markdown horizontal separators.
    text = re.sub(
        r"(?m)^\s*(?:-{3,}|\*{3,}|_{3,})\s*$",
        "",
        text,
    )

    # Remove common model self-evaluation at the end.
    text = re.sub(
        r"(?is)\n*\s*(?:this answer|this response) "
        r"(?:adheres to|follows) the rules\b.*$",
        "",
        text,
    )

    return text.strip()


# ---------------------------------------------------------
# Main RAG function
# ---------------------------------------------------------

def answer_question(question: str) -> dict:
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    # Retrieve relevant document chunks from pgvector.
    vector_store = get_vector_store()

    documents = vector_store.similarity_search(
        question,
        k=RETRIEVAL_K,
    )

    if not documents:
        return {
            "answer": (
                "I couldn't find relevant passages in the "
                "available documents."
            ),
            "sources": [],
            "citations": [],
        }

    context_sections = []
    citations = []
    seen_chunks = set()

    # Assign references to unique retrieved chunks.
    for document in documents:
        content = document.page_content.strip()

        if not content:
            continue

        source = document.metadata.get(
            "source",
            "Unknown",
        )

        page = document.metadata.get("page")

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
            "document_id": document.metadata.get(
                "document_id"
            ),
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
        citation["reference"]
        for citation in citations
    ]

    formatted_references = ", ".join(
        f"[{reference}]"
        for reference in valid_references
    )

    context = "\n\n---\n\n".join(context_sections)

    # Generate the initial answer.
    answer = answer_chain.invoke({
        "context": context,
        "question": question,
        "valid_references": formatted_references,
    }).strip()

    answer = normalize_answer_references(
        answer,
        valid_references,
    )
    answer = clean_answer_format(answer)

    # Repair the answer if it has no recognized citation.
    if not re.search(r"\[(S\d+)\]", answer):
        try:
            repaired_answer = citation_repair_chain.invoke({
                "question": question,
                "context": context,
                "answer": answer,
                "valid_references": formatted_references,
            }).strip()

            if repaired_answer:
                answer = normalize_answer_references(
                    repaired_answer,
                    valid_references,
                )
                answer = clean_answer_format(answer)

        except Exception:
            # Keep the initial answer; validation below decides
            # whether it has usable citations.
            logger.exception("Citation repair failed")

    # Validate citations against retrieved chunks.
    used_references = set(
        re.findall(r"\[(S\d+)\]", answer)
    )

    used_references.intersection_update(
        valid_references
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

    # Do not return an uncited factual answer.
    if not used_citations:
        return {
            "answer": (
                "I couldn't produce an answer with verifiable "
                "source references. Please try rephrasing "
                "your question."
            ),
            "sources": [],
            "citations": [],
        }

    # Include only source files referenced in the answer.
    used_sources = list(dict.fromkeys(
        citation["filename"]
        for citation in used_citations
    ))

    return {
        "answer": answer,
        "sources": used_sources,
        "citations": used_citations,
    }