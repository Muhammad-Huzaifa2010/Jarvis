
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.config import CHAT_MODEL, OLLAMA_BASE_URL, RETRIEVAL_K
from app.database import get_vector_store


prompt = ChatPromptTemplate.from_template(
    """
You are Jarvis, a efficient detailed question-answering assistant.

Answer the user's question using  the supplied document context.
Be clear and helpful. If the context does not contain the answer,
say that you could not find the answer in the uploaded documents.
Do not invent facts or citations.

Each context section has a source reference such as [S1].
When useful, refer to sources in your answer using that reference,
for example [S1].

Context:
{context}

Question:
{question}

Answer:
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
            "answer": "I couldn't find any relevant uploaded documents.",
            "sources": [],
            "citations": [],
        }

    context_sections = []
    citations = []
    sources = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown")
        reference = f"S{index}"

        # page is available for some loaders, but not all.
        page = document.metadata.get("page")

        citation = {
            "reference": reference,
            "filename": source,
            "page": page,
            "excerpt": document.page_content[:500],
        }
        citations.append(citation)

        if source not in sources:
            sources.append(source)

        context_sections.append(
            f"[{reference}]\n"
            f"Source: {source}\n"
            f"Content:\n{document.page_content}"
        )

    context = "\n\n---\n\n".join(context_sections)

    answer = answer_chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return {
        "answer": answer,
        "sources": sources,
        "citations": citations,
    }
