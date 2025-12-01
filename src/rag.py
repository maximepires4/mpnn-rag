import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

# Handling imports for different LangChain versions/environments
try:
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from config import get_llm, get_embeddings, get_persist_directory


def setup_rag_chain():
    persist_directory = get_persist_directory()

    if not os.path.exists(persist_directory):
        print(
            "Error: Vector database does not exist, run 'python src/ingest.py' first."
        )
        return None

    embeddings = get_embeddings()
    llm = get_llm()

    vector_db = Chroma(
        persist_directory=persist_directory, embedding_function=embeddings
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})

    system_prompt = (
        "You are an expert assistant tasked with helping developers use the 'MPNeuralNetwork' library. "
        "Use the code snippets and documentation retrieved below to answer the user's question. "
        "If you don't know the answer, just say so. "
        "Be concise and prioritize code examples."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain
