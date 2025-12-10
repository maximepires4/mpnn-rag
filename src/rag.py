import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.chains import (
    create_retrieval_chain,
    create_history_aware_retriever,
)

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings

from utils import get_llm
import config


def format_docs_with_metadata(docs):
    """
    Format documents for the LLM, including rich metadata context.
    """
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        context = doc.metadata.get("context", "")
        context_type = doc.metadata.get("context_type", "")
        start_line = doc.metadata.get("start_line", "?")
        end_line = doc.metadata.get("end_line", "?")

        # Create a header with file info and code context (Class/Function)
        header = f"File: {source}"
        if start_line != "?":
            header += f" (Lines {start_line}-{end_line})"
        if context:
            header += f"\nContext: {context_type.capitalize()} '{context}'"

        formatted.append(f"{header}\n---\n{doc.page_content}")

    return "\n\n".join(formatted)


def setup_rag_chain(k=4, temperature=0.7):
    persist_directory = config.CHROMA_DIR

    if not os.path.exists(persist_directory):
        print(
            "Error: Vector database does not exist, run 'python src/ingest.py' first."
        )
        return None

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    llm = get_llm(temperature=temperature)

    # Connect to the correct collection where children vectors are stored
    vector_db = Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

    # 1. Hybrid Search (Dense + Sparse/BM25)
    print("Initializing Hybrid Search...")

    initial_k = k * 3
    dense_retriever = vector_db.as_retriever(search_kwargs={"k": initial_k})

    collection_data = vector_db.get()

    if not collection_data["documents"]:
        print("Warning: Vector DB is empty. Using dense retriever only.")
        base_retriever = dense_retriever
    else:
        bm25_retriever = BM25Retriever.from_texts(
            texts=collection_data["documents"],
            metadatas=collection_data["metadatas"],
        )
        bm25_retriever.k = initial_k

        base_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, bm25_retriever], weights=[0.5, 0.5]
        )

    # 2. Re-ranking
    # Using a high-quality Cross-Encoder model
    compressor_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    document_compressor = CrossEncoderReranker(model=compressor_model, top_n=k)

    retriever = ContextualCompressionRetriever(
        base_compressor=document_compressor, base_retriever=base_retriever
    )

    # 3. History Handling
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 4. Answer Generation
    system_prompt = (
        "You are an expert assistant tasked with helping developers use the 'MPNeuralNetwork' library. "
        "Use the provided code snippets and documentation context to answer the user's question. "
        "The context includes file paths and line numbers; refer to them if helpful. "
        "If you don't know the answer, just say so. "
        "Be concise and prioritize code examples."
        "\n\n"
        "--- CONTEXT ---"
        "\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )

    # Custom LCEL Chain
    # Using RunnablePassthrough to format context manually using our custom function
    question_answer_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs_with_metadata(x["context"])
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

