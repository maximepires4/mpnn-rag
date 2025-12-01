import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever

from config import get_llm, get_embeddings, get_persist_directory


def setup_rag_chain(k=4, temperature=0.7):
    persist_directory = get_persist_directory()

    if not os.path.exists(persist_directory):
        print(
            "Error: Vector database does not exist, run 'python src/ingest.py' first."
        )
        return None

    embeddings = get_embeddings()
    llm = get_llm(temperature=temperature)

    vector_db = Chroma(
        persist_directory=persist_directory, embedding_function=embeddings
    )

    initial_k = k * 3
    dense_retriever = vector_db.as_retriever(search_kwargs={"k": initial_k})

    try:
        print("Initializing BM25 Retriever for Hybrid Search...")
        collection_data = vector_db.get()
        if not collection_data["documents"]:
            print("Warning: Vector DB is empty. Skipping Hybrid Search.")
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
    except Exception as e:
        print(f"Error initializing Hybrid Search (falling back to dense only): {e}")
        base_retriever = dense_retriever

    compressor_model = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    document_compressor = CrossEncoderReranker(model=compressor_model, top_n=k)

    retriever = ContextualCompressionRetriever(
        base_compressor=document_compressor, base_retriever=base_retriever
    )

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

    system_prompt = (
        "You are an expert assistant tasked with helping developers use the 'MPNeuralNetwork' library. "
        "Use the code snippets and documentation retrieved below to answer the user's question. "
        "If you don't know the answer, just say so. "
        "Be concise and prioritize code examples."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain
