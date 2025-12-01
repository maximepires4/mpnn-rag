import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever

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

    dense_retriever = vector_db.as_retriever(search_kwargs={"k": 10})

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
            bm25_retriever.k = 10

            base_retriever = EnsembleRetriever(
                retrievers=[dense_retriever, bm25_retriever], weights=[0.5, 0.5]
            )
    except Exception as e:
        print(f"Error initializing Hybrid Search (falling back to dense only): {e}")
        base_retriever = dense_retriever

    compressor_model = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    document_compressor = CrossEncoderReranker(model=compressor_model, top_n=4)

    retriever = ContextualCompressionRetriever(
        base_compressor=document_compressor, base_retriever=base_retriever
    )

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
