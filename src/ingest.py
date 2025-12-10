import os
import shutil
from git import Repo
from langchain_community.document_loaders import (
    DirectoryLoader,
    PythonLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from utils import get_repo_config
from enrichment import enrich_python_metadata
import config


def clone_repository(repo_url, repo_path, branch):
    if os.path.exists(repo_path):
        print(f"Directory {repo_path} already exists. Removing to start fresh...")
        shutil.rmtree(repo_path)

    print(f"Cloning {repo_url} (branch: {branch})...")
    Repo.clone_from(repo_url, repo_path, branch=branch)
    print("Cloning finished.")


def load_documents(repo_path):
    documents = []

    print("Loading Python files...")
    py_loader = DirectoryLoader(repo_path, glob="**/*.py", loader_cls=PythonLoader)
    loaded_py = py_loader.load()

    print("Loading Markdown files...")
    md_loader = DirectoryLoader(
        repo_path, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
    )
    loaded_md = md_loader.load()

    all_docs = loaded_py + loaded_md

    documents = []
    for doc in all_docs:
        source = doc.metadata.get("source", "")
        if any(x in source for x in ["/tests/", "/benchmark/"]):
            continue
        documents.append(doc)

    print(f"{len(documents)} relevant documents loaded (out of {len(all_docs)} found).")
    return documents


def ingest_documents(documents):
    if os.path.exists(config.CHROMA_DIR):
        shutil.rmtree(config.CHROMA_DIR)

    print("Processing documents and creating Vector DB...")

    embedding = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

    chunks = []

    python_docs = [d for d in documents if d.metadata.get("source", "").endswith(".py")]
    md_docs = [d for d in documents if d.metadata.get("source", "").endswith(".md")]
    other_docs = [d for d in documents if d not in python_docs and d not in md_docs]

    if python_docs:
        py_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )

        for doc in python_docs:
            doc_chunks = py_splitter.split_documents([doc])
            doc_chunks = enrich_python_metadata(doc, doc_chunks)
            chunks.extend(doc_chunks)

    if md_docs:
        md_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        chunks.extend(md_splitter.split_documents(md_docs))

    if other_docs:
        generic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        chunks.extend(generic_splitter.split_documents(other_docs))

    print(f"Split documents into {len(chunks)} enriched chunks.")

    print("Creating vector database...")
    Chroma.from_documents(
        collection_name=config.CHROMA_COLLECTION_NAME,
        documents=chunks,
        embedding=embedding,
        persist_directory=config.CHROMA_DIR,
    )
    print(f"Database successfully created in {config.CHROMA_DIR}")


def main():
    repo_config = get_repo_config()
    repo_path = config.REPO_DIR

    clone_repository(repo_config["url"], repo_path, repo_config["branch"])
    documents = load_documents(repo_path)

    ingest_documents(documents)


if __name__ == "__main__":
    main()
