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
from config import get_embeddings, get_repo_config, get_persist_directory


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
        if any(x in source for x in ["/tests/", "/benchmark/", "/test_", "_test.py"]):
            continue
        documents.append(doc)

    print(f"{len(documents)} relevant documents loaded (out of {len(all_docs)} found).")
    return documents


def split_documents(documents):
    python_docs = []
    markdown_docs = []
    other_docs = []

    for doc in documents:
        source = doc.metadata.get("source", "").lower()
        if source.endswith(".py"):
            python_docs.append(doc)
        elif source.endswith(".md"):
            markdown_docs.append(doc)
        else:
            other_docs.append(doc)

    chunks = []

    if python_docs:
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=1000, chunk_overlap=200
        )
        chunks.extend(python_splitter.split_documents(python_docs))

    if markdown_docs:
        markdown_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN, chunk_size=1000, chunk_overlap=200
        )
        chunks.extend(markdown_splitter.split_documents(markdown_docs))

    if other_docs:
        generic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks.extend(generic_splitter.split_documents(other_docs))

    print(
        f"Split into {len(chunks)} chunks (Python: {len(python_docs)} docs, Markdown: {len(markdown_docs)} docs)."
    )
    return chunks


def create_vector_db(chunks):
    persist_directory = get_persist_directory()

    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    print("Creating vector database ...")
    embedding_function = get_embeddings()

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=persist_directory,
    )
    print(f"Database successfully created in {persist_directory}")


def main():
    repo_config = get_repo_config()
    repo_path = os.path.join(os.getcwd(), "data", "repo")

    clone_repository(repo_config["url"], repo_path, repo_config["branch"])
    documents = load_documents(repo_path)
    if not documents:
        print("No documents found!")
        return

    chunks = split_documents(documents)
    create_vector_db(chunks)


if __name__ == "__main__":
    main()
