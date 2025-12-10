DATA_DIR = "data"
REPO_DIR = f"{DATA_DIR}/repo"
CHROMA_DIR = f"{DATA_DIR}/chroma_db"
CHROMA_COLLECTION_NAME = "split_children"

AVAILABLE_RERANKERS = {
    "large": "BAAI/bge-reranker-v2-m3",
    "small": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}
RERANKER_MODEL_NAME = AVAILABLE_RERANKERS["large"]
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
