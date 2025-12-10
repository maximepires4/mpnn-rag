# MPNeuralNetwork RAG Agent

This advanced RAG (Retrieval-Augmented Generation) agent allows developers to chat with the [MPNeuralNetwork](https://github.com/maximepires4/mp-neural-network) deep learning library. It understands the codebase structure (classes, functions) and provides precise, context-aware answers with code examples.

## ✨ Features

* **Smart Code Understanding**:
  * **AST Enrichment**: Parses Python code to attach context (e.g., "This chunk is from `MPNetwork.train`").
  * **Context-Aware**: Knows exactly which file and lines a snippet comes from.
* **Hybrid Search Engine**:
  * **Ensemble Retrieval**: Combines Semantic Search (MPNet) + Keyword Search (BM25) for maximum accuracy.
  * **State-of-the-Art Reranking**: Uses **BGE-M3** (Large) or **MS-Marco** (Small) to filter results.
* **High Performance**:
  * **Streaming**: Real-time answer generation.
  * **Configurable**: Switch between Accuracy (Large Reranker) and Speed (Small Reranker) instantly.
* **Dual Interface**:
  * **Web App (Streamlit)**: Complete UI with sidebar settings.
  * **CLI**: Fast terminal tool for power users.

## Quick Start

### 1. Installation

```bash
# Clone the repo
git clone https://github.com/maximepires4/mpnn-rag.git
cd mpnn-rag

# Setup Virtual Env
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file:

```bash
cp .env.example .env
```

Add your API Key (Google Gemini is default and free tier friendly):

```env
GOOGLE_API_KEY="your_api_key_here"
```

### 3. Ingestion (Important!)

You must build the knowledge base first. This downloads the repo and indexes the code.

```bash
python src/ingest.py
```

### 4. Run the Agent

**Web Interface:**

```bash
streamlit run src/app.py
```

**Terminal Interface:**

```bash
python src/main.py --reranker large
```

## Configuration Options

| Parameter | CLI Flag | Web UI | Description |
| :--- | :--- | :--- | :--- |
| **Retrieval Count** | `--k 4` | Slider | Number of code snippets to fetch. |
| **Creativity** | `--temperature 0.7` | Slider | 0.0 (Factual) to 1.0 (Creative). |
| **Reranker** | `--reranker large` | Selectbox | **Large** (BGE-M3, SOTA) or **Small** (MS-Marco, Fast). |

## Architecture

* **Ingestion**: `Documents` -> `Split (1000 chars)` -> `AST Enrichment` -> `ChromaDB`
* **Retrieval**: `Query` -> `Hybrid Search (Dense + BM25)` -> `Top-3*k Candidates`
* **Refinement**: `Candidates` -> `Cross-Encoder Reranking` -> `Top-k Context`
* **Generation**: `Context + History` -> `LLM` -> `Streamed Answer`

## Author

**Maxime Pires** - *AI Engineer | CentraleSupelec*
[LinkedIn](https://www.linkedin.com/in/maximepires) | [GitHub](https://github.com/maximepires4)
