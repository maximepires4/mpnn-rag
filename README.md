# MPNN RAG Agent

This project implements a Retrieval-Augmented Generation (RAG) agent designed to answer questions about the [MPNeuralNetwork](https://github.com/maximepires4/mp-neural-network) library, my Deep Learning framework built from scratch. It leverages various LLMs (Gemini, OpenAI, Ollama) and embedding models to create a vector database from the library's codebase and documentation, providing relevant information and code examples to developers.

## Features

* **Repository Ingestion**: Clones a specified Git repository, loads Python source files and Markdown documentation, and filters out test/benchmark files.
* **Vector Database Creation**: Splits documents into manageable chunks and stores them in a Chroma vector database using configurable embedding models.
* **Configurable LLMs**: Supports Google Gemini, OpenAI, and local Ollama models for generating answers.
* **Streamlit Web Interface**: An interactive and user-friendly web application for asking questions and viewing sources.
* **Command Line Interface (CLI)**: A simple terminal-based interface for interacting with the RAG agent.
* **Modular Design**: Clean separation of concerns with dedicated modules for configuration, data ingestion, RAG chain setup, and user interfaces.

## Setup

Follow these steps to get the project up and running locally.

### Prerequisites

* Python 3.9+
* Git

### 1. Clone the repository

```bash
git clone https://github.com/maximepires4/mpnn-rag.git
cd mpnn-rag
```

### 2. Create and activate a virtual environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Environment Variables

Create a `.env` file in the project root directory based on `.env.example` and fill in the necessary API keys and configurations.

```
# .env file example

# Google API Key
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"

# OpenAI API Key
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

# LLM Configuration ("gemini", "openai", "ollama")
LLM_TYPE="gemini" 

# Ollama Model Name (if LLM_TYPE is "ollama")
OLLAMA_MODEL="llama3" 

# Embeddings Configuration ("local", "gemini", "openai")
# "local" uses 'all-MiniLM-L6-v2' via HuggingFaceEmbeddings
EMBEDDINGS_TYPE="local"

# Repository to ingest
REPO_URL="https://github.com/maximepires4/mp-neural-network"
REPO_BRANCH="main"
```

## Usage

### 1. Data Ingestion (Build the Knowledge Base)

Before you can query the RAG agent, you need to ingest the target repository and build the vector database.

```bash
python src/ingest.py
```

This script will clone the `REPO_URL` specified in your `.env` file, process the code and documentation, and create a vector database in `data/chroma_db/`.

### 2. Run the Streamlit Web Application

For an interactive chat interface:

```bash
streamlit run src/app.py
```

This will open the application in your web browser, usually at `http://localhost:8501`. You can ask questions about the `MPNeuralNetwork` library and see the generated answers along with the sources.

### 3. Run the Command Line Interface (CLI) Application

For a text-based interactive experience:

```bash
python src/main.py
```

You can type your questions in the terminal, and the agent will respond. Type `exit` or `quit` to end the session.

## Project Structure

* `src/`: Contains the main application source code.
  * `config.py`: Centralized configuration for LLMs, embeddings, and repository settings.
  * `ingest.py`: Script to clone the repository, load documents, split them, and build the Chroma vector database.
  * `rag.py`: Defines the RAG chain setup, including prompt engineering and retriever configuration.
  * `main.py`: The command-line interface (CLI) for the RAG agent.
  * `app.py`: The Streamlit web application interface.
* `data/`: Directory for storing the cloned repository and the Chroma vector database.
  * `repo/`: Cloned `MPNeuralNetwork` repository.
  * `chroma_db/`: Persistent storage for the Chroma vector database.
* `.env`: Your environment variables (API keys, configuration).
* `.env.example`: Example file for `.env`.
* `requirements.txt`: List of Python dependencies.

## **Author**

**Maxime Pires** - *AI Engineer | CentraleSupelec*

[LinkedIn](https://www.linkedin.com/in/maximepires) | [Portfolio](https://github.com/maximepires4)
