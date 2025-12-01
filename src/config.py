import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.chat_models import ChatOpenAI, ChatOllama

load_dotenv()


def get_llm(temperature=0.7):
    llm_type = os.getenv("LLM_TYPE", "gemini").lower()
    llm_model = os.getenv("LLM_MODEL")

    if llm_type == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is missing in .env file")
        return ChatGoogleGenerativeAI(
            model=llm_model or "gemini-2.5-flash",
            google_api_key=api_key,
            temperature=temperature,
        )
    elif llm_type == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is missing in .env file")
        return ChatOpenAI(model=llm_model or "gpt-5-nano", temperature=temperature)
    elif llm_type == "ollama":
        return ChatOllama(model=llm_model or "llama3", temperature=temperature)
    else:
        raise ValueError(f"Unsupported or unconfigured LLM type: {llm_type}")


def get_embeddings():
    embeddings_type = os.getenv("EMBEDDINGS_TYPE", "local").lower()

    if embeddings_type == "local":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    elif embeddings_type == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is missing in .env file")
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=api_key
        )

    elif embeddings_type == "openai":
        from langchain_community.embeddings import OpenAIEmbeddings

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing in .env file")
        return OpenAIEmbeddings()

    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def get_repo_config():
    return {
        "url": os.getenv(
            "REPO_URL", "https://github.com/maximepires4/mp-neural-network"
        ),
        "branch": os.getenv("REPO_BRANCH", "main"),
    }


def get_persist_directory():
    return os.path.join(os.getcwd(), "data", "chroma_db")
