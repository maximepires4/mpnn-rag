import os
import argparse
from termcolor import cprint
from langchain_core.messages import HumanMessage, AIMessage
from rag import setup_rag_chain
import config


def main():
    parser = argparse.ArgumentParser(description="MPNeuralNetwork RAG CLI Agent")
    parser.add_argument(
        "--k", type=int, default=4, help="Number of documents to retrieve (default: 4)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature/creativity (default: 0.7)",
    )
    parser.add_argument(
        "--reranker",
        type=str,
        default="large",
        help="Reranker model: 'large', 'small', or HuggingFace ID (default: large)",
    )
    args = parser.parse_args()

    # Resolve reranker shortcut
    reranker_model = config.AVAILABLE_RERANKERS.get(args.reranker, args.reranker)

    cprint("Initializing MPNeuralNetwork RAG Agent...", "cyan")
    cprint(f"Settings: k={args.k}, temp={args.temperature}, reranker={reranker_model}", "grey")

    try:
        rag_chain = setup_rag_chain(
            k=args.k, 
            temperature=args.temperature,
            reranker_model=reranker_model
        )
    except Exception as e:
        cprint(f"Initialization error: {e}", "red")
        import traceback

        traceback.print_exc()
        return

    if not rag_chain:
        return

    cprint(
        "\n🤖 Agent ready! Ask your questions about the library (type 'exit' to quit).",
        "green",
    )

    chat_history = []

    while True:
        try:
            query = input("\n👉 You: ")
            if query.lower() in ["exit", "quit", "q"]:
                cprint("Goodbye!", "yellow")
                break

            if not query.strip():
                continue

            cprint("Searching...", "grey")
            print("\n🤖 \033[1mAssistant:\033[0m")

            full_answer = ""
            sources = []

            # Streaming loop
            for chunk in rag_chain.stream({"input": query, "chat_history": chat_history}):
                if "answer" in chunk:
                    content = chunk["answer"]
                    print(content, end="", flush=True)
                    full_answer += content
                if "context" in chunk:
                    sources = chunk["context"]
            
            print() # Newline after streaming

            chat_history.append(HumanMessage(content=query))
            chat_history.append(AIMessage(content=full_answer))

            # Display detailed sources
            print("\n\033[90m--- Sources used ---\033[0m")
            if sources:
                # Deduplicate based on unique content location
                unique_citations = {}
                for doc in sources:
                    src = doc.metadata.get("source", "Unknown")
                    fname = os.path.basename(src)
                    start = doc.metadata.get("start_line", "?")
                    end = doc.metadata.get("end_line", "?")
                    ctx = doc.metadata.get("context", "")
                    
                    citation_key = f"{fname}:{start}-{end}"
                    if citation_key not in unique_citations:
                        details = f"\033[90m- {fname}"
                        if start != "?":
                            details += f" (L{start}-{end})"
                        if ctx:
                            details += f" [{ctx}]"
                        details += "\033[0m"
                        unique_citations[citation_key] = details
                        print(details)

        except KeyboardInterrupt:
            cprint("\nInterruption detected. Goodbye!", "yellow")
            break
        except Exception as e:
            cprint(f"\nError during generation: {e}", "red")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
