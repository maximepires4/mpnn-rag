import os
from termcolor import cprint
from rag import setup_rag_chain


def main():
    cprint("Initializing MPNeuralNetwork RAG Agent...", "cyan")
    try:
        rag_chain = setup_rag_chain()
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

    while True:
        try:
            query = input("\n👉 You: ")
            if query.lower() in ["exit", "quit", "q"]:
                cprint("Goodbye!", "yellow")
                break

            if not query.strip():
                continue

            cprint("Searching...", "grey")

            response = rag_chain.invoke({"input": query})

            print("\n🤖 \033[1mAssistant:\033[0m")
            print(response["answer"])

            # Display sources
            print("\n\033[90m--- Sources used ---\033[0m")
            if "context" in response:
                seen_sources = set()
                for doc in response["context"]:
                    source = doc.metadata.get("source", "Unknown")
                    if source not in seen_sources:
                        print(f"\033[90m- {os.path.basename(source)}\033[0m")
                        seen_sources.add(source)

        except KeyboardInterrupt:
            cprint("\nInterruption detected. Goodbye!", "yellow")
            break
        except Exception as e:
            cprint(f"\nError during generation: {e}", "red")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()

