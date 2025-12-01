import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage
from rag import setup_rag_chain
from ingest import main as run_ingestion

st.set_page_config(
    page_title="MPNeuralNetwork Assistant", page_icon="🤖", layout="wide"
)

st.title("🤖 MPNeuralNetwork Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")
    st.markdown("Manage the knowledge base and configuration.")

    st.subheader("Parameters")
    k_param = st.slider(
        "Retrieval Count (k)",
        min_value=1,
        max_value=10,
        value=4,
        help="Number of documents to retrieve for context.",
    )

    temperature_param = st.slider(
        "Temperature (Creativity)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Controls the randomness of the output.",
    )

    if st.button(
        "🔄 Update Knowledge Base",
        help="Clone/Pull the repo and rebuild the vector database.",
    ):
        with st.status("Updating Knowledge Base...", expanded=True) as status:
            st.write("Starting ingestion process...")
            try:
                run_ingestion()
                st.write("Ingestion complete!")
                status.update(
                    label="Knowledge Base Updated", state="complete", expanded=False
                )

                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Error during update: {e}")
                status.update(label="Update Failed", state="error")


@st.cache_resource
def get_rag_chain(k, temperature):
    return setup_rag_chain(k=k, temperature=temperature)


try:
    rag_chain = get_rag_chain(k_param, temperature_param)
except Exception as e:
    rag_chain = None
    st.error(f"Failed to initialize RAG chain: {e}")

if not rag_chain:
    st.warning("The Vector Database is missing or empty.")
    st.info(
        "Please click 'Update Knowledge Base' in the sidebar to initialize the system."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How do I create a custom layer?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if rag_chain:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            chat_history = []
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history.append(AIMessage(content=msg["content"]))

            with st.spinner("Analyzing codebase..."):
                try:
                    response = rag_chain.invoke(
                        {"input": prompt, "chat_history": chat_history}
                    )
                    answer = response["answer"]

                    message_placeholder.markdown(answer)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                    if "context" in response and response["context"]:
                        with st.expander("📚 Sources Used"):
                            seen_sources = set()
                            for doc in response["context"]:
                                source = doc.metadata.get("source", "Unknown")
                                display_source = os.path.relpath(source, os.getcwd())
                                if display_source not in seen_sources:
                                    st.markdown(f"- `{display_source}`")
                                    seen_sources.add(display_source)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
    else:
        st.error("Agent is not ready. Please update the knowledge base.")
