import streamlit as st
import os
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
                    label="Knowledge Base Updated!", state="complete", expanded=False
                )

                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Error during update: {e}")
                status.update(label="Update Failed", state="error")


@st.cache_resource
def get_rag_chain():
    return setup_rag_chain()


try:
    rag_chain = get_rag_chain()
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

            with st.spinner("Analyzing codebase..."):
                try:
                    response = rag_chain.invoke({"input": prompt})
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
