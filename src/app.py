import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage
from rag import setup_rag_chain
from ingest import main as run_ingestion
import config

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
    
    reranker_option = st.selectbox(
        "Reranker Model",
        options=list(config.AVAILABLE_RERANKERS.keys()),
        index=0, # Default to large
        format_func=lambda x: f"{x.capitalize()} ({config.AVAILABLE_RERANKERS[x].split('/')[-1]})",
        help="Choose 'Large' for better accuracy or 'Small' for speed."
    )
    reranker_param = config.AVAILABLE_RERANKERS[reranker_option]

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
def get_rag_chain(k, temperature, reranker):
    return setup_rag_chain(k=k, temperature=temperature, reranker_model=reranker)


try:
    rag_chain = get_rag_chain(k_param, temperature_param, reranker_param)
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
                    # Stream the response
                    full_response = ""
                    context_docs = []
                    
                    stream = rag_chain.stream(
                        {"input": prompt, "chat_history": chat_history}
                    )
                    
                    for chunk in stream:
                        if "answer" in chunk:
                            content = chunk["answer"]
                            full_response += content
                            message_placeholder.markdown(full_response + "▌")
                        if "context" in chunk:
                            context_docs = chunk["context"]
                    
                    message_placeholder.markdown(full_response)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )

                    if context_docs:
                        with st.expander("📚 Sources Used"):
                            unique_citations = {}
                            for doc in context_docs:
                                src = doc.metadata.get("source", "Unknown")
                                fname = os.path.basename(src)
                                start = doc.metadata.get("start_line", "?")
                                end = doc.metadata.get("end_line", "?")
                                ctx = doc.metadata.get("context", "")
                                ctx_type = doc.metadata.get("context_type", "")

                                citation_key = f"{fname}:{start}-{end}"
                                if citation_key not in unique_citations:
                                    # Format: file.py (L10-20) [Function 'train']
                                    details = f"**{fname}**"
                                    if start != "?":
                                        details += f" (L{start}-{end})"
                                    if ctx:
                                        details += f" `[{ctx}]`"
                                    
                                    st.markdown(f"- {details}")
                                    unique_citations[citation_key] = True

                except Exception as e:
                    st.error(f"An error occurred: {e}")
    else:
        st.error("Agent is not ready. Please update the knowledge base.")
