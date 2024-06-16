import os
import tempfile
import streamlit as st
from streamlit_chat import message
from utils import set_nvidia_api_key, load_documents, split_documents, create_vector_store, create_qa_chain

# Set the page configuration
st.set_page_config(page_title="Conversational Retrieval")


def display_messages():
    """
    Displays chat messages in the Streamlit app.
    """
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))

    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    """
    Processes user input and updates the chat messages in the Streamlit app.
    """
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()

        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            try:
                result = st.session_state["qa_chain"].invoke(
                    {"question": user_text})
                agent_text = result.get("answer")
            except Exception as e:
                agent_text = f"An error occurred: {e}"

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def main():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    st.title("Chit Chat with Tribalscale AI")

    # API Key input
    nvapi_key = st.text_input("Enter your NVIDIA API key", type="password")
    if nvapi_key:
        valid_key = set_nvidia_api_key(nvapi_key)
        if valid_key:
            st.success("NVIDIA API key is valid.")
        else:
            st.error("Invalid NVIDIA API key. Please check your key and try again.")
            return

    # URLs input
    urls = st.text_area("Enter URLs to load documents from, separated by commas",
                        "https://www.tribalscale.com/, https://www.tribalscale.com/about, https://www.tribalscale.com/careers")
    urls = [url.strip() for url in urls.split(",")]

    if st.button("Load and Process Documents"):
        with st.spinner("Loading documents..."):
            documents = load_documents(urls)
            st.success(f"Loaded {len(documents)} documents.")

        with st.spinner("Splitting documents..."):
            docs = split_documents(documents)
            st.success(f"Split into {len(docs)} chunks.")

        with st.spinner("Creating vector store..."):
            vector_store = create_vector_store(docs)
            st.success("Vector store created.")

        with st.spinner("Creating QA chain..."):
            qa_chain = create_qa_chain(vector_store)
            st.success("QA chain created.")

        st.session_state['qa_chain'] = qa_chain

    if 'qa_chain' in st.session_state:
        display_messages()
        st.text_input("Message", key="user_input", on_change=process_input)


if __name__ == "__main__":
    main()
