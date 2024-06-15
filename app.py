import os
import tempfile
import streamlit as st
from streamlit_chat import message
from rag_app import ChatPDF

# adds a title for the web page
st.set_page_config(page_title="Resume Chatbot")


def display_messages():
    """
    Displays chat messages in the Streamlit app.
    This function assumes that chat messages are stored in the Streamlit session state
    under the key "messages" as a list of tuples, where each tuple contains the message
    content and a boolean indicating whether it's a user message or not.
    """
    # Display a subheader for the chat.
    st.subheader("Chat")

    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))

    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    """
    Processes user input and updates the chat messages in the Streamlit app.
    """
    # Check if there is user input and it is not empty.
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()

        # Display a thinking spinner while the assistant processes the input.
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def read_and_save_file():
    """
    Reads and saves the uploaded file, performs ingestion, and clears the assistant state.
    This function assumes that the question-answering assistant is stored in the Streamlit
    session state under the key "assistant," and file-related information is stored under
    the key "file_uploader."
    """
    st.session_state["assistant"].clear()

    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        # Display a spinner while ingesting the file.
        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["assistant"].ingest(file_path)
        os.remove(file_path)


def page():
    """
    Defines the content of the Streamlit app page for ChatPDF.

    """
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatPDF()

    st.header("ChatPDF")

    st.subheader("Upload a PDF file")
    st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )
    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()

    st.text_input("Message", key="user_input", on_change=process_input)


if __name__ == "__main__":
    page()
