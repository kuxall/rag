"""
TechInnovate AI Assistant

This script implements an advanced chatbot for TechInnovate AI, a fictional AI company.
It uses Langchain for the RAG (Retrieval Augmented Generation) process, OpenAI's GPT model
for language generation, and Streamlit for the user interface.

The system includes:
1. Synthetic data generation for the knowledge base
2. Vector storage using FAISS for efficient similarity search
3. A RAG pipeline for context-aware responses
4. A Streamlit-based user interface

Requirements:
- Python 3.7+
- Libraries: streamlit, langchain, langchain-community, langchain-openai, faiss-cpu, tiktoken

Usage:
1. Set your OpenAI API key in the .env file
2. Run the script using: streamlit run script_name.py

Author: AI Assistant
Date: June 28, 2024
"""

import os
import streamlit as st
from typing import List, Dict
from dotenv import load_dotenv

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# Load environment variables from the .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# Ensure the OpenAI API key is set
if openai_api_key is None:
    raise EnvironmentError(
        "Please set the OPENAI_API_KEY environment variable in the .env file.")

# Constants
COMPANY_NAME = "GOOGLE"
MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0.7
MAX_TOKENS = 150


def generate_synthetic_data() -> List[str]:
    """
    Generate synthetic data about the company using OpenAI's GPT model.

    Returns:
        List[str]: A list of sentences describing the company.
    """
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE,
                     api_key=openai_api_key)

    prompt = PromptTemplate(
        input_variables=[],
        template=f"""Generate 20 detailed and unique sentences about a  company called {COMPANY_NAME}.
    Include the following information:
    - The company’s mission and vision
    - A brief history and founding story
    - Key products and services offered
    - Unique selling points and technological innovations
    - Pricing models and tiers for their products/services
    - Customer support structure and policies
    - Major clients and partnerships
    - Company’s achievements and awards
    - Future goals and upcoming projects
    - Community involvement and social responsibility initiatives
    - Detailed information on the latest large language models 

    Each sentence should be informative, concise, and cover diverse aspects of the company."""
    )
    response = llm.invoke(prompt.format())
    return [sentence.strip() for sentence in response.content.split('\n') if sentence.strip()]


def initialize_knowledge_base(data: List[str]) -> FAISS:
    """
    Initialize the knowledge base with the given data.

    Args:
        data (List[str]): List of text data to be added to the knowledge base.

    Returns:
        FAISS: Initialized FAISS vector store.
    """
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = [Document(page_content=t) for t in data]
    splits = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = FAISS.from_documents(splits, embeddings)

    return vectorstore


def setup_rag_chain(vectorstore: FAISS) -> ConversationalRetrievalChain:
    """
    Set up the RAG chain for question answering.

    Args:
        vectorstore (FAISS): The vector store containing the knowledge base.

    Returns:
        ConversationalRetrievalChain: The configured RAG chain.
    """
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE,
                     api_key=openai_api_key)
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return chain


def get_response(chain: ConversationalRetrievalChain, query: str) -> str:
    """
    Get a response from the RAG chain for the given query.

    Args:
        chain (ConversationalRetrievalChain): The RAG chain.
        query (str): The user's question.

    Returns:
        str: The generated response.
    """
    response = chain.invoke({"question": query})
    return response['answer']


def update_knowledge_base(vectorstore: FAISS, new_info: str):
    """
    Update the knowledge base with new information.

    Args:
        vectorstore (FAISS): The current vector store.
        new_info (str): New information to be added.
    """
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(new_info)
    docs = [Document(page_content=s) for s in splits]
    vectorstore.add_documents(docs)


# Streamlit UI
st.title(f"{COMPANY_NAME} Assistant")

# Initialize session state
if 'vectorstore' not in st.session_state:
    data = generate_synthetic_data()

    st.session_state.vectorstore = initialize_knowledge_base(data)
    st.session_state.chain = setup_rag_chain(st.session_state.vectorstore)
    st.session_state.chat_history = []

# Chat interface
user_input = st.text_input("You:", key="user_input")

if st.button("Send"):
    response = get_response(st.session_state.chain, user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(
        (f"{COMPANY_NAME} Assistant", response))

# Display chat history
st.subheader("Conversation History")
for role, content in st.session_state.chat_history:
    st.write(f"**{role}:** {content}")

# Sidebar for knowledge base management
st.sidebar.subheader("Knowledge Base Management")

new_info = st.sidebar.text_area("Add new information:")
if st.sidebar.button("Add to Knowledge Base"):
    if new_info:
        update_knowledge_base(st.session_state.vectorstore, new_info)
        st.sidebar.success("Information added successfully!")
        st.experimental_rerun()

if st.sidebar.button("Reset Conversation"):
    st.session_state.chat_history = []
    st.experimental_rerun()

# Display current knowledge base
st.sidebar.subheader("Current Knowledge Base")
docs = st.session_state.vectorstore.similarity_search("", k=5)
for i, doc in enumerate(docs):
    st.sidebar.text(f"{i+1}. {doc.page_content[:50]}...")
