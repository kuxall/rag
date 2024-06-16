import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain_nvidia_ai_endpoints import ChatNVIDIA


def set_nvidia_api_key(api_key):
    if api_key.startswith("nvapi-"):
        os.environ["NVIDIA_API_KEY"] = api_key
        return True
    else:
        return False


def load_documents(urls):
    loader = WebBaseLoader(urls)
    documents = loader.load()
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs


def create_vector_store(docs):
    embeddings = NVIDIAEmbeddings()
    vector_store = FAISS.from_documents(documents=docs, embedding=embeddings)
    return vector_store


def create_qa_chain(vector_store):
    llm = ChatNVIDIA(model="meta/llama2-70b")
    chat = ChatNVIDIA(model="meta/llama2-70b",
                      temperature=0.1, max_tokens=1000, top_p=1.0)
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_chain(chat, chain_type="stuff", prompt=QA_PROMPT)

    qa_chain = ConversationalRetrievalChain(
        retriever=vector_store.as_retriever(),
        combine_docs_chain=doc_chain,
        memory=memory,
        question_generator=question_generator,
    )
    return qa_chain
