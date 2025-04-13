import os
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

from lib.modules.speech import SpeechToText


# Step 1: Load All Documents from a Directory
def load_documents_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        loader = TextLoader(file_path)
        documents.extend(loader.load())
    return documents


def create_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Create an embedding model using HuggingFace."""
    return HuggingFaceEmbeddings(model_name=model_name)


def create_vector_store(documents, embedding_model):
    """Create a vector store from documents and an embedding model."""
    return FAISS.from_documents(documents, embedding_model)


def create_retriever(vector_store, k=3):
    """Create a retriever from a vector store."""
    return vector_store.as_retriever(search_kwargs={"k": k})


def create_llm(api_key=None):
    """Create an OpenAI language model."""
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    return ChatOpenAI(api_key=api_key)


def create_history_aware_retriever_chain(llm, retriever):
    """Create a history-aware retriever chain."""
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return create_history_aware_retriever(llm, retriever, contextualize_q_prompt)


def create_qa_chain(llm):
    """Create a question answering chain."""
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use "
        "the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, just say that you "
        "don't know. Use three sentences maximum and keep the answer "
        "concise.\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return create_stuff_documents_chain(llm, qa_prompt)


def create_rag_chain(history_aware_retriever, question_answer_chain):
    """Create a retrieval augmented generation chain."""
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


def query_rag_chain(rag_chain, query, chat_history=None):
    """Query the RAG chain with a query and optional chat history."""
    if chat_history is None:
        chat_history = []
    return rag_chain.invoke({"input": query, "chat_history": chat_history})


def setup_rag_system(directory_path="docs"):
    """Set up the complete RAG system."""
    # Load documents
    documents = load_documents_from_directory(directory_path)
    
    # Create embeddings and vector store
    embedding_model = create_embedding_model()
    vector_store = create_vector_store(documents, embedding_model)
    
    # Create retriever
    retriever = create_retriever(vector_store)
    
    # Create LLM
    llm = create_llm()
    
    # Create chains
    history_aware_retriever = create_history_aware_retriever_chain(llm, retriever)
    question_answer_chain = create_qa_chain(llm)
    rag_chain = create_rag_chain(history_aware_retriever, question_answer_chain)
    
    # Initialize chat history
    chat_history = []
    
    return rag_chain, chat_history


# Only execute if script is run directly, not when imported
if __name__ == "__main__":
    rag_chain, chat_history = setup_rag_system()
    query = "What is discussed in the documents?"
    response = query_rag_chain(rag_chain, query, chat_history)
    
    speech = SpeechToText()
    speech.speech(response['answer'])
    print(response['answer'])
