import os

import ebooklib
from ebooklib import epub
from langchain.chains import (create_history_aware_retriever,
                              create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings


def load_documents_from_directory(directory_path):
    """Load all EPUB documents from a directory.
    
    Args:
        directory_path: Path to the directory containing EPUB files.
        
    Returns:
        A list of Document objects containing the text from the EPUB files.
    """
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".epub"):
            file_path = os.path.join(directory_path, filename)
            # Load EPUB file
            book = epub.read_epub(file_path)
            text = ""
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    text += item.get_content().decode("utf-8") + "\n"
            # Create a Document object for LangChain
            documents.append(Document(page_content=text, metadata={"source": filename}))
    return documents


def create_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Create an embedding model using HuggingFaceEmbeddings.
    
    Args:
        model_name: Name of the model to use for embeddings.
        
    Returns:
        An instance of HuggingFaceEmbeddings.
    """
    return HuggingFaceEmbeddings(model_name=model_name)


def create_vector_store(documents, embedding_model):
    """Create a FAISS vector store from documents.
    
    Args:
        documents: List of Document objects.
        embedding_model: Model to use for embeddings.
        
    Returns:
        A FAISS vector store instance.
    """
    return FAISS.from_documents(documents, embedding_model)


def create_retriever(vector_store, k=3):
    """Create a retriever from a vector store.
    
    Args:
        vector_store: Vector store to use for retrieval.
        k: Number of documents to retrieve.
        
    Returns:
        A retriever instance.
    """
    return vector_store.as_retriever(search_kwargs={"k": k})


def create_llm(model="gemma3", temperature=0, base_url="http://localhost:11434"):
    """Create an Ollama LLM instance.
    
    Args:
        model: Name of the model to use.
        temperature: Temperature parameter for generation.
        base_url: URL of the Ollama server.
        
    Returns:
        A ChatOllama instance.
    """
    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=base_url,
    )


def create_history_aware_retriever_chain(llm, retriever):
    """Create a history-aware retriever.
    
    Args:
        llm: Language model to use.
        retriever: Retriever to use.
        
    Returns:
        A history-aware retriever instance.
    """
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
    return create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )


def create_qa_chain(llm):
    """Create a question-answering chain.
    
    Args:
        llm: Language model to use.
        
    Returns:
        A question-answering chain.
    """
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use "
        "the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, just say that you "
        "don't know. Use concise language and avoid asking for more input."
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}\n\nContext:\n{context}"),
        ]
    )
    return create_stuff_documents_chain(llm, qa_prompt)


def create_rag_chain(history_aware_retriever, qa_chain):
    """Create a RAG chain.
    
    Args:
        history_aware_retriever: History-aware retriever to use.
        qa_chain: Question-answering chain to use.
        
    Returns:
        A RAG chain.
    """
    return create_retrieval_chain(
        history_aware_retriever, qa_chain
    )


def query_rag_chain(rag_chain, query, chat_history=None):
    """Query a RAG chain.
    
    Args:
        rag_chain: RAG chain to query.
        query: Query to use.
        chat_history: Chat history to use.
        
    Returns:
        The response from the RAG chain.
    """
    if chat_history is None:
        chat_history = []
    return rag_chain.invoke({"input": query, "chat_history": chat_history})


def setup_rag_system(directory_path=None):
    """Set up a complete RAG system.
    
    Args:
        directory_path: Path to the directory containing documents.
        
    Returns:
        A tuple of (rag_chain, chat_history).
    """
    if directory_path is None:
        directory_path = os.path.expanduser("~/Documents/ebooks")
    
    # Load documents
    documents = load_documents_from_directory(directory_path)
    
    # Create embeddings and vector store
    embedding_model = create_embedding_model()
    vector_store = create_vector_store(documents, embedding_model)
    
    # Create retriever
    retriever = create_retriever(vector_store)
    
    # Create LLM
    llm = create_llm()
    
    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever_chain(llm, retriever)
    
    # Create QA chain
    qa_chain = create_qa_chain(llm)
    
    # Create RAG chain
    rag_chain = create_rag_chain(history_aware_retriever, qa_chain)
    
    # Initialize chat history
    chat_history = []
    
    return rag_chain, chat_history


# Only execute this code when the module is run directly, not when imported
if __name__ == "__main__":
    rag_chain, chat_history = setup_rag_system()
    
    # Query the Chain
    query = "What is discussed in the documents?"
    response = query_rag_chain(rag_chain, query, chat_history)
    print(response['answer'])
    
    # Test direct LLM query
    llm = create_llm()
    response = llm(messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ])
    print(f"LLM Response: {response}")
