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


# Step 1: Load All Documents from a Directory
def load_documents_from_directory(directory_path):
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


directory_path = os.path.expanduser("~/Documents/ebooks")
documents = load_documents_from_directory(directory_path)

# Create Embeddings and Vector Store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(documents, embedding_model)

# Define Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Configure History-Aware Retriever
# Load the LLM model from local Ollama instance
llm = ChatOllama(
    model="gemma3",
    temperature=0,
    base_url="http://localhost:11434",
)
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
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


# Define Question Answering Chain
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
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create Retrieval Chain
rag_chain = create_retrieval_chain(
    history_aware_retriever, question_answer_chain
)

# Query the Chain
chat_history = []
query = "What is discussed in the documents?"
response = rag_chain.invoke({"input": query, "chat_history": chat_history})

print(response['answer'])

# Test direct LLM query
response = llm(messages=[
    {"role": "user", "content": "Hello, how are you?"}
])
print(f"LLM Response: {response}")
