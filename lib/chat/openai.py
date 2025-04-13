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


directory_path = "docs"  # Replace with your directory path
documents = load_documents_from_directory(directory_path)

# Step 2: Create Embeddings and Vector Store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(documents, embedding_model)

# Step 3: Define Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Adjust k as needed

# Step 4: Configure History-Aware Retriever
llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
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

# Step 5: Define Question Answering Chain
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
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Step 6: Create Retrieval Chain
rag_chain = create_retrieval_chain(
    history_aware_retriever, question_answer_chain
)

# Step 7: Query the Chain
chat_history = []  # Initialize chat history
query = "What is discussed in the documents?"
response = rag_chain.invoke({"input": query, "chat_history": chat_history})

speech = SpeechToText()
speech.speech(response['answer'])
print(response['answer'])
