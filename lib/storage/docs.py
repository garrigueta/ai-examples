import os
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


CHAT_HISTORY = []
RET_CHAIN = None


def load_documents_from_directory(directory_path):
    """ Load all documents from a specified directory. """
    documents = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        loader = TextLoader(file_path)
        documents.extend(loader.load())
    return documents


def generate_retriever(
        directory_path,
        llm,
        model="sentence-transformers/all-MiniLM-L6-v2"):
    """ Generate a retriever for question-answering tasks. """
    documents = load_documents_from_directory(directory_path)

    # Step 2: Create Embeddings and Vector Store
    embedding_model = HuggingFaceEmbeddings(model_name=model)
    vector_store = FAISS.from_documents(documents, embedding_model)

    # Step 3: Define Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Step 4: Configure History-Aware Retriever
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
    global RET_CHAIN
    RET_CHAIN = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )

    def ask_question(question):
        """ Ask a question using the retrieval chain. """
        response = RET_CHAIN({"input": question, "chat_history": CHAT_HISTORY})
        CHAT_HISTORY.append((question, response["output"]))
        return response["output"]
    return ask_question
