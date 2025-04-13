import os
import pytest
from unittest.mock import patch, MagicMock, mock_open

# Import functions to test
from lib.chat.openai import (
    load_documents_from_directory,
    create_embedding_model,
    create_vector_store,
    create_retriever,
    create_llm,
    create_history_aware_retriever_chain,
    create_qa_chain,
    create_rag_chain,
    query_rag_chain,
    setup_rag_system
)


class TestOpenAI:
    """Tests for the OpenAI-based RAG system."""

    @patch('os.path.join', lambda dir, file: f"{dir}/{file}")
    @patch('os.listdir')
    @patch('lib.chat.openai.TextLoader')
    def test_load_documents_from_directory(self, mock_text_loader, mock_listdir):
        """Test loading text documents from a directory."""
        # Setup mocks
        mock_listdir.return_value = ["file1.txt", "file2.txt"]
        
        # Create mock loaders and documents
        mock_loader = MagicMock()
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "Content from file1"
        mock_doc2 = MagicMock()
        mock_doc2.page_content = "Content from file2"
        
        # Set up the mock loader to return different documents based on file path
        mock_loader.load.side_effect = [[mock_doc1], [mock_doc2]]
        mock_text_loader.return_value = mock_loader
        
        # Call the function
        documents = load_documents_from_directory("/test/directory")
        
        # Verify results
        mock_listdir.assert_called_once_with("/test/directory")
        assert mock_text_loader.call_count == 2
        mock_text_loader.assert_any_call("/test/directory/file1.txt")
        mock_text_loader.assert_any_call("/test/directory/file2.txt")
        assert len(documents) == 2
        assert documents[0].page_content == "Content from file1"
        assert documents[1].page_content == "Content from file2"

    @patch('lib.chat.openai.HuggingFaceEmbeddings')
    def test_create_embedding_model(self, mock_embeddings_class):
        """Test creating an embedding model."""
        # Configure the mock
        mock_model = MagicMock()
        mock_embeddings_class.return_value = mock_model
        
        # Test the function with default model name
        result = create_embedding_model()
        
        # Verify the embeddings class was called with the right parameters
        mock_embeddings_class.assert_called_once_with(model_name="sentence-transformers/all-MiniLM-L6-v2")
        assert result == mock_model
        
        # Reset mock and test with custom model name
        mock_embeddings_class.reset_mock()
        mock_embeddings_class.return_value = mock_model
        
        result_custom = create_embedding_model("custom-model")
        mock_embeddings_class.assert_called_once_with(model_name="custom-model")
        assert result_custom == mock_model

    @patch('lib.chat.openai.FAISS.from_documents')
    def test_create_vector_store(self, mock_from_documents):
        """Test creating a vector store."""
        # Setup mocks
        mock_documents = [MagicMock()]
        mock_embedding_model = MagicMock()
        mock_vector_store = MagicMock()
        mock_from_documents.return_value = mock_vector_store
        
        # Call the function
        result = create_vector_store(mock_documents, mock_embedding_model)
        
        # Verify the results
        mock_from_documents.assert_called_once_with(mock_documents, mock_embedding_model)
        assert result == mock_vector_store

    def test_create_retriever(self):
        """Test creating a retriever."""
        # Setup mock
        mock_vector_store = MagicMock()
        mock_retriever = MagicMock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        
        # Test with default k
        result = create_retriever(mock_vector_store)
        mock_vector_store.as_retriever.assert_called_once_with(search_kwargs={"k": 3})
        assert result == mock_retriever
        
        # Test with custom k
        mock_vector_store.reset_mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        
        result_custom = create_retriever(mock_vector_store, k=5)
        mock_vector_store.as_retriever.assert_called_once_with(search_kwargs={"k": 5})
        assert result_custom == mock_retriever

    @patch('lib.chat.openai.ChatOpenAI')
    def test_create_llm(self, mock_chat_openai):
        """Test creating an LLM."""
        # Setup mock
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Test with default (using environment variable)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
            result = create_llm()
            
            mock_chat_openai.assert_called_once_with(api_key="test-api-key")
            assert result == mock_llm
        
        # Test with custom API key
        mock_chat_openai.reset_mock()
        mock_chat_openai.return_value = mock_llm
        
        result_custom = create_llm(api_key="custom-api-key")
        mock_chat_openai.assert_called_once_with(api_key="custom-api-key")
        assert result_custom == mock_llm

    @patch('lib.chat.openai.create_history_aware_retriever')
    def test_create_history_aware_retriever_chain(self, mock_create_history_aware_retriever):
        """Test creating a history-aware retriever chain."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        mock_chain = MagicMock()
        mock_create_history_aware_retriever.return_value = mock_chain
        
        # Call the function
        result = create_history_aware_retriever_chain(mock_llm, mock_retriever)
        
        # Verify results
        assert mock_create_history_aware_retriever.called
        # Check that the first two arguments are correct
        args, _ = mock_create_history_aware_retriever.call_args
        assert args[0] == mock_llm
        assert args[1] == mock_retriever
        assert result == mock_chain

    @patch('lib.chat.openai.create_stuff_documents_chain')
    def test_create_qa_chain(self, mock_create_stuff_documents_chain):
        """Test creating a question answer chain."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_chain = MagicMock()
        mock_create_stuff_documents_chain.return_value = mock_chain
        
        # Call the function
        result = create_qa_chain(mock_llm)
        
        # Verify results
        assert mock_create_stuff_documents_chain.called
        # Check the first argument is correct
        args, _ = mock_create_stuff_documents_chain.call_args
        assert args[0] == mock_llm
        assert result == mock_chain

    @patch('lib.chat.openai.create_retrieval_chain')
    def test_create_rag_chain(self, mock_create_retrieval_chain):
        """Test creating a RAG chain."""
        # Setup mocks
        mock_retriever = MagicMock()
        mock_qa_chain = MagicMock()
        mock_rag_chain = MagicMock()
        mock_create_retrieval_chain.return_value = mock_rag_chain
        
        # Call the function
        result = create_rag_chain(mock_retriever, mock_qa_chain)
        
        # Verify results
        mock_create_retrieval_chain.assert_called_once_with(mock_retriever, mock_qa_chain)
        assert result == mock_rag_chain

    def test_query_rag_chain(self):
        """Test querying a RAG chain."""
        # Setup mocks
        mock_rag_chain = MagicMock()
        mock_response = {"answer": "Test answer"}
        mock_rag_chain.invoke.return_value = mock_response
        
        # Test with default chat history
        response = query_rag_chain(mock_rag_chain, "test query")
        mock_rag_chain.invoke.assert_called_once_with({"input": "test query", "chat_history": []})
        assert response == mock_response
        
        # Test with custom chat history
        mock_rag_chain.reset_mock()
        mock_rag_chain.invoke.return_value = mock_response
        custom_history = [("user", "previous question"), ("ai", "previous answer")]
        
        response_custom = query_rag_chain(mock_rag_chain, "follow-up query", custom_history)
        mock_rag_chain.invoke.assert_called_once_with({"input": "follow-up query", "chat_history": custom_history})
        assert response_custom == mock_response

    @patch('lib.chat.openai.load_documents_from_directory')
    @patch('lib.chat.openai.create_embedding_model')
    @patch('lib.chat.openai.create_vector_store')
    @patch('lib.chat.openai.create_retriever')
    @patch('lib.chat.openai.create_llm')
    @patch('lib.chat.openai.create_history_aware_retriever_chain')
    @patch('lib.chat.openai.create_qa_chain')
    @patch('lib.chat.openai.create_rag_chain')
    def test_setup_rag_system(self, mock_create_rag, mock_create_qa, mock_create_history, 
                             mock_create_llm, mock_create_retriever, mock_create_vector, 
                             mock_create_embedding, mock_load_docs):
        """Test setting up a complete RAG system."""
        # Setup return values for the mocked functions
        mock_docs = [MagicMock()]
        mock_load_docs.return_value = mock_docs
        
        mock_embedding = MagicMock()
        mock_create_embedding.return_value = mock_embedding
        
        mock_vector = MagicMock()
        mock_create_vector.return_value = mock_vector
        
        mock_retriever = MagicMock()
        mock_create_retriever.return_value = mock_retriever
        
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm
        
        mock_history = MagicMock()
        mock_create_history.return_value = mock_history
        
        mock_qa = MagicMock()
        mock_create_qa.return_value = mock_qa
        
        mock_rag = MagicMock()
        mock_create_rag.return_value = mock_rag
        
        # Call the function
        rag_chain, chat_history = setup_rag_system(directory_path="/test/directory")
        
        # Verify all functions were called with correct arguments
        mock_load_docs.assert_called_once_with("/test/directory")
        mock_create_embedding.assert_called_once_with()
        mock_create_vector.assert_called_once_with(mock_docs, mock_embedding)
        mock_create_retriever.assert_called_once_with(mock_vector)
        mock_create_llm.assert_called_once_with()
        mock_create_history.assert_called_once_with(mock_llm, mock_retriever)
        mock_create_qa.assert_called_once_with(mock_llm)
        mock_create_rag.assert_called_once_with(mock_history, mock_qa)
        
        # Verify the return values
        assert rag_chain == mock_rag
        assert chat_history == []