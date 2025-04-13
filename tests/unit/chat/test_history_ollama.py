import os
import sys
from unittest.mock import patch, MagicMock, mock_open
from langchain.schema import Document
from lib.chat.history_ollama import (
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


class TestHistoryOllama:
    """Tests for the Ollama-based history-aware retrieval system."""

    @patch('os.listdir')
    @patch('ebooklib.epub.read_epub')
    def test_load_documents_from_directory(self, mock_read_epub, mock_listdir):
        """Test loading EPUB documents from a directory."""
        # Setup mocks
        mock_listdir.return_value = ["book1.epub", "book2.epub", "other.txt"]
        
        # Create mock book and items
        mock_book1 = MagicMock()
        mock_item1 = MagicMock()
        mock_item1.get_type.return_value = 'OEBPS_document'  # This matches ebooklib.ITEM_DOCUMENT
        mock_item1.get_content.return_value = b"<html><body>Book content 1</body></html>"
        mock_book1.get_items.return_value = [mock_item1]
        
        mock_book2 = MagicMock()
        mock_item2 = MagicMock()
        mock_item2.get_type.return_value = 'OEBPS_document'
        mock_item2.get_content.return_value = b"<html><body>Book content 2</body></html>"
        mock_book2.get_items.return_value = [mock_item2]
        
        # Configure read_epub to return different books for different file paths
        mock_read_epub.side_effect = lambda path: mock_book1 if "book1.epub" in path else mock_book2
        
        # Call the function
        with patch.dict('sys.modules', {'ebooklib': MagicMock()}):
            with patch('lib.chat.history_ollama.ebooklib') as mock_ebooklib:
                mock_ebooklib.ITEM_DOCUMENT = 'OEBPS_document'
                documents = load_documents_from_directory("/test/epub/directory")
        
        # Verify results
        mock_listdir.assert_called_once_with("/test/epub/directory")
        assert mock_read_epub.call_count == 2  # Only called for epub files
        
        # Verify we have 2 documents (one per EPUB file)
        assert len(documents) == 2
        
        # Check that the documents were created correctly
        assert isinstance(documents[0], Document)
        assert "<html><body>Book content 1</body></html>" in documents[0].page_content
        assert documents[0].metadata["source"] == "book1.epub"
        
        assert isinstance(documents[1], Document)
        assert "<html><body>Book content 2</body></html>" in documents[1].page_content
        assert documents[1].metadata["source"] == "book2.epub"

    def test_create_embedding_model(self):
        """Test creating an embedding model."""
        # Use patch to mock the entire HuggingFaceEmbeddings class
        with patch('lib.chat.history_ollama.HuggingFaceEmbeddings') as mock_embeddings_class:
            # Configure the mock to return a mock embedding model
            mock_model = MagicMock()
            mock_embeddings_class.return_value = mock_model
            
            # Test calling the function with a custom model name
            result = create_embedding_model("test-model")
            
            # Verify the embeddings class was called with the right parameters
            mock_embeddings_class.assert_called_once_with(model_name="test-model")
            
            # Check that the function returns the mock model
            assert result == mock_model

    def test_create_vector_store(self):
        """Test creating a vector store."""
        mock_documents = [MagicMock()]
        mock_embedding_model = MagicMock()
        
        with patch('langchain_community.vectorstores.FAISS.from_documents') as mock_faiss:
            vector_store = create_vector_store(mock_documents, mock_embedding_model)
            mock_faiss.assert_called_once_with(mock_documents, mock_embedding_model)

    def test_create_retriever(self):
        """Test creating a retriever."""
        mock_vector_store = MagicMock()
        mock_vector_store.as_retriever.return_value = "mock_retriever"
        
        retriever = create_retriever(mock_vector_store, k=5)
        mock_vector_store.as_retriever.assert_called_once_with(search_kwargs={"k": 5})
        assert retriever == "mock_retriever"

    def test_create_llm(self):
        """Test creating an LLM."""
        with patch('lib.chat.history_ollama.ChatOllama') as mock_chat_ollama:
            mock_chat_ollama.return_value = "mock_llm"
            
            llm = create_llm(model="test-model", temperature=0.7, base_url="http://test-url")
            
            mock_chat_ollama.assert_called_once_with(
                model="test-model",
                temperature=0.7,
                base_url="http://test-url",
            )
            assert llm == "mock_llm"

    def test_create_history_aware_retriever_chain(self):
        """Test creating a history-aware retriever chain."""
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        mock_chain = MagicMock()
        
        with patch('lib.chat.history_ollama.create_history_aware_retriever') as mock_create:
            mock_create.return_value = mock_chain
            
            result = create_history_aware_retriever_chain(mock_llm, mock_retriever)
            
            assert mock_create.called
            # Check the first two arguments
            args, _ = mock_create.call_args
            assert args[0] == mock_llm
            assert args[1] == mock_retriever
            assert result == mock_chain

    def test_create_qa_chain(self):
        """Test creating a QA chain."""
        mock_llm = MagicMock()
        mock_chain = MagicMock()
        
        with patch('lib.chat.history_ollama.create_stuff_documents_chain') as mock_create:
            mock_create.return_value = mock_chain
            
            qa_chain = create_qa_chain(mock_llm)
            
            assert mock_create.called
            # Check the first argument
            args, _ = mock_create.call_args
            assert args[0] == mock_llm
            assert qa_chain == mock_chain

    def test_create_rag_chain(self):
        """Test creating a RAG chain."""
        mock_retriever = MagicMock()
        mock_qa_chain = MagicMock()
        mock_rag_chain = MagicMock()
        
        with patch('lib.chat.history_ollama.create_retrieval_chain') as mock_create:
            mock_create.return_value = mock_rag_chain
            
            rag_chain = create_rag_chain(mock_retriever, mock_qa_chain)
            
            mock_create.assert_called_once_with(mock_retriever, mock_qa_chain)
            assert rag_chain == mock_rag_chain

    def test_query_rag_chain(self):
        """Test querying a RAG chain."""
        mock_rag_chain = MagicMock()
        mock_rag_chain.invoke.return_value = {"answer": "Test answer"}
        
        # Test with default chat history
        response = query_rag_chain(mock_rag_chain, "test query")
        mock_rag_chain.invoke.assert_called_once_with({"input": "test query", "chat_history": []})
        assert response == {"answer": "Test answer"}
        
        # Test with custom chat history
        mock_rag_chain.reset_mock()
        custom_history = [("user", "previous question"), ("ai", "previous answer")]
        response = query_rag_chain(mock_rag_chain, "follow-up query", custom_history)
        mock_rag_chain.invoke.assert_called_once_with({"input": "follow-up query", "chat_history": custom_history})

    @patch('lib.chat.history_ollama.load_documents_from_directory')
    @patch('lib.chat.history_ollama.create_embedding_model')
    @patch('lib.chat.history_ollama.create_vector_store')
    @patch('lib.chat.history_ollama.create_retriever')
    @patch('lib.chat.history_ollama.create_llm')
    @patch('lib.chat.history_ollama.create_history_aware_retriever_chain')
    @patch('lib.chat.history_ollama.create_qa_chain')
    @patch('lib.chat.history_ollama.create_rag_chain')
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
        with patch('os.path.expanduser', return_value="/mock/path"):
            rag_chain, chat_history = setup_rag_system()
        
        # Verify all functions were called with correct arguments
        mock_load_docs.assert_called_once_with("/mock/path")
        mock_create_embedding.assert_called_once()
        mock_create_vector.assert_called_once_with(mock_docs, mock_embedding)
        mock_create_retriever.assert_called_once_with(mock_vector)
        mock_create_llm.assert_called_once()
        mock_create_history.assert_called_once_with(mock_llm, mock_retriever)
        mock_create_qa.assert_called_once_with(mock_llm)
        mock_create_rag.assert_called_once_with(mock_history, mock_qa)
        
        # Verify the return values
        assert rag_chain == mock_rag
        assert chat_history == []