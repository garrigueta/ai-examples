import pytest
from unittest.mock import patch, MagicMock
from lib.modules.ai import AiWrapper


class TestAiWrapper:
    """Tests for the AiWrapper class."""

    def test_initialization(self):
        """Test that AiWrapper initializes with correct default values."""
        ai = AiWrapper()
        assert ai.client is None
        assert "asistente de vuelo" in ai.system_base_data
        assert ai.system_content == ""
        assert ai.user_content == ""
        assert ai.ai_response == ""

    def test_set_system_content(self):
        """Test setting system content."""
        ai = AiWrapper()
        test_content = '{"altitude": 10000, "speed": 250}'
        ai.set_system_content(test_content)
        assert ai.system_content == test_content

    def test_set_user_content(self):
        """Test setting user content."""
        ai = AiWrapper()
        test_content = "What's my current altitude?"
        ai.set_user_content(test_content)
        assert ai.user_content == test_content

    def test_initi_ai(self):
        """Test initializing the OpenAI client."""
        # Since OpenAI is imported directly at the top level in the module,
        # we need to patch it there directly
        with patch('lib.modules.ai.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            
            # Mock the environment variables
            with patch.dict('os.environ', {
                'OPENAI_API_KEY': 'test-key',
                'OPENAI_ORG': 'test-org',
                'OPENAI_PROJ': 'test-proj'
            }):
                # Test the initialization
                ai = AiWrapper()
                ai.initi_ai()
                
                # Verify OpenAI was called with the right parameters
                mock_openai.assert_called_once_with(
                    organization='test-org',
                    project='test-proj',
                    api_key='test-key'
                )
                assert ai.client == mock_client

    def test_get_ai_response(self, mock_openai_client):
        """Test getting an AI response."""
        with patch('lib.modules.ai.OpenAI', return_value=mock_openai_client):
            ai = AiWrapper()
            ai.client = mock_openai_client
            
            response = ai.get_ai_response("What's my altitude?")
            
            # Verify the API was called with the right parameters
            mock_openai_client.chat.completions.create.assert_called_once()
            call_args = mock_openai_client.chat.completions.create.call_args[1]
            assert call_args["model"] == "gpt-4-turbo"
            assert len(call_args["messages"]) == 2
            assert call_args["messages"][0]["role"] == "system"
            assert call_args["messages"][1]["role"] == "user"
            assert call_args["messages"][1]["content"] == "What's my altitude?"
            
            # Check the response is what we expect from our mock
            assert response == "Test AI response"