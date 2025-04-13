import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Add the project root to the path so imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock modules that require hardware or external dependencies
sys.modules['pyaudio'] = MagicMock()
sys.modules['pyaudio'].paInt16 = 8
sys.modules['pyaudio'].PyAudio = MagicMock
sys.modules['vosk'] = MagicMock()
sys.modules['vosk'].Model = MagicMock
sys.modules['vosk'].KaldiRecognizer = MagicMock
sys.modules['SimConnect'] = MagicMock()
sys.modules['SimConnect'].SimConnect = MagicMock
sys.modules['SimConnect'].AircraftRequests = MagicMock
sys.modules['SimConnect'].AircraftEvents = MagicMock
sys.modules['pyttsx3'] = MagicMock()
sys.modules['pyttsx3'].init = MagicMock(return_value=MagicMock())

@pytest.fixture
def mock_openai_client():
    """Fixture that provides a mock OpenAI client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test AI response"
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client

@pytest.fixture
def mock_simconnect():
    """Fixture that provides a mock SimConnect instance."""
    mock_sim = MagicMock()
    mock_requests = MagicMock()
    # Set up some basic flight data returns
    mock_requests.get.return_value = 10000  # Default value for all data points
    mock_sim.AircraftRequests.return_value = mock_requests
    return mock_sim

@pytest.fixture(autouse=True)
def mock_external_services():
    """Automatically mock external services for all tests."""
    # Mock requests.post for Ollama API calls
    with patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "Mocked Ollama response"
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        # Set environment variables for testing
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'OPENAI_ORG': 'test-org',
            'OPENAI_PROJ': 'test-proj',
            'OLLAMA_API_URL': 'http://localhost:11434/api/generate',
            'OLLAMA_MODEL': 'test-model'
        }):
            yield