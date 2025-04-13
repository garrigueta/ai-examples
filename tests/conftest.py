import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Add the project root to the path so imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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