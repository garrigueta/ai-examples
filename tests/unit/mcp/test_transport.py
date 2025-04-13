import json
import uuid
import datetime
from unittest.mock import patch, MagicMock
import pytest

from lib.mcp.transport import MCPMessage


class TestMCPMessage:
    """Tests for the MCPMessage class in the transport module."""

    def test_init_with_default_values(self):
        """Test MCPMessage initialization with default values for id and timestamp."""
        with patch('uuid.uuid4') as mock_uuid:
            mock_uuid.return_value = uuid.UUID('12345678-1234-5678-1234-567812345678')
            with patch('datetime.datetime') as mock_datetime:
                mock_dt = MagicMock()
                mock_dt.isoformat.return_value = "2025-04-13T12:34:56.789012"
                mock_datetime.utcnow.return_value = mock_dt

                msg = MCPMessage(role="user", content="Hello")

                # Check that default values were set correctly
                assert msg.id == "12345678-1234-5678-1234-567812345678"
                assert msg.timestamp == "2025-04-13T12:34:56.789012"
                assert msg.role == "user"
                assert msg.content == "Hello"
                assert msg.context == {}  # Default empty dict

    def test_init_with_custom_values(self):
        """Test MCPMessage initialization with custom values."""
        custom_id = "custom-id-123"
        custom_timestamp = "2025-04-13T10:11:12.000Z"
        custom_context = {"system": "test", "session": "abc123"}

        msg = MCPMessage(
            role="assistant",
            content="I'm an assistant",
            context=custom_context,
            msg_id=custom_id,
            timestamp=custom_timestamp
        )

        # Check that custom values were set correctly
        assert msg.id == custom_id
        assert msg.timestamp == custom_timestamp
        assert msg.role == "assistant"
        assert msg.content == "I'm an assistant"
        assert msg.context == custom_context

    def test_to_dict(self):
        """Test conversion of MCPMessage to dictionary."""
        msg = MCPMessage(
            role="system",
            content="System message",
            context={"mode": "test"},
            msg_id="sys-123",
            timestamp="2025-04-13T09:08:07.000Z"
        )

        result = msg.to_dict()

        # Check that the dictionary has the correct structure and values
        expected = {
            "id": "sys-123",
            "timestamp": "2025-04-13T09:08:07.000Z",
            "role": "system",
            "content": "System message",
            "context": {"mode": "test"}
        }
        assert result == expected

    def test_to_json(self):
        """Test serialization of MCPMessage to JSON."""
        msg = MCPMessage(
            role="user",
            content="User query",
            context={"query_id": "q1"},
            msg_id="user-123",
            timestamp="2025-04-13T15:16:17.000Z"
        )

        json_str = msg.to_json()

        # Parse the JSON and compare with expected structure
        parsed = json.loads(json_str)
        expected = {
            "id": "user-123",
            "timestamp": "2025-04-13T15:16:17.000Z",
            "role": "user",
            "content": "User query",
            "context": {"query_id": "q1"}
        }
        assert parsed == expected

    def test_from_json(self):
        """Test deserialization from JSON to MCPMessage."""
        json_str = '''
        {
            "id": "assistant-456",
            "timestamp": "2025-04-13T18:19:20.000Z",
            "role": "assistant",
            "content": "I can help with that",
            "context": {"response_type": "text"}
        }
        '''

        msg = MCPMessage.from_json(json_str)

        # Check that all attributes were correctly parsed
        assert msg.id == "assistant-456"
        assert msg.timestamp == "2025-04-13T18:19:20.000Z"
        assert msg.role == "assistant"
        assert msg.content == "I can help with that"
        assert msg.context == {"response_type": "text"}

    def test_full_serialization_cycle(self):
        """Test the full cycle of serialization and deserialization."""
        original = MCPMessage(
            role="user",
            content="What's the weather?",
            context={"location": "San Francisco"},
            msg_id="weather-query-1",
            timestamp="2025-04-13T21:22:23.000Z"
        )

        # Serialize to JSON
        json_str = original.to_json()

        # Deserialize back to an object
        deserialized = MCPMessage.from_json(json_str)

        # Compare all attributes
        assert deserialized.id == original.id
        assert deserialized.timestamp == original.timestamp
        assert deserialized.role == original.role
        assert deserialized.content == original.content
        assert deserialized.context == original.context

    def test_from_json_minimal(self):
        """Test deserialization from JSON with minimal fields."""
        # Note: id and timestamp are required according to the implementation
        json_str = '''
        {
            "id": "min-789",
            "timestamp": "2025-04-13T23:59:59.000Z",
            "role": "system",
            "content": "Minimal message"
        }
        '''

        msg = MCPMessage.from_json(json_str)

        # Check that required fields are set and optional fields have defaults
        assert msg.id == "min-789"
        assert msg.timestamp == "2025-04-13T23:59:59.000Z"
        assert msg.role == "system"
        assert msg.content == "Minimal message"
        assert msg.context == {}  # Should default to empty dict

    def test_invalid_json_raises_exception(self):
        """Test that invalid JSON raises an exception."""
        invalid_json = "{ this is not valid JSON }"

        with pytest.raises(json.JSONDecodeError):
            MCPMessage.from_json(invalid_json)