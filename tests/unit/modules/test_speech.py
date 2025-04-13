import unittest
from unittest.mock import patch, MagicMock
from lib.modules.speech import SpeechToText


class TestSpeechToText:
    """Tests for the SpeechToText class."""

    @patch('pyttsx3.init')
    def test_initialization(self, mock_init):
        """Test that SpeechToText initializes correctly."""
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine
        
        speech = SpeechToText()
        
        mock_init.assert_called_once()
        assert speech.engine == mock_engine

    @patch('pyttsx3.init')
    def test_speech(self, mock_init):
        """Test text-to-speech conversion."""
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine
        
        speech = SpeechToText()
        speech.speech("Hello, world!")
        
        mock_engine.say.assert_called_once_with("Hello, world!")
        mock_engine.runAndWait.assert_called_once()
        mock_engine.stop.assert_called_once()

    @patch('pyttsx3.init')
    def test_change_voice(self, mock_init):
        """Test changing the voice of the speech engine."""
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine
        
        speech = SpeechToText()
        result = speech.change_voice("test-voice-id")
        
        mock_engine.setProperty.assert_called_once_with('voice', "test-voice-id")
        assert result is True

    @patch('pyttsx3.init')
    def test_get_voices(self, mock_init):
        """Test getting available voices."""
        mock_engine = MagicMock()
        mock_voice1 = MagicMock()
        mock_voice1.id = "voice1"
        mock_voice2 = MagicMock()
        mock_voice2.id = "voice2"
        mock_voices = [mock_voice1, mock_voice2]
        
        mock_engine.getProperty.return_value = mock_voices
        mock_init.return_value = mock_engine
        
        speech = SpeechToText()
        voices = speech.get_voices()
        
        mock_engine.getProperty.assert_called_once_with('voices')
        assert voices == mock_voices