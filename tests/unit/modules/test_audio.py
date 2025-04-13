import pytest
from unittest.mock import patch, MagicMock, mock_open
import json
import pyaudio
import vosk
from lib.modules.audio import Audio


class TestAudio:
    """Tests for the Audio class."""

    def test_initialization(self):
        """Test that Audio initializes with correct default values."""
        audio = Audio()
        assert audio.model_path == "vosk-model-es-0.42"
        assert audio.output_file_path == "recognized_text.txt"
        assert audio.recognized_text == ""
        assert audio.data is None
        assert audio.stream is None
        assert audio.p is None
        assert audio.model is None
        assert audio.rec is None

    def test_set_model_path(self):
        """Test setting the model path."""
        audio = Audio()
        audio.set_model_path("new-model-path")
        assert audio.model_path == "new-model-path"

    @patch('pyaudio.PyAudio')
    @patch('vosk.Model')
    @patch('vosk.KaldiRecognizer')
    def test_init_audio(self, mock_recognizer, mock_model, mock_pyaudio):
        """Test initializing the audio stream."""
        # Setup mocks
        mock_stream = MagicMock()
        mock_pa_instance = MagicMock()
        mock_pa_instance.open.return_value = mock_stream
        mock_pyaudio.return_value = mock_pa_instance
        
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_recognizer_instance = MagicMock()
        mock_recognizer.return_value = mock_recognizer_instance
        
        # Initialize audio
        audio = Audio()
        audio.init_audio()
        
        # Verify the pyaudio stream was initialized correctly
        mock_pyaudio.assert_called_once()
        mock_pa_instance.open.assert_called_once_with(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=8192
        )
        
        # Verify the model and recognizer were initialized
        mock_model.assert_called_once_with(audio.model_path)
        mock_recognizer.assert_called_once_with(mock_model_instance, 16000)
        
        # Check that the audio object has the correct attributes
        assert audio.p == mock_pa_instance
        assert audio.stream == mock_stream
        assert audio.model == mock_model_instance
        assert audio.rec == mock_recognizer_instance

    @patch('pyaudio.PyAudio')
    @patch('vosk.Model')
    @patch('vosk.KaldiRecognizer')
    def test_wait_for_audio(self, mock_recognizer, mock_model, mock_pyaudio):
        """Test waiting for audio input."""
        # Setup mocks
        mock_stream = MagicMock()
        mock_stream.read.return_value = b'audio data'
        
        mock_pa_instance = MagicMock()
        mock_pa_instance.open.return_value = mock_stream
        mock_pyaudio.return_value = mock_pa_instance
        
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_recognizer_instance = MagicMock()
        # First call to AcceptWaveform returns False (no speech recognized)
        # Second call returns True (speech recognized)
        mock_recognizer_instance.AcceptWaveform.side_effect = [False, True]
        mock_recognizer_instance.Result.return_value = json.dumps({
            "text": "recognized text"
        })
        mock_recognizer.return_value = mock_recognizer_instance
        
        # Initialize and test waiting for audio
        audio = Audio()
        audio.init_audio()
        audio.wait_for_audio()
        
        # Verify that the recognizer processed audio data
        assert mock_recognizer_instance.AcceptWaveform.call_count == 2
        mock_recognizer_instance.Result.assert_called_once()
        
        # Check that the recognized text was set correctly
        assert audio.recognized_text == "recognized text"
        assert audio.data == b'audio data'