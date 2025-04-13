import unittest
from unittest.mock import patch, MagicMock

from lib.sim import FlightSimAi


class TestFlightSimAi(unittest.TestCase):
    """Tests for the FlightSimAi class in the sim module."""

    @patch('lib.sim.AiWrapper')
    @patch('lib.sim.Audio')
    @patch('lib.sim.SpeechToText')
    def test_initialization(self, mock_speech, mock_audio, mock_ai):
        """Test FlightSimAi initialization with properly mocked dependencies."""
        # Setup mocks
        mock_ai_instance = MagicMock()
        mock_audio_instance = MagicMock()
        mock_speech_instance = MagicMock()
        
        mock_ai.return_value = mock_ai_instance
        mock_audio.return_value = mock_audio_instance
        mock_speech.return_value = mock_speech_instance
        
        # Patch the start method to prevent it from running during initialization
        with patch.object(FlightSimAi, 'start'):
            flight_sim = FlightSimAi()
            
            # Verify instance attributes are set correctly
            self.assertEqual(flight_sim.ai, mock_ai_instance)
            self.assertEqual(flight_sim.audio, mock_audio_instance)
            self.assertEqual(flight_sim.speech, mock_speech_instance)
            
            # Verify start was called
            FlightSimAi.start.assert_called_once()

    @patch('lib.sim.AiWrapper')
    @patch('lib.sim.Audio')
    @patch('lib.sim.SpeechToText')
    def test_start_method_initialization(self, mock_speech, mock_speech_to_text, mock_ai):
        """Test that start method initializes the AI and Audio subsystems."""
        # Setup mocks
        mock_ai_instance = MagicMock()
        mock_audio_instance = MagicMock()
        mock_speech_instance = MagicMock()
        
        mock_ai.return_value = mock_ai_instance
        mock_speech_to_text.return_value = mock_audio_instance
        mock_speech.return_value = mock_speech_instance
        
        # Patch the continuous loop to exit immediately
        mock_audio_instance.wait_for_audio.side_effect = [Exception("Exit test early")]
        
        with patch('builtins.print') as mock_print:
            # Create instance but expect an exception from our side effect
            with self.assertRaises(Exception) as context:
                flight_sim = FlightSimAi()
            
            self.assertEqual(str(context.exception), "Exit test early")
            
            # Verify initialization calls
            mock_ai_instance.initi_ai.assert_called_once()
            mock_audio_instance.init_audio.assert_called_once()
            mock_print.assert_called_with("Listening...", flush=True)

    @patch('lib.sim.AiWrapper')
    @patch('lib.sim.Audio')
    @patch('lib.sim.SpeechToText')
    def test_audio_processing_loop(self, mock_speech, mock_audio, mock_ai):
        """Test the main processing loop for audio recognition and AI response."""
        # Setup mocks
        mock_ai_instance = MagicMock()
        mock_audio_instance = MagicMock()
        mock_speech_instance = MagicMock()
        
        mock_ai.return_value = mock_ai_instance
        mock_audio.return_value = mock_audio_instance
        mock_speech.return_value = mock_speech_instance
        
        # Setup a sequence of recognized text values that will be processed
        # First iteration: process normal text
        # Second iteration: include termination keyword to exit loop
        mock_audio_instance.recognized_text = ""
        def wait_for_audio_side_effect():
            # Toggle between empty and non-empty recognized text
            if not hasattr(wait_for_audio_side_effect, "call_count"):
                wait_for_audio_side_effect.call_count = 0
            wait_for_audio_side_effect.call_count += 1
            
            if wait_for_audio_side_effect.call_count == 1:
                mock_audio_instance.recognized_text = "What's my altitude?"
            elif wait_for_audio_side_effect.call_count == 2:
                mock_audio_instance.recognized_text = "finalizar"
        
        mock_audio_instance.wait_for_audio.side_effect = wait_for_audio_side_effect
        mock_ai_instance.get_ai_response.return_value = "Your altitude is 10,000 feet."
        
        # Execute the method under test
        flight_sim = FlightSimAi()
        
        # Verify the interactions
        self.assertEqual(mock_audio_instance.wait_for_audio.call_count, 2)
        mock_ai_instance.get_ai_response.assert_called_once_with("What's my altitude?")
        mock_speech_instance.speech.assert_called_once_with("Your altitude is 10,000 feet.")

    @patch('lib.sim.AiWrapper')
    @patch('lib.sim.Audio')
    @patch('lib.sim.SpeechToText')
    def test_empty_recognized_text(self, mock_speech, mock_audio, mock_ai):
        """Test that empty recognized text is properly handled."""
        # Setup mocks
        mock_ai_instance = MagicMock()
        mock_audio_instance = MagicMock()
        mock_speech_instance = MagicMock()
        
        mock_ai.return_value = mock_ai_instance
        mock_audio.return_value = mock_audio_instance
        mock_speech.return_value = mock_speech_instance
        
        # Setup for two iterations:
        # First with empty recognized text (should not trigger AI)
        # Second with termination keyword
        def wait_for_audio_side_effect():
            if not hasattr(wait_for_audio_side_effect, "call_count"):
                wait_for_audio_side_effect.call_count = 0
            wait_for_audio_side_effect.call_count += 1
            
            if wait_for_audio_side_effect.call_count == 1:
                mock_audio_instance.recognized_text = ""  # Empty text
            elif wait_for_audio_side_effect.call_count == 2:
                mock_audio_instance.recognized_text = "finalizar"
        
        mock_audio_instance.wait_for_audio.side_effect = wait_for_audio_side_effect
        
        # Execute the method under test
        flight_sim = FlightSimAi()
        
        # Verify AI was not called with empty text
        mock_ai_instance.get_ai_response.assert_not_called()
        mock_speech_instance.speech.assert_not_called()

    @patch('lib.sim.AiWrapper')
    @patch('lib.sim.Audio')
    @patch('lib.sim.SpeechToText')
    def test_termination_keyword_detection(self, mock_speech, mock_audio, mock_ai):
        """Test that the termination keyword is properly detected."""
        # Setup mocks
        mock_ai_instance = MagicMock()
        mock_audio_instance = MagicMock()
        mock_speech_instance = MagicMock()
        
        mock_ai.return_value = mock_ai_instance
        mock_audio.return_value = mock_audio_instance
        mock_speech.return_value = mock_speech_instance
        
        # Directly set recognized_text to termination keyword
        mock_audio_instance.recognized_text = "finalizar"
        
        # Execute the method under test with print patched
        with patch('builtins.print') as mock_print:
            flight_sim = FlightSimAi()
            
            # Verify termination message was printed
            mock_print.assert_any_call("Termination keyword detected. Stopping...", flush=True)
            
            # Verify AI was not called for termination keyword
            mock_ai_instance.get_ai_response.assert_not_called()
            mock_speech_instance.speech.assert_not_called()


if __name__ == '__main__':
    unittest.main()