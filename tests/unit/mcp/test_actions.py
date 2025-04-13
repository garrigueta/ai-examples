from unittest.mock import patch, MagicMock
import os
import subprocess
from lib.mcp.actions import (
    ask_llm_to_explain_error,
    ask_llm_for_command,
    run_command,
    run_ai_command,
    prompt_ollama_http
)


class TestMCPActions:
    """Tests for the MCP actions module."""

    @patch('requests.post')
    def test_ask_llm_to_explain_error(self, mock_post):
        """Test asking LLM to explain an error."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "This error occurs when the package is not found"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = ask_llm_to_explain_error("pip install nonexistent", "Package not found")

        # Verify the API was called correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args[1]
        assert 'json' in call_args
        assert call_args['json']['model'] == os.getenv('OLLAMA_MODEL', 'gemma3')
        assert 'nonexistent' in call_args['json']['prompt']
        assert 'Package not found' in call_args['json']['prompt']

        # Check the result
        assert result == "This error occurs when the package is not found"

    @patch('requests.post')
    def test_ask_llm_for_command(self, mock_post):
        """Test asking LLM for a command."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "ls -la"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = ask_llm_for_command("list all files in detail")

        # Verify the API was called correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args[1]
        assert 'json' in call_args
        assert call_args['json']['model'] == os.getenv('OLLAMA_MODEL', 'gemma3')
        assert 'list all files in detail' in call_args['json']['prompt']

        # Check the result
        assert result == "ls -la"

    @patch('subprocess.check_output')
    def test_run_command_success(self, mock_check_output):
        """Test running a command successfully."""
        mock_check_output.return_value = "file1.txt\nfile2.txt"

        result = run_command("ls")

        mock_check_output.assert_called_once_with(
            "ls", shell=True, stderr=subprocess.STDOUT, text=True, timeout=5
        )
        assert result == "file1.txt\nfile2.txt"

    @patch('subprocess.check_output')
    def test_run_command_error(self, mock_check_output):
        """Test running a command that fails."""
        mock_check_output.side_effect = subprocess.CalledProcessError(
            1, "invalid_command", output="Command not found"
        )

        result = run_command("invalid_command")

        assert "Command error" in result
        assert "Command not found" in result

    @patch('lib.mcp.actions.ask_llm_for_command')
    @patch('lib.mcp.actions.run_command')
    def test_run_ai_command(self, mock_run_command, mock_ask_llm):
        """Test running an AI-generated command."""
        mock_ask_llm.return_value = "ls -la"
        mock_run_command.return_value = "total 20\ndrwxr-xr-x 4 user user 4096 Apr 13 10:00 ."

        command, output = run_ai_command("list all files with details")

        mock_ask_llm.assert_called_once_with("list all files with details")
        mock_run_command.assert_called_once_with("ls -la")
        assert command == "ls -la"
        assert "total 20" in output

    @patch('requests.post')
    def test_prompt_ollama_http(self, mock_post):
        """Test sending a prompt to Ollama via HTTP."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Vector embeddings are numerical representations of text"}
        mock_post.return_value = mock_response

        with patch.dict('os.environ', {'OLLAMA_MODEL': 'test-model'}):
            result = prompt_ollama_http("Explain vector embeddings")

            mock_post.assert_called_once()
            call_args = mock_post.call_args[1]
            assert call_args['json']['model'] == 'test-model'
            assert call_args['json']['prompt'] == "Explain vector embeddings"
            assert result == "Vector embeddings are numerical representations of text"