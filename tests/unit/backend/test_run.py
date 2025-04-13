import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from lib.backend.run import app, CommandRequest, CommandResponse


class TestBackendRun:
    """Tests for the backend run API."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    def test_command_request_model(self):
        """Test the CommandRequest model."""
        cmd = CommandRequest(command="ls -la")
        assert cmd.command == "ls -la"

    def test_command_response_model(self):
        """Test the CommandResponse model."""
        resp = CommandResponse(stdout="output", stderr="error", exit_code=0)
        assert resp.stdout == "output"
        assert resp.stderr == "error"
        assert resp.exit_code == 0

    @patch('subprocess.run')
    def test_run_command_success(self, mock_run, client):
        """Test running a command successfully."""
        # Setup mock subprocess result
        mock_result = MagicMock()
        mock_result.stdout = "command output"
        mock_result.stderr = ""
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        # Make API request
        response = client.post("/run", json={"command": "echo hello"})
        
        # Verify the subprocess was called correctly
        mock_run.assert_called_once_with(
            "echo hello",
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
            check=False
        )
        
        # Check the API response
        assert response.status_code == 200
        json_response = response.json()
        assert json_response["stdout"] == "command output"
        assert json_response["stderr"] == ""
        assert json_response["exit_code"] == 0

    @patch('subprocess.run')
    def test_run_command_error(self, mock_run, client):
        """Test running a command that returns an error."""
        # Setup mock subprocess result
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = "command not found"
        mock_result.returncode = 127
        mock_run.return_value = mock_result

        # Make API request
        response = client.post("/run", json={"command": "invalid_command"})
        
        # Check the API response
        assert response.status_code == 200  # API still returns 200 even for command errors
        json_response = response.json()
        assert json_response["stdout"] == ""
        assert json_response["stderr"] == "command not found"
        assert json_response["exit_code"] == 127

    @patch('subprocess.run')
    def test_run_command_timeout(self, mock_run, client):
        """Test running a command that times out."""
        # Setup mock subprocess to raise a timeout exception
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="sleep 100", timeout=10)

        # Make API request
        response = client.post("/run", json={"command": "sleep 100"})
        
        # Check the API response
        assert response.status_code == 408  # Request Timeout
        json_response = response.json()
        assert json_response["detail"] == "Command timed out"