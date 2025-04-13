from unittest.mock import patch, MagicMock
import json
from lib.modules.msfs import MSFSWrapper


class TestMSFSWrapper:
    """Tests for the MSFSWrapper class."""

    @patch('lib.modules.msfs.SimConnect')
    @patch('lib.modules.msfs.AircraftRequests')
    @patch('lib.modules.msfs.AircraftEvents')
    def test_initialization(self, mock_events, mock_requests, mock_simconnect):
        """Test that MSFSWrapper initializes correctly."""
        # Setup mocks
        mock_sim_instance = MagicMock()
        mock_simconnect.return_value = mock_sim_instance
        
        mock_req_instance = MagicMock()
        mock_requests.return_value = mock_req_instance
        
        mock_events_instance = MagicMock()
        mock_events.return_value = mock_events_instance
        
        # Initialize MSFSWrapper
        msfs = MSFSWrapper()
        
        # Verify SimConnect was initialized
        mock_simconnect.assert_called_once()
        mock_requests.assert_called_once_with(mock_sim_instance, _time=0)
        mock_events.assert_called_once_with(mock_sim_instance)
        
        # Check that the wrapper has the right attributes
        assert msfs.simconnect == mock_sim_instance
        assert msfs.requests == mock_req_instance
        assert msfs.events == mock_events_instance
        assert msfs.running is False
        assert msfs.data == {}
        assert hasattr(msfs, 'data_lock')

    @patch('lib.modules.msfs.SimConnect')
    @patch('lib.modules.msfs.AircraftRequests')
    @patch('lib.modules.msfs.AircraftEvents')
    def test_fetch_flight_data(self, mock_events, mock_requests, mock_simconnect):
        """Test fetching flight data."""
        # Setup mocks
        mock_sim_instance = MagicMock()
        mock_simconnect.return_value = mock_sim_instance
        
        mock_req_instance = MagicMock()
        # Set up the get method to return test values
        mock_req_instance.get.return_value = 10000
        mock_requests.return_value = mock_req_instance
        
        # Initialize MSFSWrapper
        msfs = MSFSWrapper()
        
        # Fetch flight data
        msfs.fetch_flight_data()
        
        # Verify data was fetched correctly
        assert len(msfs.data) > 0
        assert msfs.data["altitud de vuelo"] == 10000
        assert msfs.data["latitud actual"] == 10000
        assert msfs.data["airspeed(velocidad)"] == 10000
        
        # Verify the correct parameters were requested
        mock_req_instance.get.assert_any_call("PLANE_ALTITUDE")
        mock_req_instance.get.assert_any_call("PLANE_LATITUDE")
        mock_req_instance.get.assert_any_call("PLANE_LONGITUDE")

    @patch('lib.modules.msfs.SimConnect')
    @patch('lib.modules.msfs.AircraftRequests')
    @patch('lib.modules.msfs.AircraftEvents')
    @patch('threading.Thread')
    def test_start_data_loop(self, mock_thread, mock_events, mock_requests, mock_simconnect):
        """Test starting the data loop thread."""
        # Setup mocks
        mock_sim_instance = MagicMock()
        mock_simconnect.return_value = mock_sim_instance
        
        mock_req_instance = MagicMock()
        mock_requests.return_value = mock_req_instance
        
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance
        
        # Initialize MSFSWrapper
        msfs = MSFSWrapper()
        
        # Start the data loop
        msfs.start_data_loop(interval=2)
        
        # Verify the thread was started correctly
        assert msfs.running is True
        mock_thread.assert_called_once()
        call_args = mock_thread.call_args[1]
        assert call_args["target"] == msfs._data_loop
        assert call_args["args"] == (2,)
        assert call_args["daemon"] is True
        mock_thread_instance.start.assert_called_once()

    @patch('lib.modules.msfs.SimConnect')
    @patch('lib.modules.msfs.AircraftRequests')
    @patch('lib.modules.msfs.AircraftEvents')
    @patch('lib.modules.msfs.json.dumps')
    def test_get_flight_data(self, mock_dumps, mock_events, mock_requests, mock_simconnect):
        """Test getting flight data as JSON."""
        # Setup mocks
        mock_sim_instance = MagicMock()
        mock_simconnect.return_value = mock_sim_instance
        
        mock_req_instance = MagicMock()
        mock_requests.return_value = mock_req_instance
        
        mock_dumps.return_value = '{"altitude": 10000, "speed": 250}'
        
        # Initialize MSFSWrapper and set some data
        msfs = MSFSWrapper()
        msfs.data = {"altitude": 10000, "speed": 250}
        
        # Get flight data
        data_json = msfs.get_flight_data()
        
        # Verify JSON was returned correctly
        mock_dumps.assert_called_once_with(msfs.data.copy())
        assert data_json == '{"altitude": 10000, "speed": 250}'

    @patch('lib.modules.msfs.SimConnect')
    @patch('lib.modules.msfs.AircraftRequests')
    @patch('lib.modules.msfs.AircraftEvents')
    def test_trigger_event(self, mock_events, mock_requests, mock_simconnect):
        """Test triggering an MSFS event."""
        # Setup mocks
        mock_sim_instance = MagicMock()
        mock_simconnect.return_value = mock_sim_instance
        
        mock_req_instance = MagicMock()
        mock_requests.return_value = mock_req_instance
        
        mock_events_instance = MagicMock()
        mock_event = MagicMock()
        mock_events_instance.find.return_value = mock_event
        mock_events.return_value = mock_events_instance
        
        # Initialize MSFSWrapper
        msfs = MSFSWrapper()
        
        # Trigger an event
        msfs.trigger_event("AP_MASTER")
        
        # Verify the event was triggered
        mock_events_instance.find.assert_called_once_with("AP_MASTER")
        mock_event.assert_called_once()