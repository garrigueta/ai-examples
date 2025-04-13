""" A Python script to interact with MSFS and OpenAI GPT """
# Custom Libraries
#from lib.msfs import MSFSWrapper
from lib.modules.speech import SpeechToText
from lib.modules.audio import Audio
from lib.modules.ai import AiWrapper


class FlightSimAi:
    """A class to interact with MSFS and OpenAI GPT-4."""
    def __init__(self):
        """ Initialize the class """
        # self.msfs = MSFSWrapper()
        self.ai = AiWrapper()
        self.audio = Audio()
        self.speech = SpeechToText()

        self.start()

    def start(self):
        """ Start the audio stream and recognize speech """
        # Start the data loop
        # self.msfs.start_data_loop()
        # Initialize the AI
        self.ai.initi_ai()
        # Initialize the audio
        self.audio.init_audio()

        print("Listening...", flush=True)
        while True:
            self.audio.wait_for_audio()
            if self.audio.recognized_text != "":
                # Check for termination keyword first
                if "finalizar" in self.audio.recognized_text.lower():
                    print("Termination keyword detected. Stopping...", flush=True)
                    break
                
                # Set the system content to the current flight data
                #n self.ai.set_system_content(self.msfs.get_flight_data())
                # Get the AI response
                response = self.ai.get_ai_response(self.audio.recognized_text)
                # Speak the response
                self.speech.speech(response)
