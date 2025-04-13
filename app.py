from lib.sim import FlightSimAi

try:
    MOD = FlightSimAi()
except Exception as e:
    print(f"Error: {e}")
    exit(1)
