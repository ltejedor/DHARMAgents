import asyncio
import serial_asyncio
from typing import Any, Dict, Optional
from smolagents.tools import Tool

class SonarDistanceSensorTool(Tool):
    """
    A SmolaAgents tool for reading a distance measurement from the HC-SR04 Sonar Distance Sensor.
    Assumes the sensor is connected via a microcontroller to a serial port.
    """
    
    def __init__(
        self, 
        name: str = "sonar_distance_sensor", 
        description: str = None
    ):
        """
        Initialize the Sonar Distance Sensor Tool.
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
        """
        if description is None:
            description = (
                "Reads a distance measurement from the HC-SR04 sonar sensor. "
                "Provide the serial port (e.g., '/dev/ttyUSB0' or 'COM3') "
                "and optionally the baud rate (default is 115200)."
            )
            
        super().__init__(name=name, description=description)
    
    def _run(self, port: str, baud: int = 115200) -> str:
        """
        Connects to the sensor over the provided serial port and reads one line of data.
        
        Args:
            port: The serial port for the sensor (e.g., '/dev/ttyUSB0' on Linux or 'COM3' on Windows)
            baud: The baud rate for the serial connection (default: 115200)
            
        Returns:
            The sensor reading as a string
        """
        try:
            # Run the async function using asyncio
            return asyncio.run(self._async_read_sensor(port, baud))
        except Exception as e:
            return f"Error reading from sensor: {str(e)}"
    
    async def _async_read_sensor(self, port: str, baud: int = 115200) -> str:
        """
        Asynchronous function to read from the sensor.
        
        Args:
            port: The serial port for the sensor
            baud: The baud rate for the serial connection
            
        Returns:
            The sensor reading as a string
        """
        try:
            # Open an asynchronous serial connection using pyserial-asyncio
            reader, writer = await serial_asyncio.open_serial_connection(url=port, baudrate=baud)
        except Exception as e:
            return f"Error opening serial port: {str(e)}"

        try:
            # Wait up to 5 seconds for a line of sensor data
            try:
                line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            except asyncio.TimeoutError:
                return "Timeout while reading from sensor."

            # Decode the line (assuming UTF-8 encoding)
            reading = line.decode("utf-8", errors="replace").strip()
            
            # Try to parse the reading as a number if possible
            try:
                distance_cm = float(reading)
                return f"Distance: {distance_cm} cm"
            except ValueError:
                # If it can't be parsed as a number, return the raw reading
                return f"Sensor reading: {reading}"
                
        finally:
            # Always close the writer to clean up resources
            writer.close()
            await writer.wait_closed()


# Example usage
if __name__ == "__main__":
    from smolagents import ToolCallingAgent, HfApiModel
    
    # Create the sonar sensor tool
    sensor_tool = SonarDistanceSensorTool()
    
    # Create a model - use your preferred model
    model = HfApiModel()
    
    # Create an agent with the sensor tool
    sensor_agent = ToolCallingAgent(
        tools=[sensor_tool],
        model=model,
        name="sensor_agent",
        description="This is an agent that can read distance measurements from a sonar sensor.",
        max_steps=3,
        verbosity_level=1
    )
    
    # Test the agent (note: this will only work if you have a sensor connected)
    try:
        # Assuming you have a sonar sensor connected to COM3 or your specific port
        result = sensor_agent.run(
            "What is the current distance reading from the sonar sensor connected to COM3?"
        )
        print(result)
    except Exception as e:
        print(f"Error testing the sensor agent: {str(e)}")
        
    # Direct tool usage example
    try:
        print("Direct tool reading:", sensor_tool._run(port="COM3"))
    except Exception as e:
        print(f"Error directly using the tool: {str(e)}")