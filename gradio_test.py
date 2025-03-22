from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    DuckDuckGoSearchTool,
    VisitWebpageTool,
    HfApiModel, 
    GradioUI
)
# Optional OpenTelemetry instrumentation
try:
    from openinference.instrumentation.smolagents import SmolagentsInstrumentor
    from phoenix.otel import register
    register()
    SmolagentsInstrumentor().instrument()
    print("OpenTelemetry instrumentation enabled")
except ImportError:
    print("OpenTelemetry instrumentation not available (phoenix and openinference packages required)")

# Import your custom SuperGradioUI
from smolagents_extender import SuperGradioUI

# Set up your model
model = HfApiModel()

# Create your search agent
search_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    model=model,
    name="search_agent",
    description="This is an agent that can do web search.",
    max_steps=10,
    verbosity_level=1
)

# Create a manager agent that can create more agents
manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[search_agent],
    name="manager_agent",
    description="This agent can solve problems using code and delegate to other agents when needed.",
    max_steps=10,
    verbosity_level=1
)

# Make sure uploads directory exists
import os
os.makedirs("./uploads", exist_ok=True)

# Launch the enhanced UI with the manager agent
# ui = SuperGradioUI(manager_agent, file_upload_folder="./uploads")
ui = GradioUI(manager_agent, file_upload_folder="./uploads")
ui.launch(share=True)  # Set share=False if you don't want to create a public link