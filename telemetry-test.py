from phoenix.otel import register
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    DuckDuckGoSearchTool,
    VisitWebpageTool,
    HfApiModel,
)

register()
SmolagentsInstrumentor().instrument()

model = HfApiModel()

search_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    model=model,
    name="search_agent",
    description="This is an agent that can do web search.",
)

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[search_agent],
)
manager_agent.run(
    "If the US keeps its 2024 growth rate, how many years will it take for the GDP to double?"
)