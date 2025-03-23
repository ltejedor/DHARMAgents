# wallet/wallet_tool.py

from wallet.create_agent import create_agent
from langchain_core.messages import HumanMessage

# Initialize your langgraph wallet agent once.
agent_executor, config = create_agent()

def wallet_agent_tool(input_text: str) -> str:
    """
    Uses the langgraph wallet agent to process wallet-related commands.
    For example: "Send 0.1 ETH to 0xABCDEF..."
    """
    messages = [HumanMessage(content=input_text)]
    output = ""
    # Run the langgraph agent; collect output from its stream.
    for chunk in agent_executor.stream({"messages": messages}, config):
        if "agent" in chunk:
            output += chunk["agent"]["messages"][0].content
        elif "tools" in chunk:
            output += chunk["tools"]["messages"][0].content
    return output
