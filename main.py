#!/usr/bin/env python3
"""
Multi-Agent System for Web Browsing and Research

This script implements a multi-agent system with a Manager Agent coordinating a Web Search Agent.
The system can perform web searches, visit webpages, and execute code to solve complex problems.

Architecture:
              +----------------+
              | Manager agent  |
              +----------------+
                       |
        _______________|______________
       |                              |
Code Interpreter            +------------------+
    tool                    | Web Search agent |
                            +------------------+
                               |            |
                        Web Search tool     |
                                   Visit webpage tool
"""

import re
import os
import requests
from markdownify import markdownify
from requests.exceptions import RequestException
from huggingface_hub import login
from dotenv import load_dotenv
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    HfApiModel,
    DuckDuckGoSearchTool,
    tool
)

@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Convert the HTML content to Markdown
        markdown_content = markdownify(response.text).strip()

        # Remove multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        return markdown_content

    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def setup_multi_agent_system():
    """Set up the multi-agent system with a manager agent and a web search agent."""
    
    # Using Qwen's model for all agents
    #model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
    model_id = "Qwen/QwQ-32B"
    model = HfApiModel(model_id=model_id)
    
    # Create the web search agent - visit_webpage is already a Tool instance due to @tool decorator
    web_agent = ToolCallingAgent(
        tools=[DuckDuckGoSearchTool(), visit_webpage],
        model=model,
        max_steps=200,
        name="web_search_agent",
        description="Runs web searches for you."
    )
    
    # Create the manager agent
    manager_agent = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[web_agent],
        additional_authorized_imports=["time", "numpy", "pandas"],
    )
    
    return manager_agent

def run_query(manager_agent, query):
    """Run a query through the manager agent and print the response."""
    print(f"Query: {query}\n")
    print("Processing...\n")
    answer = manager_agent.run(query)
    print("Answer:")
    print("-" * 80)
    print(answer)
    print("-" * 80)
    return answer

def main():
    """Main function to run the multi-agent system."""
    
    # Install dependencies if needed
    # Uncomment the following line if you need to install dependencies
    # import subprocess
    # subprocess.run(["pip", "install", "markdownify", "duckduckgo-search", "smolagents", "python-dotenv", "--upgrade", "-q"])
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get token from environment variable
    token = os.getenv("HF_TOKEN")
    
    # Log in to Hugging Face using token from .env
    if token:
        login(token=token, add_to_git_credential=False)
        print("Successfully logged in to Hugging Face using token from .env file")
    else:
        print("No HF_TOKEN found in .env file. Please log in manually:")
        login()
    
    # Set up the multi-agent system
    print("Setting up multi-agent system...")
    manager_agent = setup_multi_agent_system()
    
    # Example query
    default_query = "If LLM training continues to scale up at the current rhythm until 2030, what would be the electric power in GW required to power the biggest training runs by 2030? What would that correspond to, compared to some countries? Please provide a source for any numbers used."
    
    # Get user query or use default
    print("\nEnter your query (or press Enter to use the default query):")
    user_query = input().strip()
    query = user_query if user_query else default_query
    
    # Run the query
    run_query(manager_agent, query)

if __name__ == "__main__":
    main()