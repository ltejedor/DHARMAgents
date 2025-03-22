#!/usr/bin/env python3
"""
SmolagentWeb - A Gradio interface for interacting with smolagents
This script creates a Gradio web interface that allows users to:
1. Choose between different agent types (CodeAgent, ToolCallingAgent)
2. Configure agent settings like model type and model ID
3. Add tools to the agent
4. Upload files for the agent to use
5. Chat with the agent and see its reasoning steps
"""
from phoenix.otel import register
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import gradio as gr
from huggingface_hub import login
from smolagents import (
    CodeAgent, 
    ToolCallingAgent,
    HfApiModel, 
    LiteLLMModel,
    DuckDuckGoSearchTool,
    tool
)
from smolagents.gradio_ui import GradioUI

register()
SmolagentsInstrumentor().instrument()

# Load environment variables from .env file
load_dotenv()

# Constants
UPLOAD_FOLDER = "uploaded_files"
AVAILABLE_MODELS = {
    "HfApiModel": [
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "meta-llama/Meta-Llama-3-70B-Instruct",
        "mistralai/Mistral-Large-2-0",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
    ],
    "LiteLLMModel": [
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
    ],
}

AVAILABLE_TOOLS = {
    "web_search": DuckDuckGoSearchTool,
}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit. Should be a valid HTTP or HTTPS URL.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    import re
    import requests
    from markdownify import markdownify
    from requests.exceptions import RequestException
    
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


def create_agent(
    agent_type: str,
    model_type: str,
    model_id: str,
    selected_tools: List[str],
    additional_imports: str = "",
    max_steps: int = 10,
) -> Any:
    """Create and configure an agent based on user selections."""
    # Create the model
    if model_type == "HfApiModel":
        # Get token from environment variable
        token = os.getenv("HF_TOKEN")
        if token:
            login(token=token, add_to_git_credential=False)
            print(f"Logged in to HuggingFace with token from .env file")
        model = HfApiModel(model_id=model_id)
    elif model_type == "LiteLLMModel":
        model = LiteLLMModel(model=model_id)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create tools list
    tools = []
    if "web_search" in selected_tools:
        tools.append(DuckDuckGoSearchTool())
    if "visit_webpage" in selected_tools:
        tools.append(visit_webpage)
    
    # Parse additional imports
    imports = []
    if additional_imports:
        imports = [imp.strip() for imp in additional_imports.split()]
    
    # Create the agent
    if agent_type == "CodeAgent":
        agent = CodeAgent(
            tools=tools,
            model=model,
            max_steps=max_steps,
            additional_authorized_imports=imports if imports else None,
        )
    elif agent_type == "ToolCallingAgent":
        agent = ToolCallingAgent(
            tools=tools,
            model=model,
            max_steps=max_steps,
        )
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    
    return agent


def launch_gradio_interface():
    """Launch the Gradio interface for interacting with smolagents."""
    with gr.Blocks(title="SmolagentWeb", theme="soft") as demo:
        gr.Markdown("# 🤖 SmolagentWeb - Interact with AI Agents")
        
        with gr.Tabs():
            with gr.TabItem("Configure Agent"):
                with gr.Row():
                    with gr.Column():
                        agent_type = gr.Radio(
                            ["CodeAgent", "ToolCallingAgent"],
                            label="Agent Type",
                            value="CodeAgent",
                            info="CodeAgent uses Python code for reasoning. ToolCallingAgent uses JSON."
                        )
                        
                        model_type = gr.Radio(
                            list(AVAILABLE_MODELS.keys()),
                            label="Model Provider",
                            value="HfApiModel",
                            info="Choose the AI model provider"
                        )
                        
                        model_id = gr.Dropdown(
                            choices=AVAILABLE_MODELS["HfApiModel"],
                            value="Qwen/Qwen2.5-Coder-32B-Instruct",
                            label="Model",
                            info="Select the specific AI model to use"
                        )
                        
                        tool_checkboxes = gr.CheckboxGroup(
                            ["web_search", "visit_webpage"],
                            label="Tools",
                            value=["web_search"],
                            info="Select tools for the agent to use"
                        )
                        
                        max_steps = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=10,
                            step=1,
                            label="Max Steps",
                            info="Maximum number of steps the agent can take"
                        )
                        
                        additional_imports = gr.Textbox(
                            label="Additional Imports (space-separated)",
                            placeholder="e.g., numpy pandas matplotlib",
                            info="For CodeAgent: additional Python modules that can be imported"
                        )
                    
                    with gr.Column():
                        gr.Markdown("""
                        ### Configuration Help
                        
                        **Agent Types:**
                        - **CodeAgent**: Uses Python code for reasoning and actions. Better for complex tasks.
                        - **ToolCallingAgent**: Uses JSON for tool calls. Simpler but less powerful.
                        
                        **Model Providers:**
                        - **HfApiModel**: Models hosted on HuggingFace. Requires a HuggingFace token in your .env file.
                        - **LiteLLMModel**: Access to models from OpenAI, Anthropic, etc. Requires API keys.
                        
                        **Tools:**
                        - **web_search**: Perform web searches using DuckDuckGo
                        - **visit_webpage**: Visit a URL and extract its content
                        
                        **Additional Imports:** 
                        For CodeAgent, specify Python modules that the agent can import (space-separated).
                        """)
                
                create_button = gr.Button("Create Agent", variant="primary")
                status_message = gr.Markdown("")
                
            with gr.TabItem("Chat with Agent"):
                gr.Markdown("Agent not created yet. Please go to the 'Configure Agent' tab first.")
                chat_interface = gr.Chatbot(label="Chat with Agent")
                chat_placeholder = gr.Markdown("Configure and create an agent first!")
        
        def update_model_choices(model_type):
            return gr.Dropdown(choices=AVAILABLE_MODELS[model_type])
        
        model_type.change(update_model_choices, inputs=model_type, outputs=model_id)
        
        def on_create_agent(agent_type, model_type, model_id, tool_checkboxes, additional_imports, max_steps):
            try:
                agent = create_agent(
                    agent_type=agent_type,
                    model_type=model_type,
                    model_id=model_id,
                    selected_tools=tool_checkboxes,
                    additional_imports=additional_imports,
                    max_steps=max_steps,
                )
                
                # Create a fresh Gradio UI tab with the agent
                ui = GradioUI(agent=agent, file_upload_folder=UPLOAD_FOLDER)
                
                return gr.Markdown(f"✅ Agent created successfully! Go to the 'Chat with Agent' tab to start interacting.")
            except Exception as e:
                return gr.Markdown(f"❌ Error creating agent: {str(e)}")
        
        create_button.click(
            fn=on_create_agent,
            inputs=[agent_type, model_type, model_id, tool_checkboxes, additional_imports, max_steps],
            outputs=status_message,
        )
        
    demo.launch(share=True)


def main():
    parser = argparse.ArgumentParser(description="Launch a Gradio interface for smolagents")
    parser.add_argument("--no-share", action="store_true", help="Don't create a public link")
    args = parser.parse_args()
    
    # Launch the Gradio interface
    launch_gradio_interface()


if __name__ == "__main__":
    main()