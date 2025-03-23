import io
from contextlib import redirect_stdout
import gradio as gr
from smolagents import (
    load_tool,
    CodeAgent,
    ToolCallingAgent,
    VLLMModel,
    DuckDuckGoSearchTool,
    VisitWebpageTool,
    HfApiModel, 
    GradioUI,
    AgentLogger,
    LogLevel
)

from langchain.agents import load_tools

from smolagents.models import MLXModel


import re
import html

# Set up your model and logger
model = HfApiModel()
logger = AgentLogger(level=LogLevel.INFO)

# Import tool from Hub
image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)


# Create your agents
search_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    model=model,
    name="search_agent",
    description="This is an agent that can do web search.",
    max_steps=12,
    verbosity_level=1
)

mentor_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    model=model,
    name="mentor_agent",
    description="This is an agent that creates a persona of a mentor based on the hackathon idea and gives feedback.",
    max_steps=12,
    planning_interval=2,
    verbosity_level=1
)

design_research_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    model=model,
    name="design_research_agent",
    description="Searches online for 3-star reviews and conversations on sites like Reddit as a form of 'user research",
    max_steps=12,
    planning_interval=2,
    verbosity_level=1
)

visual_design_agent = ToolCallingAgent(
    tools=[image_generation_tool],
    model=MLXModel(model_id="HuggingFaceTB/SmolLM-135M-Instruct"),
    name="visual_design_agent",
    description="Creates, reviews, and iterates on front-end mockups of product ideas",
    max_steps=12,
    planning_interval=2,
    verbosity_level=1
)

payment_agent = ToolCallingAgent(
    tools=[wallet_tool],
    model=model,
    name="payment_agent",
    description="Sends eth to other wallets based on contract negotiations",
    max_steps=12,
    planning_interval=1,
    verbosity_level=1
)

# Create a manager agent that can create more agents
manager_agent = CodeAgent(
    tools=[],
    model=model,
    #managed_agents=[search_agent, mentor_agent, design_research_agent, visual_design_agent],
    managed_agents=[payment_agent],
    name="manager_agent",
    description="This agent can solve problems using code and delegate to other agents when needed.",
    max_steps=12,
    verbosity_level=1,
    planning_interval=4
)

# Capture agent visualization
def get_agent_visualization():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        logger.visualize_agent_tree(manager_agent)
    return buffer.getvalue()

def clean_ansi_codes(text):
    """Remove ANSI color codes for clean display"""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

# Create a custom Gradio UI that extends the GradioUI class
class MonitoringGradioUI(GradioUI):
    def __init__(self, agent, file_upload_folder=None):
        super().__init__(agent, file_upload_folder)
        self.name = "Agent Interface with Monitoring"
        
    def launch(self, share=True, **kwargs):
        with gr.Blocks(theme="ocean", fill_height=True) as demo:
            # Add session state to store session-specific data
            session_state = gr.State({})
            stored_messages = gr.State([])
            file_uploads_log = gr.State([])
            
            with gr.Tab("Chat"):
                with gr.Sidebar():
                    gr.Markdown(
                        f"# {self.name.replace('_', ' ').capitalize()}"
                        "\n> This web UI allows you to interact with a `smolagents` agent that can use tools and execute steps to complete tasks."
                        + (f"\n\n**Agent description:**\n{self.description}" if self.description else "")
                    )

                    with gr.Group():
                        gr.Markdown("**Your request**", container=True)
                        text_input = gr.Textbox(
                            lines=3,
                            label="Chat Message",
                            container=False,
                            placeholder="Enter your prompt here and press Shift+Enter or press the button",
                        )
                        submit_btn = gr.Button("Submit", variant="primary")

                    # If an upload folder is provided, enable the upload feature
                    if self.file_upload_folder is not None:
                        upload_file = gr.File(label="Upload a file")
                        upload_status = gr.Textbox(label="Upload Status", interactive=False, visible=False)
                        upload_file.change(
                            self.upload_file,
                            [upload_file, file_uploads_log],
                            [upload_status, file_uploads_log],
                        )

                    gr.HTML("<br><br><h4><center>Powered by:</center></h4>")
                    with gr.Row():
                        gr.HTML("""<div style="display: flex; align-items: center; gap: 8px; font-family: system-ui, -apple-system, sans-serif;">
                <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png" style="width: 32px; height: 32px; object-fit: contain;" alt="logo">
                <a target="_blank" href="https://github.com/huggingface/smolagents"><b>huggingface/smolagents</b></a>
                </div>""")

                # Main chat interface
                chatbot = gr.Chatbot(
                    label="Agent",
                    type="messages",
                    avatar_images=(
                        None,
                        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
                    ),
                    resizeable=True,
                    scale=1,
                )
                
                # Set up event handlers
                text_input.submit(
                    self.log_user_message,
                    [text_input, file_uploads_log],
                    [stored_messages, text_input, submit_btn],
                ).then(self.interact_with_agent, [stored_messages, chatbot, session_state], [chatbot]).then(
                    lambda: (
                        gr.Textbox(
                            interactive=True, placeholder="Enter your prompt here and press Shift+Enter or the button"
                        ),
                        gr.Button(interactive=True),
                    ),
                    None,
                    [text_input, submit_btn],
                )

                submit_btn.click(
                    self.log_user_message,
                    [text_input, file_uploads_log],
                    [stored_messages, text_input, submit_btn],
                ).then(self.interact_with_agent, [stored_messages, chatbot, session_state], [chatbot]).then(
                    lambda: (
                        gr.Textbox(
                            interactive=True, placeholder="Enter your prompt here and press Shift+Enter or the button"
                        ),
                        gr.Button(interactive=True),
                    ),
                    None,
                    [text_input, submit_btn],
                )
            
            # Add the Monitoring tab
            with gr.Tab("Agent Monitoring"):
                # Get the visualization text
                agent_viz = get_agent_visualization()
                agent_viz_clean = clean_ansi_codes(agent_viz)
                
                # Convert the tree characters to HTML with proper formatting
                html_viz = agent_viz_clean.replace("├──", "├─ ").replace("└──", "└─ ").replace("│", "│ ")
                html_viz = html.escape(html_viz)
                html_viz = f"<pre style='font-family: monospace; white-space: pre; font-size: 14px;'>{html_viz}</pre>"
                
                viz_html = gr.HTML(value=html_viz)
                
                # Add a refresh button
                refresh_btn = gr.Button("Refresh Agent Tree")
                
                def refresh_viz():
                    new_viz = get_agent_visualization()
                    new_viz_clean = clean_ansi_codes(new_viz)
                    html_viz = new_viz_clean.replace("├──", "├─ ").replace("└──", "└─ ").replace("│", "│ ")
                    html_viz = html.escape(html_viz)
                    html_viz = f"<pre style='font-family: monospace; white-space: pre; font-size: 14px;'>{html_viz}</pre>"
                    return html_viz
                
                refresh_btn.click(refresh_viz, None, viz_html)
                
                # Add some explanatory text
                gr.Markdown("""
                ### Monitoring Information
                
                This tab shows the structure of your agent, including:
                - Hierarchical organization of agents
                - Available tools for each agent
                - Agent configurations
                
                Use the refresh button to update the visualization if you modify your agent structure.
                """)

        demo.launch(debug=True, share=share, **kwargs)

# Create and launch your UI
# Make sure uploads directory exists
import os
os.makedirs("./uploads", exist_ok=True)

ui = MonitoringGradioUI(manager_agent, file_upload_folder="./uploads")
ui.launch(share=True)  # Set share=False if you don't want to create a public link