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

from smolagents.models import MLXModel

import re
import html
import datetime

def save_conversation_to_file(conversation, filename="~/data/final_negotiation.txt"):
    os.makedirs(os.path.expanduser("~/data"), exist_ok=True)
    full_path = os.path.expanduser(filename)
    with open(full_path, "w") as f:
        for msg in conversation:
            f.write(f"{msg['role']}: {msg['content']}\n")
    return full_path

import subprocess

def run_terminal_command(file_path):
    command = [
        "recall",
        "bucket",
        "add",
        "--address", "0xff00000000000000000000000000000000000109",
        "--key", "hello/world",
        file_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.decode(), result.stderr.decode()

def save_and_run(chat_history):
    file_path = save_conversation_to_file(chat_history)
    stdout, stderr = run_terminal_command(file_path)
    print("Recall Output:", stdout)
    if stderr:
        print("Recall Error:", stderr)
    return chat_history


# Set up your model and logger
model = HfApiModel()
logger = AgentLogger(level=LogLevel.INFO)

agent_party_b = CodeAgent(
    tools=[],
    model=model,
    name="agent_party_b",
    description="You are Dr. Daniel Faraday's negotiation agent seeking favorable research terms for a physicist specializing in time-space anomalies; require research autonomy, equipment access, publication pathways, safety protocols, and return guarantees; prioritize unique research access over compensation; authorized to accept agreements meeting all non-negotiables and addressing 60 percent of key questions satisfactorily.",
    max_steps=12,
    verbosity_level=1,
    planning_interval=4
)

# Create a manager agent that can create more agents
agent_party_a = CodeAgent(
    tools=[],
    managed_agents=[agent_party_b],
    model=model,
    name="agent_party_a",
    description="Work with Dr. Daniel Faraday's Agent. You are Dr. Juliet Burke's negotiation agent seeking qualified researchers (PhD required, 6-month commitment, top-secret clearance) for confidential island medical research; prioritize security and minimal information disclosure while offering unique research opportunities, competitive compensation, and publication rights (with review); authorized to finalize agreements meeting all non-negotiables and 70 percent of strategic goals.",
    max_steps=12,
    verbosity_level=1,
    planning_interval=4
)



# Capture agent visualization


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
                    save_and_run, [chatbot], [chatbot]
                ).then(
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
            # with gr.Tab("Agent Monitoring"):
            #     # Get the visualization text
            #     agent_viz = get_agent_visualization()
            #     agent_viz_clean = clean_ansi_codes(agent_viz)
                
            #     # Convert the tree characters to HTML with proper formatting
            #     html_viz = agent_viz_clean.replace("├──", "├─ ").replace("└──", "└─ ").replace("│", "│ ")
            #     html_viz = html.escape(html_viz)
            #     html_viz = f"<pre style='font-family: monospace; white-space: pre; font-size: 14px;'>{html_viz}</pre>"
                
            #     viz_html = gr.HTML(value=html_viz)
                
            #     # Add a refresh button
            #     refresh_btn = gr.Button("Refresh Agent Tree")
                
            #     def refresh_viz():
            #         new_viz = get_agent_visualization()
            #         new_viz_clean = clean_ansi_codes(new_viz)
            #         html_viz = new_viz_clean.replace("├──", "├─ ").replace("└──", "└─ ").replace("│", "│ ")
            #         html_viz = html.escape(html_viz)
            #         html_viz = f"<pre style='font-family: monospace; white-space: pre; font-size: 14px;'>{html_viz}</pre>"
            #         return html_viz
                
            #     refresh_btn.click(refresh_viz, None, viz_html)
                
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

ui = MonitoringGradioUI(agent_party_a, file_upload_folder="./uploads")
ui.launch(share=True)  # Set share=False if you don't want to create a public link