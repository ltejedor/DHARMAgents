import os
import time
import json
from typing import Dict, List, Optional, Any, Generator

import gradio as gr

from smolagents.agents import ActionStep, MultiStepAgent
from smolagents.gradio_ui import GradioUI, stream_to_gradio, pull_messages_from_step


class SuperGradioUI(GradioUI):
    """An extended Gradio UI that visualizes the manager agent and its sub-agents with a persistent graph"""

    def __init__(self, agent: MultiStepAgent, file_upload_folder: str | None = None):
        # Don't call super().__init__() yet as we'll override the launch method completely
        self.agent = agent
        self.file_upload_folder = file_upload_folder
        self.name = getattr(agent, "name") or "Agent interface"
        self.description = getattr(agent, "description", None)
        if self.file_upload_folder is not None:
            if not os.path.exists(file_upload_folder):
                os.mkdir(file_upload_folder)
        
        # Agent tracking data
        self.agent_hierarchy = {}  # Store the hierarchy of agents {name: parent_name}
        self.agent_tasks = {}      # Store tasks assigned to each agent {name: task}
        self.agent_statuses = {}   # Store the status of each agent {name: status}
        self.agent_metrics = {}    # Store metrics like tokens, time spent, etc.
        self.agent_start_times = {}  # Track when agents started working
        
        # Initialize with the manager agent
        manager_name = getattr(agent, "name", None) or "manager"
        self.register_agent(manager_name, None, "Manager Agent", "idle")
        
        # Try to discover managed agents from the manager
        if hasattr(agent, "managed_agents") and agent.managed_agents:
            for agent_name, managed_agent in agent.managed_agents.items():
                description = getattr(managed_agent, "description", "Sub-agent")
                self.register_agent(agent_name, manager_name, description, "idle")

    def register_agent(self, agent_id: str, parent_id: str | None, 
                      description: str, status: str = "idle"):
        """Register a new agent in the hierarchy"""
        self.agent_hierarchy[agent_id] = parent_id
        self.agent_tasks[agent_id] = description
        self.agent_statuses[agent_id] = status
        self.agent_metrics[agent_id] = {
            "steps": 0,
            "tokens": 0,
            "tool_calls": 0,
            "time_spent": 0
        }
        
    def update_agent_status(self, agent_id: str, status: str, task: str = None):
        """Update the status of an agent"""
        if agent_id in self.agent_statuses:
            old_status = self.agent_statuses[agent_id]
            self.agent_statuses[agent_id] = status
            
            # Track timing when agent becomes active or completes
            if status == "active" and old_status != "active":
                self.agent_start_times[agent_id] = time.time()
            elif status == "completed" and old_status == "active" and agent_id in self.agent_start_times:
                self.agent_metrics[agent_id]["time_spent"] += time.time() - self.agent_start_times[agent_id]
                
            if task:
                self.agent_tasks[agent_id] = task

    def update_agent_metrics(self, agent_id: str, step_log: ActionStep):
        """Update metrics for an agent based on a step log"""
        if agent_id in self.agent_metrics:
            # Increment steps
            self.agent_metrics[agent_id]["steps"] += 1
            
            # Count tokens if available
            if hasattr(step_log, "input_token_count") and step_log.input_token_count:
                self.agent_metrics[agent_id]["tokens"] += step_log.input_token_count
            if hasattr(step_log, "output_token_count") and step_log.output_token_count:
                self.agent_metrics[agent_id]["tokens"] += step_log.output_token_count
                
            # Count tool calls
            if hasattr(step_log, "tool_calls") and step_log.tool_calls:
                self.agent_metrics[agent_id]["tool_calls"] += len(step_log.tool_calls)

    def track_agent_creation(self, step_log: ActionStep):
        """Track agent creation and calls from steps"""
        # Figure out which agent is active
        active_agent = None
        for agent_id, status in self.agent_statuses.items():
            if status == "active":
                active_agent = agent_id
                break
        
        # Update metrics for the active agent
        if active_agent:
            self.update_agent_metrics(active_agent, step_log)
        
        # Only process if step_log has model_output
        if not hasattr(step_log, "model_output") or not step_log.model_output:
            return
            
        # Try to detect if a managed agent is being called
        model_output = step_log.model_output
        
        # Method 1: Look for Python code calling managed agents
        if ".run(" in model_output:
            # Extract agent name from code like: search_agent.run("query")
            import re
            agent_calls = re.findall(r'(\w+)\.run\([\'"]([^\'"]+)[\'"]', model_output)
            for agent_name, task in agent_calls:
                if agent_name in self.agent_hierarchy:
                    self.update_agent_status(agent_name, "active", task)
        
        # Method 2: Look for tool calls that might be managed agent calls
        if hasattr(step_log, "tool_calls") and step_log.tool_calls:
            for tool_call in step_log.tool_calls:
                # Check if tool name matches a known agent
                if tool_call.name in self.agent_hierarchy:
                    args = tool_call.arguments
                    task = args if isinstance(args, str) else str(args)
                    self.update_agent_status(tool_call.name, "active", task)

    def generate_html_visualization(self):
        """Generate an HTML visualization of the agent hierarchy with D3.js"""
        # Create nodes and links for our agent hierarchy
        nodes = []
        links = []
        
        # Create a node for each agent
        for agent_id in self.agent_hierarchy:
            status = self.agent_statuses.get(agent_id, "unknown")
            color = "#4CAF50" if status == "active" else "#8BC34A" if status == "completed" else "#9E9E9E"  # Green, light green, gray
            task = self.agent_tasks.get(agent_id, "No task assigned")
            metrics = self.agent_metrics.get(agent_id, {})
            
            # Check if this is the manager
            is_manager = self.agent_hierarchy[agent_id] is None
            
            nodes.append({
                "id": agent_id,
                "name": agent_id,
                "status": status,
                "color": color,
                "task": task[:50] + "..." if len(task) > 50 else task,
                "steps": metrics.get("steps", 0),
                "tokens": metrics.get("tokens", 0),
                "tool_calls": metrics.get("tool_calls", 0),
                "time_spent": f"{metrics.get('time_spent', 0):.2f}s",
                "is_manager": is_manager
            })
        
        # Create links between agents
        for agent_id, parent_id in self.agent_hierarchy.items():
            if parent_id is not None:
                links.append({
                    "source": parent_id,
                    "target": agent_id
                })
        
        # Convert nodes and links to JSON strings for embedding in JavaScript
        nodes_json = json.dumps(nodes)
        links_json = json.dumps(links)
        
        # Create the HTML for the visualization
        html = f"""
        <div id="agent-visualization" style="width: 100%; height: 400px; border: 1px solid #ccc; border-radius: 8px; overflow: hidden;"></div>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <script>
        (function() {{
            // Clear any existing visualization
            d3.select("#agent-visualization").html("");
            
            // Data for the visualization
            const nodes = {nodes_json};
            const links = {links_json};
            
            // Get the container dimensions
            const container = d3.select("#agent-visualization");
            const width = container.node().getBoundingClientRect().width;
            const height = container.node().getBoundingClientRect().height;
            
            // Create the SVG
            const svg = container.append("svg")
                .attr("width", width)
                .attr("height", height)
                .attr("viewBox", [0, 0, width, height]);
                
            // Create a group for everything
            const g = svg.append("g");
            
            // Create the simulation
            const simulation = d3.forceSimulation(nodes)
                .force("link", d3.forceLink(links).id(function(d) {{ return d.id; }}).distance(100))
                .force("charge", d3.forceManyBody().strength(-300))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .on("tick", ticked);
            
            // Create the links
            const link = g.append("g")
                .selectAll("line")
                .data(links)
                .join("line")
                .attr("stroke", "#999")
                .attr("stroke-opacity", 0.6)
                .attr("stroke-width", 2);
            
            // Create the nodes
            const node = g.append("g")
                .selectAll("g")
                .data(nodes)
                .join("g")
                .call(drag(simulation));
            
            // Add circles for nodes
            node.append("circle")
                .attr("r", function(d) {{ return d.is_manager ? 30 : 25; }})
                .attr("fill", function(d) {{ return d.color; }})
                .attr("stroke", "#fff")
                .attr("stroke-width", 2);
            
            // Add text labels
            node.append("text")
                .attr("text-anchor", "middle")
                .attr("dy", ".3em")
                .attr("fill", "white")
                .attr("font-weight", "bold")
                .text(function(d) {{ return d.name; }});
            
            // Add tooltips
            node.append("title")
                .text(function(d) {{ 
                    return "Agent: " + d.name + 
                           "\\nStatus: " + d.status + 
                           "\\nTask: " + d.task + 
                           "\\nSteps: " + d.steps + 
                           "\\nTokens: " + d.tokens + 
                           "\\nTool Calls: " + d.tool_calls + 
                           "\\nTime: " + d.time_spent;
                }});
            
            // Update positions on each tick
            function ticked() {{
                link
                    .attr("x1", function(d) {{ return d.source.x; }})
                    .attr("y1", function(d) {{ return d.source.y; }})
                    .attr("x2", function(d) {{ return d.target.x; }})
                    .attr("y2", function(d) {{ return d.target.y; }});
                
                node.attr("transform", function(d) {{ return "translate(" + d.x + "," + d.y + ")"; }});
            }}
            
            // Drag behavior
            function drag(simulation) {{
                function dragstarted(event) {{
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    event.subject.fx = event.subject.x;
                    event.subject.fy = event.subject.y;
                }}
                
                function dragged(event) {{
                    event.subject.fx = event.x;
                    event.subject.fy = event.y;
                }}
                
                function dragended(event) {{
                    if (!event.active) simulation.alphaTarget(0);
                    event.subject.fx = null;
                    event.subject.fy = null;
                }}
                
                return d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended);
            }}
            
            // Add zoom capabilities
            const zoom = d3.zoom()
                .on("zoom", function(event) {{
                    g.attr("transform", event.transform);
                }});
            
            svg.call(zoom);
        }})();
        </script>
        """
        
        return html

    def generate_metrics_table(self):
        """Generate an HTML table with agent metrics"""
        rows = []
        
        # Create a row for each agent
        for agent_id in self.agent_hierarchy:
            status = self.agent_statuses.get(agent_id, "unknown")
            status_emoji = "🟢" if status == "active" else "✅" if status == "completed" else "⏸️"
            metrics = self.agent_metrics.get(agent_id, {})
            
            rows.append(f"""
            <tr>
                <td>{status_emoji} {agent_id}</td>
                <td>{status}</td>
                <td>{metrics.get('steps', 0)}</td>
                <td>{metrics.get('tokens', 0)}</td>
                <td>{metrics.get('tool_calls', 0)}</td>
                <td>{metrics.get('time_spent', 0):.2f}s</td>
            </tr>
            """)
        
        table = f"""
        <div style="margin-top: 10px; margin-bottom: 10px;">
            <h3>Agent Metrics</h3>
            <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                <thead>
                    <tr>
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Agent</th>
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Status</th>
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Steps</th>
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Tokens</th>
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Tool Calls</th>
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Time Spent</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        </div>
        """
        
        return table

    def interact_with_agent(self, prompt, messages, session_state, agent_viz_html, agent_metrics_html):
        """Override the interaction method to update the visualization"""
        # Get the agent from the session state or use the default one
        if "agent" not in session_state:
            session_state["agent"] = self.agent

        try:
            # Add the user's message to the chat
            messages.append(gr.ChatMessage(role="user", content=prompt))
            
            # Update the manager agent's status and task
            manager_name = getattr(session_state["agent"], "name", None) or "manager"
            self.update_agent_status(manager_name, "active", prompt)
            
            # Update the visualization immediately
            current_viz_html = self.generate_html_visualization()
            current_metrics_html = self.generate_metrics_table()
            
            yield messages, current_viz_html, current_metrics_html

            # Track the start time of the entire session
            session_start = time.time()

            # Process the agent's run
            for step_log in session_state["agent"].run(prompt, stream=True):
                # Track agent activity if this is an action step
                if isinstance(step_log, ActionStep):
                    self.track_agent_creation(step_log)
                    
                    # Update the visualization after tracking
                    current_viz_html = self.generate_html_visualization()
                    current_metrics_html = self.generate_metrics_table()
                    yield messages, current_viz_html, current_metrics_html
                
                # Process messages from the step
                for message in pull_messages_from_step(step_log):
                    messages.append(message)
                    yield messages, current_viz_html, current_metrics_html

            # Mark the manager agent as completed
            self.update_agent_status(manager_name, "completed")
            
            # Update the time spent for the whole session
            session_time = time.time() - session_start
            for agent_id in self.agent_hierarchy:
                if self.agent_statuses.get(agent_id) != "idle" and self.agent_metrics[agent_id]["time_spent"] == 0:
                    # If no time was tracked but agent was used, assign a proportion of the session time
                    self.agent_metrics[agent_id]["time_spent"] = session_time * 0.8 if agent_id == manager_name else session_time * 0.2
            
            # Final visualization update
            current_viz_html = self.generate_html_visualization()
            current_metrics_html = self.generate_metrics_table()
            yield messages, current_viz_html, current_metrics_html
            
        except Exception as e:
            print(f"Error in interaction: {str(e)}")
            messages.append(gr.ChatMessage(role="assistant", content=f"Error: {str(e)}"))
            yield messages, agent_viz_html, agent_metrics_html

    def log_user_message(self, text_input, file_uploads_log):
        """Process user input and inform about file uploads"""
        # Add file upload information to the prompt if any files were uploaded
        prompt = text_input
        if len(file_uploads_log) > 0:
            prompt += f"\nYou have been provided with these files: {file_uploads_log}"
        
        # Clear the input and disable the button during processing
        return prompt, "", gr.Button(interactive=False)

    def launch(self, share: bool = True, **kwargs):
        """Create and launch a custom Gradio interface with a persistent agent visualization"""
        import gradio as gr

        with gr.Blocks(theme="soft") as demo:
            # Session state
            session_state = gr.State({})
            stored_messages = gr.State([])
            file_uploads_log = gr.State([])

            # Title
            gr.Markdown("# Multi-Agent System Visualizer")
            
            with gr.Row():
                # Left column for chat
                with gr.Column(scale=5):
                    # Chat interface
                    chatbot = gr.Chatbot(
                        label="Agent Chat",
                        type="messages",
                        avatar_images=(
                            None,
                            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
                        ),
                        show_copy_button=True,
                        height=500,
                    )
                    
                    # Input area
                    with gr.Row():
                        with gr.Column(scale=8):
                            text_input = gr.Textbox(
                                lines=3,
                                label="Your message",
                                placeholder="Enter your prompt here and press Enter or click the Send button",
                                show_label=False,
                            )
                        with gr.Column(scale=1):
                            submit_btn = gr.Button("Send", variant="primary")
                
                # Right column for visualization
                with gr.Column(scale=4):
                    with gr.Tabs():
                        with gr.Tab("Agent Network"):
                            agent_viz_html = gr.HTML(self.generate_html_visualization(), label="Agent Visualization")
                        with gr.Tab("Agent Metrics"):
                            agent_metrics_html = gr.HTML(self.generate_metrics_table(), label="Agent Metrics")
                    
                    # File upload section
                    if self.file_upload_folder is not None:
                        with gr.Group():
                            gr.Markdown("## File Upload")
                            with gr.Row():
                                upload_file = gr.File(label="Upload a file")
                                upload_status = gr.Textbox(label="Upload Status", interactive=False, visible=False)
                            
                            upload_file.change(
                                self.upload_file,
                                [upload_file, file_uploads_log],
                                [upload_status, file_uploads_log],
                            )
                    
                    # Information about the agent
                    with gr.Group():
                        gr.Markdown(f"## About this Agent")
                        gr.Markdown(f"**Name:** {self.name}")
                        if self.description:
                            gr.Markdown(f"**Description:** {self.description}")
                        gr.Markdown("Powered by [smolagents](https://github.com/huggingface/smolagents)")
                        
                        gr.HTML("""<div style="display: flex; align-items: center; gap: 8px; font-family: system-ui, -apple-system, sans-serif; justify-content: center; margin-top: 20px;">
                            <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png" style="width: 32px; height: 32px; object-fit: contain;" alt="logo">
                            <a target="_blank" href="https://github.com/huggingface/smolagents"><b>huggingface/smolagents</b></a>
                            </div>""")

            # Set up event handlers
            text_input.submit(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input, submit_btn],
            ).then(
                self.interact_with_agent,
                [stored_messages, chatbot, session_state, agent_viz_html, agent_metrics_html],
                [chatbot, agent_viz_html, agent_metrics_html],
            ).then(
                lambda: (
                    gr.Textbox(
                        interactive=True,
                        placeholder="Enter your prompt here and press Enter or click the Send button",
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
            ).then(
                self.interact_with_agent,
                [stored_messages, chatbot, session_state, agent_viz_html, agent_metrics_html],
                [chatbot, agent_viz_html, agent_metrics_html],
            ).then(
                lambda: (
                    gr.Textbox(
                        interactive=True,
                        placeholder="Enter your prompt here and press Enter or click the Send button",
                    ),
                    gr.Button(interactive=True),
                ),
                None,
                [text_input, submit_btn],
            )

        # Launch the demo
        demo.launch(share=share, **kwargs)