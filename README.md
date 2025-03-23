# DHARMAgents

DHARMAgents is a framework for creating, managing, and deploying autonomous AI agents with blockchain capabilities and persistent memory.

## 🔍 Overview

DHARMAgents combines several powerful technologies:
- **SmolaAgents**: For agent orchestration and tool calling
- **Recall Network**: For decentralized, persistent agent memory
- **Blockchain Integration**: For on-chain actions via AgentKit
- **Gradio UI**: For intuitive agent interaction

The project enables the creation of sophisticated agent systems that can:
- Perform web searches and research
- Execute code for complex problem-solving
- Interact with blockchains using web3 capabilities
- Store their chain-of-thought reasoning in decentralized storage
- Access historical logs to improve reasoning over time

## 🚀 Getting Started

### Prerequisites
- Python 3.10+ 
- Poetry (for wallet component)
- Hugging Face token (for models)
- (Optional) Recall Network credentials

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DHARMAgents.git
cd DHARMAgents
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
```

4. Edit `.env` with your API keys and configuration:
```
HF_TOKEN=your_huggingface_token
RECALL_PRIVATE_KEY=your_recall_private_key  # Optional
RECALL_BUCKET_ALIAS=your_bucket_alias       # Optional
```

## 🖥️ Running the Interface

### Basic Gradio UI
```bash
python gradio_app.py
```
This launches the main Gradio interface where you can:
- Configure different agent types (CodeAgent, ToolCallingAgent)
- Select models (Hugging Face, OpenAI, Anthropic, etc.)
- Choose tools (web search, webpage visits)
- Chat with your agents

### Multi-Agent System
```bash
python main.py
```
This starts a system with:
- Manager Agent (CodeAgent)
- Web Search Agent (ToolCallingAgent)

### Memory-Enhanced UI
```bash
python logging_test.py
```
This runs an enhanced UI with:
- Chain-of-thought logging
- Memory syncing to Recall Network
- Agent visualization

## 🧩 Architecture

The project has several key components:

### Agent Types
- **CodeAgent**: Uses Python code for reasoning and actions
- **ToolCallingAgent**: Uses JSON for tool calls (simpler but less powerful)

### Model Providers
- **HfApiModel**: Models hosted on HuggingFace
- **LiteLLMModel**: Access to models from OpenAI, Anthropic, etc.

### Tools
- **DuckDuckGoSearchTool**: Perform web searches
- **VisitWebpageTool**: Visit URLs and extract content
- **Recall Storage Plugin**: Store and retrieve memory from Recall Network

### Wallet Integration
- **AgentKit**: Integration with Coinbase Developer Platform
- **Wallet Provider**: Interface for blockchain interactions
- **Action Providers**: Define what actions agents can take on-chain

## 📊 Memory and Persistence

DHARMAgents uses the Recall Network for decentralized memory:

1. **Chain-of-Thought Logging**: Agents log reasoning steps into a local database
2. **Periodic Syncing**: Logs are periodically uploaded to Recall buckets
3. **Retrieval for Context**: Thought logs are retrieved before each inference cycle
4. **Efficient Storage**: Support for bucket management, object storage, and retrieval

## 🔧 Configuration Options

### Agent Configuration
```python
manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[search_agent],
    additional_authorized_imports=["time", "numpy", "pandas"],
)
```

### Model Selection
```python
model = HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")
# OR
model = LiteLLMModel(model="gpt-4o")
```

### Memory Configuration
```
RECALL_BUCKET_ALIAS="logs"
RECALL_COT_LOG_PREFIX="cot"
RECALL_SYNC_INTERVAL=120000  # 2 minutes
RECALL_BATCH_SIZE=4  # 4KB
```

## 📚 Advanced Usage

### Creating Custom Tools
The framework supports easy creation of custom tools:

```python
@tool
def my_custom_tool(parameter: str) -> str:
    """Tool description goes here
    
    Args:
        parameter: Parameter description
    
    Returns:
        Return value description
    """
    # Tool implementation
    return result
```

### Adding Blockchain Capabilities

Connect your agent to Web3 capabilities using the wallet module:

```python
from wallet.wallet_tool import wallet_agent_tool

# Add the wallet tool to your agent
agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), wallet_agent_tool],
    model=model
)
```

### Multi-Agent Orchestration

Create sophisticated agent systems:

```python
# Create specialized agents
search_agent = ToolCallingAgent(tools=[DuckDuckGoSearchTool()])
research_agent = ToolCallingAgent(tools=[VisitWebpageTool()])

# Create manager agent that can coordinate
manager = CodeAgent(
    managed_agents=[search_agent, research_agent],
    planning_interval=4
)
```

## 🔗 Example Commands

**Create a bucket for memory storage:**
```
Create a bucket for me named "agent-memories"
```

**Store logs to Recall:**
```
Add object "logs.txt" to bucket "agent-memories"
```

**Retrieve memories:**
```
Get object "data.json" from bucket "research-data"
```

## 🙏 Acknowledgements

- [SmolaAgents](https://github.com/huggingface/smolagents)
- [Recall Network](https://recall.network/) 
- [AgentKit](https://github.com/coinbase/agentkit)
- [Gradio](https://www.gradio.app/)