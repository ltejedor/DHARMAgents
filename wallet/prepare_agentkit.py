import os
import json
import secrets
from dotenv import load_dotenv

from coinbase_agentkit import (
    AgentKit,
    AgentKitConfig,
    
    CdpWalletProvider,
    CdpWalletProviderConfig,
    cdp_wallet_action_provider,
    
    cdp_api_action_provider,
    erc20_action_provider,
    pyth_action_provider,
    wallet_action_provider,
    weth_action_provider,
)


"""
AgentKit Configuration

This file serves as the entry point for configuring AgentKit tools and wallet providers.
It handles wallet setup, persistence, and initializes AgentKit with the appropriate providers.

# Key Steps to Configure AgentKit:

1. Set up your WalletProvider:
   - Learn more: https://github.com/coinbase/agentkit/tree/main/python/agentkit#evm-wallet-providers

2. Set up your Action Providers:
   - Action Providers define what your agent can do.  
   - Choose from built-in providers or create your own:
     - Built-in: https://github.com/coinbase/agentkit/tree/main/python/coinbase-agentkit#create-an-agentkit-instance-with-specified-action-providers
     - Custom: https://github.com/coinbase/agentkit/tree/main/python/coinbase-agentkit#creating-an-action-provider

# Next Steps:

- Explore the AgentKit README: https://github.com/coinbase/agentkit
- Learn more about available WalletProviders & Action Providers.
- Experiment with custom Action Providers for your unique use case.

## Want to contribute?
Join us in shaping AgentKit! Check out the contribution guide:  
- https://github.com/coinbase/agentkit/blob/main/CONTRIBUTING.md
- https://discord.gg/CDP
"""

# Configure a file to persist wallet data
wallet_data_file = "wallet_data.txt"

def prepare_agentkit():
    """Initialize CDP Agentkit and return tools."""

    
    # Initialize WalletProvider
    wallet_data = None
    if os.path.exists(wallet_data_file):
        with open(wallet_data_file) as f:
            wallet_data = f.read()

    cdp_config = None
    if wallet_data is not None:
        cdp_config = CdpWalletProviderConfig(wallet_data=wallet_data)

    wallet_provider = CdpWalletProvider(cdp_config)

    

    # Initialize AgentKit
    agentkit = AgentKit(AgentKitConfig(
        wallet_provider=wallet_provider,
        action_providers=[
            cdp_wallet_action_provider(),
            cdp_api_action_provider(),
            erc20_action_provider(),
            wallet_action_provider(),
            weth_action_provider(),
        ]
    ))

    
    # Save wallet to file for reuse
    wallet_data_json = json.dumps(wallet_provider.export_wallet().to_dict())
    with open(wallet_data_file, "w") as f:
        f.write(wallet_data_json)
    

    return agentkit