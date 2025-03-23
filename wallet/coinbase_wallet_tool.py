import os
import json
from coinbase_agentkit import (
    AgentKit,
    AgentKitConfig,
    CdpWalletProvider,
    # Import any additional providers or helper functions if needed
)

class CoinbaseWalletTool:
    name = "coinbase_wallet_management"
    description = (
        "A tool for managing your Coinbase wallet. "
        "Supported actions include checking the wallet balance and sending funds."
    )

    def __init__(self, wallet_data_file="wallet_data.txt"):
        self.wallet_data_file = wallet_data_file
        self.agentkit = self._init_agentkit()

    def _init_agentkit(self):
        """Initialize AgentKit using your persisted wallet data."""
        wallet_data = None
        if os.path.exists(self.wallet_data_file):
            with open(self.wallet_data_file) as f:
                wallet_data = f.read()

        # Create a wallet provider configuration from your wallet data
        cdp_config = None
        if wallet_data is not None:
            # Adjust the constructor if your version differs
            from coinbase_agentkit import CdpWalletProviderConfig
            cdp_config = CdpWalletProviderConfig(wallet_data=wallet_data)
            
        # Initialize the wallet provider
        wallet_provider = CdpWalletProvider(cdp_config)
        
        # Create and return the AgentKit instance (only the wallet management functions will be used)
        return AgentKit(AgentKitConfig(
            wallet_provider=wallet_provider,
            action_providers=[]  # You can add additional action providers if needed
        ))

    def run(self, action: str, params: str = "") -> str:
        """
        Execute a wallet management action.
        
        Actions:
          - "check_balance": Returns the wallet's balance.
          - "send_funds": Sends funds to a destination address.
        
        For "send_funds", params should be in the format: "<amount> <destination_address>"
        """
        if action == "check_balance":
            # Replace get_balance() with the actual method provided by your wallet provider
            try:
                balance = self.agentkit.wallet_provider.get_balance()
                return f"Wallet balance: {balance}"
            except Exception as e:
                return f"Error checking balance: {e}"
        
        elif action == "send_funds":
            parts = params.split()
            if len(parts) < 2:
                return "Usage for send_funds: <amount> <destination_address>"
            amount, destination = parts[0], parts[1]
            try:
                # Replace send_funds() with the actual method if different
                tx = self.agentkit.wallet_provider.send_funds(amount, destination)
                return f"Sent {amount} ETH to {destination}. Transaction: {tx}"
            except Exception as e:
                return f"Error sending funds: {e}"
        
        else:
            return "Unknown action. Supported actions: check_balance, send_funds"
