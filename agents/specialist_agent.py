
import modal  # Modal cloud platform for serverless deployment
from tools.agent_colors import Agent


class SpecialistAgent(Agent):
    """
    Agent that calls a fine-tuned Llama 3.2 3B model on Modal Cloud.
    
    The model is:
    - Fine-tuned using QLoRA (efficient parameter tuning)
    - Deployed remotely on Modal Cloud (serverless inference)
    - Specialized for product price prediction
    
    Called by Ensemble Agent (in parallel with Frontier and NN agents).
    """

    name = "Specialist Agent"
    color = Agent.RED  # Will log in red color

    def __init__(self):
        self.log("Specialist Agent is initializing - connecting to Modal")
        
        try:
            # Connect to the Modal Cloud deployed class
            # This is a remote connection - no model loaded locally
            Pricer = modal.Cls.from_name("pricer-service", "Pricer")
            self.pricer = Pricer()
            self.log("Specialist Agent connected to Modal successfully")
            
        except Exception as e:
            error_msg = f"Failed to connect to Modal: {str(e)}"
            self.log(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg) from e

    def price(self, description: str) -> float:
        # Validate input
        if not description or not description.strip():
            self.log("WARNING: Empty description provided, returning $0.00")
            return 0.0
        
        self.log("Specialist Agent is calling remote fine-tuned model (llama3.2-3B)")
        
        try:
            # Make remote call to Modal Cloud
            # The .remote() method is Modal's RPC mechanism
            result = self.pricer.price.remote(description)
            
            # Validate output
            if result < 0:
                self.log(f"WARNING: Negative price ({result:.2f}), clamping to $0.00")
                result = 0.0
            elif result > 50000:
                self.log(f"WARNING: Very high price predicted: ${result:.2f}")
            
            self.log(f"Specialist Agent completed - predicting ${result:.2f}")
            return result
            
        except Exception as e:
            # Handle network errors, timeouts, Modal errors, etc.
            self.log(f"ERROR: Remote call failed: {str(e)}")
            return 0.0