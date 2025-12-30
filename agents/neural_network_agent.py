"""
Neural Network Agent - Deep Learning Based Price Prediction

This agent uses a pre-trained deep residual neural network to predict product prices.
The model is loaded once during initialization and used for all subsequent predictions.
"""

from tools.agent_colors import Agent  # Base agent class for colored logging
from tools.deep_neural_network import DeepNeuralNetworkInference  # FIXED: Was agents.deep_neural_network
import os  # For file path checking


class NeuralNetworkAgent(Agent):
    
    name = "Neural Network Agent"
    color = Agent.MAGENTA

    def __init__(self, model_path="models/deep_neural_network.pth"):

        self.log("Neural Network Agent is initializing")
        
        # Validate model file exists
        if not os.path.exists(model_path):
            error_msg = f"Model file not found: {model_path}"
            self.log(f"ERROR: {error_msg}")
            raise FileNotFoundError(error_msg)
        
        self.log(f"Found model file: {model_path}")
        
        try:
            # Create and setup inference engine
            self.neural_network = DeepNeuralNetworkInference()
            self.log("Setting up inference engine...")
            self.neural_network.setup()
            
            # Load model weights
            self.log(f"Loading model weights from {model_path}...")
            self.neural_network.load(model_path)
            
            # Warmup with dummy prediction
            self.log("Warming up model...")
            _ = self.neural_network.inference("warmup product description")
            
            self.log("Neural Network Agent is ready and weights are loaded")
            
        except Exception as e:
            error_msg = f"Failed to initialize: {str(e)}"
            self.log(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg) from e

    def price(self, description: str) -> float:
        # Validate input
        if not description or not description.strip():
            self.log("WARNING: Empty description provided, returning $0.00")
            return 0.0
        
        self.log("Neural Network Agent is starting a prediction")
        
        try:
            # Run inference
            result = self.neural_network.inference(description)
            
            # Validate output
            if result < 0:
                self.log(f"WARNING: Negative price ({result:.2f}), clamping to $0.00")
                result = 0.0
            elif result > 50000:
                self.log(f"WARNING: Very high price predicted: ${result:.2f}")
            
            self.log(f"Neural Network Agent completed - predicting ${result:.2f}")
            return result
            
        except Exception as e:
            self.log(f"ERROR: Prediction failed: {str(e)}")
            return 0.0