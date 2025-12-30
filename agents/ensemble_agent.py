from tools.agent_colors import Agent  # Base agent class
from agents.specialist_agent import SpecialistAgent  # Fine-tuned Llama 3.2 on Modal
from agents.frontier_agent import FrontierAgent  # RAG-based with OpenAI
from agents.neural_network_agent import NeuralNetworkAgent  # PyTorch local model
from tools.preprocessor import PreprocessorTool as Preprocessor  # Text preprocessing


class EnsembleAgent(Agent):
    """
    Ensemble coordinator that combines predictions from multiple agents.
    """

    name = "Ensemble Agent"
    color = Agent.YELLOW  # Will log in yellow

    def __init__(self, collection):
        self.log("Initializing Ensemble Agent")
        
        try:
            # Initialize Specialist Agent (Modal Cloud connection)
            self.log(" Creating Specialist Agent...")
            self.specialist = SpecialistAgent()
            
            # Initialize Frontier Agent (needs Chroma collection)
            self.log("Creating Frontier Agent...")
            self.frontier = FrontierAgent(collection)
            
            # Initialize Neural Network Agent
            self.log("Creating Neural Network Agent...")
            self.neural_network = NeuralNetworkAgent()
            
            # Initialize Preprocessor (for text standardization)
            self.log("Creating Preprocessor...")
            self.preprocessor = Preprocessor()
            
            self.log("Ensemble Agent is ready")
            
        except Exception as e:
            error_msg = f"Failed to initialize Ensemble Agent: {str(e)}"
            self.log(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg) from e

    def price(self, description: str) -> float:
        # Validate input
        if not description or not description.strip():
            self.log("WARNING: Empty description, returning $0.00")
            return 0.0
        
        self.log("Running Ensemble Agent - preprocessing text")
        
        try:
            # Step 1: Preprocess the description
            # Converts messy text to: "Title: X\nCategory: Y\nBrand: Z..."
            rewrite = self.preprocessor.preprocess(description)
            
            if not rewrite:
                self.log("WARNING: Preprocessing failed, using original description")
                rewrite = description
            
            self.log(f"Pre-processed text using {self.preprocessor.model_name}")
            
        except Exception as e:
            self.log(f"WARNING: Preprocessing error: {e}, using original description")
            rewrite = description
        
        predictions = {}
        
        try:
            # Specialist prediction (Modal Cloud - may be slow)
            self.log(" Getting Specialist prediction...")
            predictions['specialist'] = self.specialist.price(rewrite)
        except Exception as e:
            self.log(f"WARNING: Specialist failed: {e}, using 0.0")
            predictions['specialist'] = 0.0
        
        try:
            # Frontier prediction (RAG + OpenAI - may be slow)
            self.log(" Getting Frontier prediction...")
            predictions['frontier'] = self.frontier.price(rewrite)
        except Exception as e:
            self.log(f"WARNING: Frontier failed: {e}, using 0.0")
            predictions['frontier'] = 0.0
        
        try:
            # Neural Network prediction (local - fast)
            self.log("  Getting Neural Network prediction...")
            predictions['neural_network'] = self.neural_network.price(rewrite)
        except Exception as e:
            self.log(f"WARNING: Neural Network failed: {e}, using 0.0")
            predictions['neural_network'] = 0.0
        
        # Log individual predictions
        self.log(f"  Predictions: Frontier=${predictions['frontier']:.2f}, "
                f"Specialist=${predictions['specialist']:.2f}, "
                f"NN=${predictions['neural_network']:.2f}")
        
        combined = (
            predictions['frontier'] * 0.8 +          # 80% weight
            predictions['specialist'] * 0.1 +        # 10% weight
            predictions['neural_network'] * 0.1      # 10% weight
        )
        
        # Validate final result
        if combined < 0:
            self.log("WARNING: Negative combined price, clamping to $0.00")
            combined = 0.0
        
        self.log(f"Ensemble Agent complete - returning ${combined:.2f}")
        return combined