from litellm import completion  
from dotenv import load_dotenv  
import os  
from typing import Optional, Dict, Any  #
import logging  # For error logging and debugging

# Load environment variables, override existing ones
load_dotenv(override=True)

# Configure logging for this module
logger = logging.getLogger(__name__)

# Default model for preprocessing (can be overridden)
DEFAULT_MODEL_NAME = os.getenv("PRICER_PREPROCESSOR_MODEL", "ollama/llama3.2")
# Reasoning effort parameter (only used by OpenAI o1 models)
DEFAULT_REASONING_EFFORT = "low" if "gpt-oss" in DEFAULT_MODEL_NAME else None

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


class PreprocessorTool:
    """
    Tool for preprocessing raw product descriptions into structured format.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,  # LLM model identifier
        reasoning_effort: Optional[str] = DEFAULT_REASONING_EFFORT,  # For o1 models
        base_url: Optional[str] = None,  # Custom API endpoint
    ):
        
        # Usage tracking for monitoring and cost control
        self.total_input_tokens = 0  # Total prompt tokens used
        self.total_output_tokens = 0  # Total completion tokens generated
        self.total_cost = 0.0  # Total cost in USD
        
        # Model configuration
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url
        
        # Auto-configure Ollama endpoint if using local Ollama
        if "ollama" in model_name and not base_url:
            self.base_url = "http://localhost:11434"
            logger.info(f"Auto-configured Ollama base URL: {self.base_url}")

    def _build_messages(self, text: str) -> list[dict]:
        """
        Build the message array for the LLM API call.
        """
        return [
            {"role": "system", "content": SYSTEM_PROMPT},  # Instructions for formatting
            {"role": "user", "content": text}  # The raw product data
        ]

    def preprocess(self, text: str) -> Optional[str]:
        try:
            # Build the conversation messages
            messages = self._build_messages(text)
            
            # Call the LLM API via LiteLLM
            response = completion(
                messages=messages,
                model=self.model_name,
                reasoning_effort=self.reasoning_effort,  # Only affects o1 models
                api_base=self.base_url,  # Custom endpoint if specified
            )
            
            # Update usage statistics for monitoring
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens
            
            # Safely access cost if available (not all providers return this)
            if hasattr(response, "_hidden_params") and "response_cost" in response._hidden_params:
                self.total_cost += response._hidden_params["response_cost"]
            
            # Extract and return the generated content
            processed_text = response.choices[0].message.content
            logger.debug(f"Successfully preprocessed product description")
            return processed_text
            
        except Exception as e:
            # Log the error and return None (allows calling code to handle gracefully)
            logger.error(f"Preprocessing failed: {str(e)}")
            return None

    def get_usage_stats(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 4)
        }

    def reset_stats(self) -> None:
        """Reset usage statistics to zero."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        logger.info("Usage statistics reset")


# Convenience function for one-off preprocessing
def preprocess_product(text: str, model_name: str = DEFAULT_MODEL_NAME) -> Optional[str]:
    tool = PreprocessorTool(model_name=model_name)
    return tool.preprocess(text)