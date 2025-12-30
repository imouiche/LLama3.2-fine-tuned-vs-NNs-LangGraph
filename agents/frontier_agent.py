import re  # Regular expressions for parsing prices from text
from typing import List, Dict  # Type hints for function signatures
from openai import OpenAI  # OpenAI API client
from sentence_transformers import SentenceTransformer  # For encoding text into vectors
from tools.agent_colors import Agent  # Base agent class for colored logging


class FrontierAgent(Agent):
    """
    RAG-based price prediction agent.
    
    Uses semantic similarity search to find comparable products,
    then uses those as context for LLM-based price prediction.
    """
    
    # Agent identification for colored logging
    name = "Frontier Agent"
    color = Agent.BLUE  # Will log in blue color
    
    # Default model (can be overridden in __init__)
    MODEL = "gpt-4o-mini"

    def __init__(self, collection):
        # Log initialization start (uses colored logging from base Agent class)
        self.log("Initializing Frontier Agent")
        
        # Initialize OpenAI client (requires OPENAI_API_KEY env variable)
        self.client = OpenAI()
        
        # Override default model with newer/better model
        self.MODEL = "gpt-5.1"  # Note: gpt-5.1 might not exist yet, likely meant gpt-4o or similar
        self.log("Frontier Agent is setting up with OpenAI")

        self.collection = collection
        # "all-MiniLM-L6-v2" is a lightweight, fast model (384 dimensions)
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        self.log("Frontier Agent is ready")

    def make_context(self, similars: List[str], prices: List[float]) -> str:
        
        # Start with explanation of what the context represents
        message = "To provide some context, here are some other items that might be similar to the item you need to estimate.\n\n"
        
        # Add each similar product and its price to the context
        for similar, price in zip(similars, prices):
            message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
        
        return message

    def messages_for(
        self, description: str, similars: List[str], prices: List[float]
    ) -> List[Dict[str, str]]:
  
        # Start with the main task: estimate price for this product
        message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{description}\n\n"
        
        # Add RAG context (similar products with prices)
        message += self.make_context(similars, prices)
        
        # Return in OpenAI message format (user role)
        return [{"role": "user", "content": message}]

    def find_similars(self, description: str):
        self.log(
            "Frontier Agent is performing a RAG search of the Chroma datastore to find 5 similar products"
        )
        # encode() returns numpy array of shape (1, 384) for this model
        vector = self.model.encode([description])

        # n_results: how many similar items to return
        results = self.collection.query(
            query_embeddings=vector.astype(float).tolist(),  # Convert numpy → list for Chroma
            n_results=5  # Return top 5 most similar products
        )
        
        # Extract the similar product descriptions from results
        documents = results["documents"][0][:]
        
        # Extract prices from metadata (stored when products were added to Chroma)
        prices = [m["price"] for m in results["metadatas"][0][:]]
        
        self.log("Frontier Agent has found similar products")
        return documents, prices

    def get_price(self, s: str) -> float:
        # Remove dollar signs and commas
        s = s.replace("$", "").replace(",", "")
        
        # Use regex to find first number (int or float)
        # Pattern matches: optional +/-, optional digits, optional decimal point, digits
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        
        # Return matched number as float, or 0.0 if no match
        return float(match.group()) if match else 0.0

    def price(self, description: str) -> float:
        # Step 1: RAG - Find 5 similar products
        documents, prices = self.find_similars(description)
        
        self.log(
            f"Frontier Agent is about to call {self.MODEL} with context including 5 similar products"
        )
        
        # Step 2: Call OpenAI LLM with RAG context
        response = self.client.chat.completions.create(
            model=self.MODEL,  # GPT model to use
            messages=self.messages_for(description, documents, prices),  # Prompt with RAG context
            seed=42,  # For reproducibility (deterministic outputs)
            reasoning_effort="none",  # For o1 models, but probably not relevant for gpt-4o-mini
        )
        
        # Step 3: Extract text response from LLM
        reply = response.choices[0].message.content
        
        # Step 4: Parse price from response text
        result = self.get_price(reply)
        
        self.log(f"Frontier Agent completed - predicting ${result:.2f}")
        
        # Step 5: Return predicted price
        return result