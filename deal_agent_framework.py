import os
import sys
import logging
import json
from typing import List
from dotenv import load_dotenv
import chromadb
from tools.deals import Opportunity  # Fixed import path
from workflows.deal_workflow import build_deal_workflow, run_deal_workflow  # NEW: LangGraph
from workflows.state import create_initial_state  # NEW: State management
from sklearn.manifold import TSNE
import numpy as np

load_dotenv(override=True)

# Colors for logging
BG_BLUE = "\033[44m"
WHITE = "\033[37m"
RESET = "\033[0m"

# Colors for plot
CATEGORIES = [
    "Appliances",
    "Automotive",
    "Cell_Phones_and_Accessories",
    "Electronics",
    "Musical_Instruments",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
]
COLORS = ["red", "blue", "brown", "orange", "yellow", "green", "purple", "cyan"]


def init_logging():
    """Initialize logging configuration for the framework."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [Agents] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


class DealAgentFramework:
    """
    Deal Agent Framework - Infrastructure for deal discovery system.
    
    Responsibilities:
    - Initialize Chroma vector database
    - Manage memory (track notified deals)
    - Run the LangGraph workflow
    - Provide utility functions (plotting, memory reset)
    
    This class provides the infrastructure layer.
    The workflow orchestration is handled by LangGraph.
    """
    
    DB = "products_vectorstore"
    MEMORY_FILENAME = "memory.json"

    def __init__(self):
        init_logging()
        self.log("Initializing Deal Agent Framework")
        
        # Initialize Chroma DB
        client = chromadb.PersistentClient(path=self.DB)
        self.collection = client.get_or_create_collection("products")
        self.log(f"Connected to Chroma DB at {self.DB}")
        
        # Load memory from previous runs
        self.memory = self.read_memory()
        self.log(f"Loaded {len(self.memory)} previous opportunities from memory")
        
        # LangGraph workflow (lazy initialization)
        self.workflow = None
        
        self.log("Deal Agent Framework initialized")

    def init_workflow_as_needed(self):
        if not self.workflow:
            self.log("Building LangGraph workflow...")
            self.workflow = build_deal_workflow()
            self.log("LangGraph workflow ready")

    def read_memory(self) -> List[Opportunity]:
        if os.path.exists(self.MEMORY_FILENAME):
            try:
                with open(self.MEMORY_FILENAME, "r") as file:
                    data = json.load(file)
                opportunities = [Opportunity(**item) for item in data]
                return opportunities
            except Exception as e:
                self.log(f"WARNING: Failed to read memory: {e}")
                return []
        return []

    def write_memory(self) -> None:
        try:
            data = [opportunity.model_dump() for opportunity in self.memory]
            with open(self.MEMORY_FILENAME, "w") as file:
                json.dump(data, file, indent=2)
            self.log(f"Saved {len(self.memory)} opportunities to memory")
        except Exception as e:
            self.log(f"ERROR: Failed to write memory: {e}")

    @classmethod
    def reset_memory(cls) -> None:
        data = []
        if os.path.exists(cls.MEMORY_FILENAME):
            with open(cls.MEMORY_FILENAME, "r") as file:
                data = json.load(file)
        
        # Keep only the 2 most recent
        truncated = data[:2]
        
        with open(cls.MEMORY_FILENAME, "w") as file:
            json.dump(truncated, file, indent=2)
        
        logging.info(f"Memory reset: kept {len(truncated)} of {len(data)} opportunities")

    def log(self, message: str):
        """Log a message with framework prefix and color."""
        text = BG_BLUE + WHITE + "[Agent Framework] " + message + RESET
        logging.info(text)

    def run(self) -> List[Opportunity]:
        # Initialize workflow
        self.init_workflow_as_needed()
        
        self.log("🚀 Starting deal discovery workflow")
        
        try:
            # Create initial state
            initial_state = create_initial_state(
                chroma_collection=self.collection,
                memory=self.memory
            )
            
            # Run LangGraph workflow
            self.log(" Executing LangGraph workflow...")
            final_state = self.workflow.invoke(initial_state)
            
            # Extract results
            best_opportunity = final_state.get("best_opportunity")
            
            if best_opportunity:
                self.log(f" Workflow found opportunity: ${best_opportunity.discount:.2f} discount")
                self.log(f"   Product: {best_opportunity.deal.product_description[:60]}...")
                
                # Add to memory
                self.memory.append(best_opportunity)
                self.write_memory()
            else:
                self.log("No opportunities found above threshold")
            
            # Log any errors
            if final_state.get("errors"):
                self.log(f"Workflow completed with {len(final_state['errors'])} errors")
                for error in final_state["errors"]:
                    self.log(f"   - {error}")
            
            self.log("Workflow completed successfully")
            
        except Exception as e:
            self.log(f"❌ ERROR: Workflow failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return self.memory

    def run_legacy(self) -> List[Opportunity]:
        from agents.planning_agent import PlanningAgent
        
        self.log("Running legacy Planning Agent (non-LangGraph)")
        
        # Create planning agent
        planner = PlanningAgent(self.collection)
        
        # Run workflow
        result = planner.plan(memory=self.memory)
        
        if result:
            self.log(f"Planning Agent found opportunity: ${result.discount:.2f} discount")
            self.memory.append(result)
            self.write_memory()
        else:
            self.log("No opportunities found")
        
        return self.memory

    @classmethod
    def get_plot_data(cls, max_datapoints=2000):
        client = chromadb.PersistentClient(path=cls.DB)
        collection = client.get_or_create_collection("products")
        
        # Get embeddings and metadata from Chroma
        result = collection.get(
            include=["embeddings", "documents", "metadatas"], 
            limit=max_datapoints
        )
        
        vectors = np.array(result["embeddings"])
        documents = result["documents"]
        categories = [metadata["category"] for metadata in result["metadatas"]]
        colors = [COLORS[CATEGORIES.index(c)] for c in categories]
        
        # Reduce dimensionality for 3D plotting
        tsne = TSNE(n_components=3, random_state=42, n_jobs=-1)
        reduced_vectors = tsne.fit_transform(vectors)
        
        return documents, reduced_vectors, colors


if __name__ == "__main__":
    # Run the framework
    framework = DealAgentFramework()
    
    # Run with LangGraph (new implementation)
    memory = framework.run()
    
    # Uncomment to run legacy version for comparison:
    # memory = framework.run_legacy()
    
    print(f"\n{'='*60}")
    print(f"Total opportunities in memory: {len(memory)}")
    if memory:
        print(f"\nMost recent opportunity:")
        latest = memory[-1]
        print(f"  Product: {latest.deal.product_description[:80]}...")
        print(f"  Price: ${latest.deal.price:.2f}")
        print(f"  Estimate: ${latest.estimate:.2f}")
        print(f"  Discount: ${latest.discount:.2f}")
    print(f"{'='*60}\n")