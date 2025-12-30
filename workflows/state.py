
from typing import TypedDict, List, Optional, Any
from tools.deals import Deal, Opportunity


class DealWorkflowState(TypedDict, total=False):
    """
    State that flows through the deal discovery workflow.
    
    Workflow stages:
    1. Initialize: chroma_collection, memory
    2. Scanner: deals
    3. Preprocessing: processed_deals
    4. Ensemble split: current_deal
    5. Parallel predictions: frontier_price, specialist_price, nn_price
    6. Ensemble aggregate: estimated_price, discount, opportunities
    7. Decision: best_opportunity
    8. Messaging: notification_sent
    
    total=False allows nodes to only update fields they care about
    """
    # INITIALIZATION (provided by framework)
    
    chroma_collection: Any
    """Chroma vector database collection for Frontier Agent RAG"""
    
    memory: List[Opportunity]
    """List of previously notified deals (to avoid duplicates)"""
    
    # SCANNER NODE OUTPUT
  
    deals: List[Deal]
    """Raw deals from Scanner Agent (top 5 from RSS feeds)"""
    
    # PREPROCESSING (optional - could be part of ensemble)
    
    processed_deals: List[Deal]
    """Deals after text preprocessing (if we want a separate node)"""
    
    # ENSEMBLE - CURRENT DEAL BEING PROCESSED
    
    current_deal: Optional[Deal]
    """The deal currently being priced by ensemble"""
    
    current_deal_index: int
    """Index of current deal being processed (for iteration)"""
    
    # PARALLEL PREDICTION OUTPUTS (3 predictors)
    
    frontier_price: float
    """Price prediction from Frontier Agent (RAG-based)"""
    
    specialist_price: float
    """Price prediction from Specialist Agent (fine-tuned Llama 3.2)"""
    
    nn_price: float
    """Price prediction from Neural Network Agent (PyTorch)"""

    # ENSEMBLE AGGREGATION OUTPUT
 
    estimated_price: float
    """Final ensemble price (weighted average)"""
    
    discount: float
    """Calculated discount (estimated_price - actual_price)"""
    
    opportunities: List[Opportunity]
    """All opportunities created (one per deal)"""

    # DECISION NODE OUTPUT
    
    best_opportunity: Optional[Opportunity]
    """Best opportunity (highest discount) if above threshold"""
    
    # ============================================================
    # MESSAGING NODE OUTPUT
    # ============================================================
    
    notification_sent: bool
    """Whether notification was successfully sent"""

    # ERROR HANDLING
    
    errors: List[str]
    """List of errors encountered during workflow"""


class DealWorkflowConfig(TypedDict, total=False):
 
    deal_threshold: float
    """Minimum discount ($) to trigger notification (default: 50)"""
    
    max_deals: int
    """Maximum number of deals to process (default: 5)"""
    
    enable_notifications: bool
    """Whether to send notifications (default: True)"""
    
    test_mode: bool
    """If True, use test_scan() instead of real scan (default: False)"""
    
    parallel_predictions: bool
    """If True, run 3 predictors in parallel (default: True)"""


# HELPER FUNCTIONS FOR STATE MANAGEMENT

def create_initial_state(
    chroma_collection: Any,
    memory: List[Opportunity] = None
) -> DealWorkflowState:
    """
    Create initial state for workflow.
    
    Args:
        chroma_collection: Chroma DB collection
        memory: List of previously notified deals
        
    Returns:
        Initial state dict
    """
    return {
        "chroma_collection": chroma_collection,
        "memory": memory or [],
        "deals": [],
        "opportunities": [],
        "errors": [],
        "current_deal_index": 0,
        "notification_sent": False,
    }


def add_error(state: DealWorkflowState, error: str) -> DealWorkflowState:

    if "errors" not in state:
        state["errors"] = []
    state["errors"].append(error)
    return state

# STATE VALIDATION

def validate_state(state: DealWorkflowState, stage: str) -> bool:
    required_fields = {
        "init": ["chroma_collection", "memory"],
        "scanner": ["deals"],
        "ensemble": ["current_deal"],
        "predictions": ["frontier_price", "specialist_price", "nn_price"],
        "aggregate": ["estimated_price", "discount"],
        "decision": ["best_opportunity"],
    }
    
    if stage not in required_fields:
        return False
    
    for field in required_fields[stage]:
        if field not in state or state[field] is None:
            return False
    
    return True


# WEIGHTS FOR ENSEMBLE 

ENSEMBLE_WEIGHTS = {
    "frontier": 0.8,      # W1 - RAG-based, primary
    "specialist": 0.1,    # W2 - Fine-tuned, supplementary
    "neural_network": 0.1 # W3 - Local model, supplementary
}

DEAL_THRESHOLD = 50  # Default minimum discount ($)