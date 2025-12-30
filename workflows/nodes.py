import logging
from typing import Dict, Any
from workflows.state import DealWorkflowState, ENSEMBLE_WEIGHTS, DEAL_THRESHOLD
from agents.scanner_agent import ScannerAgent
from agents.frontier_agent import FrontierAgent
from agents.specialist_agent import SpecialistAgent
from agents.neural_network_agent import NeuralNetworkAgent
from agents.messaging_agent import MessagingAgent
from tools.preprocessor import PreprocessorTool as Preprocessor
from tools.deals import Opportunity

logger = logging.getLogger(__name__)


# ============================================================
# NODE 1: SCANNER NODE
# ============================================================

def scanner_node(state: DealWorkflowState) -> Dict[str, Any]:
    """
    Scanner Node: Fetch deals from RSS feeds.
    
    Input state:
        - memory: List of previously notified deals
        
    Output state:
        - deals: List of new deals (top 5 from RSS)
        
    This node:
    1. Creates Scanner Agent
    2. Fetches deals from RSS feeds
    3. Filters out deals already in memory
    4. Returns top 5 deals with good descriptions
    """
    logger.info("Scanner Node: Starting")
    
    try:
        # Create scanner agent
        scanner = ScannerAgent()
        
        # Scan for deals (filters by memory)
        memory = state.get("memory", [])
        selection = scanner.scan(memory=memory)
        
        # Check if we got any deals
        if not selection or not selection.deals:
            logger.warning("Scanner Node: No new deals found")
            return {"deals": []}
        
        logger.info(f"Scanner Node: Found {len(selection.deals)} deals")
        
        # Return updated state with deals
        return {"deals": selection.deals}
        
    except Exception as e:
        logger.error(f"Scanner Node failed: {str(e)}")
        return {
            "deals": [],
            "errors": state.get("errors", []) + [f"Scanner failed: {str(e)}"]
        }


# ============================================================
# NODE 2: PREPROCESSING NODE (Optional - could be in ensemble)
# ============================================================

def preprocessing_node(state: DealWorkflowState) -> Dict[str, Any]:
    logger.info("Preprocessing Node: Starting")
    
    try:
        deals = state.get("deals", [])
        
        if not deals:
            return {"processed_deals": []}
        
        # Create preprocessor
        preprocessor = Preprocessor()
        
        # Preprocess each deal's description
        processed_deals = []
        for deal in deals:
            try:
                # Preprocess the description
                cleaned = preprocessor.preprocess(deal.product_description)
                
                # Create new deal with cleaned description
                processed_deal = deal.model_copy(update={"product_description": cleaned})
                processed_deals.append(processed_deal)
                
            except Exception as e:
                logger.warning(f"Failed to preprocess deal: {e}, using original")
                processed_deals.append(deal)
        
        logger.info(f"Preprocessing Node: Processed {len(processed_deals)} deals")
        
        return {"processed_deals": processed_deals}
        
    except Exception as e:
        logger.error(f"Preprocessing Node failed: {str(e)}")
        # Fall back to unprocessed deals
        return {
            "processed_deals": state.get("deals", []),
            "errors": state.get("errors", []) + [f"Preprocessing failed: {str(e)}"]
        }


# ============================================================
# NODE 3: ENSEMBLE COORDINATOR (Sets up current deal)
# ============================================================

def ensemble_coordinator_node(state: DealWorkflowState) -> Dict[str, Any]:
    logger.info("Ensemble Coordinator: Starting")
    
    # Get deals (prefer processed, fall back to raw)
    deals = state.get("processed_deals") or state.get("deals", [])
    
    if not deals:
        logger.warning("Ensemble Coordinator: No deals to process")
        return {"current_deal": None}
    
    # Get current index
    index = state.get("current_deal_index", 0)
    
    # Check if we've processed all deals
    if index >= len(deals):
        logger.info("Ensemble Coordinator: All deals processed")
        return {"current_deal": None}
    
    # Get current deal
    current_deal = deals[index]
    
    logger.info(f"Ensemble Coordinator: Processing deal {index + 1}/{len(deals)}")
    
    return {
        "current_deal": current_deal,
        "current_deal_index": index
    }


# ============================================================
# NODE 4-6: PARALLEL PREDICTION NODES
# ============================================================

def frontier_node(state: DealWorkflowState) -> Dict[str, Any]:

    logger.info("🌐 Frontier Node: Starting")
    
    try:
        current_deal = state.get("current_deal")
        
        if not current_deal:
            logger.warning("Frontier Node: No current deal")
            return {"frontier_price": 0.0}
        
        # Get Chroma collection
        collection = state.get("chroma_collection")
        if not collection:
            logger.error("Frontier Node: No Chroma collection provided")
            return {"frontier_price": 0.0}
        
        # Create and run Frontier Agent
        agent = FrontierAgent(collection)
        price = agent.price(current_deal.product_description)
        
        logger.info(f"Frontier Node: Predicted ${price:.2f}")
        
        return {"frontier_price": price}
        
    except Exception as e:
        logger.error(f"Frontier Node failed: {str(e)}")
        return {
            "frontier_price": 0.0,
            "errors": state.get("errors", []) + [f"Frontier failed: {str(e)}"]
        }


def specialist_node(state: DealWorkflowState) -> Dict[str, Any]:
    logger.info("🎯 Specialist Node: Starting")
    
    try:
        current_deal = state.get("current_deal")
        
        if not current_deal:
            logger.warning("Specialist Node: No current deal")
            return {"specialist_price": 0.0}
        
        # Create and run Specialist Agent
        agent = SpecialistAgent()
        price = agent.price(current_deal.product_description)
        
        logger.info(f"Specialist Node: Predicted ${price:.2f}")
        
        return {"specialist_price": price}
        
    except Exception as e:
        logger.error(f"Specialist Node failed: {str(e)}")
        return {
            "specialist_price": 0.0,
            "errors": state.get("errors", []) + [f"Specialist failed: {str(e)}"]
        }


def nn_node(state: DealWorkflowState) -> Dict[str, Any]:
    """
    Neural Network Node: PyTorch-based price prediction.
    """
    logger.info("🤖 Neural Network Node: Starting")
    
    try:
        current_deal = state.get("current_deal")
        
        if not current_deal:
            logger.warning("NN Node: No current deal")
            return {"nn_price": 0.0}
        
        # Create and run Neural Network Agent
        agent = NeuralNetworkAgent()
        price = agent.price(current_deal.product_description)
        
        logger.info(f"NN Node: Predicted ${price:.2f}")
        
        return {"nn_price": price}
        
    except Exception as e:
        logger.error(f"NN Node failed: {str(e)}")
        return {
            "nn_price": 0.0,
            "errors": state.get("errors", []) + [f"NN failed: {str(e)}"]
        }


# ============================================================
# NODE 7: ENSEMBLE AGGREGATOR
# ============================================================

def ensemble_aggregator_node(state: DealWorkflowState) -> Dict[str, Any]:
    """
    Ensemble Aggregator: Combine predictions into final estimate.
    """
    logger.info("⚖️ Ensemble Aggregator: Starting")
    
    try:
        current_deal = state.get("current_deal")
        
        if not current_deal:
            logger.warning("Ensemble Aggregator: No current deal")
            return {}
        
        # Get predictions
        frontier = state.get("frontier_price", 0.0)
        specialist = state.get("specialist_price", 0.0)
        nn = state.get("nn_price", 0.0)
        
        logger.info(f"Predictions: Frontier=${frontier:.2f}, Specialist=${specialist:.2f}, NN=${nn:.2f}")
        
        # Calculate weighted average
        estimated_price = (
            ENSEMBLE_WEIGHTS["frontier"] * frontier +
            ENSEMBLE_WEIGHTS["specialist"] * specialist +
            ENSEMBLE_WEIGHTS["neural_network"] * nn
        )
        
        # Calculate discount
        discount = estimated_price - current_deal.price
        
        logger.info(f"Ensemble: Estimated=${estimated_price:.2f}, Actual=${current_deal.price:.2f}, Discount=${discount:.2f}")
        
        # Create opportunity
        opportunity = Opportunity(
            deal=current_deal,
            estimate=estimated_price,
            discount=discount
        )
        
        # Add to opportunities list
        opportunities = state.get("opportunities", [])
        opportunities.append(opportunity)
        
        # Increment index for next deal
        current_index = state.get("current_deal_index", 0)
        
        return {
            "estimated_price": estimated_price,
            "discount": discount,
            "opportunities": opportunities,
            "current_deal_index": current_index + 1  # Move to next deal
        }
        
    except Exception as e:
        logger.error(f"Ensemble Aggregator failed: {str(e)}")
        return {
            "errors": state.get("errors", []) + [f"Aggregator failed: {str(e)}"]
        }


# ============================================================
# NODE 8: DECISION NODE
# ============================================================

def decision_node(state: DealWorkflowState) -> Dict[str, Any]:
    """
    Decision Node: Select best opportunity if above threshold.
    """
    logger.info("Decision Node: Starting")
    
    try:
        opportunities = state.get("opportunities", [])
        
        if not opportunities:
            logger.warning("Decision Node: No opportunities to evaluate")
            return {"best_opportunity": None}
        
        # Sort by discount (highest first)
        opportunities.sort(key=lambda opp: opp.discount, reverse=True)
        best = opportunities[0]
        
        logger.info(f"Decision Node: Best discount is ${best.discount:.2f}")
        
        # Check threshold
        threshold = DEAL_THRESHOLD
        
        if best.discount > threshold:
            logger.info(f"Decision Node: Discount exceeds threshold ${threshold}, selecting opportunity")
            return {"best_opportunity": best}
        else:
            logger.info(f"Decision Node: Discount below threshold ${threshold}, no notification")
            return {"best_opportunity": None}
        
    except Exception as e:
        logger.error(f"Decision Node failed: {str(e)}")
        return {
            "best_opportunity": None,
            "errors": state.get("errors", []) + [f"Decision failed: {str(e)}"]
        }


# ============================================================
# NODE 9: MESSAGING NODE
# ============================================================

def messaging_node(state: DealWorkflowState) -> Dict[str, Any]:
    """
    Messaging Node: Send notification to user.

    """
    logger.info("📲 Messaging Node: Starting")
    
    try:
        best_opportunity = state.get("best_opportunity")
        
        if not best_opportunity:
            logger.warning("Messaging Node: No opportunity to notify about")
            return {"notification_sent": False}
        
        # Create and run Messaging Agent
        agent = MessagingAgent()
        success = agent.alert(best_opportunity)
        
        if success:
            logger.info("Messaging Node: Notification sent successfully")
        else:
            logger.warning("Messaging Node: Notification failed")
        
        return {"notification_sent": success}
        
    except Exception as e:
        logger.error(f"Messaging Node failed: {str(e)}")
        return {
            "notification_sent": False,
            "errors": state.get("errors", []) + [f"Messaging failed: {str(e)}"]
        }


# ============================================================
# HELPER: CHECK IF MORE DEALS TO PROCESS
# ============================================================

def should_continue_processing(state: DealWorkflowState) -> str:
    """
    Conditional edge function: Check if more deals to process.
    """
    deals = state.get("processed_deals") or state.get("deals", [])
    current_index = state.get("current_deal_index", 0)
    
    if current_index < len(deals):
        logger.info(f"More deals to process: {current_index + 1}/{len(deals)}")
        return "continue"
    else:
        logger.info("All deals processed")
        return "done"


# ============================================================
# HELPER: CHECK IF SHOULD NOTIFY
# ============================================================

def should_notify(state: DealWorkflowState) -> str:
    """
    Conditional edge function: Check if should send notification.
    
    Returns:
        - "notify" if best_opportunity exists
        - "skip" if no opportunity or below threshold
        
    Used as a conditional edge after decision_node.
    """
    best_opportunity = state.get("best_opportunity")
    
    if best_opportunity:
        logger.info("Opportunity found, sending notification")
        return "notify"
    else:
        logger.info("No notification needed")
        return "skip"