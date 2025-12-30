import logging
from langgraph.graph import StateGraph, END
from workflows.state import DealWorkflowState, create_initial_state
from workflows.nodes import (
    scanner_node,
    preprocessing_node,
    ensemble_coordinator_node,
    frontier_node,
    specialist_node,
    nn_node,
    ensemble_aggregator_node,
    decision_node,
    messaging_node,
    should_continue_processing,
    should_notify,
)

logger = logging.getLogger(__name__)


def build_deal_workflow() -> StateGraph:
    logger.info("🏗️  Building LangGraph workflow...")
    
    # Create the state graph
    workflow = StateGraph(DealWorkflowState)

    # Node 1: Scanner (find deals from RSS)
    workflow.add_node("scanner", scanner_node)
    
    # Node 2: Preprocessing (clean descriptions) - Optional
    workflow.add_node("preprocessing", preprocessing_node)
    
    # Node 3: Ensemble Coordinator (set current deal)
    workflow.add_node("ensemble_coordinator", ensemble_coordinator_node)
    
    # Nodes 4-6: Parallel Prediction Agents
    workflow.add_node("frontier", frontier_node)
    workflow.add_node("specialist", specialist_node)
    workflow.add_node("nn", nn_node)
    
    # Node 7: Ensemble Aggregator (combine predictions)
    workflow.add_node("ensemble_aggregator", ensemble_aggregator_node)
    
    # Node 8: Decision (select best deal)
    workflow.add_node("decision", decision_node)
    
    # Node 9: Messaging (send notification)
    workflow.add_node("messaging", messaging_node)
    
    # Entry point: START → scanner
    workflow.set_entry_point("scanner")
    
    # scanner → preprocessing
    workflow.add_edge("scanner", "preprocessing")
    
    # preprocessing → ensemble_coordinator
    workflow.add_edge("preprocessing", "ensemble_coordinator")
    
    # Ensemble coordinator → all 3 predictors (parallel fan-out)
    workflow.add_edge("ensemble_coordinator", "frontier")
    workflow.add_edge("ensemble_coordinator", "specialist")
    workflow.add_edge("ensemble_coordinator", "nn")
    
    # All 3 predictors → ensemble aggregator (fan-in)
    workflow.add_edge("frontier", "ensemble_aggregator")
    workflow.add_edge("specialist", "ensemble_aggregator")
    workflow.add_edge("nn", "ensemble_aggregator")
    
    # After aggregator: check if more deals to process
    workflow.add_conditional_edges(
        "ensemble_aggregator",
        should_continue_processing,
        {
            "continue": "ensemble_coordinator",  # Loop back to process next deal
            "done": "decision"                    # All deals processed, move to decision
        }
    )
    
    # After decision: check if should send notification
    workflow.add_conditional_edges(
        "decision",
        should_notify,
        {
            "notify": "messaging",  # Send notification
            "skip": END             # No notification needed, end workflow
        }
    )
    
    
    # After messaging → END
    workflow.add_edge("messaging", END)
    
    logger.info("✅ Workflow built successfully")
    logger.info("📊 Workflow structure:")
    logger.info("   START → scanner → preprocessing → ensemble_coordinator")
    logger.info("      ↓")
    logger.info("   [frontier || specialist || nn] (PARALLEL)")
    logger.info("      ↓")
    logger.info("   ensemble_aggregator → (loop or continue)")
    logger.info("      ↓")
    logger.info("   decision → (notify or skip)")
    logger.info("      ↓")
    logger.info("   messaging → END")
    
    # Compile the graph
    return workflow.compile()


def run_deal_workflow(chroma_collection, memory=None):

    logger.info("tarting deal discovery workflow...")
    
    # Build the graph
    graph = build_deal_workflow()
    
    # Create initial state
    initial_state = create_initial_state(chroma_collection, memory)
    
    # Run the workflow
    logger.info("▶Executing workflow...")
    final_state = graph.invoke(initial_state)
    
    # Log results
    logger.info("Workflow completed")
    
    if final_state.get("best_opportunity"):
        opp = final_state["best_opportunity"]
        logger.info(f"🎉 Found opportunity: {opp.deal.product_description[:50]}...")
        logger.info(f"   Price: ${opp.deal.price:.2f}, Estimate: ${opp.estimate:.2f}, Discount: ${opp.discount:.2f}")
    else:
        logger.info("No opportunities found above threshold")
    
    if final_state.get("errors"):
        logger.warning(f"Workflow had {len(final_state['errors'])} errors:")
        for error in final_state["errors"]:
            logger.warning(f"   - {error}")
    
    return final_state


# Exported names

__all__ = [
    "build_deal_workflow",
    "run_deal_workflow",
]