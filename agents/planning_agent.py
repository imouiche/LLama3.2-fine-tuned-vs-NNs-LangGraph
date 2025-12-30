from typing import Optional, List  # Type hints
from tools.agent_colors import Agent  
from tools.deals import Deal, Opportunity  
from agents.scanner_agent import ScannerAgent  
from agents.ensemble_agent import EnsembleAgent  
from agents.messaging_agent import MessagingAgent 


class PlanningAgent(Agent):
    """
    Main orchestrator agent that runs the complete deal discovery workflow.
    
    Workflow:
    1. Scanner finds deals → List[Deal]
    2. Ensemble prices each deal → List[Opportunity]
    3. Find best deal (highest discount)
    4. If discount > threshold → Send notification
    5. Return best opportunity
    
    This agent is typically called:
    - On a schedule (e.g., every hour)
    - By a web UI (manual trigger)
    - By the Deal Agent Framework
    """

    name = "Planning Agent"
    color = Agent.GREEN  # Will log in green
    DEAL_THRESHOLD = 50  # Minimum discount ($) to trigger notification

    def __init__(self, collection):
        self.log("Planning Agent is initializing")
        
        try:
            # Initialize Scanner Agent (RSS + OpenAI)
            self.log("  Creating Scanner Agent...")
            self.scanner = ScannerAgent()
            
            # Initialize Ensemble Agent (coordinates 3 predictors)
            self.log("  Creating Ensemble Agent...")
            self.ensemble = EnsembleAgent(collection)
            
            # Initialize Messaging Agent (Pushover notifications)
            self.log("  Creating Messaging Agent...")
            self.messenger = MessagingAgent()
            
            self.log("Planning Agent is ready")
            
        except Exception as e:
            error_msg = f"Failed to initialize Planning Agent: {str(e)}"
            self.log(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg) from e

    def run(self, deal: Deal) -> Opportunity:
        self.log("Planning Agent is pricing up a potential deal")
        
        try:
            # Get price estimate from Ensemble (which coordinates 3 predictors)
            estimate = self.ensemble.price(deal.product_description)
            
            # Calculate discount (how much we estimate user would save)
            discount = estimate - deal.price
            
            self.log(f"Planning Agent has processed a deal with discount ${discount:.2f}")
            
            # Create and return Opportunity
            return Opportunity(deal=deal, estimate=estimate, discount=discount)
            
        except Exception as e:
            # If pricing fails, return Opportunity with 0 estimate/discount
            self.log(f"ERROR: Failed to price deal: {str(e)}")
            return Opportunity(deal=deal, estimate=0.0, discount=0.0)

    def plan(self, memory: List = None) -> Optional[Opportunity]:
  
        # Default to empty list if no memory provided
        if memory is None:
            memory = []
        
        self.log("Planning Agent is kicking off a run")
        
        try:
            # Step 1: Scanner Agent - Find new deals from RSS feeds
            self.log("  Step 1: Scanning for deals...")
            selection = self.scanner.scan(memory=memory)
            
            # If no new deals found, return None
            if not selection or not selection.deals:
                self.log("Planning Agent found no new deals")
                return None
            
            self.log(f"  Found {len(selection.deals)} deals to analyze")
            
            # Step 2: Ensemble Agent - Price each deal
            self.log("  Step 2: Pricing deals with Ensemble...")
            opportunities = []
            
            for i, deal in enumerate(selection.deals[:5], 1):  # Process up to 5 deals
                self.log(f"    Processing deal {i}/5...")
                try:
                    opp = self.run(deal)  # Get estimate and discount
                    opportunities.append(opp)
                except Exception as e:
                    self.log(f"    WARNING: Failed to process deal {i}: {e}")
                    continue
            
            # If all deals failed to process, return None
            if not opportunities:
                self.log("Planning Agent failed to process any deals")
                return None
            
            # Step 3: Sort by discount (highest first)
            opportunities.sort(key=lambda opp: opp.discount, reverse=True)
            best = opportunities[0]
            
            self.log(f"Planning Agent has identified the best deal has discount ${best.discount:.2f}")
            
            # Step 4: Check if discount meets threshold
            if best.discount > self.DEAL_THRESHOLD:
                self.log(f"  Discount ${best.discount:.2f} exceeds threshold ${self.DEAL_THRESHOLD}")
                
                try:
                    # Step 5: Send notification
                    self.log("  Step 3: Sending notification...")
                    self.messenger.alert(best)
                except Exception as e:
                    self.log(f" WARNING: Failed to send notification: {e}")
                
                self.log("Planning Agent has completed a run")
                return best  # Return the opportunity
            else:
                # Discount not high enough
                self.log(f"  Discount ${best.discount:.2f} below threshold ${self.DEAL_THRESHOLD}, not notifying")
                self.log("Planning Agent has completed a run")
                return None
            
        except Exception as e:
            self.log(f"ERROR: Planning workflow failed: {str(e)}")
            return None