import os 
from tools.deals import Opportunity  # Data model for opportunities
from tools.agent_colors import Agent 
from litellm import completion 
import requests 

# Pushover API endpoint
pushover_url = "https://api.pushover.net/1/messages.json"


class MessagingAgent(Agent):
    """
    Agent that sends notifications about deals to users.
    """

    name = "Messaging Agent"
    color = Agent.WHITE  # Will log in white
    MODEL = "gpt-5-mini"  # LLM for message crafting

    def __init__(self):
        """
        Initialize messaging agent with Pushover credentials.
        
        Reads from environment variables:
        - PUSHOVER_USER: User key from Pushover
        - PUSHOVER_TOKEN: App token from Pushover
        
        Get these from: https://pushover.net/
        """
        self.log("Messaging Agent is initializing")
        
        # Load Pushover credentials from environment
        self.pushover_user = os.getenv("PUSHOVER_USER", "your-pushover-user-if-not-using-env")
        self.pushover_token = os.getenv("PUSHOVER_TOKEN", "your-pushover-token-if-not-using-env")
        
        # Warn if using default values (likely won't work)
        if "your-pushover" in self.pushover_user or "your-pushover" in self.pushover_token:
            self.log("WARNING: Using default Pushover credentials - notifications may fail")
        
        self.log("Messaging Agent has initialized Pushover and LLM")

    def push(self, text: str) -> bool:
        """
        Send a push notification via Pushover API.
        """
        self.log("Messaging Agent is sending a push notification")
        
        try:
            # Prepare payload for Pushover API
            payload = {
                "user": self.pushover_user,      # User key
                "token": self.pushover_token,    # App token
                "message": text,                  # Message text
                "sound": "cashregister",         # Sound effect for deals
            }
            
            # Send POST request to Pushover
            response = requests.post(pushover_url, data=payload, timeout=10)
            
            # Check if successful
            if response.status_code == 200:
                self.log("Push notification sent successfully")
                return True
            else:
                self.log(f"WARNING: Push notification failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.log(f"ERROR: Failed to send push notification: {str(e)}")
            return False

    def alert(self, opportunity: Opportunity) -> bool:
        # Format message with deal details
        text = f"Deal Alert! Price=${opportunity.deal.price:.2f}, "
        text += f"Estimate=${opportunity.estimate:.2f}, "
        text += f"Discount=${opportunity.discount:.2f}: "
        text += opportunity.deal.product_description[:100] + "... "  # First 100 chars
        text += opportunity.deal.url
        
        # Send push notification
        success = self.push(text)
        
        if success:
            self.log("Messaging Agent has completed")
        
        return success

    def craft_message(
        self, description: str, deal_price: float, estimated_true_value: float
    ) -> str:
        """
        Use LLM to craft an exciting deal notification message.
        """
        # Build prompt for LLM
        user_prompt = "Please summarize this great deal in 2-3 sentences to be sent as an exciting push notification alerting the user about this deal.\n"
        user_prompt += f"Item Description: {description}\n"
        user_prompt += f"Offered Price: ${deal_price:.2f}\n"
        user_prompt += f"Estimated true value: ${estimated_true_value:.2f}\n"
        user_prompt += "\n\nRespond only with the 2-3 sentence message which will be used to alert & excite the user about this deal"
        
        try:
            # Call LLM to generate message
            response = completion(
                model=self.MODEL,
                messages=[{"role": "user", "content": user_prompt}],
                timeout=30  # Add timeout
            )
            
            # Extract message from response
            message = response.choices[0].message.content
            self.log("Successfully crafted message using LLM")
            return message
            
        except Exception as e:
            # Fallback to simple message if LLM fails
            self.log(f"WARNING: LLM message crafting failed: {e}, using fallback")
            discount_pct = ((estimated_true_value - deal_price) / estimated_true_value) * 100
            return f"Great deal alert! {description[:50]}... Only ${deal_price:.2f} (estimated value ${estimated_true_value:.2f}, save {discount_pct:.0f}%)"

    def notify(
        self, description: str, deal_price: float, estimated_true_value: float, url: str
    ) -> bool:
        self.log("Messaging Agent is using LLM to craft the message")
        # Craft message using LLM
        text = self.craft_message(description, deal_price, estimated_true_value)
        # Pushover has ~1024 char limit, so truncate message to 200 chars
        message = text[:200] + "... " + url
        
        # Send push notification
        success = self.push(message)
        
        if success:
            self.log("Messaging Agent has completed")
        
        return success