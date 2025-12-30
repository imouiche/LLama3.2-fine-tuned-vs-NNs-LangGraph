from typing import Optional, List 
from openai import OpenAI 
from tools.deals import ScrapedDeal, DealSelection
from tools.agent_colors import Agent 


class ScannerAgent(Agent):
    """
    Agent that discovers and filters product deals from RSS feeds.
    
    Process:
    1. Fetch deals from RSS feeds (dealnews.com)
    2. Filter out deals already seen (using memory)
    3. Use OpenAI Structured Outputs to select top 5 deals
    4. Return deals with clear prices and detailed descriptions
    
    Called by Planning Agent to find new deal opportunities.
    """

    MODEL = "gpt-5-mini"  # 

    # System prompt instructs LLM on deal selection criteria
    SYSTEM_PROMPT = """You identify and summarize the 5 most detailed deals from a list, by selecting deals that have the most detailed, high quality description and the most clear price.
    Respond strictly in JSON with no explanation, using this format. You should provide the price as a number derived from the description. If the price of a deal isn't clear, do not include that deal in your response.
    Most important is that you respond with the 5 deals that have the most detailed product description with price. It's not important to mention the terms of the deal; most important is a thorough description of the product.
    Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price. 
    """

    # User prompt prefix (deals will be inserted between prefix and suffix)
    USER_PROMPT_PREFIX = """Respond with the most promising 5 deals from this list, selecting those which have the most detailed, high quality product description and a clear price that is greater than 0.
    You should rephrase the description to be a summary of the product itself, not the terms of the deal.
    Remember to respond with a short paragraph of text in the product_description field for each of the 5 items that you select.
    Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price. 
    
    Deals:
    
    """

    # User prompt suffix
    USER_PROMPT_SUFFIX = "\n\nInclude exactly 5 deals, no more."

    name = "Scanner Agent"
    color = Agent.CYAN  # Will log in cyan

    def __init__(self):
        self.log("Scanner Agent is initializing")
        
        try:
            # Initialize OpenAI client (requires OPENAI_API_KEY env var)
            self.openai = OpenAI()
            self.log("Scanner Agent is ready")
        except Exception as e:
            error_msg = f"Failed to initialize Scanner Agent: {str(e)}"
            self.log(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg) from e

    def fetch_deals(self, memory: List) -> List[ScrapedDeal]:
        self.log("Scanner Agent is about to fetch deals from RSS feed")
        
        try:
            # Extract URLs from memory (deals we've already alerted about)
            urls = [opp.deal.url for opp in memory] if memory else []
            
            # Fetch all deals from RSS feeds (may take 30-60 seconds)
            scraped = ScrapedDeal.fetch()
            
            # Filter out deals already in memory
            result = [scrape for scrape in scraped if scrape.url not in urls]
            
            self.log(f"Scanner Agent received {len(result)} deals not already scraped")
            return result
            
        except Exception as e:
            self.log(f"ERROR: Failed to fetch deals: {str(e)}")
            return []

    def make_user_prompt(self, scraped: List[ScrapedDeal]) -> str:
        user_prompt = self.USER_PROMPT_PREFIX
        # Add each deal's description (title, details, features, URL)
        user_prompt += "\n\n".join([scrape.describe() for scrape in scraped])
        user_prompt += self.USER_PROMPT_SUFFIX
        return user_prompt

    def scan(self, memory: List = None) -> Optional[DealSelection]:
        # Default to empty list if no memory provided
        if memory is None:
            memory = []
        
        # Step 1: Fetch new deals
        scraped = self.fetch_deals(memory)
        
        # If no new deals, return None
        if not scraped:
            self.log("Scanner Agent found no new deals")
            return None
        
        try:
            # Step 2: Build prompt with all deals
            user_prompt = self.make_user_prompt(scraped)
            
            self.log("Scanner Agent is calling OpenAI using Structured Outputs")
            
            # Step 3: Call OpenAI with Structured Outputs
            # response_format=DealSelection ensures response matches our Pydantic model
            result = self.openai.chat.completions.parse(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=DealSelection,  # Structured output (Pydantic model)
                reasoning_effort="low",  # Minimal reasoning (was "minimal" - fixed)
            )
            
            # Extract parsed response (automatically validated against DealSelection)
            result = result.choices[0].message.parsed
            
            # Step 4: Filter out invalid deals (price <= 0)
            result.deals = [deal for deal in result.deals if deal.price > 0]
            
            self.log(
                f"Scanner Agent received {len(result.deals)} selected deals with price>0 from OpenAI"
            )
            
            # Return top deals (should be 5, but could be fewer)
            return result if result.deals else None
            
        except Exception as e:
            self.log(f"ERROR: OpenAI call failed: {str(e)}")
            return None

    def test_scan(self, memory: List = None) -> Optional[DealSelection]:
        # Hardcoded test deals (real products from past scrapes)
        results = {
            "deals": [
                {
                    "product_description": "The Hisense R6 Series 55R6030N is a 55-inch 4K UHD Roku Smart TV that offers stunning picture quality with 3840x2160 resolution. It features Dolby Vision HDR and HDR10 compatibility, ensuring a vibrant and dynamic viewing experience. The TV runs on Roku's operating system, allowing easy access to streaming services and voice control compatibility with Google Assistant and Alexa. With three HDMI ports available, connecting multiple devices is simple and efficient.",
                    "price": 178,
                    "url": "https://www.dealnews.com/products/Hisense/Hisense-R6-Series-55-R6030-N-55-4-K-UHD-Roku-Smart-TV/484824.html?iref=rss-c142",
                },
                {
                    "product_description": "The Poly Studio P21 is a 21.5-inch LED personal meeting display designed specifically for remote work and video conferencing. With a native resolution of 1080p, it provides crystal-clear video quality, featuring a privacy shutter and stereo speakers. This display includes a 1080p webcam with manual pan, tilt, and zoom control, along with an ambient light sensor to adjust the vanity lighting as needed. It also supports 5W wireless charging for mobile devices, making it an all-in-one solution for home offices.",
                    "price": 30,
                    "url": "https://www.dealnews.com/products/Poly-Studio-P21-21-5-1080-p-LED-Personal-Meeting-Display/378335.html?iref=rss-c39",
                },
                {
                    "product_description": "The Lenovo IdeaPad Slim 5 laptop is powered by a 7th generation AMD Ryzen 5 8645HS 6-core CPU, offering efficient performance for multitasking and demanding applications. It features a 16-inch touch display with a resolution of 1920x1080, ensuring bright and vivid visuals. Accompanied by 16GB of RAM and a 512GB SSD, the laptop provides ample speed and storage for all your files. This model is designed to handle everyday tasks with ease while delivering an enjoyable user experience.",
                    "price": 446,
                    "url": "https://www.dealnews.com/products/Lenovo/Lenovo-Idea-Pad-Slim-5-7-th-Gen-Ryzen-5-16-Touch-Laptop/485068.html?iref=rss-c39",
                },
                {
                    "product_description": "The Dell G15 gaming laptop is equipped with a 6th-generation AMD Ryzen 5 7640HS 6-Core CPU, providing powerful performance for gaming and content creation. It features a 15.6-inch 1080p display with a 120Hz refresh rate, allowing for smooth and responsive gameplay. With 16GB of RAM and a substantial 1TB NVMe M.2 SSD, this laptop ensures speedy performance and plenty of storage for games and applications. Additionally, it includes the Nvidia GeForce RTX 3050 GPU for enhanced graphics and gaming experiences.",
                    "price": 650,
                    "url": "https://www.dealnews.com/products/Dell/Dell-G15-Ryzen-5-15-6-Gaming-Laptop-w-Nvidia-RTX-3050/485067.html?iref=rss-c39",
                },
            ]
        }
        
        return DealSelection(**results)