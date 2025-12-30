from pydantic import BaseModel, Field  
from typing import List, Dict, Self  
from bs4 import BeautifulSoup  # HTML parsing
import re  
import feedparser  # RSS feed parsing
from tqdm import tqdm  # Progress bar for scraping
import requests  # HTTP requests to fetch deal pages
import time  

# RSS feed sources for deal scraping (electronics, computers, smart home)
feeds = [
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    # "https://www.dealnews.com/c39/Computers/?rss=1",
    # "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
]

# Additional feeds available but commented out:
# "https://www.dealnews.com/c238/Automotive/?rss=1"
# "https://www.dealnews.com/c196/Home-Garden/?rss=1"


def extract(html_snippet: str) -> str:
    """
    Clean HTML snippet and extract plain text using BeautifulSoup.
    Removes HTML tags and extracts content from 'snippet summary' div.
    """
    soup = BeautifulSoup(html_snippet, "html.parser")
    snippet_div = soup.find("div", class_="snippet summary")

    if snippet_div:
        description = snippet_div.get_text(strip=True)
        description = BeautifulSoup(description, "html.parser").get_text()
        description = re.sub("<[^<]+?>", "", description)
        result = description.strip()
    else:
        result = html_snippet
    return result.replace("\n", " ")


class ScrapedDeal:
    """
    Raw deal data scraped from RSS feeds.
    Fetches full deal details by following the URL to the deal page.
    """

    category: str
    title: str
    summary: str
    url: str
    details: str
    features: str

    def __init__(self, entry: Dict[str, str]):
        """
        Initialize deal from RSS feed entry.
        Fetches full page content to extract details and features.
        """
        self.title = entry["title"]
        self.summary = extract(entry["summary"])
        self.url = entry["links"][0]["href"]
        # Fetch the full deal page
        stuff = requests.get(self.url).content
        soup = BeautifulSoup(stuff, "html.parser")
        content = soup.find("div", class_="content-section").get_text()
        content = content.replace("\nmore", "").replace("\n", " ")
        # Split content into details and features if "Features" section exists
        if "Features" in content:
            self.details, self.features = content.split("Features", 1)
        else:
            self.details = content
            self.features = ""
        self.truncate()

    def truncate(self):
        self.title = self.title[:100]
        self.details = self.details[:500]
        self.features = self.features[:500]

    def __repr__(self):
        """String representation for debugging."""
        return f"<{self.title}>"

    def describe(self):
        return f"Title: {self.title}\nDetails: {self.details.strip()}\nFeatures: {self.features.strip()}\nURL: {self.url}"

    @classmethod
    def fetch(cls, show_progress: bool = False) -> List[Self]:
        deals = []
        feed_iter = tqdm(feeds) if show_progress else feeds
        for feed_url in feed_iter:
            feed = feedparser.parse(feed_url)
            # Only take first 10 entries from each feed
            for entry in feed.entries[:10]:
                deals.append(cls(entry))
                time.sleep(0.05)  # Rate limiting
        return deals


class Deal(BaseModel):

    product_description: str = Field(
        description="Your clearly expressed summary of the product in 3-4 sentences. Details of the item are much more important than why it's a good deal. Avoid mentioning discounts and coupons; focus on the item itself. There should be a short paragraph of text for each item you choose."
    )
    price: float = Field(
        description="The actual price of this product, as advertised in the deal. Be sure to give the actual price; for example, if a deal is described as $100 off the usual $300 price, you should respond with $200"
    )
    url: str = Field(description="The URL of the deal, as provided in the input")


class DealSelection(BaseModel):

    deals: List[Deal] = Field(
        description="Your selection of the 5 deals that have the most detailed, high quality description and the most clear price. You should be confident that the price reflects the deal, that it is a good deal, with a clear description"
    )


class Opportunity(BaseModel):

    deal: Deal  
    estimate: float 
    discount: float  
