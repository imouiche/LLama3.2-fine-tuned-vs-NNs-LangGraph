import os
import sys
import logging
from dotenv import load_dotenv
from huggingface_hub import login
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.items import Item

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class VectorDBPopulator:
    """
    Populates Chroma vector database with Amazon product data.
    
    Process:
    1. Load dataset from Hugging Face Hub
    2. Initialize Chroma DB and encoder
    3. Encode products into vectors (384-dimensional)
    4. Add to Chroma collection in batches
    """
    
    DB_PATH = "products_vectorstore"
    COLLECTION_NAME = "products"
    ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    BATCH_SIZE = 1000
    
    def __init__(self, lite_mode=False):

        self.lite_mode = lite_mode
        self.dataset_name = None
        self.train = None
        self.val = None
        self.test = None
        self.client = None
        self.collection = None
        self.encoder = None
        
    def setup_huggingface(self):
        """Login to Hugging Face Hub."""
        load_dotenv(override=True)
        hf_token = os.environ.get('HF_TOKEN')
        
        if not hf_token:
            logger.error("HF_TOKEN not found in environment variables!")
            logger.error("Please set HF_TOKEN in your .env file")
            sys.exit(1)
        
        logger.info("Logging in to Hugging Face Hub...")
        login(token=hf_token, add_to_git_credential=False)
        logger.info(" Logged in to Hugging Face Hub")
    
    def load_dataset(self):
        username = "Inoussa-guru"
        self.dataset_name = f"{username}/items_lite" if self.lite_mode else f"{username}/items_full"
        
        logger.info(f"Loading dataset: {self.dataset_name}")
        logger.info("This may take a few minutes...")
        
        try:
            self.train, self.val, self.test = Item.from_hub(self.dataset_name)
            
            logger.info(f"Loaded dataset:")
            logger.info(f" Training: {len(self.train):,} items")
            logger.info(f" Validation: {len(self.val):,} items")
            logger.info(f"Test: {len(self.test):,} items")
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            sys.exit(1)
    
    def initialize_chroma(self):
        """Initialize Chroma DB client and encoder."""
        logger.info(f"Initializing Chroma DB at: {self.DB_PATH}")
        self.client = chromadb.PersistentClient(path=self.DB_PATH)
        
        logger.info(f"Loading encoder model: {self.ENCODER_MODEL}")
        self.encoder = SentenceTransformer(self.ENCODER_MODEL)
        logger.info("✅ Encoder loaded (384-dimensional embeddings)")
    
    def check_existing_collection(self):
        try:
            # Try to get the collection
            collection = self.client.get_collection(self.COLLECTION_NAME)
            count = collection.count()
            
            logger.warning(f"Collection '{self.COLLECTION_NAME}' already exists with {count:,} items")
            response = input("Do you want to delete and recreate it? [y/N]: ")
            
            if response.lower() == 'y':
                logger.info("Deleting existing collection...")
                self.client.delete_collection(self.COLLECTION_NAME)
                logger.info("✅ Collection deleted")
                return False  # Continue with population
            else:
                logger.info("Keeping existing collection. Exiting.")
                sys.exit(0)
                
        except Exception as e:
            # Collection doesn't exist, continue
            logger.info("No existing collection found. Will create new one.")
            return False
    
    def populate_collection(self):
        logger.info(f"Creating collection: {self.COLLECTION_NAME}")
        self.collection = self.client.create_collection(self.COLLECTION_NAME)
        
        total_items = len(self.train)
        logger.info(f"Starting to encode and add {total_items:,} products...")
        logger.info(f"Batch size: {self.BATCH_SIZE}")
        
        if self.lite_mode:
            logger.info("⏱️  Estimated time: 10-15 minutes")
        else:
            logger.info("⏱️  Estimated time: 30-40 minutes (on GPU)")
        
        # Process in batches
        for i in tqdm(range(0, total_items, self.BATCH_SIZE), desc="Encoding products"):
            batch = self.train[i:i + self.BATCH_SIZE]
            
            # Extract documents (product summaries)
            documents = [item.summary for item in batch]
            
            # Encode to vectors (384-dimensional)
            vectors = self.encoder.encode(documents).astype(float).tolist()
            
            # Prepare metadata (category and price for each product)
            metadatas = [
                {"category": item.category, "price": item.price} 
                for item in batch
            ]
            
            # Generate unique IDs
            ids = [f"doc_{j}" for j in range(i, i + len(documents))]
            
            # Add to collection
            try:
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=vectors,
                    metadatas=metadatas
                )
            except Exception as e:
                logger.error(f"Failed to add batch {i//self.BATCH_SIZE}: {e}")
                continue
        
        # Verify
        final_count = self.collection.count()
        logger.info(f"Successfully added {final_count:,} products to Chroma DB")
        
        if final_count < total_items:
            logger.warning(f" Expected {total_items:,} but got {final_count:,}")
    
    def run(self):
        """Run the complete population process."""
        logger.info("="*60)
        logger.info("CHROMA VECTOR DB POPULATION")
        logger.info("="*60)
        logger.info(f"Mode: {'LITE (200K products)' if self.lite_mode else 'FULL (800K products)'}")
        logger.info("")
        
        try:
            # Step 1: Setup
            self.setup_huggingface()
            
            # Step 2: Load dataset
            self.load_dataset()
            
            # Step 3: Initialize Chroma
            self.initialize_chroma()
            
            # Step 4: Check if collection exists
            self.check_existing_collection()
            
            # Step 5: Populate
            self.populate_collection()
            
            logger.info("")
            logger.info("="*60)
            logger.info("POPULATION COMPLETE!")
            logger.info("="*60)
            logger.info(f"Database location: {self.DB_PATH}")
            logger.info(f"Collection name: {self.COLLECTION_NAME}")
            logger.info("")
            logger.info("You can now run the deal discovery system:")
            logger.info("  python deal_agent_framework.py")
            logger.info("  python price_is_right.py")
            logger.info("")
            
        except KeyboardInterrupt:
            logger.warning("\nProcess interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Populate Chroma vector database with Amazon product data'
    )
    parser.add_argument(
        '--lite',
        action='store_true',
        help='Use lite dataset (200K products) instead of full (800K)'
    )
    
    args = parser.parse_args()
    
    populator = VectorDBPopulator(lite_mode=args.lite)
    populator.run()


if __name__ == "__main__":
    main()