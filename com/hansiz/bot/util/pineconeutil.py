from pinecone import Pinecone

from com.hansiz.bot.util.config import GLOBAL_CONFIG
from com.hansiz.bot.util.logger import logger


def initialize_pinecone():
    """
    Initialize Pinecone with proper error handling
    """
    try:
        api_key = GLOBAL_CONFIG.pinecone.api_key
        region = GLOBAL_CONFIG.pinecone.region

        if not api_key or api_key.startswith("${"):
            raise ValueError("PINECONE_API_KEY is not properly set in environment variables")

        if not region or region.startswith("${"):
            raise ValueError("PINECONE_ENV is not properly set in environment variables")

        pc = Pinecone(api_key=api_key, environment=region)
        return pc

    except Exception as e:
        logger.error(f"Error initializing Pinecone: {str(e)}")
        return False
