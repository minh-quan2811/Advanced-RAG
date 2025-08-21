import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Optional

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http.models import PayloadSchemaType
import qdrant_client

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NodeStorageHandler:
    """
    Handles the setup of data stores, document processing, and index construction.
    Its primary responsibility is to prepare the storage and index for querying.
    """
    def __init__(self, collection_name: str = "sailing_test"):
        self.collection_name = collection_name

        # Initialize models and settings
        self._setup_models()

        # Setup Qdrant client
        self.qdrant_client = self._setup_qdrant_client(
            os.getenv("QDRANT_URL"), 
            os.getenv("QDRANT_API_KEY")
        )

        # Check connections to services
        self._check_connections()

        self.index: Optional[VectorStoreIndex] = None
        self.storage_context: Optional[StorageContext] = None

    def _setup_models(self):
        """Initializes and configures the LLM and embedding models."""
        logger.info("Initializing LLM and embedding models...")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY not found. Please set it to use Gemini models.")

        self.llm = GoogleGenAI(
            model_name="gemini-2.0-flash",
            temperature=0.1,
        )
        self.embed_model = BedrockEmbedding(
            model_name="amazon.titan-embed-text-v2:0",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION"),
        )

        # Cấu hình Settings global
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

    def _setup_qdrant_client(self, qdrant_url: Optional[str], qdrant_api_key: Optional[str]):
        """Sets up the Qdrant client for either cloud or local instance."""
        logger.info("Setting up Qdrant client...")
        if qdrant_url and qdrant_api_key:
            client = qdrant_client.QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)
            logger.info("Connected to Qdrant Cloud.")
        else:
            client = qdrant_client.QdrantClient(path="./qdrant_storage")
            logger.info("Initialized local Qdrant client.")
        return client

    def _check_connections(self):
        """Verifies connection to Qdrant."""
        logger.info("Checking external service connections...")
        try:
            collections = self.qdrant_client.get_collections()
            logger.info(f"Qdrant connection OK - Found {len(collections.collections)} collections.")
        except Exception as e:
            logger.error(f"Qdrant connection failed: {e}")
            raise

    def build_or_load_index(self, nodes: Optional[List] = None, insert_batch_size: int = 20):
        """
        Builds the index from documents if it doesn't exist, otherwise loads it from disk.
        """
        # Qdrant Vector store 
        vector_store = QdrantVectorStore(
            client=self.qdrant_client, 
            collection_name=self.collection_name
        )

        # Try to load existing index
        if self.index is None:
            logger.info(f"Trying to get index from collection '{self.collection_name}'...")
            try:
                self.index = VectorStoreIndex.from_vector_store(vector_store)
                logger.info("Successfully load index.")
            except Exception:
                logger.info("Can't load index.")
                self.index = None

        if nodes:
            leaf_nodes = get_leaf_nodes(nodes)
            if self.index:
                # If index exists, insert new nodes
                logger.info(f"Insert {len(leaf_nodes)} new nodes...")
                self.index.insert_nodes(leaf_nodes)
            else:
                # Index isn't built yet, create a new one
                logger.info(f"Creating Index with {len(leaf_nodes)} node...")
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                self.index = VectorStoreIndex(
                    nodes=leaf_nodes,
                    storage_context=storage_context,
                    insert_batch_size=insert_batch_size
                )

        logger.info("Index built and stored in Qdrant successfully.")

        self._ensure_payload_indexes()

        return self.index

    def _ensure_payload_indexes(self):
        """Ensures that Qdrant has indexes on metadata fields for efficient filtering."""
        logger.info("Ensuring payload indexes for 'Keywords' and 'Category' exist...")
        try:
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="Keywords",
                field_schema=PayloadSchemaType.KEYWORD
            )
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="Category",
                field_schema=PayloadSchemaType.KEYWORD
            )
            logger.info("Payload indexes are confirmed.")
        except Exception as e:
            logger.warning(f"Could not create payload indexes (this might be okay if they already exist or an operation is in progress): {e}")