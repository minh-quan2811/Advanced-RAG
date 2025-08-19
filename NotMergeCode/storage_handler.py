import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Optional

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http.models import PayloadSchemaType
import qdrant_client

from caller import MarketingDocs 

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NodeStorageHandler:
    """
    Handles the setup of data stores, document processing, and index construction.
    Its primary responsibility is to prepare the storage and index for querying.
    """
    def __init__(self, collection_name: str = "sailing"):
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
            logger.info(f"✅ Qdrant connection OK - Found {len(collections.collections)} collections.")
        except Exception as e:
            logger.error(f"❌ Qdrant connection failed: {e}")
            raise

    def build_or_load_index(self, persist_dir: str = "./storage", insert_batch_size: int = 20):
        """
        Builds the index from documents if it doesn't exist, otherwise loads it from disk.
        Ensures the necessary payload indexes are created in Qdrant.
        """
        vector_store = QdrantVectorStore(client=self.qdrant_client, collection_name=self.collection_name)

        if os.path.exists(persist_dir):
            logger.info(f"Found existing storage at {persist_dir}. Loading index...")
            self.storage_context = StorageContext.from_defaults(persist_dir=persist_dir, vector_store=vector_store)
            self.index = load_index_from_storage(self.storage_context)
            logger.info("Index loaded successfully.")
        else:
            logger.info("No existing storage found. Building new index from scratch...")
            logger.info("Fetching nodes from MarketingDocs...")
            md = MarketingDocs()
            all_nodes = md.get_nodes()
            logger.info(f"Fetched {len(all_nodes)} nodes.")

            self.storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self.storage_context.docstore.add_documents(all_nodes)

            leaf_nodes = get_leaf_nodes(all_nodes)
            self.index = VectorStoreIndex(
                nodes=leaf_nodes,
                storage_context=self.storage_context,
                insert_batch_size=insert_batch_size,
                show_progress=True
            )

            logger.info(f"Persisting storage context to {persist_dir}...")
            self.storage_context.persist(persist_dir=persist_dir)
            logger.info("Index built and saved successfully.")

        # Ensure payload indexes exist for filtering after index is created/loaded
        self._ensure_payload_indexes()

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