import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Optional

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader, 
    load_index_from_storage,
)
from llama_index.core import PromptTemplate
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core import SelectorPromptTemplate
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.retrievers import AutoMergingRetriever, VectorIndexAutoRetriever
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores import VectorStoreInfo
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.storage.index_store.postgres import PostgresIndexStore
import psycopg2
import qdrant_client
from qdrant_client.http.models import PayloadSchemaType


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NodeStorageHandler:
    def __init__(self, qdrant_url: str = None, qdrant_api_key: str = None, 
                 collection_name: str = "store_testing", index_id: str = "marketing_docs_index",
                 postgres_config: dict = None):
        """
        Handler for storing pre-processed nodes
        Args:
            qdrant_url: Qdrant server URL
            qdrant_api_key: Qdrant API key for cloud
            collection_name: Name of the collection in Qdrant
            postgres_config: PostgreSQL configuration dictionary (keys: host, port, user, password, database)
        """
        self.collection_name = collection_name
        self.index_id = index_id
        self.postgres_config = postgres_config or {
            "host": os.getenv("POSTGRES_HOST"),
            "port": os.getenv("POSTGRES_PORT"),
            "user": os.getenv("POSTGRES_USER"),
            "password": os.getenv("POSTGRES_PASSWORD"),
            "database": os.getenv("POSTGRES_DB"),
        }

        # Initialize LLM and Embedding model
        self.llm = GoogleGenAI(
            model_name="gemini-2.0-flash",
            temperature=0.1,
        )
        self.embed_model = BedrockEmbedding(
            model_name="amazon.titan-embed-text-v2:0",
            region_name='us-west-2',
        )

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        # Setting Qdrant client
        qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.qdrant_client = self._setup_qdrant_client(qdrant_url, qdrant_api_key)
        
        # Check connections
        self._check_connections()
        
        self.index = None
        self.auto_merging_retriever = None
        self.storage_context = None

    def _setup_qdrant_client(self, qdrant_url: Optional[str] = None, qdrant_api_key: Optional[str] = None):
        """
        Setup Qdrant client
        """
        logger.debug("Setting up Qdrant client...")
        
        if qdrant_url and qdrant_api_key:
            client = qdrant_client.QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                timeout=60
            )
            logger.debug("Connected to Qdrant Cloud")
        else:
            logger.warning("Can't connect to Qdrant Cloud")
            
        return client

    def _check_connections(self):
        """
        Check connections to services
        """
        logger.debug("Checking connections...")
    
        # Check Qdrant
        try:
            collections = self.qdrant_client.get_collections()
            logger.debug(f"Qdrant connection OK - {len(collections.collections)} collections")
        except Exception as e:
            logger.error(f"Qdrant connection failed: {e}")

        # Check PostgreSQL
        try:
            conn = psycopg2.connect(**self.postgres_config)
            conn.close()
            logger.debug("PostgreSQL connection OK")
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")

    def setup_stores(self):
        """
        Sets up Qdrant Vector Store, PostgreSQL DocStore, and IndexStore.
        """
        # Setup Qdrant Vector Store
        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name
        )

        # Setup PostgreSQL DocStore and IndexStore
        docstore = PostgresDocumentStore.from_params(
            host=self.postgres_config["host"],
            port=self.postgres_config["port"],
            database=self.postgres_config["database"],
            user=self.postgres_config["user"],
            password=self.postgres_config["password"]
        )

        index_store = PostgresIndexStore.from_params(
            host=self.postgres_config["host"],
            port=self.postgres_config["port"],
            database=self.postgres_config["database"],
            user=self.postgres_config["user"],
            password=self.postgres_config["password"]
        )

        return vector_store, docstore, index_store

    def _prepare_metadata_nodes(self, nodes: List[BaseNode]) -> List[BaseNode]:
        """
        Applies custom templates and metadata exclusions to nodes before indexing.
        This controls exactly what the LLM and Embedding models will process.
        """
        keys_to_excluded = ["file_path"]

        for i, node in enumerate(nodes):
            node.text_template = "Metadata:\n{metadata_str}\n-----\nContent:\n{content}"
            node.metadata_seperator = "\n"

            exclusion_lists = [
                node.excluded_llm_metadata_keys,
                node.excluded_embed_metadata_keys
            ]

            # Exclude metadata being seen by the LLM and EMBED
            for lst in exclusion_lists:
                for key in keys_to_excluded:
                    if key not in lst:
                        lst.append(key)

        return nodes

    def build_automerging_index(self, nodes: Optional[List] = None, insert_batch_size: int = 20):
        """
        Adding nodes to PostgreSQL and Qdrant or Load AutoMerging Index if no nodes provided.
        """
        # Create Storage Context
        vector_store, docstore, index_store = self.setup_stores()

        self.storage_context = StorageContext.from_defaults(
            docstore=docstore,
            index_store=index_store,
            vector_store=vector_store
        )

        if nodes:
            logger.debug(f"Adding {len(nodes)} nodes to the stores.")
            docstore.add_documents(nodes)

            prepared_nodes = get_leaf_nodes(nodes)
            leaf_nodes = self._prepare_metadata_nodes(prepared_nodes)

            self.index = VectorStoreIndex(
                nodes=leaf_nodes,
                storage_context=self.storage_context,
                insert_batch_size=insert_batch_size,
                show_progress=True
            )
            self.index.set_index_id(self.index_id)
            self._ensure_payload_indexes()
        else:
            logger.info("Loading existing index from storage.")
            self.index = load_index_from_storage(self.storage_context, index_id=self.index_id)

        logger.info("Finished building/updating AutoMerging Index.")
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
            logger.warning(f"Could not create payload indexes: {e}")