"""
Ví dụ về AutoMerging Index trong LlamaIndex với Google Gemini Flash 2.0 và Qdrant
AutoMerging Index cho phép tự động merge các chunks liên quan để tạo ra context tốt hơn
"""

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
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes, get_root_nodes
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo

# from caller import MarketingDocs

import qdrant_client

# Load environment variables
load_dotenv()

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NodeStorageHandler:
    def __init__(self, google_api_key: str = None, 
                 qdrant_url: str = None, qdrant_api_key: str = None, collection_name: str = "sailing"):
        """
        Handler để storing nodes đã được xử lý sẵn

        Args:
            google_api_key: Google API key (nếu không có sẽ lấy từ environment)
            qdrant_url: Qdrant server URL (nếu không có sẽ sử dụng local)
            qdrant_api_key: Qdrant API key cho cloud
            collection_name: Tên collection trong Qdrant
        """
        self.collection_name = collection_name

        # Khởi tạo LLM và Embedding model với Google Gemini
        self.llm = GoogleGenAI(
            model_name="gemini-2.0-flash",
            temperature=0.1,
        )
        self.embed_model = BedrockEmbedding(
            model_name="amazon.titan-embed-text-v2:0",
            region_name='us-west-2',
        )

        # Cấu hình Settings global
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        # Thiết lập Qdrant client
        qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.qdrant_client = self._setup_qdrant_client(qdrant_url, qdrant_api_key)
        
        # Check connection
        self._check_connections()
        
        self.index = None
        self.auto_merging_retriever = None
        self.storage_context = None

    def _setup_qdrant_client(self, qdrant_url: Optional[str] = None, qdrant_api_key: Optional[str] = None):
        """
        Thiết lập Qdrant client
        """
        logger.debug("Thiết lập Qdrant client...")
        
        if qdrant_url and qdrant_api_key:
            # Qdrant Cloud
            client = qdrant_client.QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                timeout=60
            )
            logger.debug("Đã kết nối với Qdrant Cloud")
        else:
            # Local Qdrant
            client = qdrant_client.QdrantClient(path="./qdrant_storage")
            logger.debug("Đã khởi tạo Qdrant local")
            
        return client

    def _check_connections(self):
        """
        Kiểm tra kết nối đến các services
        """
        logger.debug("Kiểm tra kết nối...")
    
        # Check Qdrant
        try:
            collections = self.qdrant_client.get_collections()
            logger.debug(f"✅ Qdrant connection OK - {len(collections.collections)} collections")
        except Exception as e:
            logger.error(f"❌ Qdrant connection failed: {e}")

    def build_automerging_index(self, nodes: List, persist_dir: str = "./storage_testing", insert_batch_size: int = 20):
        """
        Xây dựng hoặc cập nhật AutoMerging Index bằng cách thêm các node mới vào các store hiện có.
        """
        # Bắt buộc phải có nodes để thêm vào
        if not nodes:
            logger.warning("Không có node nào được cung cấp để thêm vào. Bỏ qua.")
            # Nếu index chưa được load, cố gắng load nó từ storage hiện có
            if not self.index and os.path.exists(persist_dir):
                logger.debug(f"Đang thử tải index hiện có từ {persist_dir}...")
                vector_store = QdrantVectorStore(client=self.qdrant_client, collection_name=self.collection_name)
                self.storage_context = StorageContext.from_defaults(persist_dir=persist_dir, vector_store=vector_store)
                self.index = load_index_from_storage(self.storage_context)
                logger.debug("Đã tải index thành công.")
            return self.index

        logger.debug(f"Chuẩn bị thêm {len(nodes)} nodes mới vào các kho lưu trữ...")

        # Thiết lập Vector Store
        vector_store = QdrantVectorStore(
            client=self.qdrant_client, 
            collection_name=self.collection_name
        )

        # Tạo mới Storage Context (để quản lý docstore và index_store)
        if not self.storage_context:
            if os.path.exists(persist_dir):
                logger.debug(f"Tìm thấy storage tại {persist_dir}, đang tải...")
                self.storage_context = StorageContext.from_defaults(
                    persist_dir=persist_dir,
                    vector_store=vector_store
                )
            else:
                logger.debug("Không tìm thấy storage, đang tạo mới...")
                self.storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Tạo mới Index
        if not self.index:
            try:
                logger.debug("Đang thử tải index từ storage context...")
                self.index = load_index_from_storage(self.storage_context)
                logger.debug("Đã tải index hiện có thành công.")
            except Exception:
                logger.debug("Không thể tải index (có thể là lần chạy đầu tiên). Sẽ tạo index mới.")
                self.index = None

        leaf_nodes = get_leaf_nodes(nodes)
        logger.debug(f"Chuẩn bị chèn {len(leaf_nodes)} leaf nodes vào vector store.")
        
        if self.index is None:
            # Nếu chưa có index nào -> tạo mới hoàn toàn với các node đầu tiên
            logger.debug("Tạo VectorStoreIndex mới...")
            self.index = VectorStoreIndex(
                nodes=leaf_nodes,
                storage_context=self.storage_context,
                insert_batch_size=insert_batch_size,
                show_progress=True
            )
        else:
            # Nếu đã có index -> chỉ chèn các node mới vào
            logger.debug("Chèn các node mới vào VectorStoreIndex hiện có...")
            self.index.insert_nodes(
                leaf_nodes,
                show_progress=True
            )

        # 5. Lưu lại trạng thái mới của storage context (quan trọng!)
        self.storage_context.persist(persist_dir=persist_dir)
        logger.debug(f"Đã lưu lại storage context tại {persist_dir}")
        
        logger.debug("Đã xây dựng/cập nhật xong AutoMerging Index.")
        return self.index

    def _create_custom_auto_retriever(self, similarity_top_k: int = 30):
        """
        Tạo ra auto retrieve 
        """
        logger.debug("Định nghĩa VectorStoreInfo cho auto-retriever")
        # Our metadata
        vector_store_info = VectorStoreInfo(
            content_info="A collection of marketing documents, analyses, and case studies focused on marketing strategy and customer value management.",
            metadata_info=[
                {
                    "name": "Category",
                    "type": "List[str]",
                    "description": (
                        "A list of high-level marketing domains or topics the document belongs to. "
                        "A single document can belong to multiple categories. "
                        "Use this filter when the user's query refers to a specific field, topic, or type of document. "
                        "Examples include: 'Customer Value Analysis', 'Customer Segmentation', 'Loyalty & Retention', "
                        "'Branding & Positioning' and many more"   
                    ),
                },
                {
                    "name": "Keywords",
                    "type": "List[str]",
                    "description": (
                        "A list of specific marketing terms, models, or concepts discussed in the document. "
                        "Use this filter when the user's query includes a specific technical term, acronym, or methodology. "
                        "Examples include: 'Customer Lifetime Value (CLV)', 'Churn Rate Analysis', 'Retention Rate', "
                        "'Customer Investment Management (CIM)', 'Marketing ROI', 'Cross-sell & Up-sell', "
                        "'Decile Analysis' and many more"
                    ),
                },
            ]
        )

        logger.debug("Mẫu prompt tùy chỉnh với các ví dụ")
        # Redefine the default prompt
        prompt_tmpl_str = """
        Your goal is to structure the user's query to match the request schema provided below.

        << Structured Request Schema >>
        When responding use a markdown code snippet with a JSON object formatted in the following schema:

        {schema_str}

        The query string should contain only text that is expected to match the contents of documents. Any conditions in the filter should not be mentioned in the query as well.
        - Make sure that filters only refer to attributes that exist in the data source.
        - Make sure that filters are only used as needed. If there are no filters that should be applied, return [] for the filter value.

        << Example 1: Filtering by a single Category >>
        User Query: "Tell me about loyalty and retention"
        Structured Request:
        ```json
        {{
            "query": "strategies for loyalty and retention",
            "filters": [
                {{"key": "Category", "value": "Loyalty & Retention", "operator": "=="}}
            ]
        }}
        ```
        << Example 2: Filtering by both Category and Keyword >>
        User Query: "Show me case studies related to Marketing ROI"
        Structured Request:
        ```json
        {{
            "query": "Marketing ROI analysis",
            "filters": [
                {{"key": "Category", "value": "Case Study", "operator": "=="}},
                {{"key": "Keywords", "value": "Marketing ROI", "operator": "=="}}
            ]
        }}
        ```
        User's Request
        Data Source:
        ```json
        {info_str}

        User Query:
        {query_str}

        Structured Request:
        """
        
        custom_prompt = PromptTemplate(prompt_tmpl_str)

        logger.debug("Khởi tạo VectorIndexAutoRetriever với custom prompt")
        
        base_retriever = VectorIndexAutoRetriever(
            self.index,
            vector_store_info=vector_store_info,
            output_parser_prompt=custom_prompt,
            similarity_top_k=similarity_top_k,
            verbose=True
        )
        return base_retriever

    def create_query_engine(self, persist_dir: str = "./storage_testing", similarity_top_k: int = 30):
        """
        Tạo Query Engine với AutoMerging Retriever
        """
        logger.debug("Tạo query engine từ stored nodes...")
        
        # Load từ storage nếu chưa có index
        if not self.index:
            self.build_automerging_index(persist_dir)
        
        # Tạo AutoRetriever
        base_retriever = self._create_custom_auto_retriever(similarity_top_k=similarity_top_k)

        # Use this if custome retriever not working
        # base_retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)

        # Tạo AutoMerging Retriever
        retriever = AutoMergingRetriever(base_retriever, 
                                         self.storage_context,
                                         verbose=True)
        
        # Tạo Query Engine
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            response_mode=ResponseMode.TREE_SUMMARIZE,
            callback_manager=CallbackManager([LlamaDebugHandler(print_trace_on_end=True)])
        )
        
        logger.debug("Query engine đã sẵn sàng!")
        return query_engine

    def setup_complete_pipeline(self, persist_dir: str = "./storage_testing", similarity_top_k: int = 30):
        """
        Thiết lập pipeline hoàn chỉnh từ nodes đến query engine
        """
        logger.debug("Thiết lập complete pipeline...")
        
        logger.debug("Lấy nodes từ MarketingDocs...")

        # md = MarketingDocs()
        # all_nodes = md.get_nodes()

        # logger.info(f"Đã lấy {len(all_nodes)} nodes")
        
        # # Build index
        # logger.info("Xây dựng AutoMerging Index...")
        # self.build_automerging_index(all_nodes, persist_dir)
        
        # # Tạo query engine
        # logger.info("Tạo query engine...")
        # query_engine = self.create_query_engine(persist_dir, similarity_top_k)
        
        # logger.info("omplete pipeline đã sẵn sàng!")
        # return query_engine
        pass
