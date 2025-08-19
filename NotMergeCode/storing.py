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
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from llama_index.core import PromptTemplate
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes, get_root_nodes
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from qdrant_client.http.models import Distance, VectorParams, PointStruct, UpdateStatus, PayloadSchemaType
from llama_index.core.schema import QueryBundle

from caller import MarketingDocs

import qdrant_client

# Load environment variables
load_dotenv()

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryFilters(BaseModel):
    """Data model for query filters."""
    categories: List[str] = Field(description="List of categories to filter on.")
    keywords: List[str] = Field(description="List of keywords to filter on.")

def get_valid_filters_from_db():
    # This would query to PostgreSQL database
    # SELECT DISTINCT "Category" FROM ...; SELECT DISTINCT "Keywords" FROM ...;
    return {
        "categories": [
            "Customer Value Analysis",
            "Customer Segmentation",
            "Loyalty & Retention",
            "Branding & Positioning",
            "Retail Marketing",
            "Promotional Tactics",
            "Performance Measurement",
            "Case Study"
        ],
        "keywords": [
            "Customer Lifetime Value (CLV)",
            "Churn Rate Analysis",
            "Retention Rate",
            "Customer Investment Management (CIM)",
            "Marketing ROI",
            "Cross-sell & Up-sell",
            "Decile Analysis",
            "Customer Profiling",
            "Targeted Promotion",
            "Sales Forecasting",
            "High-Value Customer Identification",
            "Marketing Framework (Big Picture)",
            "STP (Segmentation, Targeting, Positioning)",
            "Customer-centric Service",
            "Brand Storytelling",
            "Moment of Truth (真実の瞬間)",
            "Marketing Mix (4Ps)",
            "Relationship Marketing",
            "Lifestyle Segmentation",
            "Demand Creation (需要創造)",
            "Marketing Myopia (マーケティング近視眼)"
        ]
        }


def generate_filters_for_query(query_str: str, llm: any) -> QueryFilters:
    """Uses an LLM to extract relevant categories and keywords from a user query."""
    logger.debug(f"Generating filters for query: {query_str}")
    valid_filters = get_valid_filters_from_db()
    
    prompt = PromptTemplate(
        "You're an expert at determine keywords and category for filtering metadata"
        "Your task is to base on the user's query, identify relevant filters from the available options.\n"
        "Respond ONLY with a JSON object. If no filters are relevant, return empty lists.\n\n"
        "Available Categories: {categories}\n"
        "Available Keywords: {keywords}\n\n"
        "User Query: \"{query_str}\""
    )
    
    program = LLMTextCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(output_cls=QueryFilters),
        prompt=prompt.partial_format(
            categories=valid_filters['categories'],
            keywords=valid_filters['keywords']
        ),
        llm=llm,
    )
    
    filter_object = program(query_str=query_str)
    logger.info(f"LLM generated filters: {filter_object.dict()}")
    return filter_object

class NodeStorageHandler:
    def __init__(self, google_api_key: str = None, 
                 qdrant_url: str = None, qdrant_api_key: str = None, collection_name: str = "sailing_test"):
        """
        Handler để storing nodes đã được xử lý sẵn

        Args:
            google_api_key: Google API key (nếu không có sẽ lấy từ environment)
            qdrant_url: Qdrant server URL (nếu không có sẽ sử dụng local)
            qdrant_api_key: Qdrant API key cho cloud
            collection_name: Tên collection trong Qdrant
        """
        self.collection_name = collection_name
        
        # Thiết lập Google API key
        if google_api_key:
            os.getenv["GOOGLE_API_KEY"] = google_api_key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("Chưa có GOOGLE_API_KEY. Vui lòng thiết lập API key để sử dụng Google Gemini models.")

        # Khởi tạo LLM và Embedding model với Google Gemini
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
        logger.info("Thiết lập Qdrant client...")
        
        if qdrant_url and qdrant_api_key:
            # Qdrant Cloud
            client = qdrant_client.QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                timeout=60
            )
            logger.info("Đã kết nối với Qdrant Cloud")
        else:
            # Local Qdrant
            client = qdrant_client.QdrantClient(path="./qdrant_storage")
            logger.info("Đã khởi tạo Qdrant local")

        try:
            collection_info = client.get_collection(collection_name=self.collection_name)
            
            # Check for 'Keywords' index
            if "Keywords" not in collection_info.payload_schema:
                logger.info(f"Creating payload index for 'Keywords' in collection '{self.collection_name}'...")
                client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="Keywords",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                logger.info("Index for 'Keywords' created successfully.")
            
            # Check for 'Category' index
            if "Category" not in collection_info.payload_schema:
                logger.info(f"Creating payload index for 'Category' in collection '{self.collection_name}'...")
                client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="Category",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                logger.info("Index for 'Category' created successfully.")
                
        except Exception as e:
            logger.warning(f"Could not check/create payload indexes (collection might not exist yet): {e}")
            
        return client

    def _check_connections(self):
        """
        Kiểm tra kết nối đến các services
        """
        logger.info("Kiểm tra kết nối...")
    
        # Check Qdrant
        try:
            collections = self.qdrant_client.get_collections()
            logger.info(f"✅ Qdrant connection OK - {len(collections.collections)} collections")
        except Exception as e:
            logger.error(f"❌ Qdrant connection failed: {e}")

    def build_automerging_index(self, nodes: List, persist_dir: str = "./storage", insert_batch_size: int = 20):
        """
        Xây dựng hoặc cập nhật AutoMerging Index và đảm bảo payload indexes tồn tại.
        """
        if not nodes:
            logger.warning("Không có node nào được cung cấp để thêm vào. Bỏ qua.")
            if not self.index and os.path.exists(persist_dir):
                logger.info(f"Đang thử tải index hiện có từ {persist_dir}...")
                vector_store = QdrantVectorStore(client=self.qdrant_client, collection_name=self.collection_name)
                self.storage_context = StorageContext.from_defaults(persist_dir=persist_dir, vector_store=vector_store)
                self.index = load_index_from_storage(self.storage_context)
                logger.info("Đã tải index thành công.")
            return self.index

        logger.info(f"Chuẩn bị thêm {len(nodes)} nodes mới vào các kho lưu trữ...")
        vector_store = QdrantVectorStore(client=self.qdrant_client, collection_name=self.collection_name)

        if not self.storage_context:
            if os.path.exists(persist_dir):
                self.storage_context = StorageContext.from_defaults(persist_dir=persist_dir, vector_store=vector_store)
            else:
                self.storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if not self.index:
            try:
                self.index = load_index_from_storage(self.storage_context)
                logger.info("Đã tải index hiện có thành công.")
            except Exception:
                logger.info("Không thể tải index. Sẽ tạo index mới.")
                self.index = None

        self.storage_context.docstore.add_documents(nodes)
        leaf_nodes = get_leaf_nodes(nodes)
        
        if self.index is None:
            logger.info("Tạo VectorStoreIndex mới...")
            self.index = VectorStoreIndex(
                nodes=leaf_nodes,
                storage_context=self.storage_context,
                insert_batch_size=insert_batch_size,
                show_progress=True
            )
        else:
            logger.info("Chèn các node mới vào VectorStoreIndex hiện có...")
            self.index.insert_nodes(leaf_nodes, show_progress=True)

        # This code runs AFTER the index (and the Qdrant collection) is guaranteed to exist.
        logger.info("Đảm bảo payload indexes cho 'Keywords' và 'Category' tồn tại...")
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
            logger.info("Payload indexes đã được xác nhận/tạo thành công.")
        except Exception as e:
            # This might raise an error if an operation is already in progress, which is okay.
            logger.warning(f"Could not create payload indexes (this might be okay if they already exist): {e}")

        self.storage_context.persist(persist_dir=persist_dir)
        logger.info(f"Đã lưu lại storage context tại {persist_dir}")
        
        return self.index


    def create_retrieval_pipeline(self, persist_dir: str = "./storage_testing", similarity_top_k: int = 30):
        """
        Returned query retrieval.
        """
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY not found in environment variables.")
        
        cohere_rerank = CohereRerank(
            api_key=cohere_api_key,
            top_n=5
        )
        
        if not self.index:
            self.build_automerging_index(persist_dir)

        logger.info("Creating a dynamic query function...")

        def dynamic_query_function(query_str: str):
            # Generate filters dynamically from the user's query
            filters = generate_filters_for_query(query_str, self.llm)
            
            # Construct metadata filters list
            metadata_filters_list = []
            if filters.categories:
                metadata_filters_list.extend([ExactMatchFilter(key="Category", value=c) for c in filters.categories])
            if filters.keywords:
                metadata_filters_list.extend([ExactMatchFilter(key="Keywords", value=k) for k in filters.keywords])
            
            # Create the retriever ON-THE-FLY for this specific query
            base_retriever = self.index.as_retriever(
                similarity_top_k=similarity_top_k,
                filters=MetadataFilters(filters=metadata_filters_list)
            )
            
            # retriever = AutoMergingRetriever(
            #     base_retriever, 
            #     self.storage_context,
            #     verbose=True
            # )

            # Perform the retrieval
            logger.info("Retrieving initial documents...")
            retrieved_nodes = base_retriever.retrieve(query_str)
            logger.info(f"Retrieved {len(retrieved_nodes)} initial nodes.")

            # Perform the reranking
            logger.info("Reranking retrieved documents with Cohere...")
            query_bundle = QueryBundle(query_str=query_str)
            reranked_nodes = cohere_rerank.postprocess_nodes(
                retrieved_nodes, query_bundle=query_bundle
            )
            logger.info(f"Reranked down to {len(reranked_nodes)} nodes.")

            return reranked_nodes


        return dynamic_query_function

    def setup_complete_pipeline(self, persist_dir: str = "./storage", similarity_top_k: int = 30):
        """
        Thiết lập pipeline hoàn chỉnh từ nodes đến query engine
        """
        logger.info("Thiết lập complete pipeline...")
        
        # Lấy nodes từ MarketingDocs
        logger.info("Lấy nodes từ MarketingDocs...")

        md = MarketingDocs()
        all_nodes = md.get_nodes()

        logger.info(f"Đã lấy {len(all_nodes)} nodes")
        
        # Build index
        logger.info("Xây dựng AutoMerging Index...")
        self.build_automerging_index(all_nodes, persist_dir)
        
        # Tạo query engine
        retrieve = self.create_retrieval_pipeline(persist_dir, similarity_top_k)
        
        logger.info("Complete pipeline đã sẵn sàng!")
        return retrieve

def main():
    """
    Hàm main để testing NodeStorageHandler với MarketingDocs
    """
    try:
        print("=== KHỞI TẠO NODE STORAGE HANDLER ===")
        storage_handler = NodeStorageHandler(
            collection_name="sailing_test" # Or "sailing_test"
        )

        # This now returns your retrieval function
        print("=== THIẾT LẬP COMPLETE RETRIEVAL PIPELINE ===")
        retrieval = storage_handler.setup_complete_pipeline()

        print("\n" + "="*50)
        print("=== CHẾ ĐỘ INTERACTIVE RETRIEVAL ===")
        print("Nhập câu hỏi của bạn (gõ 'quit' để thoát):")
        print("="*50)
        
        while True:
            user_query = input("\nCâu hỏi: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q', 'thoát']:
                print("\nCảm ơn bạn đã sử dụng!!")
                break
                
            if not user_query:
                print("Vui lòng nhập câu hỏi!")
                continue
                
            try:
                print("\nĐang tìm kiếm và rerank...")
                # The pipeline now returns a list of nodes
                retrieve_nodes = retrieval(user_query)
                
                print("\n✅ Top Reranked Documents:\n")
                if not retrieve_nodes:
                    print("--> Không tìm thấy tài liệu nào liên quan.")
                else:
                    for i, node_with_score in enumerate(retrieve_nodes):
                        print(f"--- Document {i+1}---")
                        print(f"\nMetadata: {node_with_score.node.metadata}")
                        print(f"\nContent: \n{node_with_score.node.get_content().strip()}")
                        print("-" * 80)
                
            except Exception as e:
                print(f"Lỗi khi xử lý câu hỏi: {e}")

    except Exception as e:
        logger.error(f"Lỗi trong main: {e}", exc_info=True)
        print(f"Đã xảy ra lỗi nghiêm trọng: {e}")

if __name__ == "__main__":
    main()