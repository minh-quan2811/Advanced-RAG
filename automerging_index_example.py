"""
Ví dụ về AutoMerging Index trong LlamaIndex với Google Gemini Flash 1.5
AutoMerging Index cho phép tự động merge các chunks liên quan để tạo ra context tốt hơn
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import List

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader, load_index_from_storage,
)
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes, get_root_nodes
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.bedrock import BedrockEmbedding

# Load environment variables
load_dotenv()

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoMergingIndexDemo:
    def __init__(self, data_folder: str = "./client data", google_api_key: str = None):
        """
        Khởi tạo AutoMerging Index Demo với Google Gemini

        Args:
            data_folder: Đường dẫn đến folder chứa dữ liệu
            google_api_key: Google API key (nếu không có sẽ lấy từ environment)
        """
        self.data_folder = Path(data_folder)

        # Thiết lập Google API key
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
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

        self.index = None
        self.auto_merging_retriever = None

    def load_documents(self) -> List:
        """
        Load tất cả documents từ data folder
        """
        logger.info(f"Đang load documents từ {self.data_folder}")

        # Kiểm tra folder tồn tại
        if not self.data_folder.exists():
            raise FileNotFoundError(f"Folder {self.data_folder} không tồn tại!")

        # Sử dụng SimpleDirectoryReader để load documents
        reader = SimpleDirectoryReader(
            input_dir=str(self.data_folder),
            recursive=True,
            required_exts=[".docx", ".doc"]
        )

        documents = reader.load_data()
        logger.info(f"Đã load {len(documents)} documents")

        return documents

    def create_hierarchical_nodes(self, documents: List):
        """
        Tạo hierarchical nodes từ documents
        AutoMerging cần hierarchical structure để hoạt động hiệu quả
        """
        logger.info("Tạo hierarchical nodes...")

        # Tạo HierarchicalNodeParser với nhiều level chunk sizes
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[8192, 4096, 1024],  # 3 levels: coarse, medium, fine
            chunk_overlap=20
        )

        # Parse documents thành nodes
        nodes = node_parser.get_nodes_from_documents(documents)

        leaf_nodes = get_leaf_nodes(nodes)
        print("leaf_nodes:", len(leaf_nodes))

        root_nodes = get_root_nodes(nodes)
        print("root_nodes:", len(root_nodes))
        logger.info(f"Đã tạo {len(nodes)} hierarchical nodes")
        return nodes

    def build_automerging_index(self, nodes: List):
        """
        Xây dựng AutoMerging Index từ hierarchical nodes
        """
        logger.info("Đang xây dựng AutoMerging Index...")
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])
        # Tạo storage context
        if os.path.exists("./context"):
            storage_context = StorageContext.from_defaults(persist_dir="./context")
            self.index = load_index_from_storage(storage_context)

        else:
            storage_context = StorageContext.from_defaults()
            storage_context.docstore.add_documents(nodes)

            # Tạo VectorStoreIndex
            self.index = VectorStoreIndex(
                nodes=get_leaf_nodes(nodes),
                storage_context=storage_context,
                callback_manager=callback_manager,
                show_progress=True
            )
            storage_context.persist("./context")
        # Thêm nodes vào storage context

        logger.info("Đã xây dựng xong AutoMerging Index")

    def create_automerging_retriever(self, similarity_top_k: int = 60, verbose: bool = True):
        """
        Tạo AutoMerging Retriever

        Args:
            similarity_top_k: Số lượng nodes để retrieve
            verbose: Có hiển thị thông tin debug không
        """
        if not self.index:
            raise ValueError("Index chưa được xây dựng! Hãy gọi build_automerging_index() trước.")

        logger.info("Tạo AutoMerging Retriever...")

        # Lấy base retriever từ index
        base_retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)

        # Tạo AutoMerging Retriever
        self.auto_merging_retriever = AutoMergingRetriever(
            base_retriever,
            self.index.storage_context,
            verbose=verbose
        )

        logger.info("Đã tạo AutoMerging Retriever")

    def create_query_engine(self):
        """
        Tạo Query Engine với AutoMerging Retriever
        """
        if not self.auto_merging_retriever:
            raise ValueError("AutoMerging Retriever chưa được tạo! Hãy gọi create_automerging_retriever() trước.")

        logger.info("Tạo Query Engine...")

        # Tạo query engine với AutoMerging Retriever
        query_engine = RetrieverQueryEngine.from_args(
            retriever=self.auto_merging_retriever,
            response_mode=ResponseMode.TREE_SUMMARIZE,
            callback_manager=CallbackManager([LlamaDebugHandler(print_trace_on_end=True)])
        )

        logger.info("Đã tạo Query Engine")
        return query_engine

    def setup_complete_pipeline(self):
        """
        Thiết lập pipeline hoàn chỉnh cho AutoMerging Index
        """
        logger.info("=== Bắt đầu thiết lập AutoMerging Index Pipeline ===")

        # 1. Load documents
        documents = self.load_documents()

        # 2. Tạo hierarchical nodes
        nodes = self.create_hierarchical_nodes(documents)

        # 3. Xây dựng index
        self.build_automerging_index(nodes)

        # 4. Tạo retriever
        self.create_automerging_retriever()

        # 5. Tạo query engine
        query_engine = self.create_query_engine()

        logger.info("=== Hoàn thành thiết lập pipeline ===")
        return query_engine


def main():
    """
    Hàm main để demo AutoMerging Index với Google Gemini
    """
    try:
        # Khởi tạo demo với Google Gemini
        demo = AutoMergingIndexDemo(data_folder="./client data")

        # Thiết lập pipeline
        query_engine = demo.setup_complete_pipeline()

        print("\n=== CHẾ ĐỘ TƯƠNG TÁC ===")
        print("Nhập câu hỏi của bạn (gõ 'quit' để thoát):")

        while True:
            user_query = input("\nCâu hỏi: ").strip()

            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Cảm ơn bạn đã sử dụng AutoMerging Index Demo với Google Gemini!")
                break

            if not user_query:
                continue

            try:
                response = query_engine.query(user_query)
                print(f"\nTrả lời: {response}")

            except Exception as e:
                print(f"Lỗi: {e}")

    except Exception as e:
        logger.error(f"Lỗi trong main: {e}", exc_info=True)
        print(f"Đã xảy ra lỗi nghiêm trọng: {e}")


if __name__ == "__main__":
    main()