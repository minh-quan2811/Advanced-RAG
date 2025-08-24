import os
import logging
from dotenv import load_dotenv
from typing import List

from pydantic import BaseModel, Field
from llama_index.core import PromptTemplate, QueryBundle
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters, FilterCondition
from llama_index.postprocessor.cohere_rerank import CohereRerank

from storage_handler import NodeStorageHandler

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryFilters(BaseModel):
    """Data model for query filters."""
    categories: List[str] = Field(description="List of categories to filter on.")
    keywords: List[str] = Field(description="List of keywords to filter on.")

def get_valid_filters_from_db():
    """Mocks fetching available filter options from a database."""
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

class QueryPipeline:
    """
    Handles all query-time operations using a pre-built NodeStorageHandler.
    """
    def __init__(self, storage_handler: NodeStorageHandler):
        if not storage_handler.index:
            raise ValueError("NodeStorageHandler must be built or loaded before creating a QueryPipeline.")

        self.storage_handler = storage_handler
        self.llm = storage_handler.llm

        cohere_api_key = os.getenv("COHERE_API_KEY")
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY must be set in the environment.")

        self.reranker = CohereRerank(api_key=cohere_api_key,
                                     top_n=10,
                                     model="rerank-multilingual-v3.0")
        logger.info("QueryPipeline initialized with Cohere Reranker.")

    def retrieve(self, query_str: str, filters: QueryFilters, similarity_top_k: int = 30):
        """
        Performs retrieval and reranking for a given query string.
        Returns a list of the top reranked nodes.
        """
        # Dynamic filters
        metadata_filters_list = []
        if filters.categories:
            metadata_filters_list.extend([ExactMatchFilter(key="Category", value=c) for c in filters.categories])
        if filters.keywords:
            metadata_filters_list.extend([ExactMatchFilter(key="Keywords", value=k) for k in filters.keywords])

        # Retriever with filters
        retriever = self.storage_handler.index.as_retriever(
            similarity_top_k=similarity_top_k,
            filters=MetadataFilters(filters=metadata_filters_list,
                                    condition=FilterCondition.OR)
        )

        # Initial retrieval
        logger.info("Retrieving initial documents from vector store...")
        retrieved_nodes = retriever.retrieve(query_str)
        logger.info(f"Retrieved {len(retrieved_nodes)} initial nodes.")

        if not retrieved_nodes:
            return []

        # Reranking
        logger.info("Reranking retrieved documents with Cohere...")
        query_bundle = QueryBundle(query_str=query_str)
        reranked_nodes = self.reranker.postprocess_nodes(
            retrieved_nodes, query_bundle=query_bundle
        )
        logger.info(f"Reranked down to {len(reranked_nodes)} nodes.")

        return reranked_nodes

# def main():
#     """Main function to run the interactive retrieval session."""
#     try:
#         # Setup the storage and index
#         print("=== INITIALIZING STORAGE HANDLER ===")
#         storage_handler = NodeStorageHandler(collection_name="sailing_test")
#         storage_handler.build_automerging_index(persist_dir="./storage")

#         print("\n=== INITIALIZING QUERY PIPELINE ===")
#         pipeline = QueryPipeline(storage_handler)

#         print("\n" + "="*50)
#         print("=== INTERACTIVE RETRIEVAL MODE ===")
#         print("Enter your query to retrieve the most relevant documents. Type 'quit' to exit.")
#         print("="*50)
        
#         while True:
#             user_query = input("\nQuery: ").strip()
#             if user_query.lower() in ['quit', 'exit', 'q']:
#                 print("\nExiting. Goodbye!")
#                 break
#             if not user_query:
#                 print("Please enter a query.")
#                 continue
                
#             try:
#                 print("\nRetrieving and reranking...")
#                 reranked_nodes = pipeline.retrieve(user_query)
                
#                 print("\nTop Reranked Documents:\n")
#                 if not reranked_nodes:
#                     print("--> No relevant documents found.")
#                 else:
#                     for i, node_with_score in enumerate(reranked_nodes):
#                         print(f"--- Document {i+1} ---")
#                         print(f"Metadata: {node_with_score.node.metadata}")
#                         print(f"Content: \n{node_with_score.node.get_content().strip()}")
#                         print("-" * 80)
                
#             except Exception as e:
#                 logger.error(f"Error during query processing: {e}", exc_info=True)
#                 print(f"An error occurred: {e}")

#     except Exception as e:
#         logger.error(f"A critical error occurred in main: {e}", exc_info=True)
#         print(f"A critical error occurred: {e}")


# if __name__ == "__main__":
#     main()