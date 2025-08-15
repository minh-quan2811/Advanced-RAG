import logging
from pathlib import Path
from typing import List
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes, get_root_nodes
from llama_index.core.schema import Document
from llama_cloud_services import LlamaParse
from typing import List, Union, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import asyncio
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import threading

class LLamaParser:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(LLamaParser, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.llama_parse = LlamaParse(num_workers=4, verbose=True)
llama_parser = LLamaParser()

class DocumentChunker:
    def __init__(self, meta_extractor, chunk_sizes: List[int] = None, chunk_overlap: int = 100):
        """
        Initialize DocumentChunker
        
        Args:
            chunk_sizes: List of chunk sizes for hierarchical parsing (default: [8192, 4096, 1024])
            chunk_overlap: Overlap between chunks (default: 20)
        """
        self.meta_extractor = meta_extractor
        self.chunk_sizes = chunk_sizes if chunk_sizes else [8192, 4096, 1024]
        self.chunk_overlap = chunk_overlap
        self.processing_status = {}  # Track background processing status
        self.processed_documents = {}  # Store processed documents
        self.executor = ThreadPoolExecutor(max_workers=2)  # For background processing
        
        self.node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=self.chunk_sizes,
            chunk_overlap=self.chunk_overlap
        )
        
        logger.info(f"DocumentChunker initialized with chunk sizes: {self.chunk_sizes}")

    def load_documents_from_folder(self, data_folder: str) -> List[Document]:
        data_folder_path = Path(data_folder)
        
        logger.info(f"Loading documents from {data_folder_path}")

        if not data_folder_path.exists():
            raise FileNotFoundError(f"Folder {data_folder_path} does not exist!")

        reader = SimpleDirectoryReader(
            input_dir=str(data_folder_path),
            recursive=True,
            required_exts=[".docx", ".doc", ".txt", ".pdf"]
        )

        documents = reader.load_data()
        logger.info(f"Loaded {len(documents)} documents")

        return documents

    def load_documents_from_text(self, text_content: str, metadata: dict = None) -> List[Document]:
        logger.info("Loading document from text content")
        
        if not text_content.strip():
            raise ValueError("Text content is empty!")
        
        document = Document(
            text=text_content,
            metadata=metadata or {}
        )
        
        return [document]

    async def _parse_file_background(self, file_path: str, task_id: str) -> None:
        """Background task to parse files using LlamaParse"""
        try:
            self.processing_status[task_id] = "parsing"
            logger.info(f"Starting background parsing for {file_path} (task_id: {task_id})")
            
            # Directly await the async aparse method
            parse_result = await llama_parser.llama_parse.aparse(str(file_path))
            
            # Process the parse result
            if hasattr(parse_result, "text"):
                documents = [Document(text=parse_result.text, metadata={})]
            elif isinstance(parse_result, list):
                documents = [Document(text=item.text, metadata={}) for item in parse_result if hasattr(item, "text")]
            else:
                logger.error(f'Unexpected parse result for {file_path}: type {type(parse_result)}')
                raise ValueError("Unable to convert parse_result to Document(s)")

            # Store the processed documents
            self.processed_documents[task_id] = documents
            self.processing_status[task_id] = "completed"
            logger.info(f"Background parsing completed for {file_path} (task_id: {task_id})")
            
        except Exception as e:
            logger.error(f"Error in background parsing for {file_path}: {e}")
            self.processing_status[task_id] = "error"
            self.processed_documents[task_id] = None

    async def load_documents_from_file(self, file_path: str) -> Union[List[Document], Dict[str, str]]:
        file_path_obj = Path(file_path)
        
        logger.info(f"Loading document from {file_path_obj}")

        if not file_path_obj.exists():
            raise FileNotFoundError(f"File {file_path_obj} does not exist!")
        
        file_extension = file_path_obj.suffix.lower()
        
        if file_extension in [".xlsx", ".xls"]:
            task_id = f"{file_path_obj.stem}_{hash(str(file_path_obj))}"
            self.processing_status[task_id] = "queued"
            
            # Start background parsing
            asyncio.create_task(self._parse_file_background(str(file_path_obj), task_id))
            
            return {
                "status": "processing",
                "message": f"File {file_path_obj.name} is being processed in the background",
                "task_id": task_id
            }
        
        # For other file types that need LlamaParse
        elif file_extension not in [".docx", ".doc", ".txt", ".pdf"]:
            # Directly await the async aparse method
            parse_result = await llama_parser.llama_parse.aparse(str(file_path_obj))
            
            if hasattr(parse_result, "text"):
                documents = [Document(text=parse_result.text, metadata={})]
            elif isinstance(parse_result, list):
                documents = [Document(text=item.text, metadata={}) for item in parse_result if hasattr(item, "text")]
            else:
                logger.error(f'Parse result: {parse_result}')
                raise ValueError("Unable to convert parse_result to Document(s)")

            logger.info(f"Parsed {len(documents)} documents from {file_path_obj}")
            return documents
            
        # For standard file types, use SimpleDirectoryReader
        else:
            reader = SimpleDirectoryReader(input_files=[str(file_path_obj)])
            documents = reader.load_data()
        
        logger.info(f"Loaded {len(documents)} documents from file")
        return documents

    def get_processing_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a background processing task"""
        status = self.processing_status.get(task_id, "not_found")
        result = {
            "task_id": task_id,
            "status": status
        }
        
        if status == "completed" and task_id in self.processed_documents:
            result["documents"] = self.processed_documents[task_id]
            result["document_count"] = len(self.processed_documents[task_id]) if self.processed_documents[task_id] else 0
        elif status == "error":
            result["error"] = "Processing failed"
            
        return result

    def get_processed_documents(self, task_id: str) -> Union[List[Document], None]:
        """Retrieve processed documents by task_id"""
        if task_id in self.processed_documents and self.processing_status.get(task_id) == "completed":
            return self.processed_documents[task_id]
        return None

    def create_hierarchical_chunks(self, documents: List[Document], metadata: dict) -> List:
        logger.info("Creating hierarchical chunks...")

        if not documents:
            raise ValueError("No documents provided for chunking!")

        nodes = self.node_parser.get_nodes_from_documents(documents)

        leaf_nodes = get_leaf_nodes(nodes)
        root_nodes = get_root_nodes(nodes)
        
        logger.info(f"Created {len(nodes)} hierarchical nodes")
        logger.info(f"Leaf nodes: {len(leaf_nodes)}, Root nodes: {len(root_nodes)}")
        
        return self.apply_metadata(nodes, metadata)

    def get_leaf_nodes(self, nodes: List) -> List:
        return get_leaf_nodes(nodes)

    def get_root_nodes(self, nodes: List) -> List:
        return get_root_nodes(nodes)

    def chunk_from_folder(self, data_folder: str, metadata: dict = None) -> List:
        documents = self.load_documents_from_folder(data_folder)
        return self.create_hierarchical_chunks(documents, metadata)

    def chunk_from_text(self, text_content: str, metadata: dict = None) -> List:
        documents = self.load_documents_from_text(text_content, metadata)
        return self.create_hierarchical_chunks(documents, metadata)

    async def chunk_from_file(self, file_path: str, metadata: dict = None, db_categories = None, db_keywords = None) -> Union[List, Dict[str, str]]:
        documents_or_status = await self.load_documents_from_file(file_path)
        
        # If it's a processing status (for xlsx files), return it immediately
        if isinstance(documents_or_status, dict) and "status" in documents_or_status:
            return documents_or_status
        
        # Otherwise, process the documents normally
        documents = documents_or_status

        if len(documents) > 1:
            merged_document = Document(
                text="\n\n".join(doc.text for doc in documents),
                metadata={k: v for doc in documents for k, v in doc.metadata.items()}
            )
            documents = [merged_document]

        if not metadata:
            metadata = self.meta_extractor.extract_metadata(documents[0].text, db_categories=db_categories, db_keywords=db_keywords) if documents else {}

        if metadata == None:
            return None

        return self.create_hierarchical_chunks(documents, metadata)

    async def chunk_from_file_with_task_id(self, task_id: str, metadata: dict = None, db_categories = None, db_keywords = None) -> Union[List, Dict[str, str]]:
        """Process chunks from a completed background task"""
        documents = self.get_processed_documents(task_id)
        
        if documents is None:
            status = self.get_processing_status(task_id)
            return {
                "status": status["status"],
                "message": f"Task {task_id} is not ready or failed",
                "task_id": task_id
            }

        if len(documents) > 1:
            merged_document = Document(
                text="\n\n".join(doc.text for doc in documents),
                metadata={k: v for doc in documents for k, v in doc.metadata.items()}
            )
            documents = [merged_document]

        if not metadata:
            metadata = self.meta_extractor.extract_metadata(documents[0].text, db_categories=db_categories, db_keywords=db_keywords) if documents else {}

        if metadata == None:
            return None

        return self.create_hierarchical_chunks(documents, metadata)

    def cleanup_completed_tasks(self, max_age_hours: int = 24) -> None:
        """Clean up old completed/failed tasks to prevent memory buildup"""
        import time
        current_time = time.time()
        tasks_to_remove = []
        
        for task_id, status in self.processing_status.items():
            if status in ["completed", "error"]:
                # For simplicity, remove tasks after they've been in the system
                # In a production system, you'd want to track creation time
                tasks_to_remove.append(task_id)
        
        # Remove old tasks (keeping only the most recent ones)
        if len(tasks_to_remove) > 10:  # Keep only 10 most recent completed tasks
            for task_id in tasks_to_remove[:-10]:
                self.processing_status.pop(task_id, None)
                self.processed_documents.pop(task_id, None)
                
        logger.info(f"Cleaned up {len(tasks_to_remove[:-10]) if len(tasks_to_remove) > 10 else 0} old tasks")

    def apply_metadata(self, nodes, metadata: dict):
        for node in self.get_leaf_nodes(nodes):
            node.metadata = metadata

        return nodes

    def analyze_hierarchical_structure(self, nodes: List) -> dict:
        analysis = {
            'total_nodes': len(nodes),
            'leaf_nodes': [],
            'parent_nodes': [],
            'root_nodes': []
        }
        
        for node in nodes:
            has_children = hasattr(node, 'child_nodes') and node.child_nodes
            has_parent = hasattr(node, 'parent_node') and node.parent_node
            
            if has_children and not has_parent:
                analysis['root_nodes'].append(node)
            elif has_children and has_parent:
                analysis['parent_nodes'].append(node)
            elif not has_children and has_parent:
                analysis['leaf_nodes'].append(node)
            else:
                analysis['leaf_nodes'].append(node)
        
        return analysis
    
    def get_node_text_preview(self, node_or_info, max_length: int = 200) -> str:
        try:
            if hasattr(node_or_info, 'node'):
                actual_node = node_or_info.node
                text = actual_node.text if hasattr(actual_node, 'text') else str(actual_node)
            elif hasattr(node_or_info, 'text'):
                text = node_or_info.text
            else:
                text = str(node_or_info)
            
            return text[:max_length] + "..." if len(text) > max_length else text
        except Exception:
            return "Unable to extract text"
    
    def print_hierarchical_structure(self, nodes: List, max_text_length: int = 200):
        analysis = self.analyze_hierarchical_structure(nodes)
        
        print("\n" + "="*80)
        print("HIERARCHICAL CHUNKING ANALYSIS")
        print("="*80)
        print(f"Total nodes: {analysis['total_nodes']}")
        print(f"Root nodes: {len(analysis['root_nodes'])}")
        print(f"Parent nodes: {len(analysis['parent_nodes'])}")
        print(f"Leaf nodes: {len(analysis['leaf_nodes'])}")
        
        if analysis['root_nodes']:
            print("\n" + "-"*60)
            print("ROOT NODES (Top-level chunks)")
            print("-"*60)
            for i, node in enumerate(analysis['root_nodes']):
                print(f"\n[ROOT {i+1}] ID: {node.node_id}")
                text_preview = node.text[:max_text_length] + "..." if len(node.text) > max_text_length else node.text
                print(f"Text: {text_preview}")
                print(f"Length: {len(node.text)} chars")
                
                if hasattr(node, 'child_nodes') and node.child_nodes:
                    print(f"Children: {len(node.child_nodes)} nodes")
                    for j, child in enumerate(node.child_nodes):
                        child_preview = self.get_node_text_preview(child, 300)
                        print(f"  ├─ Child {j+1}: {child_preview}")
        
        if analysis['parent_nodes']:
            print("\n" + "-"*60)
            print("PARENT NODES (Intermediate-level chunks)")
            print("-"*60)
            for i, node in enumerate(analysis['parent_nodes']):
                print(f"\n[PARENT {i+1}] ID: {node.node_id}")
                text_preview = node.text[:max_text_length] + "..." if len(node.text) > max_text_length else node.text
                print(f"Text: {text_preview}")
                print(f"Length: {len(node.text)} chars")
                
                if hasattr(node, 'parent_node') and node.parent_node:
                    parent_preview = self.get_node_text_preview(node.parent_node, 300)
                    print(f"Parent: {parent_preview}")
                
                if hasattr(node, 'child_nodes') and node.child_nodes:
                    print(f"Children: {len(node.child_nodes)} nodes")
                    for j, child in enumerate(node.child_nodes):
                        child_preview = self.get_node_text_preview(child, 300)
                        print(f"  ├─ Child {j+1}: {child_preview}")
        
        if analysis['leaf_nodes']:
            print("\n" + "-"*60)
            print("LEAF NODES (Finest-level chunks)")
            print("-"*60)
            for i, node in enumerate(analysis['leaf_nodes'][:5]): 
                print(f"\n[LEAF {i+1}] ID: {node.node_id}")
                text_preview = node.text[:max_text_length] + "..." if len(node.text) > max_text_length else node.text
                print(f"Text: {text_preview}")
                print(f"Length: {len(node.text)} chars")
                
                if hasattr(node, 'parent_node') and node.parent_node:
                    parent_preview = self.get_node_text_preview(node.parent_node, 300)
                    print(f"Parent: {parent_preview}")
            
            if len(analysis['leaf_nodes']) > 5:
                print(f"\n... and {len(analysis['leaf_nodes']) - 5} more leaf nodes")
        
        print("\n" + "="*80)

    @staticmethod
    def to_list_node(nodes):
        def node_to_dict(node):
            return {
                "node_id": getattr(node, "node_id", None),
                "text": getattr(node, "text", None),
                "parent_node_id": getattr(node.parent_node, "node_id", None) if hasattr(node, "parent_node") and node.parent_node else None,
                "child_node_ids": [getattr(child, "node_id", None) for child in getattr(node, "child_nodes", [])] if hasattr(node, "child_nodes") and node.child_nodes else [],
            }
        return [node_to_dict(n) for n in nodes]

