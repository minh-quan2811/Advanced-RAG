import logging
from pathlib import Path
from typing import List
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes, get_root_nodes
from llama_index.core.schema import Document
# from llama_index.core.llama_parse import LlamaParse
# from app.utils.environment_setup import EnvironmentSetup

import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



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

    def load_documents_from_file(self, file_path: str) -> List[Document]:
        file_path_obj = Path(file_path)
        
        logger.info(f"Loading document from {file_path_obj}")

        if not file_path_obj.exists():
            raise FileNotFoundError(f"File {file_path_obj} does not exist!")

        reader = SimpleDirectoryReader(input_files=[str(file_path_obj)])
        documents = reader.load_data()
        
        logger.info(f"Loaded {len(documents)} documents from file")
        return documents

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

    def chunk_from_file(self, file_path: str, metadata: dict = None, db_categories = None, db_keywords = None) -> List:
        documents = self.load_documents_from_file(file_path)

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
