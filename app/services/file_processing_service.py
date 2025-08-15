from fastapi import UploadFile
import os
import shutil
from ..utils.document_chunker import DocumentChunker
from ..services.storage_service import NodeStorageHandler
from ..utils.metadata_extractor import MetadataExtractor

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(UPLOAD_DIR, exist_ok=True)

meta_extractor = MetadataExtractor()
document_chunker = DocumentChunker(meta_extractor=meta_extractor)
storage_handler = NodeStorageHandler(collection_name="sailing_docs")

def process_uploaded_file(file: UploadFile):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    nodes = document_chunker.chunk_from_file(file_path)

    if nodes is None:
        os.remove(file_path)
        return {"error": "Could not process file."}

    storage_handler.build_automerging_index(nodes=nodes)

    return {"filename": file.filename}
