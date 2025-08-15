import asyncio
from fastapi import UploadFile
import os
import shutil
from ..utils.document_chunker import DocumentChunker
from ..services.storage_service import NodeStorageHandler
from ..utils.metadata_extractor import MetadataExtractor
from concurrent.futures import ProcessPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Use a ProcessPoolExecutor to run blocking tasks in a separate process
executor = ProcessPoolExecutor(max_workers=os.cpu_count())

def _process_file_sync(file_path: str, filename: str):
    """
    Synchronous function to handle the entire file processing logic.
    This will be run in a separate process to avoid blocking the main event loop.
    """
    try:
        logger.info(f"[{os.getpid()}] Starting synchronous processing for {filename}")
        
        # Re-initialize components for the new process
        meta_extractor = MetadataExtractor()
        document_chunker = DocumentChunker(meta_extractor=meta_extractor)
        storage_handler = NodeStorageHandler(collection_name="sailing_docs")

        # Since this is a sync function, we need to run the async code in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        chunk_result = loop.run_until_complete(document_chunker.chunk_from_file(file_path))

        nodes = None
        if isinstance(chunk_result, dict) and "task_id" in chunk_result:
            task_id = chunk_result["task_id"]
            logger.info(f"File '{filename}' processing started with task_id: {task_id}")
            
            while True:
                status_info = document_chunker.get_processing_status(task_id)
                status = status_info.get("status")
                
                if status == "completed":
                    logger.info(f"Task {task_id} completed. Retrieving nodes.")
                    nodes = loop.run_until_complete(document_chunker.chunk_from_file_with_task_id(task_id))
                    break
                elif status == "error":
                    logger.error(f"Error processing task {task_id}.")
                    return
                
                # This is a blocking sleep, which is fine since we are in a separate process
                loop.run_until_complete(asyncio.sleep(2))
        else:
            nodes = chunk_result

        if nodes is None:
            logger.error(f"Could not process file {filename}.")
            return

        storage_handler.build_automerging_index(nodes=nodes)
        logger.info(f"Successfully processed and stored {filename}")

    except Exception as e:
        logger.error(f"Error processing file {filename}: {e}", exc_info=True)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.info(f"[{os.getpid()}] Finished synchronous processing for {filename}")
        if 'loop' in locals() and not loop.is_closed():
            loop.close()

async def process_uploaded_file(file_path: str, filename: str):
    """
    Asynchronous wrapper that submits the blocking task to the process pool.
    """
    loop = asyncio.get_running_loop()
    logger.info(f"Submitting {filename} to process pool.")
    # Run the synchronous function in the process pool
    await loop.run_in_executor(
        executor, _process_file_sync, file_path, filename
    )
