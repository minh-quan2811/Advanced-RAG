from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil

from storing import NodeStorageHandler
from data_chunker import DocumentChunker
from new_metadata_extractor import MetadataExtractor

app = FastAPI()

meta_extractor = MetadataExtractor()

document_chunker = DocumentChunker(meta_extractor=meta_extractor)
storage_handler = NodeStorageHandler(collection_name="sailing_docs_simple")


UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    allowed_ext = {".docx", ".doc", ".pdf", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail="File type not allowed.")
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    nodes = document_chunker.chunk_from_file(file_path)

    if nodes == None:
        os.remove(file_path)

    storage_handler.build_automerging_index(nodes=nodes)

    return {"filename": file.filename, "saved_to": file_path}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
