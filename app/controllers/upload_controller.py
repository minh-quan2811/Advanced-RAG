from fastapi import APIRouter, File, UploadFile, HTTPException
from ..services.file_processing_service import process_uploaded_file
import os

router = APIRouter()

@router.post("/upload")
def upload_file(file: UploadFile = File(...)):
    allowed_ext = {".docx", ".doc", ".pdf", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail="File type not allowed.")
    
    result = process_uploaded_file(file)
    return result
