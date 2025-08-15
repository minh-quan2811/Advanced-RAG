from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from ..services.file_processing_service import process_uploaded_file, UPLOAD_DIR
import os
import shutil

router = APIRouter()

@router.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save the file immediately to a persistent location
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
    finally:
        file.file.close()

    # Pass the file path to the background task, not the UploadFile object
    background_tasks.add_task(process_uploaded_file, file_path=file_path, filename=file.filename)
    
    return {"filename": file.filename, "status": "processing in background"}
